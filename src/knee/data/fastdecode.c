/*
 * C decoding kernels (knee.data.cfast), bit-identical to the Python/NumPy path.
 *
 * Compile without FMA contraction and without fast-math: every floating-point operation must be
 * exactly the rounded one NumPy performs (`-O2 -ffp-contract=off`).
 *
 *   tails_f32       one pass: the values <= thr_lo and >= thr_hi, in order
 *                   (`flat[flat <= t]`, `flat[flat >= t]` of cache._tail_percentiles)
 *   normalize_u8    (float64(x) - lo) * scale, clipped to [0, 255], converted by truncation, on an
 *                   output with arbitrary strides (cache._to_uint8)
 *   pix_to_f32      uncompressed pixels -> float32, with masking or sign extension of the unused
 *                   bits (dicom._read_native_pixels)
 *   walk_order      the ordering tags from the header (dicom.fast_order_tags)
 *   walk_pixels     the group 0028 tags and the position of PixelData
 *                   (dicom._read_native_pixels)
 *   resize_area_f32 cv2.resize(..., INTER_AREA) bit for bit for single-channel float32, on OpenCV
 *                   4.x's generic path (ResizeArea_Invoker; cache._resize_into)
 *
 * The hot functions have an AVX/AVX2 version chosen at runtime, with the same operations in the
 * same order for every element: the vectors run over independent elements (columns, or rows in the
 * resize), never a reordered sum. Without AVX the scalar code is used.
 */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <cpuid.h>
#include <immintrin.h>
#define KNEE_X86 1
/*
 * CPUID + XGETBV (no __builtin_cpu_supports: it needs compiler-rt symbols that zig's DLL on Windows
 * does not link): AVX declared by the CPU and YMM registers enabled by the OS.
 */
static int cpu_avx(void) {
    unsigned a, b, c, d, lo, hi;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return 0;
    if (!((c >> 27) & 1) || !((c >> 28) & 1)) return 0;          /* OSXSAVE, AVX */
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (lo & 6) == 6;                                         /* XMM and YMM state */
}
static int cpu_avx2(void) {
    unsigned a, b, c, d;
    if (!cpu_avx()) return 0;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return 0;
    return (b >> 5) & 1;
}
#else
static int cpu_avx(void) { return 0; }
static int cpu_avx2(void) { return 0; }
#endif
static int g_avx = -1, g_avx2 = -1;
static int has_avx(void) { if (g_avx < 0) g_avx = cpu_avx(); return g_avx; }
static int has_avx2(void) { if (g_avx2 < 0) g_avx2 = cpu_avx2(); return g_avx2; }

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

/* 1 = use the vectors where available, 0 = scalar only (the tests compare the two paths). */
static int g_simd = 1;
EXPORT void knee_set_simd(int64_t on) { g_simd = (int)on; }

EXPORT int64_t knee_abi(void) { return 1; }

static void tails_scalar(const float *flat, int64_t n, float thr_lo, float thr_hi,
                         float *low, int64_t cap_lo, float *high, int64_t cap_hi,
                         int64_t *out_counts) {
    int64_t n_lo = 0, n_hi = 0;
    for (int64_t i = 0; i < n; i++) {
        float v = flat[i];
        if (v <= thr_lo) {
            if (n_lo < cap_lo) low[n_lo] = v;
            n_lo++;
        }
        if (v >= thr_hi) {
            if (n_hi < cap_hi) high[n_hi] = v;
            n_hi++;
        }
    }
    out_counts[0] = n_lo;
    out_counts[1] = n_hi;
}

/*
 * out[k*s0 + i*s1 + j*s2] (strides in elements, possibly negative) from the contiguous stack
 * (K, H, W).
 */
static void normalize_scalar(const float *stack, int64_t kk, int64_t hh, int64_t ww, double lo,
                             double scale, uint8_t *out, int64_t s0, int64_t s1, int64_t s2) {
    for (int64_t k = 0; k < kk; k++) {
        for (int64_t i = 0; i < hh; i++) {
            const float *row = stack + (k * hh + i) * ww;
            uint8_t *o = out + k * s0 + i * s1;
            for (int64_t j = 0; j < ww; j++) {
                double v = ((double)row[j] - lo) * scale;
                if (v < 0.0) v = 0.0;
                else if (v > 255.0) v = 255.0;
                o[j * s2] = (uint8_t)v;
            }
        }
    }
}

/*
 * bits 8 or 16; rep 0 unsigned, 1 signed; shift = bits - stored if pydicom fixes the unused bits
 * (0 otherwise). Unsigned: mask of the low bits; signed: shift left then arithmetic shift right,
 * like np.left_shift + np.right_shift on the integer type.
 */
static void pix_scalar(const uint8_t *src, int64_t n, int64_t bits, int64_t rep, int64_t shift,
                       float *dst) {
    if (bits == 16) {
        if (!rep) {
            uint16_t mask = shift > 0 ? (uint16_t)((1u << (16 - shift)) - 1u) : (uint16_t)0xFFFFu;
            for (int64_t i = 0; i < n; i++) {
                uint16_t v;
                memcpy(&v, src + 2 * i, 2);
                dst[i] = (float)(uint16_t)(v & mask);
            }
        } else {
            for (int64_t i = 0; i < n; i++) {
                uint16_t u;
                memcpy(&u, src + 2 * i, 2);
                int16_t v = (int16_t)(uint16_t)(u << shift);
                v = (int16_t)(v >> shift);
                dst[i] = (float)v;
            }
        }
    } else {
        if (!rep) {
            uint8_t mask = shift > 0 ? (uint8_t)((1u << (8 - shift)) - 1u) : (uint8_t)0xFFu;
            for (int64_t i = 0; i < n; i++) dst[i] = (float)(uint8_t)(src[i] & mask);
        } else {
            for (int64_t i = 0; i < n; i++) {
                int8_t v = (int8_t)(uint8_t)(src[i] << shift);
                v = (int8_t)(v >> shift);
                dst[i] = (float)v;
            }
        }
    }
}

static uint16_t rd16(const uint8_t *b, int64_t p) { return (uint16_t)(b[p] | (b[p + 1] << 8)); }
static uint32_t rd32(const uint8_t *b, int64_t p) {
    return (uint32_t)b[p] | ((uint32_t)b[p + 1] << 8) | ((uint32_t)b[p + 2] << 16) |
           ((uint32_t)b[p + 3] << 24);
}

static int vr_long(const uint8_t *vr) {
    static const char *L[] = {"OB", "OW", "OF", "OD", "OL", "OV", "SQ", "SV", "UC", "UN",
                              "UR", "UT", "UV"};
    for (int i = 0; i < 13; i++)
        if (vr[0] == (uint8_t)L[i][0] && vr[1] == (uint8_t)L[i][1]) return 1;
    return 0;
}

/* The transfer syntax without trailing padding ("\0" and " "), like `rstrip(b"\x00 ")`. */
static int ts_is(const uint8_t *b, int64_t val, int64_t len, const char *s) {
    while (len > 0 && (b[val + len - 1] == 0 || b[val + len - 1] == ' ')) len--;
    int64_t m = (int64_t)strlen(s);
    return len == m && memcmp(b + val, s, (size_t)m) == 0;
}

#define TS_EXPLICIT_LE "1.2.840.10008.1.2.1"
#define TS_IMPLICIT_LE "1.2.840.10008.1.2"
#define TS_BIG_ENDIAN "1.2.840.10008.1.2.2"
#define UNDEF 0xFFFFFFFFu

/*
 * out: [ipp_off, ipp_len, iop_off, iop_len, inst_off, inst_len] (-1 = absent).
 * Returns 0, or -1 where `fast_order_tags` returns None.
 */
EXPORT int64_t walk_order(const uint8_t *buf, int64_t n, int64_t *out) {
    for (int i = 0; i < 6; i++) out[i] = -1;
    if (n < 140 || memcmp(buf + 128, "DICM", 4) != 0) return -1;
    int64_t pos = 132;
    int explicit_vr = 1;
    int64_t ts_off = -1, ts_len = 0;     /* ts "not None" = ts_off >= 0 */
    int broke = 0;
    while (pos + 8 <= n) {
        uint16_t group = rd16(buf, pos), elem = rd16(buf, pos + 2);
        if (group != 0x0002 && ts_off >= 0) {
            if (ts_is(buf, ts_off, ts_len, TS_BIG_ENDIAN)) return -1;
            explicit_vr = !ts_is(buf, ts_off, ts_len, TS_IMPLICIT_LE);
            ts_off = -1;
        } else if (group != 0x0002 && pos == 132) {
            return -1;
        }
        if (group > 0x0020 || (group == 0x0020 && elem > 0x0037)) {
            broke = 1;
            break;
        }
        uint32_t length;
        int64_t hdr;
        if (group == 0x0002 || explicit_vr) {
            if (vr_long(buf + pos + 4)) {
                if (pos + 12 > n) return -1;   /* struct.unpack_from would fail: exception */
                length = rd32(buf, pos + 8);
                hdr = 12;
            } else {
                length = rd16(buf, pos + 6);
                hdr = 8;
            }
        } else {
            length = rd32(buf, pos + 4);
            hdr = 8;
        }
        if (length == UNDEF) return -1;
        int64_t val = pos + hdr, end = val + (int64_t)length;
        if (end > n) return -1;
        if (group == 0x0002 && elem == 0x0010) {
            ts_off = val;
            ts_len = (int64_t)length;
        } else if (group == 0x0020) {
            if (elem == 0x0013) { out[4] = val; out[5] = (int64_t)length; }
            else if (elem == 0x0032) { out[0] = val; out[1] = (int64_t)length; }
            else if (elem == 0x0037) { out[2] = val; out[3] = (int64_t)length; }
        }
        pos = end;
    }
    if (!broke) return -1;
    if (ts_off >= 0) return -1;
    return 0;
}

/*
 * Skips one Explicit VR LE element (dicom._skip_element), including undefined-length sequences.
 * Returns the position after the element or -1 (Python's exceptions).
 */
static int64_t skip_element(const uint8_t *b, int64_t pos, int64_t n, int depth);

static int64_t skip_undefined_item(const uint8_t *b, int64_t pos, int64_t n, int depth) {
    while (pos + 8 <= n) {
        uint16_t group = rd16(b, pos), elem = rd16(b, pos + 2);
        if (group == 0xFFFE) {
            if (elem != 0xE00D) return -1;
            return pos + 8;
        }
        pos = skip_element(b, pos, n, depth + 1);
        if (pos < 0) return -1;
    }
    return -1;
}

static int64_t skip_undefined_sequence(const uint8_t *b, int64_t pos, int64_t n, int depth) {
    while (pos + 8 <= n) {
        uint16_t group = rd16(b, pos), elem = rd16(b, pos + 2);
        uint32_t length = rd32(b, pos + 4);
        if (group != 0xFFFE) return -1;
        pos += 8;
        if (elem == 0xE0DD) return pos;
        if (elem != 0xE000) return -1;
        pos = length == UNDEF ? skip_undefined_item(b, pos, n, depth + 1) : pos + (int64_t)length;
        if (pos < 0) return -1;
    }
    return -1;
}

static int64_t skip_element(const uint8_t *b, int64_t pos, int64_t n, int depth) {
    if (depth > 64) return -1;
    uint32_t length;
    int64_t hdr;
    int is_sq = b[pos + 4] == 'S' && b[pos + 5] == 'Q';
    if (vr_long(b + pos + 4)) {
        if (pos + 12 > n) return -1;
        length = rd32(b, pos + 8);
        hdr = 12;
    } else {
        length = rd16(b, pos + 6);
        hdr = 8;
    }
    if (length == UNDEF) {
        if (!is_sq) return -1;
        return skip_undefined_sequence(b, pos + hdr, n, depth + 1);
    }
    int64_t end = pos + hdr + (int64_t)length;
    if (end > n) return -1;
    return end;
}

/*
 * The _PIX_TAGS tags in order: 0002 0004 0008 0010 0011 0100 0101 0103 1052 1053.
 * out: [pix_off, pix_len] then per tag [kind, a, b]: kind 0 absent, 1 US (a = value), 2 bytes
 * (a = offset, b = length). Returns 0, or -1 where `_read_native_pixels` returns None or raises.
 */
EXPORT int64_t walk_pixels(const uint8_t *buf, int64_t n, int64_t *out) {
    static const uint16_t T[10] = {0x0002, 0x0004, 0x0008, 0x0010, 0x0011,
                                   0x0100, 0x0101, 0x0103, 0x1052, 0x1053};
    for (int i = 0; i < 32; i++) out[i] = 0;
    out[0] = out[1] = -1;
    if (n < 140 || memcmp(buf + 128, "DICM", 4) != 0) return -1;
    int64_t pos = 132, ts_off = -1, ts_len = 0;
    int ts_explicit = 0;
    while (pos + 8 <= n) {
        uint16_t group = rd16(buf, pos), elem = rd16(buf, pos + 2);
        if (group != 0x0002 && !ts_explicit) return -1;
        if (group == 0x7FE0 && elem == 0x0010) {
            const uint8_t *vr = buf + pos + 4;
            if (!((vr[0] == 'O' && vr[1] == 'W') || (vr[0] == 'O' && vr[1] == 'B'))) return -1;
            if (pos + 12 > n) return -1;
            uint32_t length = rd32(buf, pos + 8);
            if (length == UNDEF) return -1;
            out[0] = pos + 12;
            out[1] = (int64_t)length;
            return 0;
        }
        if (group == 0x0002 && elem == 0x0010) {
            uint16_t length = rd16(buf, pos + 6);
            int64_t len = length;
            if (pos + 8 + len > n) len = n - (pos + 8);   /* Python slicing stops */
            ts_off = pos + 8;
            ts_len = len;
            ts_explicit = ts_is(buf, ts_off, ts_len, TS_EXPLICIT_LE);
        } else if (group == 0x0028) {
            int t = -1;
            for (int i = 0; i < 10; i++)
                if (T[i] == elem) t = i;
            if (t >= 0) {
                const uint8_t *vr = buf + pos + 4;
                if (vr_long(vr)) return -1;
                uint16_t length = rd16(buf, pos + 6);
                int64_t val = pos + 8, len = length;
                if (val + len > n) len = n - val < 0 ? 0 : n - val;
                if (vr[0] == 'U' && vr[1] == 'S') {
                    if (length != 2) return -1;
                    if (val + 2 > n) return -1;        /* unpack_from on 2 bytes */
                    out[2 + 3 * t] = 1;
                    out[3 + 3 * t] = rd16(buf, val);
                } else {
                    out[2 + 3 * t] = 2;
                    out[3 + 3 * t] = val;
                    out[4 + 3 * t] = len;
                }
            }
        }
        pos = skip_element(buf, pos, n, 0);
        if (pos < 0) return -1;
    }
    return -1;                             /* PixelData never found */
}

/* ------------------------------------------------------------------ vector versions */
#ifdef KNEE_X86
__attribute__((target("avx")))
static void tails_avx(const float *flat, int64_t n, float thr_lo, float thr_hi,
                      float *low, int64_t cap_lo, float *high, int64_t cap_hi,
                      int64_t *out_counts) {
    /*
     * Eight values at a time are discarded together if none is in the tails (almost always);
     * otherwise they are passed in order as in the scalar code.
     */
    int64_t n_lo = 0, n_hi = 0, i = 0;
    const __m256 vlo = _mm256_set1_ps(thr_lo), vhi = _mm256_set1_ps(thr_hi);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(flat + i);
        __m256 m = _mm256_or_ps(_mm256_cmp_ps(v, vlo, _CMP_LE_OQ), _mm256_cmp_ps(v, vhi, _CMP_GE_OQ));
        if (!_mm256_movemask_ps(m)) continue;
        for (int64_t k = i; k < i + 8; k++) {
            float x = flat[k];
            if (x <= thr_lo) { if (n_lo < cap_lo) low[n_lo] = x; n_lo++; }
            if (x >= thr_hi) { if (n_hi < cap_hi) high[n_hi] = x; n_hi++; }
        }
    }
    for (; i < n; i++) {
        float x = flat[i];
        if (x <= thr_lo) { if (n_lo < cap_lo) low[n_lo] = x; n_lo++; }
        if (x >= thr_hi) { if (n_hi < cap_hi) high[n_hi] = x; n_hi++; }
    }
    out_counts[0] = n_lo;
    out_counts[1] = n_hi;
}

__attribute__((target("avx")))
static void normalize_avx(const float *stack, int64_t kk, int64_t hh, int64_t ww, double lo,
                          double scale, uint8_t *out, int64_t s0, int64_t s1, int64_t s2) {
    /*
     * (double(x) - lo) * scale, clipped to [0, 255], truncated: the same double operations four at a
     * time. Contiguous output (s2 = 1) or flipped (s2 = -1, right knees).
     */
    const __m256d vlo = _mm256_set1_pd(lo), vsc = _mm256_set1_pd(scale);
    const __m256d zero = _mm256_setzero_pd(), top = _mm256_set1_pd(255.0);
    for (int64_t k = 0; k < kk; k++) {
        for (int64_t i = 0; i < hh; i++) {
            const float *row = stack + (k * hh + i) * ww;
            uint8_t *o = out + k * s0 + i * s1;
            int64_t j = 0;
            if (s2 == 1 || s2 == -1) {
                for (; j + 8 <= ww; j += 8) {
                    __m256 f = _mm256_loadu_ps(row + j);
                    __m256d a = _mm256_cvtps_pd(_mm256_castps256_ps128(f));
                    __m256d b = _mm256_cvtps_pd(_mm256_extractf128_ps(f, 1));
                    a = _mm256_mul_pd(_mm256_sub_pd(a, vlo), vsc);
                    b = _mm256_mul_pd(_mm256_sub_pd(b, vlo), vsc);
                    a = _mm256_min_pd(_mm256_max_pd(a, zero), top);
                    b = _mm256_min_pd(_mm256_max_pd(b, zero), top);
                    __m128i w16 = _mm_packs_epi32(_mm256_cvttpd_epi32(a), _mm256_cvttpd_epi32(b));
                    uint64_t bytes = (uint64_t)_mm_cvtsi128_si64(_mm_packus_epi16(w16, w16));
                    if (s2 == 1) {
                        memcpy(o + j, &bytes, 8);
                    } else {
                        bytes = __builtin_bswap64(bytes);
                        memcpy(o - (j + 7), &bytes, 8);
                    }
                }
            }
            for (; j < ww; j++) {
                double v = ((double)row[j] - lo) * scale;
                if (v < 0.0) v = 0.0;
                else if (v > 255.0) v = 255.0;
                o[j * s2] = (uint8_t)v;
            }
        }
    }
}

__attribute__((target("avx2")))
static void pix_avx2(const uint8_t *src, int64_t n, int64_t bits, int64_t rep, int64_t shift,
                     float *dst) {
    int64_t i = 0;
    if (bits == 16 && !rep) {
        const __m128i mask = _mm_set1_epi16(
            (short)(shift > 0 ? (uint16_t)((1u << (16 - shift)) - 1u) : (uint16_t)0xFFFFu));
        for (; i + 8 <= n; i += 8) {
            __m128i v = _mm_and_si128(_mm_loadu_si128((const __m128i *)(src + 2 * i)), mask);
            _mm256_storeu_ps(dst + i, _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(v)));
        }
    } else if (bits == 16) {
        const __m128i sh = _mm_cvtsi32_si128((int)shift);
        for (; i + 8 <= n; i += 8) {
            __m128i v = _mm_loadu_si128((const __m128i *)(src + 2 * i));
            v = _mm_sra_epi16(_mm_sll_epi16(v, sh), sh);
            _mm256_storeu_ps(dst + i, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v)));
        }
    }
    if (i < n) pix_scalar(src + (bits == 16 ? 2 : 1) * i, n - i, bits, rep, shift, dst + i);
}
#endif

EXPORT void tails_f32(const float *flat, int64_t n, float thr_lo, float thr_hi,
                      float *low, int64_t cap_lo, float *high, int64_t cap_hi,
                      int64_t *out_counts) {
#ifdef KNEE_X86
    if (g_simd && has_avx()) {
        tails_avx(flat, n, thr_lo, thr_hi, low, cap_lo, high, cap_hi, out_counts);
        return;
    }
#endif
    tails_scalar(flat, n, thr_lo, thr_hi, low, cap_lo, high, cap_hi, out_counts);
}

EXPORT void normalize_u8(const float *stack, int64_t kk, int64_t hh, int64_t ww, double lo,
                         double scale, uint8_t *out, int64_t s0, int64_t s1, int64_t s2) {
#ifdef KNEE_X86
    if (g_simd && has_avx()) {
        normalize_avx(stack, kk, hh, ww, lo, scale, out, s0, s1, s2);
        return;
    }
#endif
    normalize_scalar(stack, kk, hh, ww, lo, scale, out, s0, s1, s2);
}

EXPORT void pix_to_f32(const uint8_t *src, int64_t n, int64_t bits, int64_t rep, int64_t shift,
                       float *dst) {
#ifdef KNEE_X86
    if (g_simd && bits == 16 && has_avx2()) {
        pix_avx2(src, n, bits, rep, shift, dst);
        return;
    }
#endif
    pix_scalar(src, n, bits, rep, shift, dst);
}

/*
 * ------------------------------------------------------------ area resize (OpenCV 4.x)
 * The generic path of cv::resize with INTER_AREA when both scales are >= 1 and not both integer
 * (the integer one is resizeAreaFast; upscaling is a linear interpolation; INTER_AREA does not go
 * through IPP because it is not exact). Tables as in computeResizeAreaTab (in double, float
 * weights), sums as in ResizeArea_Invoker: for every source row buf[dx] = 0 + S[si] * a + ... in
 * table order; for every output row sum = beta0 * buf0, then sum = sum + beta * buf. OpenCV redoes
 * the horizontal pass for every entry of the vertical table and accumulates in memory entry by
 * entry; here every source row is processed once, and the AVX vectors run over 8 ROWS together
 * (same weights, same order for every pixel), with the stack transposed in 8x8 blocks: no reordered
 * sum, bit-identical.
 * Returns 0; 1 if a scale is < 1, 2 if both are integer (the caller uses cv2), -1 on error.
 */
typedef struct { int si, di; float alpha; } dtab_t;

static int cv_round_d(double v) {
#ifdef KNEE_X86
    return _mm_cvtsd_si32(_mm_set_sd(v));
#else
    double r = v - (double)(int64_t)v;
    int i = (int)v;
    if (r > 0.5 || (r == 0.5 && (i & 1))) return i + 1;
    if (r < -0.5 || (r == -0.5 && (i & 1))) return i - 1;
    return i;
#endif
}
static int cv_floor_d(double v) { int i = (int)v; return i - (i > v); }
static int cv_ceil_d(double v) { int i = (int)v; return i + (i < v); }

static int area_tab(int ssize, int dsize, double scale, dtab_t *tab) {
    int k = 0;
    for (int dx = 0; dx < dsize; dx++) {
        double fsx1 = dx * scale;
        double fsx2 = fsx1 + scale;
        double rest = ssize - fsx1;
        double cellWidth = (rest < scale) ? rest : scale;            /* std::min(scale, rest) */
        int sx1 = cv_ceil_d(fsx1), sx2 = cv_floor_d(fsx2);
        sx2 = (ssize - 1 < sx2) ? ssize - 1 : sx2;
        sx1 = (sx2 < sx1) ? sx2 : sx1;
        if (sx1 - fsx1 > 1e-3) {
            if (k >= ssize * 2) return -1;
            tab[k].di = dx;
            tab[k].si = sx1 - 1;
            tab[k++].alpha = (float)((sx1 - fsx1) / cellWidth);
        }
        for (int sx = sx1; sx < sx2; sx++) {
            if (k >= ssize * 2) return -1;
            tab[k].di = dx;
            tab[k].si = sx;
            tab[k++].alpha = (float)(1.0 / cellWidth);
        }
        if (fsx2 - sx2 > 1e-3) {
            if (k >= ssize * 2) return -1;
            double m = fsx2 - sx2;
            m = (1. < m) ? 1. : m;
            m = (cellWidth < m) ? cellWidth : m;
            tab[k].di = dx;
            tab[k].si = sx2;
            tab[k++].alpha = (float)(m / cellWidth);
        }
    }
    return k;
}

static void area_h_rows(const float *src, int64_t sstride, int r0, int r1, int dw,
                        const int *xs, const int *xsi, const float *xal, float *H) {
    for (int r = r0; r < r1; r++) {
        const float *S = src + (int64_t)r * sstride;
        float *h = H + (int64_t)r * dw;
        for (int dx = 0; dx < dw; dx++) {
            float acc = 0.f;
            for (int t = xs[dx]; t < xs[dx + 1]; t++) acc = acc + S[xsi[t]] * xal[t];
            h[dx] = acc;
        }
    }
}

static void area_v_scalar(const float *H, int dw, int dh, const dtab_t *ytab, const int *tabofs,
                          float *dst, int64_t dstride) {
    for (int dy = 0; dy < dh; dy++) {
        int j0 = tabofs[dy], j1 = tabofs[dy + 1];
        float *o = dst + (int64_t)dy * dstride;
        for (int dx = 0; dx < dw; dx++) {
            float s = ytab[j0].alpha * H[(int64_t)ytab[j0].si * dw + dx];
            for (int j = j0 + 1; j < j1; j++) s = s + ytab[j].alpha * H[(int64_t)ytab[j].si * dw + dx];
            o[dx] = s;
        }
    }
}

#ifdef KNEE_X86
__attribute__((target("avx")))
static void tr8(__m256 *r) {
    __m256 t0 = _mm256_unpacklo_ps(r[0], r[1]), t1 = _mm256_unpackhi_ps(r[0], r[1]);
    __m256 t2 = _mm256_unpacklo_ps(r[2], r[3]), t3 = _mm256_unpackhi_ps(r[2], r[3]);
    __m256 t4 = _mm256_unpacklo_ps(r[4], r[5]), t5 = _mm256_unpackhi_ps(r[4], r[5]);
    __m256 t6 = _mm256_unpacklo_ps(r[6], r[7]), t7 = _mm256_unpackhi_ps(r[6], r[7]);
    __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    r[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
    r[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
    r[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
    r[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
    r[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
    r[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
    r[6] = _mm256_permute2f128_ps(s2, s6, 0x31);
    r[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
}

/* The horizontal pass 8 rows at a time (lane = row). Returns the rows done. */
__attribute__((target("avx")))
static int area_h_avx(const float *src, int64_t sstride, int sh, int sw, int dw,
                      const int *xs, const int *xsi, const float *xal, float *H, float *T) {
    int r0 = 0;
    for (; r0 + 8 <= sh; r0 += 8) {
        const float *S0 = src + (int64_t)r0 * sstride;
        int x = 0;
        for (; x + 8 <= sw; x += 8) {
            __m256 r[8];
            for (int k = 0; k < 8; k++) r[k] = _mm256_loadu_ps(S0 + (int64_t)k * sstride + x);
            tr8(r);
            for (int k = 0; k < 8; k++) _mm256_store_ps(T + (int64_t)(x + k) * 8, r[k]);
        }
        for (; x < sw; x++)
            for (int k = 0; k < 8; k++) T[(int64_t)x * 8 + k] = S0[(int64_t)k * sstride + x];
        int dx0 = 0;
        for (; dx0 + 8 <= dw; dx0 += 8) {
            __m256 acc[8];
            for (int d = 0; d < 8; d++) {
                const int dx = dx0 + d;
                __m256 a = _mm256_setzero_ps();
                for (int t = xs[dx]; t < xs[dx + 1]; t++)
                    a = _mm256_add_ps(a, _mm256_mul_ps(_mm256_load_ps(T + (int64_t)xsi[t] * 8),
                                                       _mm256_set1_ps(xal[t])));
                acc[d] = a;
            }
            tr8(acc);
            for (int k = 0; k < 8; k++) _mm256_storeu_ps(H + (int64_t)(r0 + k) * dw + dx0, acc[k]);
        }
        for (; dx0 < dw; dx0++) {
            __m256 a = _mm256_setzero_ps();
            for (int t = xs[dx0]; t < xs[dx0 + 1]; t++)
                a = _mm256_add_ps(a, _mm256_mul_ps(_mm256_load_ps(T + (int64_t)xsi[t] * 8),
                                                   _mm256_set1_ps(xal[t])));
            float tmp[8];
            _mm256_storeu_ps(tmp, a);
            for (int k = 0; k < 8; k++) H[(int64_t)(r0 + k) * dw + dx0] = tmp[k];
        }
    }
    return r0;
}

__attribute__((target("avx")))
static void area_v_avx(const float *H, int dw, int dh, const dtab_t *ytab, const int *tabofs,
                       float *dst, int64_t dstride) {
    for (int dy = 0; dy < dh; dy++) {
        const int j0 = tabofs[dy], j1 = tabofs[dy + 1];
        float *o = dst + (int64_t)dy * dstride;
        const float *h0 = H + (int64_t)ytab[j0].si * dw;
        const __m256 b0 = _mm256_set1_ps(ytab[j0].alpha);
        int dx = 0;
        for (; dx + 8 <= dw; dx += 8) {
            __m256 s = _mm256_mul_ps(b0, _mm256_loadu_ps(h0 + dx));
            for (int j = j0 + 1; j < j1; j++)
                s = _mm256_add_ps(s, _mm256_mul_ps(_mm256_set1_ps(ytab[j].alpha),
                                                   _mm256_loadu_ps(H + (int64_t)ytab[j].si * dw + dx)));
            _mm256_storeu_ps(o + dx, s);
        }
        for (; dx < dw; dx++) {
            float s = ytab[j0].alpha * h0[dx];
            for (int j = j0 + 1; j < j1; j++) s = s + ytab[j].alpha * H[(int64_t)ytab[j].si * dw + dx];
            o[dx] = s;
        }
    }
}
#endif

EXPORT int64_t resize_area_f32(const float *src, int64_t sh, int64_t sw, int64_t sstride,
                               float *dst, int64_t dh, int64_t dw, int64_t dstride) {
    if (sh <= 0 || sw <= 0 || dh <= 0 || dw <= 0) return -1;
    const double inv_x = (double)dw / (double)sw, inv_y = (double)dh / (double)sh;
    const double scale_x = 1. / inv_x, scale_y = 1. / inv_y;
    if (!(scale_x >= 1 && scale_y >= 1)) return 1;
    const int ix = cv_round_d(scale_x), iy = cv_round_d(scale_y);
    if (fabs(scale_x - ix) < DBL_EPSILON && fabs(scale_y - iy) < DBL_EPSILON) return 2;
    int64_t rc = -1;
    dtab_t *xtab = (dtab_t *)malloc(sizeof(dtab_t) * 2 * (size_t)sw);
    dtab_t *ytab = (dtab_t *)malloc(sizeof(dtab_t) * 2 * (size_t)sh);
    int *xs = (int *)malloc(sizeof(int) * ((size_t)dw + 1));
    int *xsi = (int *)malloc(sizeof(int) * 2 * (size_t)sw);
    float *xal = (float *)malloc(sizeof(float) * 2 * (size_t)sw);
    int *tabofs = (int *)malloc(sizeof(int) * ((size_t)dh + 1));
    float *H = (float *)malloc(sizeof(float) * (size_t)sh * (size_t)dw);
    float *Traw = (float *)malloc(sizeof(float) * 8 * (size_t)sw + 32);
    if (!xtab || !ytab || !xs || !xsi || !xal || !tabofs || !H || !Traw) goto done;
    {
        float *T = (float *)(((uintptr_t)Traw + 31) & ~(uintptr_t)31);
        const int nx = area_tab((int)sw, (int)dw, scale_x, xtab);
        const int ny = area_tab((int)sh, (int)dh, scale_y, ytab);
        if (nx < 0 || ny < 0) goto done;
        int k = 0;
        for (int dx = 0; dx < dw; dx++) {
            xs[dx] = k;
            while (k < nx && xtab[k].di == dx) {
                xsi[k] = xtab[k].si;
                xal[k] = xtab[k].alpha;
                k++;
            }
        }
        xs[dw] = k;
        if (k != nx) goto done;
        int dy = 0;
        for (k = 0; k < ny; k++) {
            if (k == 0 || ytab[k].di != ytab[k - 1].di) {
                if (ytab[k].di != dy) goto done;
                tabofs[dy++] = k;
            }
        }
        tabofs[dy] = ny;
        if (dy != dh) goto done;
        int r_done = 0;
#ifdef KNEE_X86
        if (g_simd && has_avx())
            r_done = area_h_avx(src, sstride, (int)sh, (int)sw, (int)dw, xs, xsi, xal, H, T);
#endif
        area_h_rows(src, sstride, r_done, (int)sh, (int)dw, xs, xsi, xal, H);
#ifdef KNEE_X86
        if (g_simd && has_avx()) area_v_avx(H, (int)dw, (int)dh, ytab, tabofs, dst, dstride);
        else
#endif
            area_v_scalar(H, (int)dw, (int)dh, ytab, tabofs, dst, dstride);
        rc = 0;
    }
done:
    free(xtab);
    free(ytab);
    free(xs);
    free(xsi);
    free(xal);
    free(tabofs);
    free(H);
    free(Traw);
    return rc;
}

/*
 * --- scan_prefix: the header prefixes of a block of files, in C ----------------------
 * For every file: open, posix_fadvise(RANDOM), read of the first `cap` bytes (fewer if the file ends
 * first), the `walk_order` walk on the prefix, and the descriptor kept open if `keep` (like
 * `dicom._read_prefix(path, _HEADER_BYTES, keep)`), closed otherwise. This replaces one task per file
 * in a pool of 16 Python threads (open, fadvise, read, join, ctypes, futures), with the workers on 4
 * vCPUs fighting for the GIL. ctypes calls release the GIL: several blocks in parallel keep the
 * latency of the mount overlapped.
 * paths: NUL-terminated strings, path_off[i] = start of the i-th; heads: n * cap bytes; head_len[i] =
 * bytes read or -1 if open or read fail; fds[i] = kept descriptor or -1; offs: n * 6 as in
 * walk_order; walk_rc[i] = the return value of walk_order (0 = ok). Returns 0, or -2 where
 * unsupported (Windows).
 */
#if defined(_WIN32)
EXPORT int64_t scan_prefix(const char *paths, const int64_t *path_off, int64_t n, int64_t cap,
                           uint8_t *heads, int64_t *head_len, int64_t *fds, int64_t keep,
                           int64_t *offs, int64_t *walk_rc) {
    (void)paths; (void)path_off; (void)n; (void)cap; (void)heads; (void)head_len; (void)fds;
    (void)keep; (void)offs; (void)walk_rc;
    return -2;
}
#else
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
EXPORT int64_t scan_prefix(const char *paths, const int64_t *path_off, int64_t n, int64_t cap,
                           uint8_t *heads, int64_t *head_len, int64_t *fds, int64_t keep,
                           int64_t *offs, int64_t *walk_rc) {
    for (int64_t i = 0; i < n; i++) {
        uint8_t *h = heads + i * cap;
        fds[i] = -1;
        walk_rc[i] = -1;
        int fd = open(paths + path_off[i], O_RDONLY);
        if (fd < 0) { head_len[i] = -1; for (int k = 0; k < 6; k++) offs[6 * i + k] = -1; continue; }
#ifdef POSIX_FADV_RANDOM
        posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
#endif
        int64_t got = 0;
        int bad = 0;
        while (got < cap) {
            ssize_t r = read(fd, h + got, (size_t)(cap - got));
            if (r < 0 && errno == EINTR) continue;
            if (r < 0) { bad = 1; break; }
            if (r == 0) break;
            got += r;
        }
        if (bad) {                  /* like an `os.read` that raises: the caller redoes the file in Python */
            close(fd);
            head_len[i] = -1;
            for (int k = 0; k < 6; k++) offs[6 * i + k] = -1;
            continue;
        }
        head_len[i] = got;
        walk_rc[i] = walk_order(h, got, offs + 6 * i);
        if (keep) fds[i] = fd;
        else close(fd);
    }
    return 0;
}
#endif

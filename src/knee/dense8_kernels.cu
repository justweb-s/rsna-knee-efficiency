// Kernels of the `knee.dense8` engine (the first INT8 DenseNet, superseded by `knee.dense8s`),
// compiled with NVRTC (`knee.cudart`) for sm_75.
//
// In a DenseNet every layer applies its own BatchNorm + ReLU to ALL previous features of the block
// and quantizes them for its conv1: in a chained TensorRT engine these are two memory passes per
// layer over the whole concatenation (58% of the time per study), a cost that grows with the square
// of the number of layers. Here it is turned around: every feature segment (the block input or the
// 32 outputs of a layer) is read ONCE, right after it is produced, and written already normalised
// and quantized into the INT8 input of every layer that will consume it (`fan_dense`; also the
// transition in the "full" and "fan" forms), of the transition in the "pool" form (`fan_trans`: 2x2
// mean before the 1x1 conv, they are linear and commute) or of the final features (`fan_final`:
// norm5, ReLU, mean). The concatenation no longer exists: every consumer has its own buffer.
//
// The segment is read in the format written by the TensorRT engine that produces it (fp16 or fp32;
// NCHW, NC/xHWx vectorised over channels, HWC with padded channels): element (n, c, s) is at
// n*sN + (c / V)*sG + s*sS + (c % V)*sC (`Fmt`). No reformatting in between. The segment can also be
// the source seen through a pooling done here (`Src.mode`): 1 = 2x2 mean (the transition AvgPool,
// outside the engine), 2 = 3x3 max with stride 2 and padding 1 (the stem max-pool, outside the
// engine): so the engines have neither the pooling nor the reformatting TensorRT would put in front.
// INT8 buffers are NC/32HW32 (TensorRT kCHW32): for every pixel the 32 channels of a group are 32
// contiguous bytes, and a 32-channel segment is exactly one group. A thread handles HALF a group
// (16 channels, 16 bytes): consecutive threads write 16 consecutive bytes, and one warp write covers
// 512 contiguous bytes.
// Quantization (that of `knee.quant`, per tensor, symmetric): q = clamp(rint(v / s)), here with the
// scale folded into the affine: a' = a / s, b' = b / s, q = clamp(rint(a' y + b'), 0, 127) (ReLU is
// the clamp at 0). Round-half-to-even like torch (`__float2int_rn`).
// No includes: fp16 and vector types by hand.

typedef unsigned short half_t;

struct alignas(16) U4 { unsigned int x, y, z, w; };
struct alignas(16) F4 { float x, y, z, w; };

__device__ __forceinline__ float h2f(half_t h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ unsigned int q8(float v) {
    int i = __float2int_rn(v);
    i = i < 0 ? 0 : (i > 127 ? 127 : i);
    return (unsigned int)i;
}

// Source format (see above); f32 = 1 if fp32, 0 if fp16.
struct Fmt { long long sN, sG, sS, sC; int V, f32; };
// Pooling applied here on the source: mode 0 none, 1 2x2 mean, 2 3x3/2 max with padding 1;
// sh x sw = source pixels (the segment has h x w).
struct Src { int mode, sh, sw; };

__device__ __forceinline__ float ld(const void* y, int f32, long long off) {
    return f32 ? ((const float*)y)[off] : h2f(((const half_t*)y)[off]);
}

// 16 channels (from c0) of the source at pixel s of image n, as float.
__device__ __forceinline__ void load16(const void* y, const Fmt& f, int n, int s, int c0,
                                       float* v) {
    const long long base = (long long)n * f.sN + (long long)s * f.sS;
    if (f.sC == 1 && (c0 % f.V) + 16 <= f.V) {
        const long long off = base + (long long)(c0 / f.V) * f.sG + (c0 % f.V);
        if (f.f32) {
            const F4* p = (const F4*)((const float*)y + off);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                F4 t = p[q];
                v[4 * q] = t.x; v[4 * q + 1] = t.y; v[4 * q + 2] = t.z; v[4 * q + 3] = t.w;
            }
        } else {
            const U4* p = (const U4*)((const half_t*)y + off);
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                U4 t = p[q];
                unsigned int w[4] = {t.x, t.y, t.z, t.w};
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    v[8 * q + 2 * r] = h2f((half_t)(w[r] & 0xFFFFu));
                    v[8 * q + 2 * r + 1] = h2f((half_t)(w[r] >> 16));
                }
            }
        }
    } else {
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            const int gc = c0 + c;
            v[c] = ld(y, f.f32, base + (long long)(gc / f.V) * f.sG + (long long)(gc % f.V) * f.sC);
        }
    }
}

// 16 channels of the segment at its pixel (oy, ox), with the source pooling if any.
__device__ __forceinline__ void seg16(const void* y, const Fmt& f, const Src& r, int n, int oy,
                                      int ox, int w, int c0, float* v) {
    if (r.mode == 0) {
        load16(y, f, n, oy * w + ox, c0, v);
        return;
    }
    float t[16];
    if (r.mode == 1) {
        const int s00 = (2 * oy) * r.sw + 2 * ox;
        load16(y, f, n, s00, c0, v);
        load16(y, f, n, s00 + 1, c0, t);
#pragma unroll
        for (int c = 0; c < 16; ++c) v[c] += t[c];
        load16(y, f, n, s00 + r.sw, c0, t);
#pragma unroll
        for (int c = 0; c < 16; ++c) v[c] += t[c];
        load16(y, f, n, s00 + r.sw + 1, c0, t);
#pragma unroll
        for (int c = 0; c < 16; ++c) v[c] = 0.25f * (v[c] + t[c]);
        return;
    }
    bool first = true;
    for (int dy = -1; dy <= 1; ++dy) {
        const int yy = 2 * oy + dy;
        if (yy < 0 || yy >= r.sh) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            const int xx = 2 * ox + dx;
            if (xx < 0 || xx >= r.sw) continue;
            load16(y, f, n, yy * r.sw + xx, c0, t);
#pragma unroll
            for (int c = 0; c < 16; ++c) v[c] = first ? t[c] : fmaxf(v[c], t[c]);
            first = false;
        }
    }
}

// One channel of the segment at its pixel (oy, ox), with the source pooling (fan_final).
__device__ __forceinline__ float seg1(const void* y, const Fmt& f, const Src& r, int n, int oy,
                                      int ox, int w, int c) {
    const long long cb = (long long)n * f.sN + (long long)(c / f.V) * f.sG
                         + (long long)(c % f.V) * f.sC;
    if (r.mode == 0) return ld(y, f.f32, cb + (long long)(oy * w + ox) * f.sS);
    if (r.mode == 1) {
        const long long s00 = (long long)((2 * oy) * r.sw + 2 * ox);
        float a = ld(y, f.f32, cb + s00 * f.sS);
        a += ld(y, f.f32, cb + (s00 + 1) * f.sS);
        a += ld(y, f.f32, cb + (s00 + r.sw) * f.sS);
        return 0.25f * (a + ld(y, f.f32, cb + (s00 + r.sw + 1) * f.sS));
    }
    float m = -3.0e38f;
    for (int dy = -1; dy <= 1; ++dy) {
        const int yy = 2 * oy + dy;
        if (yy < 0 || yy >= r.sh) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            const int xx = 2 * ox + dx;
            if (xx < 0 || xx >= r.sw) continue;
            m = fmaxf(m, ld(y, f.f32, cb + (long long)(yy * r.sw + xx) * f.sS));
        }
    }
    return m;
}

// One thread = one pixel x half a group (16 channels) of the segment; for every consumer k of its
// range (blockIdx.z) it writes the 16 bytes into k's buffer. grid = (ceil(2*npix / blockDim),
// segment groups, consumer ranges). par: [ncons][2][cseg] (a' then b', the channels of the
// segment). dst[k]: NC/32HW32 buffer of the consumer, groups[k] groups per image; the segment
// occupies the groups starting at g0.
extern "C" __global__ void __launch_bounds__(128) fan_dense(
    const void* __restrict__ y, Fmt f, Src r, int npix, int h, int w, int cseg, int g0,
    int ncons, int kch, const unsigned long long* __restrict__ dst,
    const int* __restrict__ groups, const float* __restrict__ par)
{
    extern __shared__ float sp[];
    const int g = blockIdx.y;
    const int k0 = blockIdx.z * kch;
    const int nk = min(kch, ncons - k0);
    for (int i = threadIdx.x; i < nk * 64; i += blockDim.x) {
        const int k = i >> 6, q = i & 63;
        sp[i] = par[(size_t)(k0 + k) * 2 * cseg + (size_t)(q >> 5) * cseg + g * 32 + (q & 31)];
    }
    __syncthreads();
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int p = t >> 1, hf = t & 1;
    if (p >= npix) return;
    const int hw = h * w;
    const int n = p / hw, s = p - n * hw;
    const int oy = s / w, ox = s - oy * w;
    float v[16];
    seg16(y, f, r, n, oy, ox, w, g * 32 + hf * 16, v);
    for (int k = 0; k < nk; ++k) {
        const float* a = sp + k * 64 + hf * 16;
        const float* b = a + 32;
        unsigned int wv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned int q = 0;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int c = j * 4 + u;
                q |= q8(fmaf(a[c], v[c], b[c])) << (8 * u);
            }
            wv[j] = q;
        }
        U4* d = (U4*)((signed char*)dst[k0 + k]
                      + (((size_t)n * groups[k0 + k] + g0 + g) * hw + s) * 32 + hf * 16);
        *d = U4{wv[0], wv[1], wv[2], wv[3]};
    }
}

// The transition in the "pool" form: 2x2 mean of relu(a' y + b') (scale folded in), then rint ->
// its INT8 input at half resolution, NC/32HW32 with gt groups. One thread = one output pixel x half
// a group. grid = (ceil(2*n*hw2 / blockDim), groups). par: [2][cseg].
extern "C" __global__ void __launch_bounds__(128) fan_trans(
    const void* __restrict__ y, Fmt f, Src r, int nimg, int h, int w, int cseg, int g0, int gt,
    signed char* __restrict__ dst, const float* __restrict__ par)
{
    __shared__ float sa[32], sb[32];
    const int g = blockIdx.y;
    if (threadIdx.x < 32) {
        sa[threadIdx.x] = par[g * 32 + threadIdx.x];
        sb[threadIdx.x] = par[cseg + g * 32 + threadIdx.x];
    }
    __syncthreads();
    const int h2 = h >> 1, w2 = w >> 1, hw2 = h2 * w2;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int p = t >> 1, hf = t & 1;
    if (p >= nimg * hw2) return;
    const int n = p / hw2, s2 = p - n * hw2;
    const int oy = s2 / w2, ox = s2 - oy * w2;
    const float* a = sa + hf * 16;
    const float* b = sb + hf * 16;
    const int c0 = g * 32 + hf * 16;
    float acc[16], v[16];
    seg16(y, f, r, n, 2 * oy, 2 * ox, w, c0, v);
#pragma unroll
    for (int c = 0; c < 16; ++c) acc[c] = fmaxf(fmaf(a[c], v[c], b[c]), 0.f);
    seg16(y, f, r, n, 2 * oy, 2 * ox + 1, w, c0, v);
#pragma unroll
    for (int c = 0; c < 16; ++c) acc[c] += fmaxf(fmaf(a[c], v[c], b[c]), 0.f);
    seg16(y, f, r, n, 2 * oy + 1, 2 * ox, w, c0, v);
#pragma unroll
    for (int c = 0; c < 16; ++c) acc[c] += fmaxf(fmaf(a[c], v[c], b[c]), 0.f);
    seg16(y, f, r, n, 2 * oy + 1, 2 * ox + 1, w, c0, v);
#pragma unroll
    for (int c = 0; c < 16; ++c) acc[c] += fmaxf(fmaf(a[c], v[c], b[c]), 0.f);
    unsigned int wv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        unsigned int q = 0;
#pragma unroll
        for (int u = 0; u < 4; ++u) q |= q8(0.25f * acc[j * 4 + u]) << (8 * u);
        wv[j] = q;
    }
    U4* d = (U4*)(dst + (((size_t)n * gt + g0 + g) * hw2 + s2) * 32 + hf * 16);
    *d = U4{wv[0], wv[1], wv[2], wv[3]};
}

// The final features: out[n, off + c] = mean over pixels of relu(a y + b) (norm5 + ReLU + global
// mean). grid = (images, segment groups), 128 threads: 32 channels x 4 pixel rows.
// par: [2][cseg] (a and b of norm5, unscaled).
extern "C" __global__ void __launch_bounds__(128) fan_final(
    const void* __restrict__ y, Fmt f, Src r, int h, int w, int cseg, int off, int ctot,
    float* __restrict__ out, const float* __restrict__ par)
{
    __shared__ float part[4][32];
    const int n = blockIdx.x, g = blockIdx.y;
    const int c = threadIdx.x & 31, q = threadIdx.x >> 5;
    const float a = par[g * 32 + c], b = par[cseg + g * 32 + c];
    const int hw = h * w;
    float acc = 0.f;
    for (int s = q; s < hw; s += 4) {
        const int oy = s / w, ox = s - oy * w;
        acc += fmaxf(fmaf(a, seg1(y, f, r, n, oy, ox, w, g * 32 + c), b), 0.f);
    }
    part[q][c] = acc;
    __syncthreads();
    if (q == 0)
        out[(size_t)n * ctot + off + g * 32 + c] =
            (part[0][c] + part[1][c] + part[2][c] + part[3][c]) / (float)hw;
}

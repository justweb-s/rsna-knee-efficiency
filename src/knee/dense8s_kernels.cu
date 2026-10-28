// Kernels of the `knee.dense8s` engine, compiled with NVRTC (`knee.cudart`) for sm_75.
//
// In `knee.dense8` every feature segment was copied, already normalised and quantized, into the INT8
// input of EVERY layer consuming it (fan kernels): in block 1 ~1,100 channels written per pixel instead
// of 256, 15.6 ms out of 49.9 per pass, plus TensorRT's conv1 re-reading all those copies (~13 ms).
// Here every segment is written ONCE in INT8 with a per-channel scale (the block "store"), and the conv1
// of every layer is one of our tensor-core kernels (mma.sync m8n8k16 u8 x s8) that applies the
// BatchNorm, ReLU and quantization of ITS consumer while loading the data (the "prologue"): no copies.
//
// The store of a block, two byte slabs with rows of 32 channels (one group) per pixel:
//   in0: the block input (stem or transition), [N][G0][HW][32] (TensorRT kCHW32);
//   seg: the layer outputs, [L][N][HW][32]: layer l writes its group as a kCHW32 tensor
//        [N][32][H][W] of its own, so TensorRT's conv2 can write straight into it.
// Channel c of pixel (n, p) is x = sigma_c * q (q int8, per-channel sigma).
//
// Prologue (layer or transition): u = rint(255 * sat(a'_c q + b'_c)) in [0, 255] (u8), with
// a' = a_c sigma_c / (255 s), b' = b_c / (255 s): relu(a x + b) / s, s = per-tensor scale of the conv
// input. The ReLU is the lower saturation. The byte is converted to an exact float through the
// mantissa (2^23 + byte), the rounding (half to even) comes from the FMA with 2^23.
//
// conv1 (mode 0): acc = sum_c W_oc u_c (int32, int8 weights per output channel, sign of norm2 inside)
// -> v = rint(127 * sat(g'_o acc + d'_o)) in [0, 127]: the kCHW32 bottleneck [N][4][HW][32], INT8
// input of TensorRT's conv2 (norm2, ReLU, Q).
// Transition (mode 1): the same GEMM over all channels of the block; the tile rows are the four pixels
// of every 2x2 square, the epilogue sums the four accumulators (exact, int32) and quantizes the mean
// per channel -> in0 of the next block: q' = rint(clamp(k_o * sum, +-127)).
//
// No includes: vector types and conversions by hand.

typedef unsigned short half_t;

struct alignas(16) U4 { unsigned int x, y, z, w; };
struct alignas(8) F2 { float x, y; };
struct alignas(8) U2 { unsigned int x, y; };

#define DEV __device__ __forceinline__
#define OROWS_HI(bm) ((bm) * 144)

DEV unsigned smem_addr(const void* p) {
    unsigned r;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p));
    return r;
}

DEV void ldsm4(unsigned addr, unsigned* r) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

DEV void ldsm2(unsigned addr, unsigned* r) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

DEV void ldsm1(unsigned addr, unsigned& r) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(r) : "r"(addr));
}

DEV void mma_us(int* c, unsigned a, unsigned b) {
    asm("mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
        : "+r"(c[0]), "+r"(c[1]) : "r"(a), "r"(b));
}

// --- phase tracing (measurements only, *_t kernels) -------------------------------------------------
// Every warp sums in registers the SM cycles (clock64) elapsed between one mark and the next, per
// phase; at the end of the kernel lane 0 adds them to tr[0 .. TR_N - 1] and tr[TR_N] counts the warps.
// A mark costs ~4 instructions and acts as a compiler barrier: the shares matter, not the absolute
// times of the traced kernel.
#define TR_N 12
struct TraceW {
    unsigned long long acc[TR_N];
    long long t;
};
template <bool TR>
DEV void tr_init(TraceW& w) {
    if (TR) {
#pragma unroll
        for (int k = 0; k < TR_N; ++k) w.acc[k] = 0ull;
        w.t = clock64();
    }
}
template <bool TR>
DEV void tr_mark(TraceW& w, int k) {
    if (TR) {
        const long long c = clock64();
        w.acc[k] += (unsigned long long)(c - w.t);
        w.t = c;
    }
}
template <bool TR>
DEV void tr_done(const TraceW& w, unsigned long long* tr) {
    if (TR && (threadIdx.x & 31) == 0) {
#pragma unroll
        for (int k = 0; k < TR_N; ++k) atomicAdd(tr + k, w.acc[k]);
        atomicAdd(tr + TR_N, 1ull);
    }
}

// 16 signed bytes -> 16 u8: u = rint(255 * sat(ab.x * q + ab.y)).
DEV U4 pro16(U4 x, const F2* ab) {
    const unsigned w[4] = {x.x ^ 0x80808080u, x.y ^ 0x80808080u, x.z ^ 0x80808080u,
                           x.w ^ 0x80808080u};
    unsigned o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        unsigned r[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            // byte u (with 0x80 added) in the mantissa of 2^23: the float is 2^23 + q + 128
            const float f = __int_as_float(__byte_perm(w[j], 0x4B000000u, 0x7540 | u))
                            - 8388736.0f;
            const F2 p = ab[4 * j + u];
            const float v = __saturatef(fmaf(p.x, f, p.y));
            r[u] = __float_as_uint(fmaf(v, 255.0f, 8388608.0f));   // low byte = rint(255 v)
        }
        o[j] = __byte_perm(__byte_perm(r[0], r[1], 0x0040), __byte_perm(r[2], r[3], 0x0040),
                           0x5410);
    }
    return U4{o[0], o[1], o[2], o[3]};
}

// Prologue in fp16x2 (2.5 instructions per byte instead of ~5.25, a quarter of the floating-point
// work): the byte with 0x80 added goes into the mantissa of 1024 (exact fp16 1024 + 128 + q), 1152 is
// subtracted (exact q), v = sat(A q + B) with A, B in fp16 (one rounding, saturating to [0, 1]),
// u = rint(255 v) from the fma with 1024 (ulp 1): the low byte of each half. Compared with the fp32
// prologue the rounding of v changes (<= 2^-12, i.e. <= 0.06 LSB of u) and A, B are rounded to fp16:
// `emulate(f16_layers=...)` reproduces it. p = (A of 2 channels, B of 2 channels).
DEV unsigned pro4h(unsigned x, U2 p01, U2 p23) {
    const unsigned w = x ^ 0x80808080u;
    unsigned lo = __byte_perm(w, 0x64646464u, 0x4140);
    unsigned hi = __byte_perm(w, 0x64646464u, 0x4342);
    asm("add.rn.f16x2 %0, %0, %1;" : "+r"(lo) : "r"(0xE480E480u));        // -1152
    asm("add.rn.f16x2 %0, %0, %1;" : "+r"(hi) : "r"(0xE480E480u));
    asm("fma.rn.sat.f16x2 %0, %0, %1, %2;" : "+r"(lo) : "r"(p01.x), "r"(p01.y));
    asm("fma.rn.sat.f16x2 %0, %0, %1, %2;" : "+r"(hi) : "r"(p23.x), "r"(p23.y));
    asm("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(lo) : "r"(0x5BF85BF8u), "r"(0x64006400u));  // 255, 1024
    asm("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(hi) : "r"(0x5BF85BF8u), "r"(0x64006400u));
    return __byte_perm(lo, hi, 0x6420);
}

DEV U4 pro16h(U4 x, const U2* ph) {                          // 16 channels = 8 pairs
    return U4{pro4h(x.x, ph[0], ph[1]), pro4h(x.y, ph[2], ph[3]), pro4h(x.z, ph[4], ph[5]),
              pro4h(x.w, ph[6], ph[7])};
}

// Arguments of the GEMM kernels, passed by value (`dense8s.KArg`).
struct KArg {
    const signed char* in0;     // block input slab [N][G0][HW][32]
    const signed char* seg;     // layer slab [L][N][HW][32]
    const signed char* w;       // weights [G][cout][32]
    const float* ab;            // prologue [G*32][2]
    const float* gd;            // epilogue [cout][2]
    signed char* out;
    int G0, G, N, HW, W;        // groups of the block input, groups read, images, pixels
    int rows;                   // GEMM rows (mode 0: N*HW; mode 1: N*HW, 4 per output)
    int cout;                   // output channels (128 for conv1)
    int G0n;                    // mode 1: groups of the next block's input
};

// MODE 0: conv1 -> kCHW32 int8 bottleneck [0, 127] for TensorRT's conv2 (QO = 127);
// MODE 1: transition; MODE 2: conv1 -> NHWC bottleneck [N][HW][128] u8 [0, 255] for `conv2_q`;
// MODE 3: like MODE 2 but the bottleneck in planes of 32 channels [4][N*HW][32]: the conv2 reads the
// halo of a channel block from contiguous rows (in NHWC it reads 32 bytes out of every 128: ~2.3 ms/GB
// at the power cap for that halo against ~1.1 for the contiguous reads of conv1).
// DIAG (measurements only, `*_d<k>`): 1 no MMA, 2 no global loads in the loop, 4 no prologue (raw
// bytes in shared memory), 8 loads always from the addresses of the first two stages (same
// instructions, data from L1 instead of DRAM / L2: the energy of the traffic), 16 no output writes to
// global memory (the cost of writing the bottleneck). Meaningless results: only time and energy count.
// PH = 1: fp16 prologue (`pro16h`, a.ab = table L*_abh in channel pairs).
// PF > 0: L2 prefetch (prefetch.global.L2) of the thread's row PF stages ahead: conv1 is memory-bound
// (~140 GB/s on average against 274 for DRAM reads) and the registers keep only one stage in flight;
// with the prefetch the real loads hit L2.
// BF = 1: steady state in pairs of stages without branches (load, mma and prologue in the same basic
// block between two barriers: the compiler can interleave the next stage's prologue with this stage's
// mma; with the `if`s of the normal loop the prologue always comes after the mma).
template <int BM, int MODE, int NWARPS = 8, int DIAG = 0, int PH = 0, int PF = 0, int BF = 0,
          bool TR = false>
DEV void gemm_q(const KArg& a, unsigned long long* tr = nullptr) {
    constexpr int NTH = NWARPS * 32;
    constexpr int WARPS_M = BM / 32;
    constexpr int WARPS_N = NWARPS / WARPS_M;
    constexpr int WN = 128 / WARPS_N;
    constexpr int NT = WN / 8;
    constexpr int A_BYTES = BM * 32, B_BYTES = 128 * 32;
    constexpr int OROWS = MODE == 1 ? BM / 4 : BM;
    constexpr float QO = (MODE == 2 || MODE == 3) ? 255.0f : 127.0f;
    constexpr int STAGE = OROWS * 144;
    constexpr int UNION = 2 * (A_BYTES + B_BYTES) > STAGE ? 2 * (A_BYTES + B_BYTES) : STAGE;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* sA = smem;
    unsigned char* sB = smem + 2 * A_BYTES;
    F2* sAB = (F2*)(smem + UNION);

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % WARPS_M, wn = warp / WARPS_M;
    // transition: the output channel blocks are the fast dimension of the grid, so those reading the same
    // rows (t2: 5) run together and the rows come from L2
    const int m0 = (MODE == 1 ? blockIdx.y : blockIdx.x) * BM;
    const int n0 = (MODE == 1 ? blockIdx.x : blockIdx.y) * 128;
    const int HW = a.HW;
    const int in0g = HW * 32;                       // stride of a group in in0
    const int segg = a.N * HW * 32;                 // stride of a group in seg
    TraceW tw;
    tr_init<TR>(tw);

    if (PH) {
        for (int i = tid; i < a.G * 16; i += NTH) ((U2*)sAB)[i] = ((const U2*)a.ab)[i];
    } else {
        for (int i = tid; i < a.G * 32; i += NTH) sAB[i] = ((const F2*)a.ab)[i];
    }

    // the row (pixel) this thread loads, half ha of the 32 channels
    const bool a_thr = tid < 2 * BM;
    const int ra = tid >> 1, ha = tid & 1;
    const int m = m0 + ra;
    const bool valid = a_thr && m < a.rows;
    int o_in0 = 0, o_seg = 0;
    if (valid) {
        int n, p;
        if (MODE != 1) {
            n = m / HW;
            p = m - n * HW;
        } else {
            const int W2 = a.W >> 1, HW4 = HW >> 2;
            const int q = m >> 2, k = m & 3;
            n = q / HW4;
            const int p4 = q - n * HW4;
            const int oy = p4 / W2, ox = p4 - oy * W2;
            p = (2 * oy + (k >> 1)) * a.W + 2 * ox + (k & 1);
        }
        o_in0 = (n * a.G0 * HW + p) * 32 + ha * 16;
        o_seg = (n * HW + p) * 32 + ha * 16;
    }
    const int rb = tid >> 1, hb = tid & 1;
    const bool b_thr = tid < 256;                   // 128 weight rows x 2 halves
    const signed char* wrow = a.w + (n0 + (b_thr ? rb : 0)) * 32 + hb * 16;
    const int wg = a.cout * 32;                     // stride of a group in the weights

    int acc[4][NT][2];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = 0;

    // Two channel blocks in flight in registers (va0/vb0, va1/vb1: no cp.async on sm_75), one ready in
    // shared memory: the load of block g+2 starts while block g is computed (with only one in flight conv1
    // ran at ~100-150 GB/s).
    U4 va0 = U4{0u, 0u, 0u, 0u}, vb0, va1 = U4{0u, 0u, 0u, 0u}, vb1;
    auto gload = [&](int g, U4& va, U4& vb) {
        if (DIAG & 2) {
            if (g > 1) return;
        }
        if (DIAG & 8) g &= 1;
        if (valid)
            va = *(const U4*)(g < a.G0 ? a.in0 + o_in0 + g * in0g
                                       : a.seg + o_seg + (g - a.G0) * segg);
        if (b_thr) vb = *(const U4*)(wrow + g * wg);
    };
    auto pfetch = [&](int g) {
        if (PF > 0 && valid && ha == 0 && g < a.G) {
            const signed char* src = g < a.G0 ? a.in0 + o_in0 + g * in0g
                                              : a.seg + o_seg + (g - a.G0) * segg;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(src));
        }
    };
    auto sstore = [&](int buf, int g, const U4& va, const U4& vb) {
        if (a_thr)
            *(U4*)(sA + buf * A_BYTES + ra * 32 + ((ha ^ ((ra >> 2) & 1)) << 4)) =
                (DIAG & 4) ? va
                           : (PH ? pro16h(va, (const U2*)sAB + g * 16 + ha * 8)
                                 : pro16(va, sAB + g * 32 + ha * 16));
        if (b_thr) *(U4*)(sB + buf * B_BYTES + rb * 32 + ((hb ^ ((rb >> 2) & 1)) << 4)) = vb;
    };
    auto compute = [&](int buf) {
        const unsigned aBase = smem_addr(sA + buf * A_BYTES);
        const unsigned bBase = smem_addr(sB + buf * B_BYTES);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            unsigned af[4], bf[NT];
            {
                const int row = wm * 32 + lane;
                ldsm4(aBase + row * 32 + ((h ^ ((row >> 2) & 1)) << 4), af);
            }
#pragma unroll
            for (int j = 0; j < NT / 4; ++j) {
                const int row = wn * WN + j * 32 + lane;
                ldsm4(bBase + row * 32 + ((h ^ ((row >> 2) & 1)) << 4), bf + 4 * j);
            }
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    if (DIAG & 1) acc[mi][ni][0] ^= (int)(af[mi] ^ bf[ni]);
                    else mma_us(acc[mi][ni], af[mi], bf[ni]);
                }
        }
    };
    gload(0, va0, vb0);
    if (a.G > 1) gload(1, va1, vb1);
    if (PF > 0) {
#pragma unroll
        for (int k = 2; k < 2 + PF; ++k) pfetch(k);
    }
    __syncthreads();                                // sAB ready
    sstore(0, 0, va0, vb0);
    __syncthreads();
    tr_mark<TR>(tw, 0);
    int g0 = 0;
    if (BF) {
        // steady state: g and g+1 ready (smem, registers), g+2 and g+3 exist
        for (; g0 + 3 < a.G; g0 += 2) {
            if (PF > 0) {
                pfetch(g0 + 2 + PF);
                pfetch(g0 + 3 + PF);
            }
            gload(g0 + 2, va0, vb0);
            tr_mark<TR>(tw, 1);
            compute(0);
            tr_mark<TR>(tw, 2);
            sstore(1, g0 + 1, va1, vb1);
            tr_mark<TR>(tw, 3);
            __syncthreads();
            tr_mark<TR>(tw, 4);
            gload(g0 + 3, va1, vb1);
            tr_mark<TR>(tw, 1);
            compute(1);
            tr_mark<TR>(tw, 2);
            sstore(0, g0 + 2, va0, vb0);
            tr_mark<TR>(tw, 3);
            __syncthreads();
            tr_mark<TR>(tw, 4);
        }
    }
    for (int g = g0; g < a.G; g += 2) {
        // even step: block g in memory (buffer 0), g+1 in va1/vb1
        if (PF > 0) {
            pfetch(g + 2 + PF);
            pfetch(g + 3 + PF);
        }
        if (g + 2 < a.G) gload(g + 2, va0, vb0);
        tr_mark<TR>(tw, 1);
        compute(0);
        tr_mark<TR>(tw, 2);
        if (g + 1 < a.G) sstore(1, g + 1, va1, vb1);
        tr_mark<TR>(tw, 3);
        __syncthreads();
        tr_mark<TR>(tw, 4);
        if (g + 1 >= a.G) break;
        // odd step: block g+1 in memory (buffer 1), g+2 in va0/vb0
        if (g + 3 < a.G) gload(g + 3, va1, vb1);
        tr_mark<TR>(tw, 1);
        compute(1);
        tr_mark<TR>(tw, 2);
        if (g + 2 < a.G) sstore(0, g + 2, va0, vb0);
        tr_mark<TR>(tw, 3);
        __syncthreads();
        tr_mark<TR>(tw, 4);
    }

    unsigned char* sO = smem;                       // [OROWS][144]: 128 channels + 16 of padding
    const F2* gd = (const F2*)a.gd;
#pragma unroll
    for (int ni = 0; ni < NT; ++ni) {
        const int c = wn * WN + ni * 8 + (lane & 3) * 2;
        const F2 p0 = gd[n0 + c], p1 = gd[n0 + c + 1];
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            if (MODE != 1) {
                const int r = wm * 32 + mi * 8 + (lane >> 2);
                const unsigned q0 = __float_as_uint(fmaf(
                    __saturatef(fmaf(p0.x, (float)acc[mi][ni][0], p0.y)), QO, 8388608.0f));
                const unsigned q1 = __float_as_uint(fmaf(
                    __saturatef(fmaf(p1.x, (float)acc[mi][ni][1], p1.y)), QO, 8388608.0f));
                *(unsigned short*)(sO + r * 144 + c) = (unsigned short)__byte_perm(q0, q1, 0x0040);
            } else {
                int s0 = acc[mi][ni][0], s1 = acc[mi][ni][1];
                s0 += __shfl_xor_sync(0xffffffffu, s0, 4);
                s1 += __shfl_xor_sync(0xffffffffu, s1, 4);
                s0 += __shfl_xor_sync(0xffffffffu, s0, 8);
                s1 += __shfl_xor_sync(0xffffffffu, s1, 8);
                if ((lane & 12) == 0) {
                    const int rq = wm * 8 + mi * 2 + (lane >> 4);
                    const float t0 = fminf(fmaxf(p0.x * (float)s0, -127.0f), 127.0f);
                    const float t1 = fminf(fmaxf(p1.x * (float)s1, -127.0f), 127.0f);
                    const unsigned q0 = __float_as_uint(t0 + 12582912.0f);
                    const unsigned q1 = __float_as_uint(t1 + 12582912.0f);
                    *(unsigned short*)(sO + rq * 144 + c) =
                        (unsigned short)__byte_perm(q0, q1, 0x0040);
                }
            }
        }
    }
    __syncthreads();
    tr_mark<TR>(tw, 5);
    for (int i = tid; i < ((DIAG & 16) ? 0 : OROWS * 8); i += NTH) {
        const int r = i >> 3, q = i & 7;
        const U4 v = *(const U4*)(sO + r * 144 + q * 16);
        if (MODE == 0) {
            const int mm = m0 + r;
            if (mm >= a.rows) continue;
            const int n = mm / HW, p = mm - n * HW;
            *(U4*)(a.out + ((n * 4 + (q >> 1)) * HW + p) * 32 + (q & 1) * 16) = v;
        } else if (MODE == 2) {
            const int mm = m0 + r;
            if (mm >= a.rows) continue;
            *(U4*)(a.out + (size_t)mm * 128 + q * 16) = v;
        } else if (MODE == 3) {
            const int mm = m0 + r;
            if (mm >= a.rows) continue;
            *(U4*)(a.out + (size_t)(q >> 1) * a.rows * 32 + (size_t)mm * 32 + (q & 1) * 16) = v;
        } else {
            const int o = (m0 >> 2) + r;
            const int HW4 = HW >> 2;
            if (o >= (a.rows >> 2)) continue;
            const int n = o / HW4, p = o - n * HW4;
            *(U4*)(a.out + ((n * a.G0n + (n0 >> 5) + (q >> 1)) * HW4 + p) * 32
                   + (q & 1) * 16) = v;
        }
    }
    tr_mark<TR>(tw, 6);
    tr_done<TR>(tw, tr);
}

extern "C" __global__ void __launch_bounds__(256, 2) conv1_q128(KArg a) { gemm_q<128, 0>(a); }
extern "C" __global__ void __launch_bounds__(256, 2) conv1_q64(KArg a) { gemm_q<64, 0>(a); }
extern "C" __global__ void __launch_bounds__(256, 2) trans_q128(KArg a) { gemm_q<128, 1>(a); }
extern "C" __global__ void __launch_bounds__(256, 2) conv1_u128(KArg a) { gemm_q<128, 2>(a); }
extern "C" __global__ void __launch_bounds__(256, 2) conv1_u128h(KArg a) { gemm_q<128, 2, 8, 0, 1>(a); }
extern "C" __global__ void __launch_bounds__(512, 1) conv1_u256h(KArg a) { gemm_q<256, 2, 16, 0, 1>(a); }
extern "C" __global__ void __launch_bounds__(256, 2) conv1_u128p(KArg a) { gemm_q<128, 2, 8, 0, 0, 4>(a); }
extern "C" __global__ void __launch_bounds__(256, 2) conv1_u128hp(KArg a) { gemm_q<128, 2, 8, 0, 1, 4>(a); }
extern "C" __global__ void __launch_bounds__(512, 1) conv1_u256hp(KArg a) { gemm_q<256, 2, 16, 0, 1, 4>(a); }
// branch-free steady state (BF)
extern "C" __global__ void __launch_bounds__(256, 2) conv1_u128hs(KArg a) { gemm_q<128, 2, 8, 0, 1, 0, 1>(a); }
// 64 rows, 8 warps (2 x 4, 32 x 32 tile per warp, 32 accumulators), 3 thread blocks per SM: phases of
// 3 independent blocks instead of 2
extern "C" __global__ void __launch_bounds__(256, 3) conv1_u64hs(KArg a) { gemm_q<64, 2, 8, 0, 1, 0, 1>(a); }
extern "C" __global__ void __launch_bounds__(256, 2) conv1_u128hs_t(KArg a, unsigned long long* tr) {
    gemm_q<128, 2, 8, 0, 1, 0, 1, true>(a, tr);
}
extern "C" __global__ void __launch_bounds__(512, 1) conv1_u256hs(KArg a) { gemm_q<256, 2, 16, 0, 1, 0, 1>(a); }
// planar bottleneck (MODE 3)
extern "C" __global__ void __launch_bounds__(512, 1) conv1_u256hs_pl(KArg a) { gemm_q<256, 3, 16, 0, 1, 0, 1>(a); }
extern "C" __global__ void __launch_bounds__(512, 1) conv1_u256hps(KArg a) { gemm_q<256, 2, 16, 0, 1, 4, 1>(a); }
extern "C" __global__ void __launch_bounds__(256, 2) conv1_u64(KArg a) { gemm_q<64, 2>(a); }
// 256 rows, 16 warps (8 x 2): the weights of every channel block serve twice as many rows (half the L2
// weight traffic); one 512-thread block per SM like two of 256.
extern "C" __global__ void __launch_bounds__(512, 1) conv1_u256(KArg a) { gemm_q<256, 2, 16>(a); }
extern "C" __global__ void __launch_bounds__(512, 1) trans_q256(KArg a) { gemm_q<256, 1, 16>(a); }
// more warps per SM (32 x 32 tile per warp, 32 accumulators): 2 x 16 warps or 4 x 8 warps per SM
extern "C" __global__ void __launch_bounds__(512, 2) conv1_u128x16(KArg a) { gemm_q<128, 2, 16>(a); }
extern "C" __global__ void __launch_bounds__(256, 4) conv1_u64x8(KArg a) { gemm_q<64, 2, 8>(a); }
// measurements (see DIAG in gemm_q)
extern "C" __global__ void __launch_bounds__(256, 2) conv1_u128_d1(KArg a) { gemm_q<128, 2, 8, 1>(a); }
extern "C" __global__ void __launch_bounds__(256, 2) conv1_u128_d2(KArg a) { gemm_q<128, 2, 8, 2>(a); }
extern "C" __global__ void __launch_bounds__(256, 2) conv1_u128_d4(KArg a) { gemm_q<128, 2, 8, 4>(a); }
extern "C" __global__ void __launch_bounds__(256, 2) conv1_u128_d6(KArg a) { gemm_q<128, 2, 8, 6>(a); }
extern "C" __global__ void __launch_bounds__(256, 2) conv1_u128_d7(KArg a) { gemm_q<128, 2, 8, 7>(a); }
// the engine's conv1 (u128hs) with parts switched off, for the energy at the power cap
extern "C" __global__ void __launch_bounds__(256, 2) conv1_u128hs_d1(KArg a) { gemm_q<128, 2, 8, 1, 1, 0, 1>(a); }
extern "C" __global__ void __launch_bounds__(256, 2) conv1_u128hs_d2(KArg a) { gemm_q<128, 2, 8, 2, 1, 0, 1>(a); }
extern "C" __global__ void __launch_bounds__(256, 2) conv1_u128hs_d4(KArg a) { gemm_q<128, 2, 8, 4, 1, 0, 1>(a); }
extern "C" __global__ void __launch_bounds__(256, 2) conv1_u128hs_d7(KArg a) { gemm_q<128, 2, 8, 7, 1, 0, 1>(a); }
extern "C" __global__ void __launch_bounds__(256, 2) conv1_u128hs_d8(KArg a) { gemm_q<128, 2, 8, 8, 1, 0, 1>(a); }
extern "C" __global__ void __launch_bounds__(512, 1) conv1_u256hs_d8(KArg a) { gemm_q<256, 2, 16, 8, 1, 0, 1>(a); }
extern "C" __global__ void __launch_bounds__(512, 1) conv1_u256hs_d16(KArg a) { gemm_q<256, 2, 16, 16, 1, 0, 1>(a); }

// --- conv1 with 64 x 64 warp tiles ---------------------------------------------------
// At the power cap the conv1 (48% of the time) executes ~1.85 instructions per IMMA besides the mma
// themselves: prologue 0.63, integers and addresses ~0.8 (at every stage the in0 / seg choice and the
// 64-bit addresses; the ldmatrix addresses recomputed under the 128-register cap of u256hs). Here 8
// warps (4 rows x 2 channels) with 64 x 64 warp tiles (8 x 8 m8n8 blocks, 128 accumulators, up to 255
// registers): per stage a warp does 128 IMMA with 8 ldmatrix (u256hs: 64 with 6), a third fewer shared
// memory reads per IMMA, and the warp's fixed per-stage work is spread over twice as many mma. Every
// thread carries half `ha` of two rows (r0 and r0 + 128: the same 16 channels, a single load of the
// prologue parameters) and half a weight row. Pointers advance by one stage (in0 then seg): no choice
// and no 64-bit multiplications per stage. Same arithmetic as conv1_u256hs (fp16 prologue `pro16h`,
// u8 bottleneck): bit-identical outputs. PL = 1: planar bottleneck [4][rows][32] (like MODE 3).
template <int PL>
DEV void conv1w_body(const KArg& a) {
    constexpr int BM = 256, NTH = 256, WARPS_M = 4;
    constexpr int A_BYTES = BM * 32, B_BYTES = 128 * 32;
    constexpr int STAGE_O = BM * 144;
    constexpr int UNION = 2 * (A_BYTES + B_BYTES) > STAGE_O ? 2 * (A_BYTES + B_BYTES) : STAGE_O;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* sA = smem;
    unsigned char* sB = smem + 2 * A_BYTES;
    U2* sAB = (U2*)(smem + UNION);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % WARPS_M, wn = warp / WARPS_M;
    const int m0 = blockIdx.x * BM;
    const int G = a.G, G0 = a.G0, HW = a.HW;

    for (int i = tid; i < G * 16; i += NTH) sAB[i] = ((const U2*)a.ab)[i];

    // A: half ha of rows r0 and r0 + 128 of the tile
    const int ha = tid & 1, r0 = tid >> 1;
    const size_t in0g = (size_t)HW * 32, segg = (size_t)a.N * HW * 32;
    bool val[2];
    const signed char* pa[2];                          // group to load next, per row
    const signed char* ps[2];                          // group 0 of seg, per row
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int m = m0 + r0 + j * 128;
        val[j] = m < a.rows;
        const int mm = val[j] ? m : 0;
        const int n = mm / HW, p = mm - n * HW;
        pa[j] = a.in0 + ((size_t)n * G0 * HW + p) * 32 + ha * 16;
        ps[j] = a.seg + ((size_t)n * HW + p) * 32 + ha * 16;
    }
    // B: half a weight row (128 rows x 2 halves)
    const int rb = tid >> 1, hb = tid & 1;
    const signed char* pb = a.w + rb * 32 + hb * 16;
    const size_t wg = (size_t)a.cout * 32;
    int gl = 0;                                        // the group of the next load

    U4 va0[2], va1[2], vb0, vb1;
    auto gload = [&](U4* va, U4& vb) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            va[j] = val[j] ? *(const U4*)pa[j] : U4{0u, 0u, 0u, 0u};
            pa[j] = (gl + 1 == G0) ? ps[j] : pa[j] + (gl + 1 < G0 ? in0g : segg);
        }
        vb = *(const U4*)pb;
        pb += wg;
        ++gl;
    };
    // shared memory addresses (swizzle of the 16-byte half with bit 2 of the row)
    const int sw0 = (ha ^ ((r0 >> 2) & 1)) << 4;                    // r0 + 128: same bit
    const int soA = r0 * 32 + sw0;
    const int soB = rb * 32 + ((hb ^ ((rb >> 2) & 1)) << 4);
    auto sstore = [&](int buf, int g, const U4* va, const U4& vb) {
        const U2* ph = sAB + g * 16 + ha * 8;
        U2 prm[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) prm[k] = ph[k];
#pragma unroll
        for (int j = 0; j < 2; ++j)
            *(U4*)(sA + buf * A_BYTES + j * 128 * 32 + soA) = pro16h(va[j], prm);
        *(U4*)(sB + buf * B_BYTES + soB) = vb;
    };
    // ldmatrix: rows (A) wm * 64 + lane (+ 32), columns (B) wn * 64 + lane (+ 32)
    const int ra = wm * 64 + lane, rbw = wn * 64 + lane;
    const unsigned aL = smem_addr(sA) + ra * 32, bL = smem_addr(sB) + rbw * 32;
    const int sa = ((ra >> 2) & 1) << 4, sb = ((rbw >> 2) & 1) << 4;
    int acc[8][8][2];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j][0] = acc[i][j][1] = 0;
    auto compute = [&](int buf) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            unsigned af[8], bf[8];
            const unsigned ao = aL + buf * A_BYTES + ((h << 4) ^ sa);
            const unsigned bo = bL + buf * B_BYTES + ((h << 4) ^ sb);
            ldsm4(ao, af);
            ldsm4(ao + 32 * 32, af + 4);
            ldsm4(bo, bf);
            ldsm4(bo + 32 * 32, bf + 4);
#pragma unroll
            for (int mi = 0; mi < 8; ++mi)
#pragma unroll
                for (int ni = 0; ni < 8; ++ni) mma_us(acc[mi][ni], af[mi], bf[ni]);
        }
    };
    gload(va0, vb0);
    if (G > 1) gload(va1, vb1);
    __syncthreads();                                   // sAB
    sstore(0, 0, va0, vb0);
    __syncthreads();
    int g0 = 0;
    for (; g0 + 3 < G; g0 += 2) {                      // steady state: g, g+1 ready; g+2, g+3 exist
        gload(va0, vb0);
        compute(0);
        sstore(1, g0 + 1, va1, vb1);
        __syncthreads();
        gload(va1, vb1);
        compute(1);
        sstore(0, g0 + 2, va0, vb0);
        __syncthreads();
    }
    for (int g = g0; g < G; g += 2) {
        if (g + 2 < G) gload(va0, vb0);
        compute(0);
        if (g + 1 < G) sstore(1, g + 1, va1, vb1);
        __syncthreads();
        if (g + 1 >= G) break;
        if (g + 3 < G) gload(va1, vb1);
        compute(1);
        if (g + 2 < G) sstore(0, g + 2, va0, vb0);
        __syncthreads();
    }

    // epilogue: u8 bottleneck = rint(255 sat(g acc + d)) into [BM][144], then 16-byte rows
    unsigned char* sO = smem;
    const F2* gd = (const F2*)a.gd;
#pragma unroll
    for (int ni = 0; ni < 8; ++ni) {
        const int c = wn * 64 + ni * 8 + (lane & 3) * 2;
        const F2 p0 = gd[c], p1 = gd[c + 1];
#pragma unroll
        for (int mi = 0; mi < 8; ++mi) {
            const int r = wm * 64 + mi * 8 + (lane >> 2);
            const unsigned q0 = __float_as_uint(fmaf(
                __saturatef(fmaf(p0.x, (float)acc[mi][ni][0], p0.y)), 255.0f, 8388608.0f));
            const unsigned q1 = __float_as_uint(fmaf(
                __saturatef(fmaf(p1.x, (float)acc[mi][ni][1], p1.y)), 255.0f, 8388608.0f));
            *(unsigned short*)(sO + r * 144 + c) = (unsigned short)__byte_perm(q0, q1, 0x0040);
        }
    }
    __syncthreads();
    const int r_out = tid >> 3, q_out = tid & 7;
    signed char* po = PL
        ? a.out + ((size_t)(q_out >> 1) * a.rows + m0 + r_out) * 32 + (q_out & 1) * 16
        : a.out + (size_t)(m0 + r_out) * 128 + q_out * 16;
    constexpr size_t o_step = PL ? 32 * 32 : 32 * 128;
    const unsigned char* ps_out = sO + r_out * 144 + q_out * 16;
#pragma unroll
    for (int k = 0; k < BM * 8 / NTH; ++k) {
        if (m0 + r_out + k * 32 < a.rows)
            *(U4*)(po + k * o_step) = *(const U4*)(ps_out + k * 32 * 144);
    }
}

extern "C" __global__ void __launch_bounds__(256, 1) conv1_w64(KArg a) { conv1w_body<0>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) conv1_w64_pl(KArg a) { conv1w_body<1>(a); }

// --- high-occupancy conv1 ---------------------------------------------------------------
// The bare "ldmatrix + mma" loop of conv1 reaches 100% of the IMMA peak already with 8 warps per SM
// (50 TMAC/s), and with a barrier per stage it loses 22% at 8 warps, 10% at 32. In the phased conv1
// the prologue and the loads do not overlap the mma: 2 warps per sub-SM, 128 registers per thread (64
// of accumulators). Here 32 warps per SM (2 blocks of 16, 32 x 32 tile per warp, 32 accumulators, <= 64
// registers): while some warps do prologue and loads, others do mma. Every thread loads and prepares 8
// bytes of A (1/4 of a row) and 8 bytes of weights per stage; branch-free loop in steady state (the
// last stage reloads itself). Same arithmetic as conv1_u128 (MODE 2): u8 NHWC128 bottleneck.
DEV unsigned pro4(unsigned x, const F2* ab) {
    const unsigned w = x ^ 0x80808080u;
    unsigned r[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const float f = __int_as_float(__byte_perm(w, 0x4B000000u, 0x7540 | u)) - 8388736.0f;
        const F2 p = ab[u];
        const float v = __saturatef(fmaf(p.x, f, p.y));
        r[u] = __float_as_uint(fmaf(v, 255.0f, 8388608.0f));
    }
    return __byte_perm(__byte_perm(r[0], r[1], 0x0040), __byte_perm(r[2], r[3], 0x0040), 0x5410);
}

// PH = 1: fp16 prologue (`pro4h`), a.ab points to table L*_abh: per channel pair (A_2p, A_2p+1, B_2p,
// B_2p+1) in fp16, 8 bytes.
template <int BM, int PH = 0>
DEV void conv1_hi_body(const KArg& a) {
    constexpr int NTH = 512, NW = 16;
    constexpr int WARPS_M = BM / 32, WARPS_N = NW / WARPS_M, WN = 128 / WARPS_N, NT = WN / 8;
    constexpr int A_BYTES = BM * 32, B_BYTES = 128 * 32;
    constexpr int STAGE = OROWS_HI(BM);
    constexpr int UNION = 2 * (A_BYTES + B_BYTES) > STAGE ? 2 * (A_BYTES + B_BYTES) : STAGE;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* sA = smem;
    unsigned char* sB = smem + 2 * A_BYTES;
    F2* sAB = (F2*)(smem + UNION);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % WARPS_M, wn = warp / WARPS_M;
    const int m0 = blockIdx.x * BM;
    const int HW = a.HW, in0g = HW * 32, segg = a.N * HW * 32;
    if (PH) {
        for (int i = tid; i < a.G * 16; i += NTH) ((U2*)sAB)[i] = ((const U2*)a.ab)[i];
    } else {
        for (int i = tid; i < a.G * 32; i += NTH) sAB[i] = ((const F2*)a.ab)[i];
    }

    // A: the thread loads 8 bytes (quarter qa) of row ra; with BM = 64 half of the threads
    constexpr int ATH = BM * 4;
    const bool a_thr = ATH >= NTH || tid < ATH;
    const int ra = (tid >> 2) % BM, qa = tid & 3;
    int m = m0 + ra;
    if (m >= a.rows) m = 0;                                 // rows beyond the end: discarded
    const int n = m / HW, px = m - n * HW;
    const int oin0 = (n * a.G0 * HW + px) * 32 + qa * 8;       // 32-bit offsets: fewer registers
    const int oseg = (n * HW + px) * 32 + qa * 8;
    // weights: 8 bytes (quarter qb) of row rb
    const int rb = (tid >> 2) & 127, qb = tid & 3;
    const bool b_thr = tid < 512;
    const int ow = rb * 32 + qb * 8;
    const int wg = a.cout * 32;
    // shared memory positions (16 B halves with the usual swizzle, 8 B inside)
    const int offA = ra * 32 + ((((qa >> 1) ^ ((ra >> 2) & 1))) << 4) + (qa & 1) * 8;
    const int offB = rb * 32 + ((((qb >> 1) ^ ((rb >> 2) & 1))) << 4) + (qb & 1) * 8;

    int acc[4][NT][2];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = 0;

    auto gload = [&](int g, unsigned& a0, unsigned& a1, unsigned& b0, unsigned& b1) {
        const signed char* src = g < a.G0 ? a.in0 + (oin0 + g * in0g)
                                          : a.seg + (oseg + (g - a.G0) * segg);
        if (a_thr) {
            const U2 v = *(const U2*)src;
            a0 = v.x;
            a1 = v.y;
        }
        if (b_thr) {
            const U2 v = *(const U2*)(a.w + (ow + g * wg));
            b0 = v.x;
            b1 = v.y;
        }
    };
    auto sstore = [&](int buf, int g, unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
        if (a_thr) {
            if (PH) {
                const U2* ph = (const U2*)sAB + g * 16 + qa * 4;
                *(U2*)(sA + buf * A_BYTES + offA) = U2{pro4h(a0, ph[0], ph[1]),
                                                       pro4h(a1, ph[2], ph[3])};
            } else {
                const F2* ab = sAB + g * 32 + qa * 8;
                *(U2*)(sA + buf * A_BYTES + offA) = U2{pro4(a0, ab), pro4(a1, ab + 4)};
            }
        }
        if (b_thr) *(U2*)(sB + buf * B_BYTES + offB) = U2{b0, b1};
    };
    auto compute = [&](int buf) {
        const unsigned aBase = smem_addr(sA + buf * A_BYTES);
        const unsigned bBase = smem_addr(sB + buf * B_BYTES);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            unsigned af[4], bf[NT];
            {
                const int row = wm * 32 + lane;
                ldsm4(aBase + row * 32 + ((h ^ ((row >> 2) & 1)) << 4), af);
            }
#pragma unroll
            for (int j = 0; j < NT / 4; ++j) {
                const int row = wn * WN + j * 32 + lane;
                ldsm4(bBase + row * 32 + ((h ^ ((row >> 2) & 1)) << 4), bf + 4 * j);
            }
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) mma_us(acc[mi][ni], af[mi], bf[ni]);
        }
    };
    unsigned a0 = 0u, a1 = 0u, b0 = 0u, b1 = 0u, c0 = 0u, c1 = 0u, d0 = 0u, d1 = 0u;
    gload(0, a0, a1, b0, b1);
    __syncthreads();                                        // sAB
    sstore(0, 0, a0, a1, b0, b1);
    gload(a.G > 1 ? 1 : 0, c0, c1, d0, d1);
    __syncthreads();
    // steady state: stage g in memory (buffer g & 1), g+1 in registers; g+2 is loaded
    for (int g = 0; g < a.G; g += 2) {
        const int g2 = g + 2 < a.G ? g + 2 : a.G - 1;
        gload(g2, a0, a1, b0, b1);
        compute(0);
        if (g + 1 >= a.G) break;
        sstore(1, g + 1, c0, c1, d0, d1);
        __syncthreads();
        const int g3 = g + 3 < a.G ? g + 3 : a.G - 1;
        gload(g3, c0, c1, d0, d1);
        compute(1);
        if (g + 2 >= a.G) break;
        sstore(0, g + 2, a0, a1, b0, b1);
        __syncthreads();
    }
    __syncthreads();
    unsigned char* sO = smem;                               // [BM][144]
    const F2* gd = (const F2*)a.gd;
#pragma unroll
    for (int ni = 0; ni < NT; ++ni) {
        const int c = wn * WN + ni * 8 + (lane & 3) * 2;
        const F2 p0 = gd[c], p1 = gd[c + 1];
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            const int r = wm * 32 + mi * 8 + (lane >> 2);
            const unsigned q0 = __float_as_uint(fmaf(
                __saturatef(fmaf(p0.x, (float)acc[mi][ni][0], p0.y)), 255.0f, 8388608.0f));
            const unsigned q1 = __float_as_uint(fmaf(
                __saturatef(fmaf(p1.x, (float)acc[mi][ni][1], p1.y)), 255.0f, 8388608.0f));
            *(unsigned short*)(sO + r * 144 + c) = (unsigned short)__byte_perm(q0, q1, 0x0040);
        }
    }
    __syncthreads();
    for (int i = tid; i < BM * 8; i += NTH) {
        const int r = i >> 3, q = i & 7;
        const int mm = m0 + r;
        if (mm < a.rows)
            *(U4*)(a.out + (size_t)mm * 128 + q * 16) = *(const U4*)(sO + r * 144 + q * 16);
    }
}

extern "C" __global__ void __launch_bounds__(512, 2) conv1_h128(KArg a) { conv1_hi_body<128>(a); }
extern "C" __global__ void __launch_bounds__(512, 2) conv1_h128h(KArg a) { conv1_hi_body<128, 1>(a); }

// --- warp-specialised conv1 ------------------------------------------------------------
// In conv1 the tensor cores are not the limit (~30% of the IMMA peak). The prologue (~5 instructions
// per byte) and the loads sit between barriers in the same warps that do the MMA, with 128 registers
// per thread (64 of accumulators): the phases do not overlap. Here the roles are separated. NP
// "producer" warps load rows and weights from global memory two stages ahead (three register sets),
// do the prologue and fill a ring of S stages (BK = 32) in shared memory; NMMA "MMA" warps (8: 32 x 64
// tile per warp; 16: 32 x 32) consume them with the tensor cores.
// Named barriers: FULL[s] = 1 + s (producers arrive, MMA warps wait), EMPTY[s] = 1 + S + s (the
// opposite), 1 + 2S among the MMA warps only for the epilogue. Persistent: one block per SM, tiles of
// BM rows with stride gridDim.x; during the MMA epilogue the producers already fill the stages of the
// next tile. Same arithmetic as conv1_u128 (MODE 2): u8 NHWC128 bottleneck.
DEV void bar_sync_n(int id, int n) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
DEV void bar_arrive_n(int id, int n) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}

// DIAG (measurements only): 1 = the MMA warps do not run the mma (the ldmatrix stay), 2 = the producers
// do not load and do no prologue (they write the stages as they were).
template <int BM, int NP, int S, int NMMA = 8, int DIAG = 0, int PH = 0>
DEV void conv1_ws_body(const KArg& a) {
    static_assert(S % 3 == 0, "l'anello si srotola a multipli dei tre giri di registri");
    constexpr int NC = NMMA * 32, NPT = NP * 32, NTH = NC + NPT;
    constexpr int WARPS_M = BM / 32, WARPS_N = NMMA / WARPS_M, WN = 128 / WARPS_N, NT = WN / 8;
    constexpr int A_BYTES = BM * 32, B_BYTES = 128 * 32, STAGE = A_BYTES + B_BYTES;
    constexpr int RPT = 2 * BM / NPT;                     // 16 B chunks of A per producer
    constexpr int WPT = 256 / NPT;                        // 16 B chunks of the weights per producer
    static_assert(RPT >= 1 && RPT <= 2 && WPT >= 1, "una riga di A per produttore al massimo");
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* sO = smem + S * STAGE;                 // [BM][144]
    F2* sAB = (F2*)(sO + BM * 144);                       // [G*32]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int tiles = (a.rows + BM - 1) / BM;
    if (PH) {
        for (int i = tid; i < a.G * 16; i += NTH) ((U2*)sAB)[i] = ((const U2*)a.ab)[i];
    } else {
        for (int i = tid; i < a.G * 32; i += NTH) sAB[i] = ((const F2*)a.ab)[i];
    }
    __syncthreads();
    const int mine = (int)blockIdx.x < tiles ? (tiles - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
    const int T = mine * a.G;                             // stages of this block

    if (warp >= NMMA) {
        // --- producers ---
        const int p = tid - NC;
        const int in0g = a.HW * 32, segg = a.N * a.HW * 32;
        int ld_tile = blockIdx.x, ld_g = 0, o_in0 = 0, o_seg = 0;
        auto set_row = [&](int tile) {
            int m = tile * BM + ((p * RPT) >> 1);
            if (m >= a.rows) m = 0;                       // rows beyond the end: output discarded
            const int n = m / a.HW, px = m - n * a.HW;
            o_in0 = (n * a.G0 * a.HW + px) * 32;
            o_seg = (n * a.HW + px) * 32;
        };
        set_row(ld_tile);
        U4 RA[3][RPT], RB[3][WPT];
        auto load = [&](U4* va, U4* vb) {                 // the stage (ld_tile, ld_g), then forward
            const signed char* src = ld_g < a.G0 ? a.in0 + o_in0 + ld_g * in0g
                                                 : a.seg + o_seg + (ld_g - a.G0) * segg;
#pragma unroll
            for (int k = 0; k < RPT; ++k) va[k] = *(const U4*)(src + (((p * RPT + k) & 1) << 4));
            const signed char* ws = a.w + (size_t)ld_g * a.cout * 32;
#pragma unroll
            for (int k = 0; k < WPT; ++k) {
                const int c = p * WPT + k;
                vb[k] = *(const U4*)(ws + (c >> 1) * 32 + ((c & 1) << 4));
            }
            if (++ld_g == a.G) {
                ld_g = 0;
                ld_tile += gridDim.x;
                set_row(ld_tile);
            }
        };
        int st_g = 0;
        auto put = [&](int it, int slot, const U4* va, const U4* vb) {
            if (it >= S) bar_sync_n(1 + S + slot, NTH);
            unsigned char* sa = smem + slot * STAGE;
            unsigned char* sb = sa + A_BYTES;
#pragma unroll
            for (int k = 0; k < RPT; ++k) {
                const int c = p * RPT + k, row = c >> 1, h = c & 1;
                *(U4*)(sa + row * 32 + ((h ^ ((row >> 2) & 1)) << 4)) =
                    (DIAG & 2) ? va[k]
                               : (PH ? pro16h(va[k], (const U2*)sAB + st_g * 16 + h * 8)
                                     : pro16(va[k], sAB + st_g * 32 + h * 16));
            }
#pragma unroll
            for (int k = 0; k < WPT; ++k) {
                const int c = p * WPT + k, row = c >> 1, h = c & 1;
                *(U4*)(sb + row * 32 + ((h ^ ((row >> 2) & 1)) << 4)) = vb[k];
            }
            bar_arrive_n(1 + slot, NTH);
            if (++st_g == a.G) st_g = 0;
        };
        if (T > 0) load(RA[0], RB[0]);
        if (T > 1) load(RA[1], RB[1]);
        for (int it0 = 0; it0 < T; it0 += S) {
#pragma unroll
            for (int k = 0; k < S; ++k) {
                const int it = it0 + k;
                if (it < T) {
                    if (it + 2 < T && !(DIAG & 2)) load(RA[(k + 2) % 3], RB[(k + 2) % 3]);
                    put(it, k, RA[k % 3], RB[k % 3]);
                }
            }
        }
        // the last arrivals of the MMA warps on EMPTY: barriers balanced before exiting
        for (int it = T > S ? T - S : 0; it < T; ++it) bar_sync_n(1 + S + it % S, NTH);
        return;
    }

    // --- MMA ---
    const int wm = warp % WARPS_M, wn = warp / WARPS_M;
    const F2* gd = (const F2*)a.gd;
    int slot = 0;
    for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        int acc[4][NT][2];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = 0;
        for (int g = 0; g < a.G; ++g) {
            bar_sync_n(1 + slot, NTH);
            const unsigned aBase = smem_addr(smem + slot * STAGE);
            const unsigned bBase = aBase + A_BYTES;
            unsigned af[2][4], bf[2][NT];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = wm * 32 + lane;
                ldsm4(aBase + row * 32 + ((h ^ ((row >> 2) & 1)) << 4), af[h]);
#pragma unroll
                for (int j = 0; j < NT / 4; ++j) {
                    const int rb = wn * WN + j * 32 + lane;
                    ldsm4(bBase + rb * 32 + ((h ^ ((rb >> 2) & 1)) << 4), bf[h] + 4 * j);
                }
            }
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni) {
                        if (DIAG & 1) {
                            if (mi == 0) acc[0][ni][0] ^= (int)(af[h][ni & 3] ^ bf[h][ni]);
                        } else {
                            mma_us(acc[mi][ni], af[h][mi], bf[h][ni]);
                        }
                    }
            bar_arrive_n(1 + S + slot, NTH);
            if (++slot == S) slot = 0;
        }
        // epilogue: rint(255 sat(g' acc + d')) -> sO -> u8 bottleneck [rows][128]
        bar_sync_n(1 + 2 * S, NC);                        // sO fully read by the previous tile
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int c = wn * WN + ni * 8 + (lane & 3) * 2;
            const F2 p0 = gd[c], p1 = gd[c + 1];
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int r = wm * 32 + mi * 8 + (lane >> 2);
                const unsigned q0 = __float_as_uint(fmaf(
                    __saturatef(fmaf(p0.x, (float)acc[mi][ni][0], p0.y)), 255.0f, 8388608.0f));
                const unsigned q1 = __float_as_uint(fmaf(
                    __saturatef(fmaf(p1.x, (float)acc[mi][ni][1], p1.y)), 255.0f, 8388608.0f));
                *(unsigned short*)(sO + r * 144 + c) = (unsigned short)__byte_perm(q0, q1, 0x0040);
            }
        }
        bar_sync_n(1 + 2 * S, NC);
        for (int i = tid; i < BM * 8; i += NC) {
            const int r = i >> 3, q = i & 7;
            const int mm = tile * BM + r;
            if (mm < a.rows)
                *(U4*)(a.out + (size_t)mm * 128 + q * 16) = *(const U4*)(sO + r * 144 + q * 16);
        }
    }
}

extern "C" __global__ void __launch_bounds__(384, 1) conv1_w4(KArg a) { conv1_ws_body<128, 4, 3>(a); }
extern "C" __global__ void __launch_bounds__(384, 1) conv1_w4f(KArg a) { conv1_ws_body<128, 4, 3, 8, 0, 1>(a); }
extern "C" __global__ void __launch_bounds__(512, 1) conv1_w8(KArg a) { conv1_ws_body<128, 8, 3>(a); }
extern "C" __global__ void __launch_bounds__(384, 1) conv1_w4h(KArg a) { conv1_ws_body<64, 4, 3>(a); }
extern "C" __global__ void __launch_bounds__(320, 1) conv1_w2h(KArg a) { conv1_ws_body<64, 2, 3>(a); }
// 16 MMA warps (4 x 4, 32 x 32 tile per warp): in case 2 warps per sub-SM are not enough for the tensor
// cores
extern "C" __global__ void __launch_bounds__(640, 1) conv1_x4(KArg a) { conv1_ws_body<128, 4, 3, 16>(a); }
extern "C" __global__ void __launch_bounds__(768, 1) conv1_x8(KArg a) { conv1_ws_body<128, 8, 3, 16>(a); }
// measurements (see DIAG in conv1_ws_body)
extern "C" __global__ void __launch_bounds__(384, 1) conv1_w4_d1(KArg a) { conv1_ws_body<128, 4, 3, 8, 1>(a); }
extern "C" __global__ void __launch_bounds__(384, 1) conv1_w4_d2(KArg a) { conv1_ws_body<128, 4, 3, 8, 2>(a); }
extern "C" __global__ void __launch_bounds__(640, 1) conv1_x4_d1(KArg a) { conv1_ws_body<128, 4, 3, 16, 1>(a); }
extern "C" __global__ void __launch_bounds__(640, 1) conv1_x4_d2(KArg a) { conv1_ws_body<128, 4, 3, 16, 2>(a); }

// --- whole block for small images (block 3, 9 x 9) ---------------------------------
// A persistent thread block holds P images (P * 81 <= 256 rows) and runs all layers of the block in a
// row: conv1 (fp16 prologue, like conv1_u256h: 16 warps 8 x 2, 32 x 64 tile) over the P * HW rows ->
// u8 bottleneck in shared memory (32 KB, 128 B rows in swizzled chunks) -> conv2 from the bottleneck
// (16 warps 8 x 2: 32 rows x 16 outputs, neighbours outside the image point to row 255, all zero) ->
// layer segment in global memory -> next layer. No per-layer launches, no queues, no bottleneck in
// DRAM. Images do not talk to each other: no barrier across thread blocks. Activation loads are
// ld.global.cg (they bypass L1: the next layer re-reads what this block has just written).
struct LayerP {
    const signed char* w1;      // [G][128][32]
    const unsigned* abh;        // fp16 prologue: G*16 U2 pairs
    const float* gd;            // [128][2] conv1 epilogue (u8 bottleneck)
    const signed char* w2;      // [4][9][32][32]
    const float* sc;            // [32] conv2 scale
    int G, pad;
};
struct ImgArg {
    const signed char* in0;     // [N][G0][HW][32]
    signed char* seg;           // [L][N][HW][32]
    const LayerP* lp;           // [L]
    int N, H, W, G0, L;
    int TH;                     // output rows per band (dense_slide)
};

DEV U4 ldcg16(const void* p) {
    U4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

template <int P>
DEV void dense_img_body(const ImgArg& a) {
    constexpr int NTH = 512, BM = 256;
    constexpr int A_BYTES = BM * 32, B_BYTES = 128 * 32;
    constexpr int ST1 = 2 * (A_BYTES + B_BYTES);                 // 24 KB (conv1)
    constexpr int ST2 = 2 * 9216;                                // 18 KB (conv2 weights)
    constexpr int STG = ST1 > ST2 ? ST1 : ST2;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* sX = smem;                                    // [256][128]
    unsigned char* sS = smem + BM * 128;                         // staging
    U2* sAB = (U2*)(sS + STG);                                   // [G*16]
    unsigned char* sA = sS;
    unsigned char* sB = sS + 2 * A_BYTES;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int HW = a.H * a.W;
    const int rows_valid = P * HW;
    const int segg = a.N * HW * 32;                             // < 2^31 (352 x 81 x 32 x 32)
    // conv1: rows loaded by the thread (row ra, half ha) and warp layout
    const int ra = tid >> 1, ha = tid & 1;
    const int rb = tid >> 1, hb = tid & 1;
    const bool b_thr = tid < 256;
    const int wm = warp & 7, wn = warp >> 3;
    // conv2: output row of every lane (4 m8 blocks of the warp, block i = lane >> 3)
    const int r2 = wm * 32 + (lane >> 3) * 8 + (lane & 7);
    const bool v2 = r2 < rows_valid;
    const int i2 = v2 ? r2 / HW : 0, p2 = v2 ? r2 - i2 * HW : 0;
    const int y2 = p2 / a.W, x2 = p2 - y2 * a.W;

    for (int grp = blockIdx.x; grp * P < a.N; grp += gridDim.x) {
        const int n0 = grp * P;
        const int ri = ra / HW, rp = ra - ri * HW;
        const bool rv = ra < rows_valid && n0 + ri < a.N;
        const int nimg = rv ? n0 + ri : n0;
        const int o_in0 = (nimg * a.G0 * HW + (rv ? rp : 0)) * 32 + ha * 16;
        const int o_seg = (nimg * HW + (rv ? rp : 0)) * 32 + ha * 16;
        const int nval = a.N - n0 < P ? a.N - n0 : P;           // real images of the group
        for (int l = 0; l < a.L; ++l) {
            const LayerP lp = a.lp[l];
            const int G = lp.G;
            __syncthreads();                                     // previous layer finished
            for (int i = tid; i < G * 16; i += NTH) sAB[i] = ((const U2*)lp.abh)[i];
            // --- conv1 (like gemm_q<256, 2, 16, 0, PH=1>) ---
            int acc[4][8][2];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j][0] = acc[i][j][1] = 0;
            U4 va0 = U4{0u, 0u, 0u, 0u}, vb0 = va0, va1 = va0, vb1 = va0;
            auto gload = [&](int g, U4& va, U4& vb) {
                va = ldcg16(g < a.G0 ? a.in0 + (o_in0 + g * HW * 32)
                                     : a.seg + (o_seg + (g - a.G0) * segg));
                if (b_thr) vb = *(const U4*)(lp.w1 + ((g * 128 + rb) * 32 + hb * 16));
            };
            auto sstore = [&](int buf, int g, const U4& va, const U4& vb) {
                *(U4*)(sA + buf * A_BYTES + ra * 32 + ((ha ^ ((ra >> 2) & 1)) << 4)) =
                    pro16h(va, sAB + g * 16 + ha * 8);
                if (b_thr)
                    *(U4*)(sB + buf * B_BYTES + rb * 32 + ((hb ^ ((rb >> 2) & 1)) << 4)) = vb;
            };
            auto compute = [&](int buf) {
                const unsigned aBase = smem_addr(sA + buf * A_BYTES);
                const unsigned bBase = smem_addr(sB + buf * B_BYTES);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    unsigned af[4], bf[8];
                    {
                        const int row = wm * 32 + lane;
                        ldsm4(aBase + row * 32 + ((h ^ ((row >> 2) & 1)) << 4), af);
                    }
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const int row = wn * 64 + j * 32 + lane;
                        ldsm4(bBase + row * 32 + ((h ^ ((row >> 2) & 1)) << 4), bf + 4 * j);
                    }
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                        for (int ni = 0; ni < 8; ++ni) mma_us(acc[mi][ni], af[mi], bf[ni]);
                }
            };
            gload(0, va0, vb0);
            if (G > 1) gload(1, va1, vb1);
            __syncthreads();                                     // sAB
            sstore(0, 0, va0, vb0);
            __syncthreads();
            for (int g = 0; g < G; g += 2) {
                if (g + 2 < G) gload(g + 2, va0, vb0);
                compute(0);
                if (g + 1 < G) sstore(1, g + 1, va1, vb1);
                __syncthreads();
                if (g + 1 >= G) break;
                if (g + 3 < G) gload(g + 3, va1, vb1);
                compute(1);
                if (g + 2 < G) sstore(0, g + 2, va0, vb0);
                __syncthreads();
            }
            // --- epilogue 1: u8 bottleneck into sX (rows outside the images set to zero, row 255 included) ---
            {
                const F2* gd = (const F2*)lp.gd;
#pragma unroll
                for (int ni = 0; ni < 8; ++ni) {
                    const int c = wn * 64 + ni * 8 + (lane & 3) * 2;
                    const F2 p0 = gd[c], p1 = gd[c + 1];
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int r = wm * 32 + mi * 8 + (lane >> 2);
                        const bool ok = r < nval * HW;
                        const unsigned q0 = __float_as_uint(fmaf(
                            __saturatef(fmaf(p0.x, (float)acc[mi][ni][0], p0.y)), 255.0f, 8388608.0f));
                        const unsigned q1 = __float_as_uint(fmaf(
                            __saturatef(fmaf(p1.x, (float)acc[mi][ni][1], p1.y)), 255.0f, 8388608.0f));
                        const unsigned short v = ok ? (unsigned short)__byte_perm(q0, q1, 0x0040) : 0;
                        *(unsigned short*)(sX + r * 128 + ((((c >> 4) ^ (r & 7))) << 4) + (c & 15)) = v;
                    }
                }
            }
            __syncthreads();
            // --- conv2 from the bottleneck: 8 x 2 warps, 32 rows x 16 outputs, 4 blocks of 32 channels ---
            int acc2[4][2][2];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) acc2[i][j][0] = acc2[i][j][1] = 0;
            U4 w0 = U4{0u, 0u, 0u, 0u}, w1 = w0;
            auto wload = [&](int c) {
                w0 = *(const U4*)(lp.w2 + c * 9216 + tid * 16);
                if (tid < 64) w1 = *(const U4*)(lp.w2 + c * 9216 + (tid + 512) * 16);
            };
            auto wstore = [&](int buf) {
                {
                    const int q = tid, r = q >> 1;
                    *(U4*)(sS + buf * 9216 + r * 32 + ((((q & 1) ^ ((r >> 2) & 1))) << 4)) = w0;
                }
                if (tid < 64) {
                    const int q = tid + 512, r = q >> 1;
                    *(U4*)(sS + buf * 9216 + r * 32 + ((((q & 1) ^ ((r >> 2) & 1))) << 4)) = w1;
                }
            };
            wload(0);
            wstore(0);
            __syncthreads();
            const unsigned xb = smem_addr(sX);
#pragma unroll 1
            for (int c = 0; c < 4; ++c) {
                if (c + 1 < 4) wload(c + 1);
                const unsigned wb = smem_addr(sS + (c & 1) * 9216);
#pragma unroll
                for (int t = 0; t < 9; ++t) {
                    const int dy = t / 3 - 1, dx = t % 3 - 1;
                    const int yy = y2 + dy, xx = x2 + dx;
                    const bool in = v2 && yy >= 0 && yy < a.H && xx >= 0 && xx < a.W;
                    const int hr = in ? i2 * HW + yy * a.W + xx : 255;
                    unsigned bf[4];
                    {
                        // rows t*32 + wn*16 + (lane & 15), half lane >> 4: n8 0-1 x half 0-1
                        const int row = t * 32 + wn * 16 + (lane & 15), hh = lane >> 4;
                        ldsm4(wb + row * 32 + ((hh ^ ((row >> 2) & 1)) << 4), bf);
                    }
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        unsigned af[4];
                        const int ch = c * 2 + h;                // 16 B chunk of the row
                        ldsm4(xb + hr * 128 + ((ch ^ (hr & 7)) << 4), af);
#pragma unroll
                        for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                            for (int ni = 0; ni < 2; ++ni)
                                mma_us(acc2[mi][ni], af[mi], bf[ni * 1 + h * 2]);
                    }
                }
                __syncthreads();                                 // buffer c read by everyone
                if (c + 1 < 4) {
                    wstore((c + 1) & 1);
                    __syncthreads();
                }
            }
            // --- epilogue 2: q = rint(clamp(acc sc, +-127)) -> sS [256][32] -> segment l ---
            unsigned char* sO = sS;
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                const int c = wn * 16 + ni * 8 + (lane & 3) * 2;
                const float s0 = lp.sc[c], s1 = lp.sc[c + 1];
#pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    const int r = wm * 32 + mi * 8 + (lane >> 2);
                    const float t0 = fminf(fmaxf((float)acc2[mi][ni][0] * s0, -127.0f), 127.0f);
                    const float t1 = fminf(fmaxf((float)acc2[mi][ni][1] * s1, -127.0f), 127.0f);
                    const unsigned q0 = __float_as_uint(t0 + 12582912.0f);
                    const unsigned q1 = __float_as_uint(t1 + 12582912.0f);
                    *(unsigned short*)(sO + r * 32 + c) = (unsigned short)__byte_perm(q0, q1, 0x0040);
                }
            }
            __syncthreads();
            signed char* out = a.seg + (l * segg + n0 * HW * 32);
            for (int i = tid; i < nval * HW * 2; i += NTH)
                *(U4*)(out + (size_t)i * 16) = *(const U4*)(sO + i * 16);
        }
    }
}

extern "C" __global__ void __launch_bounds__(512, 1) dense_img3(ImgArg a) { dense_img_body<3>(a); }

// --- whole block in bands (block 2, 18 x 18) -----------------------------------------------
// Like dense_img_body, but one image per thread block in two bands of TH = H / 2 output rows: the
// conv1 of each band covers its rows plus one halo row towards the other band (recomputed: 10 x 18 =
// 180 rows against 162, 1.11x), the band's bottleneck in shared memory (WM * 32 rows, the last all zero
// for the outside neighbours), the conv2 the TH x W output rows. No conflict between the bands of a
// layer: the conv1 reads only the segments before l.
template <int WM>
DEV void dense_band_body(const ImgArg& a) {
    constexpr int NTH = WM * 64, BM = WM * 32;
    constexpr int A_BYTES = BM * 32, B_BYTES = 128 * 32;
    constexpr int ST1 = 2 * (A_BYTES + B_BYTES);
    constexpr int ST2 = 2 * 9216;
    constexpr int STG = ST1 > ST2 ? ST1 : ST2;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* sX = smem;                                    // [BM][128]
    unsigned char* sS = smem + BM * 128;
    U2* sAB = (U2*)(sS + STG);
    unsigned char* sA = sS;
    unsigned char* sB = sS + 2 * A_BYTES;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int HW = a.H * a.W, TH = a.H >> 1;
    const int segg = a.N * HW * 32;
    const int ra = tid >> 1, ha = tid & 1;
    const int rb = tid >> 1, hb = tid & 1;
    const bool b_thr = tid < 256;
    const int wm = warp % WM, wn = warp / WM;
    const int ZR = BM - 1;                                       // zero row of the bottleneck

    for (int n = blockIdx.x; n < a.N; n += gridDim.x) {
        for (int l = 0; l < a.L; ++l) {
            const LayerP lp = a.lp[l];
            const int G = lp.G;
            __syncthreads();
            for (int i = tid; i < G * 16; i += NTH) sAB[i] = ((const U2*)lp.abh)[i];
            for (int band = 0; band < 2; ++band) {
                const int y0 = band * TH;
                const int ys = y0 > 0 ? y0 - 1 : 0;
                const int ye = y0 + TH + 1 < a.H ? y0 + TH + 1 : a.H;
                const int m1 = (ye - ys) * a.W;                  // conv1 rows (< BM)
                const bool rv = ra < m1;
                const int px = ys * a.W + (rv ? ra : 0);         // image pixels
                const int o_in0 = (n * a.G0 * HW + px) * 32 + ha * 16;
                const int o_seg = (n * HW + px) * 32 + ha * 16;
                int acc[4][8][2];
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 8; ++j) acc[i][j][0] = acc[i][j][1] = 0;
                U4 va0 = U4{0u, 0u, 0u, 0u}, vb0 = va0, va1 = va0, vb1 = va0;
                auto gload = [&](int g, U4& va, U4& vb) {
                    va = ldcg16(g < a.G0 ? a.in0 + (o_in0 + g * HW * 32)
                                         : a.seg + (o_seg + (g - a.G0) * segg));
                    if (b_thr) vb = *(const U4*)(lp.w1 + ((g * 128 + rb) * 32 + hb * 16));
                };
                auto sstore = [&](int buf, int g, const U4& va, const U4& vb) {
                    *(U4*)(sA + buf * A_BYTES + ra * 32 + ((ha ^ ((ra >> 2) & 1)) << 4)) =
                        pro16h(va, sAB + g * 16 + ha * 8);
                    if (b_thr)
                        *(U4*)(sB + buf * B_BYTES + rb * 32 + ((hb ^ ((rb >> 2) & 1)) << 4)) = vb;
                };
                auto compute = [&](int buf) {
                    const unsigned aBase = smem_addr(sA + buf * A_BYTES);
                    const unsigned bBase = smem_addr(sB + buf * B_BYTES);
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        unsigned af[4], bf[8];
                        {
                            const int row = wm * 32 + lane;
                            ldsm4(aBase + row * 32 + ((h ^ ((row >> 2) & 1)) << 4), af);
                        }
#pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            const int row = wn * 64 + j * 32 + lane;
                            ldsm4(bBase + row * 32 + ((h ^ ((row >> 2) & 1)) << 4), bf + 4 * j);
                        }
#pragma unroll
                        for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                            for (int ni = 0; ni < 8; ++ni) mma_us(acc[mi][ni], af[mi], bf[ni]);
                    }
                };
                __syncthreads();                                 // previous band / sAB
                gload(0, va0, vb0);
                if (G > 1) gload(1, va1, vb1);
                sstore(0, 0, va0, vb0);
                __syncthreads();
                for (int g = 0; g < G; g += 2) {
                    if (g + 2 < G) gload(g + 2, va0, vb0);
                    compute(0);
                    if (g + 1 < G) sstore(1, g + 1, va1, vb1);
                    __syncthreads();
                    if (g + 1 >= G) break;
                    if (g + 3 < G) gload(g + 3, va1, vb1);
                    compute(1);
                    if (g + 2 < G) sstore(0, g + 2, va0, vb0);
                    __syncthreads();
                }
                // epilogue 1: u8 bottleneck of the band (rows beyond m1 set to zero, ZR included)
                {
                    const F2* gd = (const F2*)lp.gd;
#pragma unroll
                    for (int ni = 0; ni < 8; ++ni) {
                        const int c = wn * 64 + ni * 8 + (lane & 3) * 2;
                        const F2 p0 = gd[c], p1 = gd[c + 1];
#pragma unroll
                        for (int mi = 0; mi < 4; ++mi) {
                            const int r = wm * 32 + mi * 8 + (lane >> 2);
                            const unsigned q0 = __float_as_uint(fmaf(
                                __saturatef(fmaf(p0.x, (float)acc[mi][ni][0], p0.y)), 255.0f, 8388608.0f));
                            const unsigned q1 = __float_as_uint(fmaf(
                                __saturatef(fmaf(p1.x, (float)acc[mi][ni][1], p1.y)), 255.0f, 8388608.0f));
                            const unsigned short v = r < m1 ? (unsigned short)__byte_perm(q0, q1, 0x0040) : 0;
                            *(unsigned short*)(sX + r * 128 + ((((c >> 4) ^ (r & 7))) << 4) + (c & 15)) = v;
                        }
                    }
                }
                __syncthreads();
                // conv2: output rows r2 = (y - y0) * W + x of the band
                const int r2 = wm * 32 + (lane >> 3) * 8 + (lane & 7);
                const bool v2 = r2 < TH * a.W;
                const int y2 = y0 + (v2 ? r2 / a.W : 0), x2 = v2 ? r2 % a.W : 0;
                int acc2[4][2][2];
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 2; ++j) acc2[i][j][0] = acc2[i][j][1] = 0;
                // 576 weight chunks of 16 B per block of 32 channels: WPT rounds of NTH threads (with 256 threads 3
                // are needed; 2 would miss tap 8)
                constexpr int WPT = (576 + NTH - 1) / NTH;
                U4 wr[WPT];
                auto wload = [&](int c) {
#pragma unroll
                    for (int k = 0; k < WPT; ++k) {
                        const int q = tid + k * NTH;
                        if (q < 576) wr[k] = *(const U4*)(lp.w2 + c * 9216 + q * 16);
                    }
                };
                auto wstore = [&](int buf) {
#pragma unroll
                    for (int k = 0; k < WPT; ++k) {
                        const int q = tid + k * NTH, r = q >> 1;
                        if (q < 576)
                            *(U4*)(sS + buf * 9216 + r * 32 + ((((q & 1) ^ ((r >> 2) & 1))) << 4)) =
                                wr[k];
                    }
                };
                wload(0);
                wstore(0);
                __syncthreads();
                const unsigned xb = smem_addr(sX);
#pragma unroll 1
                for (int c = 0; c < 4; ++c) {
                    if (c + 1 < 4) wload(c + 1);
                    const unsigned wb = smem_addr(sS + (c & 1) * 9216);
#pragma unroll
                    for (int t = 0; t < 9; ++t) {
                        const int yy = y2 + t / 3 - 1, xx = x2 + t % 3 - 1;
                        const bool in = v2 && yy >= ys && yy < ye && xx >= 0 && xx < a.W;
                        const int hr = in ? (yy - ys) * a.W + xx : ZR;
                        unsigned bf[4];
                        {
                            const int row = t * 32 + wn * 16 + (lane & 15), hh = lane >> 4;
                            ldsm4(wb + row * 32 + ((hh ^ ((row >> 2) & 1)) << 4), bf);
                        }
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            unsigned af[4];
                            const int ch = c * 2 + h;
                            ldsm4(xb + hr * 128 + ((ch ^ (hr & 7)) << 4), af);
#pragma unroll
                            for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                                for (int ni = 0; ni < 2; ++ni)
                                    mma_us(acc2[mi][ni], af[mi], bf[ni + h * 2]);
                        }
                    }
                    __syncthreads();
                    if (c + 1 < 4) {
                        wstore((c + 1) & 1);
                        __syncthreads();
                    }
                }
                // epilogue 2 -> segment l, rows y0 .. y0 + TH - 1 of image n
                unsigned char* sO = sS;
#pragma unroll
                for (int ni = 0; ni < 2; ++ni) {
                    const int c = wn * 16 + ni * 8 + (lane & 3) * 2;
                    const float s0 = lp.sc[c], s1 = lp.sc[c + 1];
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int r = wm * 32 + mi * 8 + (lane >> 2);
                        const float t0 = fminf(fmaxf((float)acc2[mi][ni][0] * s0, -127.0f), 127.0f);
                        const float t1 = fminf(fmaxf((float)acc2[mi][ni][1] * s1, -127.0f), 127.0f);
                        const unsigned q0 = __float_as_uint(t0 + 12582912.0f);
                        const unsigned q1 = __float_as_uint(t1 + 12582912.0f);
                        *(unsigned short*)(sO + r * 32 + c) = (unsigned short)__byte_perm(q0, q1, 0x0040);
                    }
                }
                __syncthreads();
                signed char* out = a.seg + (l * segg + (n * HW + y0 * a.W) * 32);
                for (int i = tid; i < TH * a.W * 2; i += NTH)
                    *(U4*)(out + i * 16) = *(const U4*)(sO + i * 16);
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(384, 1) dense_band2(ImgArg a) { dense_band_body<6>(a); }

// --- layer fused in rectangles (blocks 0 and 1): conv1 + conv2 in one launch per layer ----------
// The conv1 of blocks 0-2 already runs at 170-190 GB/s (close to the ~200-226 practical DRAM limit):
// the only way down is reading less. The bottleneck (u8, 128 channels) went to DRAM and back for the
// conv2 (~17 MB per image in blocks 0-2, ~27 ms of DRAM alone for 352 images), and the separate conv2
// spent half its time bringing halo and weights into shared memory. Here a persistent thread block
// loops over the TH x TW output rectangles of all images: conv1 (fp16 prologue, like dense_band_body)
// over the (TH + 2) x (TW + 2) halo rows (recomputed between neighbouring rectangles: 9 x 18 -> 11 x 20
// = 220 rows, 1.36x the conv1) -> u8 bottleneck in shared memory (rows outside the image set to zero:
// the conv2 padding) -> conv2 from the bottleneck (the tap neighbours are always inside the halo) ->
// layer segment. conv1 and conv2 weights from L2 at every rectangle. The first two conv1 stages of the
// next rectangle are loaded into registers during the conv2.
struct TileArg {
    const signed char* in0;     // [N][G0][HW][32]
    signed char* seg;           // [L][N][HW][32]
    const LayerP* lp;           // the layer
    int N, H, W, G0, l, TH, TW, pad;
};

template <int WM, bool TR = false>
DEV void dense_tile_body(const TileArg& a, unsigned long long* tr = nullptr) {
    constexpr int NTH = WM * 64, BM = WM * 32;
    constexpr int A_BYTES = BM * 32, B_BYTES = 128 * 32;
    constexpr int ST1 = 2 * (A_BYTES + B_BYTES);
    constexpr int ST2 = 2 * 9216;
    constexpr int STG = ST1 > ST2 ? ST1 : ST2;
    constexpr int WPT = (576 + NTH - 1) / NTH;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* sX = smem;                                    // [BM][128] bottleneck
    unsigned char* sS = smem + BM * 128;                         // staging
    U2* sAB = (U2*)(sS + STG);
    unsigned char* sA = sS;
    unsigned char* sB = sS + 2 * A_BYTES;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int HW = a.H * a.W, TH = a.TH, TW = a.TW, HWd = TW + 2;
    const int M1 = (TH + 2) * HWd, M2 = TH * TW;
    const int txn = (a.W + TW - 1) / TW, tyn = (a.H + TH - 1) / TH, tpi = txn * tyn;
    const int total = a.N * tpi;
    const int segg = a.N * HW * 32;
    const int ra = tid >> 1, ha = tid & 1;
    const int rb = tid >> 1, hb = tid & 1;
    const bool b_thr = tid < 256;
    const int wm = warp % WM, wn = warp / WM;
    const LayerP lp = *a.lp;
    const int G = lp.G;
    TraceW tw;
    tr_init<TR>(tw);
    for (int i = tid; i < G * 16; i += NTH) sAB[i] = ((const U2*)lp.abh)[i];
    // conv2 row of every lane (4 m8 blocks of the warp) and its 9 neighbours in the halo
    const int r2 = wm * 32 + (lane >> 3) * 8 + (lane & 7);
    const bool v2 = r2 < M2;
    const int oy = v2 ? r2 / TW : 0, ox = v2 ? r2 - (r2 / TW) * TW : 0;
    const int hbase = oy * HWd + ox;
    const bool c2w = wm * 32 < M2;                               // warps with real conv2 rows

    // conv1 row (halo pixel) of the thread for rectangle `tile`
    int o_in0 = 0, o_seg = 0;
    bool rv = false;
    auto rows_of = [&](int tile) {
        const int n = tile / tpi, t = tile - n * tpi;
        const int ty0 = (t / txn) * TH, tx0 = (t - (t / txn) * txn) * TW;
        const int hy = ra / HWd, hx = ra - hy * HWd;
        const int y = ty0 - 1 + hy, x = tx0 - 1 + hx;
        rv = ra < M1 && y >= 0 && y < a.H && x >= 0 && x < a.W;
        const int px = rv ? y * a.W + x : 0;
        o_in0 = (n * a.G0 * HW + px) * 32 + ha * 16;
        o_seg = (n * HW + px) * 32 + ha * 16;
    };
    U4 va0 = U4{0u, 0u, 0u, 0u}, vb0 = va0, va1 = va0, vb1 = va0;
    auto gload = [&](int g, U4& va, U4& vb) {
        va = rv ? ldcg16(g < a.G0 ? a.in0 + (o_in0 + g * HW * 32)
                                  : a.seg + (o_seg + (g - a.G0) * segg))
                : U4{0u, 0u, 0u, 0u};
        if (b_thr) vb = *(const U4*)(lp.w1 + ((g * 128 + rb) * 32 + hb * 16));
    };
    auto sstore = [&](int buf, int g, const U4& va, const U4& vb) {
        *(U4*)(sA + buf * A_BYTES + ra * 32 + ((ha ^ ((ra >> 2) & 1)) << 4)) =
            pro16h(va, sAB + g * 16 + ha * 8);
        if (b_thr)
            *(U4*)(sB + buf * B_BYTES + rb * 32 + ((hb ^ ((rb >> 2) & 1)) << 4)) = vb;
    };
    int tile = blockIdx.x;
    if (tile >= total) return;
    rows_of(tile);
    gload(0, va0, vb0);
    if (G > 1) gload(1, va1, vb1);
    __syncthreads();                                             // sAB
    tr_mark<TR>(tw, 0);
    for (; tile < total; tile += gridDim.x) {
        const int n = tile / tpi, t = tile - n * tpi;
        const int ty0 = (t / txn) * TH, tx0 = (t - (t / txn) * txn) * TW;
        const bool rv_t = rv;                                    // rows of this rectangle
        // --- conv1 (stages 0 and 1 already in registers) ---
        int acc[4][8][2];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[i][j][0] = acc[i][j][1] = 0;
        auto compute = [&](int buf) {
            const unsigned aBase = smem_addr(sA + buf * A_BYTES);
            const unsigned bBase = smem_addr(sB + buf * B_BYTES);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                unsigned af[4], bf[8];
                {
                    const int row = wm * 32 + lane;
                    ldsm4(aBase + row * 32 + ((h ^ ((row >> 2) & 1)) << 4), af);
                }
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int row = wn * 64 + j * 32 + lane;
                    ldsm4(bBase + row * 32 + ((h ^ ((row >> 2) & 1)) << 4), bf + 4 * j);
                }
#pragma unroll
                for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                    for (int ni = 0; ni < 8; ++ni) mma_us(acc[mi][ni], af[mi], bf[ni]);
            }
        };
        sstore(0, 0, va0, vb0);
        tr_mark<TR>(tw, 3);
        __syncthreads();
        tr_mark<TR>(tw, 4);
        for (int g = 0; g < G; g += 2) {
            if (g + 2 < G) gload(g + 2, va0, vb0);
            tr_mark<TR>(tw, 1);
            compute(0);
            tr_mark<TR>(tw, 2);
            if (g + 1 < G) sstore(1, g + 1, va1, vb1);
            tr_mark<TR>(tw, 3);
            __syncthreads();
            tr_mark<TR>(tw, 4);
            if (g + 1 >= G) break;
            if (g + 3 < G) gload(g + 3, va1, vb1);
            tr_mark<TR>(tw, 1);
            compute(1);
            tr_mark<TR>(tw, 2);
            if (g + 2 < G) sstore(0, g + 2, va0, vb0);
            tr_mark<TR>(tw, 3);
            __syncthreads();
            tr_mark<TR>(tw, 4);
        }
        // --- epilogue 1: u8 bottleneck into sX (rows outside the image or beyond M1 set to zero) ---
        {
            const F2* gd = (const F2*)lp.gd;
#pragma unroll
            for (int ni = 0; ni < 8; ++ni) {
                const int c = wn * 64 + ni * 8 + (lane & 3) * 2;
                const F2 p0 = gd[c], p1 = gd[c + 1];
#pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    const int r = wm * 32 + mi * 8 + (lane >> 2);
                    const int hy = r / HWd, hx = r - hy * HWd;
                    const int y = ty0 - 1 + hy, x = tx0 - 1 + hx;
                    const bool ok = r < M1 && y >= 0 && y < a.H && x >= 0 && x < a.W;
                    const unsigned q0 = __float_as_uint(fmaf(
                        __saturatef(fmaf(p0.x, (float)acc[mi][ni][0], p0.y)), 255.0f, 8388608.0f));
                    const unsigned q1 = __float_as_uint(fmaf(
                        __saturatef(fmaf(p1.x, (float)acc[mi][ni][1], p1.y)), 255.0f, 8388608.0f));
                    const unsigned short v = ok ? (unsigned short)__byte_perm(q0, q1, 0x0040) : 0;
                    *(unsigned short*)(sX + r * 128 + ((((c >> 4) ^ (r & 7))) << 4) + (c & 15)) = v;
                }
            }
        }
        (void)rv_t;
        // the first two conv1 stages of the next rectangle: in registers during the conv2
        const int next = tile + gridDim.x;
        if (next < total) {
            rows_of(next);
            gload(0, va0, vb0);
            if (G > 1) gload(1, va1, vb1);
        }
        tr_mark<TR>(tw, 5);
        // --- conv2 from the bottleneck: WM x 2 warps, 32 rows x 16 outputs, 4 blocks of 32 channels ---
        int acc2[4][2][2];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j) acc2[i][j][0] = acc2[i][j][1] = 0;
        U4 wr[WPT];
        auto wload = [&](int c) {
#pragma unroll
            for (int k = 0; k < WPT; ++k) {
                const int q = tid + k * NTH;
                if (q < 576) wr[k] = *(const U4*)(lp.w2 + c * 9216 + q * 16);
            }
        };
        auto wstore = [&](int buf) {
#pragma unroll
            for (int k = 0; k < WPT; ++k) {
                const int q = tid + k * NTH, r = q >> 1;
                if (q < 576)
                    *(U4*)(sS + buf * 9216 + r * 32 + ((((q & 1) ^ ((r >> 2) & 1))) << 4)) = wr[k];
            }
        };
        wload(0);
        wstore(0);                                               // sS free: conv1 finished
        tr_mark<TR>(tw, 6);
        __syncthreads();                                         // sX and weights 0
        tr_mark<TR>(tw, 8);
        const unsigned xb = smem_addr(sX);
#pragma unroll 1
        for (int c = 0; c < 4; ++c) {
            if (c + 1 < 4) wload(c + 1);
            tr_mark<TR>(tw, 6);
            const unsigned wb = smem_addr(sS + (c & 1) * 9216);
            if (c2w) {
#pragma unroll
                for (int tp = 0; tp < 9; ++tp) {
                    const int hr = v2 ? hbase + (tp / 3) * HWd + (tp % 3) : 0;
                    unsigned bf[4];
                    {
                        const int row = tp * 32 + wn * 16 + (lane & 15), hh = lane >> 4;
                        ldsm4(wb + row * 32 + ((hh ^ ((row >> 2) & 1)) << 4), bf);
                    }
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        unsigned af[4];
                        const int ch = c * 2 + h;
                        ldsm4(xb + hr * 128 + ((ch ^ (hr & 7)) << 4), af);
#pragma unroll
                        for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                            for (int ni = 0; ni < 2; ++ni)
                                mma_us(acc2[mi][ni], af[mi], bf[ni + h * 2]);
                    }
                }
            }
            tr_mark<TR>(tw, 7);
            __syncthreads();
            tr_mark<TR>(tw, 8);
            if (c + 1 < 4) {
                wstore((c + 1) & 1);
                tr_mark<TR>(tw, 6);
                __syncthreads();
                tr_mark<TR>(tw, 8);
            }
        }
        // --- epilogue 2 -> sS [BM][32] -> segment l ---
        unsigned char* sO = sS;
        if (c2w) {
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                const int c = wn * 16 + ni * 8 + (lane & 3) * 2;
                const float s0 = lp.sc[c], s1 = lp.sc[c + 1];
#pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    const int r = wm * 32 + mi * 8 + (lane >> 2);
                    const float t0 = fminf(fmaxf((float)acc2[mi][ni][0] * s0, -127.0f), 127.0f);
                    const float t1 = fminf(fmaxf((float)acc2[mi][ni][1] * s1, -127.0f), 127.0f);
                    const unsigned q0 = __float_as_uint(t0 + 12582912.0f);
                    const unsigned q1 = __float_as_uint(t1 + 12582912.0f);
                    *(unsigned short*)(sO + r * 32 + c) = (unsigned short)__byte_perm(q0, q1, 0x0040);
                }
            }
        }
        __syncthreads();
        tr_mark<TR>(tw, 9);
        signed char* out = a.seg + ((size_t)a.l * segg + (size_t)n * HW * 32);
        for (int i = tid; i < M2 * 2; i += NTH) {
            const int r = i >> 1, yy = ty0 + r / TW, xx = tx0 + (r - (r / TW) * TW);
            if (yy < a.H && xx < a.W)
                *(U4*)(out + (size_t)(yy * a.W + xx) * 32 + (i & 1) * 16) = *(const U4*)(sO + i * 16);
        }
        __syncthreads();                                         // sS reused by stage 0
        tr_mark<TR>(tw, 10);
    }
    tr_done<TR>(tw, tr);
}

extern "C" __global__ void __launch_bounds__(448) dense_tile7(TileArg a) { dense_tile_body<7>(a); }
extern "C" __global__ void __launch_bounds__(448) dense_tile7_t(TileArg a, unsigned long long* tr) {
    dense_tile_body<7, true>(a, tr);
}

// --- whole block in sliding bands (blocks 0 and 1: 72 x 72 at TH = 2, 36 x 36 at TH = 4) ----
// One image per thread block; the bottleneck lives in a ring of R = TH + 2 image rows in shared memory
// (row y in slot y mod R) plus a zero row. Step -1: conv1 of row 0; step b: conv1 of rows TH b + 1 ..
// TH b + TH (the new ones), then conv2 of output rows TH b .. TH b + TH - 1, which read rows
// TH b - 1 .. TH b + TH already in the ring. No row recomputed; TH * W = 144 GEMM rows per step (WM = 5:
// 160).
// MERGED = true: step 0 immediately does conv1 rows 0 .. TH (TH + 1 rows, no step -1), for small images
// where (TH + 1) * W fits in the block rows (block 2: 7 x 18 = 126 <= 128).
template <int WM, bool MERGED = false>
DEV void dense_slide_body(const ImgArg& a) {
    constexpr int NTH = WM * 64, BM = WM * 32;
    constexpr int A_BYTES = BM * 32, B_BYTES = 128 * 32;
    constexpr int ST1 = 2 * (A_BYTES + B_BYTES);
    constexpr int ST2 = 2 * 9216;
    constexpr int STG = ST1 > ST2 ? ST1 : ST2;
    extern __shared__ __align__(16) unsigned char smem[];
    const int TH = a.TH, R = TH + 2, W = a.W;
    const int ZR = R * W;                                        // zero row of the ring
    const int xbytes = (ZR + 1) * 128;
    unsigned char* sX = smem;
    unsigned char* sS = smem + xbytes;
    U2* sAB = (U2*)(sS + STG);
    unsigned char* sA = sS;
    unsigned char* sB = sS + 2 * A_BYTES;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int HW = a.H * W;
    const int segg = a.N * HW * 32;
    const int ra = tid >> 1, ha = tid & 1;
    const int rb = tid >> 1, hb = tid & 1;
    const bool b_thr = tid < 256;
    const int wm = warp % WM, wn = warp / WM;
    const int nsteps = a.H / TH;

    for (int n = blockIdx.x; n < a.N; n += gridDim.x) {
        for (int l = 0; l < a.L; ++l) {
            const LayerP lp = a.lp[l];
            const int G = lp.G;
            __syncthreads();
            for (int i = tid; i < G * 16; i += NTH) sAB[i] = ((const U2*)lp.abh)[i];
            if (tid < 8) *(U4*)(sX + ZR * 128 + tid * 16) = U4{0u, 0u, 0u, 0u};
            for (int step = MERGED ? 0 : -1; step < nsteps; ++step) {
                const int ya = step < 0 ? 0 : (MERGED && step == 0 ? 0 : TH * step + 1);
                const int yb = step < 0 ? 1 : TH * step + TH + 1;
                const int m1 = (yb - ya) * W;                    // GEMM rows (<= BM)
                // row ra -> pixel (ya + ra / W, ra % W), inside the image
                int yr = ya + (ra < m1 ? ra / W : 0);
                const int xr = ra < m1 ? ra % W : 0;
                if (yr >= a.H) yr = a.H - 1;
                const int px = yr * W + xr;
                const int o_in0 = (n * a.G0 * HW + px) * 32 + ha * 16;
                const int o_seg = (n * HW + px) * 32 + ha * 16;
                int acc[4][8][2];
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 8; ++j) acc[i][j][0] = acc[i][j][1] = 0;
                U4 va0 = U4{0u, 0u, 0u, 0u}, vb0 = va0, va1 = va0, vb1 = va0;
                auto gload = [&](int g, U4& va, U4& vb) {
                    va = ldcg16(g < a.G0 ? a.in0 + (o_in0 + g * HW * 32)
                                         : a.seg + (o_seg + (g - a.G0) * segg));
                    if (b_thr) vb = *(const U4*)(lp.w1 + ((g * 128 + rb) * 32 + hb * 16));
                };
                auto sstore = [&](int buf, int g, const U4& va, const U4& vb) {
                    *(U4*)(sA + buf * A_BYTES + ra * 32 + ((ha ^ ((ra >> 2) & 1)) << 4)) =
                        pro16h(va, sAB + g * 16 + ha * 8);
                    if (b_thr)
                        *(U4*)(sB + buf * B_BYTES + rb * 32 + ((hb ^ ((rb >> 2) & 1)) << 4)) = vb;
                };
                auto compute = [&](int buf) {
                    const unsigned aBase = smem_addr(sA + buf * A_BYTES);
                    const unsigned bBase = smem_addr(sB + buf * B_BYTES);
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        unsigned af[4], bf[8];
                        {
                            const int row = wm * 32 + lane;
                            ldsm4(aBase + row * 32 + ((h ^ ((row >> 2) & 1)) << 4), af);
                        }
#pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            const int row = wn * 64 + j * 32 + lane;
                            ldsm4(bBase + row * 32 + ((h ^ ((row >> 2) & 1)) << 4), bf + 4 * j);
                        }
#pragma unroll
                        for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                            for (int ni = 0; ni < 8; ++ni) mma_us(acc[mi][ni], af[mi], bf[ni]);
                    }
                };
                __syncthreads();                                 // previous step / sAB
                gload(0, va0, vb0);
                if (G > 1) gload(1, va1, vb1);
                sstore(0, 0, va0, vb0);
                __syncthreads();
                for (int g = 0; g < G; g += 2) {
                    if (g + 2 < G) gload(g + 2, va0, vb0);
                    compute(0);
                    if (g + 1 < G) sstore(1, g + 1, va1, vb1);
                    __syncthreads();
                    if (g + 1 >= G) break;
                    if (g + 3 < G) gload(g + 3, va1, vb1);
                    compute(1);
                    if (g + 2 < G) sstore(0, g + 2, va0, vb0);
                    __syncthreads();
                }
                // epilogue 1: the new rows into the ring (slot y mod R); nothing outside the image
                {
                    const F2* gd = (const F2*)lp.gd;
#pragma unroll
                    for (int ni = 0; ni < 8; ++ni) {
                        const int c = wn * 64 + ni * 8 + (lane & 3) * 2;
                        const F2 p0 = gd[c], p1 = gd[c + 1];
#pragma unroll
                        for (int mi = 0; mi < 4; ++mi) {
                            const int r = wm * 32 + mi * 8 + (lane >> 2);
                            const int y = ya + r / W, x = r % W;
                            if (r < m1 && y < a.H) {
                                const int sr = (y % R) * W + x;
                                const unsigned q0 = __float_as_uint(fmaf(
                                    __saturatef(fmaf(p0.x, (float)acc[mi][ni][0], p0.y)), 255.0f, 8388608.0f));
                                const unsigned q1 = __float_as_uint(fmaf(
                                    __saturatef(fmaf(p1.x, (float)acc[mi][ni][1], p1.y)), 255.0f, 8388608.0f));
                                *(unsigned short*)(sX + sr * 128 + ((((c >> 4) ^ (sr & 7))) << 4) + (c & 15)) =
                                    (unsigned short)__byte_perm(q0, q1, 0x0040);
                            }
                        }
                    }
                }
                __syncthreads();
                if (step < 0) continue;
                // conv2 of output rows y0 .. y0 + TH - 1
                const int y0 = TH * step;
                const int r2 = wm * 32 + (lane >> 3) * 8 + (lane & 7);
                const bool v2 = r2 < TH * W;
                const int y2 = y0 + (v2 ? r2 / W : 0), x2 = v2 ? r2 % W : 0;
                int acc2[4][2][2];
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 2; ++j) acc2[i][j][0] = acc2[i][j][1] = 0;
                // 576 weight chunks of 16 B per block of 32 channels: WPT rounds of NTH threads (with 256 threads 3
                // are needed; 2 would miss tap 8)
                constexpr int WPT = (576 + NTH - 1) / NTH;
                U4 wr[WPT];
                auto wload = [&](int c) {
#pragma unroll
                    for (int k = 0; k < WPT; ++k) {
                        const int q = tid + k * NTH;
                        if (q < 576) wr[k] = *(const U4*)(lp.w2 + c * 9216 + q * 16);
                    }
                };
                auto wstore = [&](int buf) {
#pragma unroll
                    for (int k = 0; k < WPT; ++k) {
                        const int q = tid + k * NTH, r = q >> 1;
                        if (q < 576)
                            *(U4*)(sS + buf * 9216 + r * 32 + ((((q & 1) ^ ((r >> 2) & 1))) << 4)) =
                                wr[k];
                    }
                };
                wload(0);
                wstore(0);
                __syncthreads();
                const unsigned xb = smem_addr(sX);
#pragma unroll 1
                for (int c = 0; c < 4; ++c) {
                    if (c + 1 < 4) wload(c + 1);
                    const unsigned wb = smem_addr(sS + (c & 1) * 9216);
#pragma unroll
                    for (int t = 0; t < 9; ++t) {
                        const int yy = y2 + t / 3 - 1, xx = x2 + t % 3 - 1;
                        const bool in = v2 && yy >= 0 && yy < a.H && xx >= 0 && xx < W;
                        const int hr = in ? (yy % R) * W + xx : ZR;
                        unsigned bf[4];
                        {
                            const int row = t * 32 + wn * 16 + (lane & 15), hh = lane >> 4;
                            ldsm4(wb + row * 32 + ((hh ^ ((row >> 2) & 1)) << 4), bf);
                        }
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            unsigned af[4];
                            const int ch = c * 2 + h;
                            ldsm4(xb + hr * 128 + ((ch ^ (hr & 7)) << 4), af);
#pragma unroll
                            for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                                for (int ni = 0; ni < 2; ++ni)
                                    mma_us(acc2[mi][ni], af[mi], bf[ni + h * 2]);
                        }
                    }
                    __syncthreads();
                    if (c + 1 < 4) {
                        wstore((c + 1) & 1);
                        __syncthreads();
                    }
                }
                unsigned char* sO = sS;
#pragma unroll
                for (int ni = 0; ni < 2; ++ni) {
                    const int c = wn * 16 + ni * 8 + (lane & 3) * 2;
                    const float s0 = lp.sc[c], s1 = lp.sc[c + 1];
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int r = wm * 32 + mi * 8 + (lane >> 2);
                        const float t0 = fminf(fmaxf((float)acc2[mi][ni][0] * s0, -127.0f), 127.0f);
                        const float t1 = fminf(fmaxf((float)acc2[mi][ni][1] * s1, -127.0f), 127.0f);
                        const unsigned q0 = __float_as_uint(t0 + 12582912.0f);
                        const unsigned q1 = __float_as_uint(t1 + 12582912.0f);
                        *(unsigned short*)(sO + r * 32 + c) = (unsigned short)__byte_perm(q0, q1, 0x0040);
                    }
                }
                __syncthreads();
                signed char* out = a.seg + (l * segg + (n * HW + y0 * W) * 32);
                for (int i = tid; i < TH * W * 2; i += NTH)
                    *(U4*)(out + i * 16) = *(const U4*)(sO + i * 16);
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(320, 1) dense_slide(ImgArg a) { dense_slide_body<5>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) dense_slide4(ImgArg a) { dense_slide_body<4, true>(a); }

// --- 3x3 conv2 (128 -> 32) on the tensor cores ---------------------------------------------------
// Input: the u8 bottleneck [N][H][W][128] of conv1_u*; weights [4 blocks of 32 channels][9 taps][32
// outputs][32 inputs] int8; output q = rint(clamp(acc * sc_o, +-127)) into the store slot [N][H][W][32]
// (sc = s2 w2s / sigma). A thread block = a TH x TW rectangle of outputs of one image (M = TH*TW GEMM
// rows, split into chunks of 8 among the warps: MW chunks per warp), K = 9 x 128 in four blocks of 32
// channels: for every block the (TH+2) x (TW+2) x 32-byte halo and the 9 x 32 x 32 weights in shared
// memory, double-buffered. Every GEMM row addresses its own halo row (ldmatrix takes one address per
// row): the rectangle does not need rows of 8 pixels. Outside the image the halo is 0 (the conv2
// padding on the bottleneck, after ReLU).
struct C2Arg {
    const unsigned char* x;
    const signed char* w;
    const float* sc;
    signed char* out;
    int N, H, W, TH, TW, txn;   // txn = rectangles per image row
};

DEV int c2_halo_bytes(int TH, int TW) { return (((TH + 2) * (TW + 2) * 32 + 127) / 128) * 128; }

// PD = 2: the loads of channel block c + 2 start while c is computed (two register sets): ~35-50% of
// the time was spent waiting for the data of the next block, loaded only one block ahead. TR: phase
// tracing (`*_t`).
// PL = 1: the planar bottleneck [4][N*H*W][32] (conv1 MODE 3) instead of NHWC128.
template <int MW, int WARPS, int DIAG = 0, bool UNROLL = false, int PD = 1, bool TR = false,
          int PL = 0>
DEV void conv2_body(const C2Arg& a, unsigned long long* tr = nullptr) {
    constexpr int NT = WARPS * 32;
    constexpr int NG = (MW + 3) / 4;
    constexpr int HPC = 4;                              // 16 B halo chunks per thread
    constexpr int WPC = (576 + NT - 1) / NT;            // weight chunks per thread
    extern __shared__ __align__(16) unsigned char smem[];
    const int HWd = a.TW + 2, HP = (a.TH + 2) * HWd;
    const int HB = c2_halo_bytes(a.TH, a.TW);
    unsigned char* sX = smem;                            // [2][HP][32]
    unsigned char* sW = smem + 2 * HB;                   // [2][288][32]
    float* sSc = (float*)(sW + 2 * 9216);                // [32]

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n = blockIdx.y;
    const int ty0 = (blockIdx.x / a.txn) * a.TH, tx0 = (blockIdx.x % a.txn) * a.TW;
    const int M = a.TH * a.TW;
    if (tid < 32) sSc[tid] = a.sc[tid];

    int base[NG];
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        const int i = g * 4 + (lane >> 3);
        const int m = (warp * MW + i) * 8 + (lane & 7);
        const int mm = (i < MW && m < M) ? m : 0;
        base[g] = (mm / a.TW) * HWd + (mm % a.TW);
    }
    int acc[MW][4][2];
#pragma unroll
    for (int i = 0; i < MW; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = 0;
    TraceW tw;
    tr_init<TR>(tw);

    U4 hxA[HPC], hwA[WPC], hxB[HPC], hwB[WPC];
    // the thread's halo chunks (no divisions at every channel block): offset from pixel (ty0 - 1, tx0 - 1)
    // and position (row << 16 | column, 0xffffffff = outside)
    int hoff[HPC];
    unsigned hpos[HPC];
#pragma unroll
    for (int k = 0; k < HPC; ++k) {
        const int p = tid + k * NT;
        const int hr = p >> 1, hy = hr / HWd, hxx = hr - hy * HWd;
        const bool in = p < 2 * HP;
        hoff[k] = in ? (hy * a.W + hxx) * (PL ? 32 : 128) + (p & 1) * 16 : 0;
        hpos[k] = in ? ((unsigned)hy << 16) | (unsigned)hxx : 0xffffffffu;
    }
    constexpr int PX = PL ? 32 : 128;                   // bytes of one pixel in the bottleneck
    const size_t cstep = PL ? (size_t)a.N * a.H * a.W * 32 : 32;   // stride of a channel block
    const unsigned char* xn = a.x + (size_t)n * a.H * a.W * PX
                              + ((ptrdiff_t)(ty0 - 1) * a.W + (tx0 - 1)) * PX;

    auto load = [&](int c, U4* hx, U4* hw) {
        if ((DIAG & 2) && c > 1) return;
        if (DIAG & 8) c &= 1;
#pragma unroll
        for (int k = 0; k < HPC; ++k) {
            const int y = ty0 - 1 + (int)(hpos[k] >> 16), x = tx0 - 1 + (int)(hpos[k] & 0xffffu);
            hx[k] = ((unsigned)y < (unsigned)a.H && (unsigned)x < (unsigned)a.W)
                    ? *(const U4*)(xn + hoff[k] + c * cstep) : U4{0u, 0u, 0u, 0u};
        }
#pragma unroll
        for (int k = 0; k < WPC; ++k) {
            const int q = tid + k * NT;
            if (q < 576) hw[k] = *(const U4*)(a.w + c * 9216 + q * 16);
        }
    };
    auto store = [&](int buf, const U4* hx, const U4* hw) {
        if (DIAG & 32) return;
#pragma unroll
        for (int k = 0; k < HPC; ++k) {
            const int p = tid + k * NT;
            if (p < 2 * HP) {
                const int hr = p >> 1;
                *(U4*)(sX + buf * HB + hr * 32 + (((p & 1) ^ ((hr >> 2) & 1)) << 4)) = hx[k];
            }
        }
#pragma unroll
        for (int k = 0; k < WPC; ++k) {
            const int q = tid + k * NT;
            if (q < 576) {
                const int r = q >> 1;
                *(U4*)(sW + buf * 9216 + r * 32 + (((q & 1) ^ ((r >> 2) & 1)) << 4)) = hw[k];
            }
        }
    };

    load(0, hxA, hwA);
    if (PD == 2) load(1, hxB, hwB);
    __syncthreads();                                    // sSc
    store(0, hxA, hwA);
    __syncthreads();
    tr_mark<TR>(tw, 0);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int buf = c & 1;
        if (PD == 2) {
            // set A: even blocks, B: odd; c + 2 goes into c's set (already written)
            if (c + 2 < 4) {
                if (c & 1) load(c + 2, hxB, hwB);
                else load(c + 2, hxA, hwA);
            }
        } else if (c + 1 < 4) {
            load(c + 1, hxA, hwA);
        }
        tr_mark<TR>(tw, 1);
        const unsigned xb = smem_addr(sX + buf * HB);
        const unsigned wb = smem_addr(sW + buf * 9216);
        auto tap = [&](int t) {
            const int toff = (t / 3) * HWd + (t % 3);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                unsigned bf[4];
                {
                    const int row = t * 32 + lane;
                    ldsm4(wb + row * 32 + ((h ^ ((row >> 2) & 1)) << 4), bf);
                }
#pragma unroll
                for (int g = 0; g < NG; ++g) {
                    unsigned af[4];
                    const int hr = base[g] + toff;
                    ldsm4(xb + hr * 32 + ((h ^ ((hr >> 2) & 1)) << 4), af);
                    if (DIAG & 64) {
                        acc[0][0][0] ^= (int)(af[0] ^ af[1] ^ af[2] ^ af[3] ^ bf[0] ^ bf[1] ^ bf[2] ^ bf[3]);
                    } else {
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (g * 4 + j < MW)
#pragma unroll
                            for (int ni = 0; ni < 4; ++ni) {
                                if (DIAG & 1) acc[g * 4 + j][ni][0] ^= (int)(af[j] ^ bf[ni]);
                                else mma_us(acc[g * 4 + j][ni], af[j], bf[ni]);
                            }
                    }
                }
            }
        };
        if (UNROLL) {
#pragma unroll
            for (int t = 0; t < 9; ++t) tap(t);
        } else {
#pragma unroll 1
            for (int t = 0; t < 9; ++t) tap(t);
        }
        tr_mark<TR>(tw, 2);
        if (c + 1 < 4) {
            if (PD == 2 && !(c & 1)) store(buf ^ 1, hxB, hwB);
            else store(buf ^ 1, hxA, hwA);
        }
        tr_mark<TR>(tw, 3);
        __syncthreads();
        tr_mark<TR>(tw, 4);
    }
    unsigned char* sO = smem;                            // [M][32]
#pragma unroll
    for (int i = 0; i < MW; ++i) {
        const int m = (warp * MW + i) * 8 + (lane >> 2);
        if (m >= M) continue;
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int c = ni * 8 + (lane & 3) * 2;
            const float t0 = fminf(fmaxf((float)acc[i][ni][0] * sSc[c], -127.0f), 127.0f);
            const float t1 = fminf(fmaxf((float)acc[i][ni][1] * sSc[c + 1], -127.0f), 127.0f);
            const unsigned q0 = __float_as_uint(t0 + 12582912.0f);
            const unsigned q1 = __float_as_uint(t1 + 12582912.0f);
            *(unsigned short*)(sO + m * 32 + c) = (unsigned short)__byte_perm(q0, q1, 0x0040);
        }
    }
    __syncthreads();
    tr_mark<TR>(tw, 5);
    signed char* on = a.out + (size_t)n * a.H * a.W * 32;
    if (DIAG & 16) return;
    for (int p = tid; p < 2 * M; p += NT) {
        const int m = p >> 1;
        const int y = ty0 + m / a.TW, x = tx0 + m % a.TW;
        if (y < a.H && x < a.W)
            *(U4*)(on + ((size_t)(y * a.W + x)) * 32 + (p & 1) * 16) = *(const U4*)(sO + p * 16);
    }
    tr_mark<TR>(tw, 6);
    tr_done<TR>(tw, tr);
}

extern "C" __global__ void __launch_bounds__(256) conv2_q3w8(C2Arg a) { conv2_body<3, 8>(a); }
extern "C" __global__ void __launch_bounds__(256) conv2_q4w8(C2Arg a) { conv2_body<4, 8>(a); }
extern "C" __global__ void __launch_bounds__(256) conv2_q6w8(C2Arg a) { conv2_body<6, 8>(a); }
extern "C" __global__ void __launch_bounds__(128) conv2_q3w4(C2Arg a) { conv2_body<3, 4>(a); }
extern "C" __global__ void __launch_bounds__(128) conv2_q4w4(C2Arg a) { conv2_body<4, 4>(a); }
extern "C" __global__ void __launch_bounds__(128) conv2_q5w4(C2Arg a) { conv2_body<5, 4>(a); }
extern "C" __global__ void __launch_bounds__(128) conv2_q4w4u(C2Arg a) { conv2_body<4, 4, 0, true>(a); }
// two-stage prefetch
extern "C" __global__ void __launch_bounds__(128) conv2_q4w4v(C2Arg a) { conv2_body<4, 4, 0, true, 2>(a); }
extern "C" __global__ void __launch_bounds__(256) conv2_q3w8v(C2Arg a) { conv2_body<3, 8, 0, true, 2>(a); }
extern "C" __global__ void __launch_bounds__(128) conv2_q4w4u_t(C2Arg a, unsigned long long* tr) {
    conv2_body<4, 4, 0, true, 1, true>(a, tr);
}
extern "C" __global__ void __launch_bounds__(128) conv2_q4w4v_t(C2Arg a, unsigned long long* tr) {
    conv2_body<4, 4, 0, true, 2, true>(a, tr);
}
extern "C" __global__ void __launch_bounds__(256) conv2_q3w8u(C2Arg a) { conv2_body<3, 8, 0, true>(a); }
// measurements: 1 no MMA, 2 no global loads, 3 neither
extern "C" __global__ void __launch_bounds__(128) conv2_q4w4_d1(C2Arg a) { conv2_body<4, 4, 1>(a); }
extern "C" __global__ void __launch_bounds__(128) conv2_q4w4_d2(C2Arg a) { conv2_body<4, 4, 2>(a); }
extern "C" __global__ void __launch_bounds__(128) conv2_q4w4_d3(C2Arg a) { conv2_body<4, 4, 3>(a); }
// q4w4u with parts switched off: 2 no loads (c > 1), 16 no epilogue writes, 32 no STS, 64 no mma
// (ldmatrix consumed by an XOR per tap), and combinations
extern "C" __global__ void __launch_bounds__(128) conv2_q4w4u_x2(C2Arg a) { conv2_body<4, 4, 2, true>(a); }
extern "C" __global__ void __launch_bounds__(128) conv2_q4w4u_x16(C2Arg a) { conv2_body<4, 4, 16, true>(a); }
extern "C" __global__ void __launch_bounds__(128) conv2_q4w4u_x34(C2Arg a) { conv2_body<4, 4, 34, true>(a); }
extern "C" __global__ void __launch_bounds__(128) conv2_q4w4u_x64(C2Arg a) { conv2_body<4, 4, 64, true>(a); }
extern "C" __global__ void __launch_bounds__(128) conv2_q4w4u_x114(C2Arg a) { conv2_body<4, 4, 114, true>(a); }
// q4w4v (the engine's block-2 conv2) with parts switched off, for the energy at the power cap
extern "C" __global__ void __launch_bounds__(128) conv2_q4w4v_d1(C2Arg a) { conv2_body<4, 4, 1, true, 2>(a); }
extern "C" __global__ void __launch_bounds__(128) conv2_q4w4v_d2(C2Arg a) { conv2_body<4, 4, 2, true, 2>(a); }
extern "C" __global__ void __launch_bounds__(128) conv2_q4w4v_d16(C2Arg a) { conv2_body<4, 4, 16, true, 2>(a); }
extern "C" __global__ void __launch_bounds__(128) conv2_q4w4v_d8(C2Arg a) { conv2_body<4, 4, 8, true, 2>(a); }
extern "C" __global__ void __launch_bounds__(128) conv2_q4w4v_pl(C2Arg a) { conv2_body<4, 4, 0, true, 2, false, 1>(a); }

// --- high-occupancy conv2: blocks of 16 channels ------------------------------------------
// Like conv2_body, but every stage carries 16 bottleneck channels instead of 32: halo (TH+2)(TW+2) x
// 16 bytes and weights 9 x 32 x 16 = 4.6 KB, double buffer ~18-21 KB per thread block, so 3 blocks fit
// per SM (18-27 warps) instead of 2 of 4 warps (with 8 warps per SM loads and ldmatrix do not overlap
// the mma). One k16 per stage: ldmatrix x4 of A = 4 m8 blocks, of B = the 4 n8 blocks of the 32
// outputs. Consecutive 16 B halo rows: 8 rows in a row cover the 32 banks. HPC halo chunks and WPC
// weight chunks per thread per stage.
template <int MW, int WARPS>
DEV void conv2_k16_body(const C2Arg& a) {
    constexpr int NT = WARPS * 32;
    constexpr int NG = (MW + 3) / 4;
    constexpr int HPC = (4 * 128 + NT - 1) / NT;        // halo <= 512 rows
    constexpr int WPC = (288 + NT - 1) / NT;
    extern __shared__ __align__(16) unsigned char smem[];
    const int HWd = a.TW + 2, HP = (a.TH + 2) * HWd;
    const int HB = ((HP * 16 + 127) / 128) * 128;
    unsigned char* sX = smem;                            // [2][HP][16]
    unsigned char* sW = smem + 2 * HB;                   // [2][288][16]
    float* sSc = (float*)(sW + 2 * 4608);                // [32]

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n = blockIdx.y;
    const int ty0 = (blockIdx.x / a.txn) * a.TH, tx0 = (blockIdx.x % a.txn) * a.TW;
    const int M = a.TH * a.TW;
    if (tid < 32) sSc[tid] = a.sc[tid];

    int base[NG];
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        const int i = g * 4 + (lane >> 3);
        const int m = (warp * MW + i) * 8 + (lane & 7);
        const int mm = (i < MW && m < M) ? m : 0;
        base[g] = (mm / a.TW) * HWd + (mm % a.TW);
    }
    int acc[MW][4][2];
#pragma unroll
    for (int i = 0; i < MW; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = 0;

    U4 hx[HPC], hw[WPC];
    const unsigned char* xn = a.x + (size_t)n * a.H * a.W * 128;
    auto load = [&](int c) {                             // c = half block (16 channels), 0..7
#pragma unroll
        for (int k = 0; k < HPC; ++k) {
            const int p = tid + k * NT;
            hx[k] = U4{0u, 0u, 0u, 0u};
            if (p < HP) {
                const int hy = p / HWd, hxx = p - hy * HWd;
                const int y = ty0 - 1 + hy, x = tx0 - 1 + hxx;
                if (y >= 0 && y < a.H && x >= 0 && x < a.W)
                    hx[k] = *(const U4*)(xn + ((size_t)(y * a.W + x)) * 128 + c * 16);
            }
        }
        const signed char* wc = a.w + (c >> 1) * 9216 + (c & 1) * 16;
#pragma unroll
        for (int k = 0; k < WPC; ++k) {
            const int q = tid + k * NT;
            if (q < 288) hw[k] = *(const U4*)(wc + q * 32);
        }
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int k = 0; k < HPC; ++k) {
            const int p = tid + k * NT;
            if (p < HP) *(U4*)(sX + buf * HB + p * 16) = hx[k];
        }
#pragma unroll
        for (int k = 0; k < WPC; ++k) {
            const int q = tid + k * NT;
            if (q < 288) *(U4*)(sW + buf * 4608 + q * 16) = hw[k];
        }
    };

    load(0);
    __syncthreads();                                    // sSc
    store(0);
    __syncthreads();
#pragma unroll 1
    for (int c = 0; c < 8; ++c) {
        const int buf = c & 1;
        if (c + 1 < 8) load(c + 1);
        const unsigned xb = smem_addr(sX + buf * HB);
        const unsigned wb = smem_addr(sW + buf * 4608);
#pragma unroll
        for (int t = 0; t < 9; ++t) {
            const int toff = (t / 3) * HWd + (t % 3);
            unsigned bf[4];
            ldsm4(wb + (t * 32 + lane) * 16, bf);
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                unsigned af[4];
                ldsm4(xb + (base[g] + toff) * 16, af);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (g * 4 + j < MW)
#pragma unroll
                        for (int ni = 0; ni < 4; ++ni) mma_us(acc[g * 4 + j][ni], af[j], bf[ni]);
            }
        }
        if (c + 1 < 8) store(buf ^ 1);
        __syncthreads();
    }
    unsigned char* sO = smem;                            // [M][32]
#pragma unroll
    for (int i = 0; i < MW; ++i) {
        const int m = (warp * MW + i) * 8 + (lane >> 2);
        if (m >= M) continue;
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int c = ni * 8 + (lane & 3) * 2;
            const float t0 = fminf(fmaxf((float)acc[i][ni][0] * sSc[c], -127.0f), 127.0f);
            const float t1 = fminf(fmaxf((float)acc[i][ni][1] * sSc[c + 1], -127.0f), 127.0f);
            const unsigned q0 = __float_as_uint(t0 + 12582912.0f);
            const unsigned q1 = __float_as_uint(t1 + 12582912.0f);
            *(unsigned short*)(sO + m * 32 + c) = (unsigned short)__byte_perm(q0, q1, 0x0040);
        }
    }
    __syncthreads();
    signed char* on = a.out + (size_t)n * a.H * a.W * 32;
    for (int p = tid; p < 2 * M; p += NT) {
        const int m = p >> 1;
        const int y = ty0 + m / a.TW, x = tx0 + m % a.TW;
        if (y < a.H && x < a.W)
            *(U4*)(on + ((size_t)(y * a.W + x)) * 32 + (p & 1) * 16) = *(const U4*)(sO + p * 16);
    }
}

extern "C" __global__ void __launch_bounds__(288, 3) conv2_k4w9(C2Arg a) { conv2_k16_body<4, 9>(a); }
extern "C" __global__ void __launch_bounds__(192, 4) conv2_k4w6(C2Arg a) { conv2_k16_body<4, 6>(a); }
extern "C" __global__ void __launch_bounds__(288, 3) conv2_k3w9(C2Arg a) { conv2_k16_body<3, 9>(a); }
extern "C" __global__ void __launch_bounds__(224, 4) conv2_k3w7(C2Arg a) { conv2_k16_body<3, 7>(a); }
extern "C" __global__ void __launch_bounds__(192, 4) conv2_k3w6(C2Arg a) { conv2_k16_body<3, 6>(a); }
extern "C" __global__ void __launch_bounds__(128, 6) conv2_k3w4(C2Arg a) { conv2_k16_body<3, 4>(a); }

// Final features: out[n, c] = mean over pixels of relu(a_c q + b_c) (norm5 with sigma folded in).
// grid = (images, block groups), 128 threads: 32 channels x 4 pixel rows.
extern "C" __global__ void __launch_bounds__(128) final_q(
    const signed char* __restrict__ in0, int G0, const signed char* __restrict__ seg, int N,
    int HW, const float* __restrict__ ab, float* __restrict__ out, int ctot)
{
    __shared__ float part[4][32];
    const int n = blockIdx.x, g = blockIdx.y;
    const int c = threadIdx.x & 31, q = threadIdx.x >> 5;
    const F2 p = ((const F2*)ab)[g * 32 + c];
    const signed char* base = g < G0 ? in0 + (size_t)(n * G0 + g) * HW * 32
                                     : seg + (size_t)((g - G0) * N + n) * HW * 32;
    float acc = 0.f;
    for (int s = q; s < HW; s += 4) acc += fmaxf(fmaf(p.x, (float)base[s * 32 + c], p.y), 0.f);
    part[q][c] = acc;
    __syncthreads();
    if (q == 0)
        out[(size_t)n * ctot + g * 32 + c] =
            (part[0][c] + part[1][c] + part[2][c] + part[3][c]) / (float)HW;
}

// --- quantization from fp16 (fallback if TensorRT does not give INT8 output) --------------------
DEV float h2f(half_t h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

// Format of a TensorRT output (as in dense8): element (n, c, s) at
// n*sN + (c / V)*sG + s*sS + (c % V)*sC; f32 = 1 if fp32.
struct Fmt { long long sN, sG, sS, sC; int V, f32; };

DEV float ldv(const void* y, const Fmt& f, int n, int c, int s) {
    const long long off = (long long)n * f.sN + (long long)(c / f.V) * f.sG
                          + (long long)s * f.sS + (long long)(c % f.V) * f.sC;
    return f.f32 ? ((const float*)y)[off] : h2f(((const half_t*)y)[off]);
}

// 16 channels (from c0) at pixel s of image n, as float: 16 B loads when the channels are contiguous
// (sC = 1, all in the same group of V), otherwise one by one (as in dense8).
DEV void load16(const void* y, const Fmt& f, int n, int s, int c0, float* v) {
    const long long base = (long long)n * f.sN + (long long)s * f.sS;
    if (f.sC == 1 && (c0 % f.V) + 16 <= f.V && !f.f32) {
        const long long off = base + (long long)(c0 / f.V) * f.sG + (c0 % f.V);
        const U4* p = (const U4*)((const half_t*)y + off);
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const U4 t = p[q];
            const unsigned w[4] = {t.x, t.y, t.z, t.w};
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                v[8 * q + 2 * r] = h2f((half_t)(w[r] & 0xFFFFu));
                v[8 * q + 2 * r + 1] = h2f((half_t)(w[r] >> 16));
            }
        }
        return;
    }
#pragma unroll
    for (int c = 0; c < 16; ++c) v[c] = ldv(y, f, n, c0 + c, s);
}

// One thread = one pixel x 16 channels (half a group). pool: 0 none, 2 3x3/2 max with padding 1 from
// the sh x sw source. q = rint(clamp(x * inv_c, +-127)) -> kCHW32 dst with G groups per image (group
// g0 + blockIdx.y). grid = (ceil(2*N*h*w / 128), groups).
extern "C" __global__ void __launch_bounds__(128) quant_seg(
    const void* __restrict__ y, Fmt f, int pool, int sh, int sw, int N, int h, int w, int g0,
    int G, const float* __restrict__ inv, signed char* __restrict__ dst)
{
    const int g = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int px = t >> 1, hf = t & 1;
    const int hw = h * w;
    if (px >= N * hw) return;
    const int n = px / hw, s = px - n * hw;
    const int oy = s / w, ox = s - oy * w;
    const int c0 = g * 32 + hf * 16;
    float v[16];
    if (pool == 0) {
        load16(y, f, n, s, c0, v);
    } else {
        float tv[16];
        bool first = true;
        for (int dy = -1; dy <= 1; ++dy) {
            const int yy = 2 * oy + dy;
            if (yy < 0 || yy >= sh) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                const int xx = 2 * ox + dx;
                if (xx < 0 || xx >= sw) continue;
                load16(y, f, n, yy * sw + xx, c0, tv);
#pragma unroll
                for (int c = 0; c < 16; ++c) v[c] = first ? tv[c] : fmaxf(v[c], tv[c]);
                first = false;
            }
        }
    }
    unsigned wv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        unsigned word = 0;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int c = j * 4 + u;
            const float tq = fminf(fmaxf(v[c] * inv[c0 + c], -127.0f), 127.0f);
            word |= (__float_as_uint(tq + 12582912.0f) & 0xFFu) << (8 * u);
        }
        wv[j] = word;
    }
    *(U4*)(dst + (((size_t)n * G + g0 + g) * hw + s) * 32 + hf * 16) = U4{wv[0], wv[1], wv[2], wv[3]};
}

// --- persistent conv2: resident weights -----------------------------------------------------
// Like conv2_q (same arguments, same result), but one thread block per SM looping over the
// rectangles: the 36 KB of weights are loaded once instead of for every rectangle (in block 1 that was
// ~146 MB of L2 traffic per layer), and the pipeline of the halo channel blocks continues from one
// rectangle to the next (block 0 of the next one loads while block 3 of this one is computed). 8
// warps, MW chunks of 8 rows per warp.
DEV int c2p_smem(int TH, int TW) {
    return 36864 + 2 * c2_halo_bytes(TH, TW) + 128 + TH * TW * 32;
}

// UNROLL: the 9 taps unrolled (the separate conv2 spent more than half its time bringing halo and
// weights into shared memory; here the weights stay put and the next rectangle's halo loads while
// this one is computed).
// PD = 2: the halo of channel block k + 2 (in the continuous rectangle x block sequence) starts while k
// is computed, in two register sets (even / odd: 4 blocks per rectangle).
// DIAG (measurements only): 1 no mma (XOR), 2 no halo loads after the first two blocks, 8 halo always
// from the addresses of the thread block's first rectangle (same instructions, data from L1: the cost
// of the traffic), 16 no epilogue writes. Meaningless results: only time and energy count.
// PL = 1: the planar bottleneck [4][N*H*W][32] (conv1 MODE 3) instead of NHWC128.
template <int MW, bool UNROLL = false, int PD = 1, bool TR = false, int DIAG = 0, int PL = 0>
DEV void conv2p_body(const C2Arg& a, unsigned long long* tr = nullptr) {
    constexpr int NT = 256;
    constexpr int NG = (MW + 3) / 4;
    constexpr int HPC = 4;
    extern __shared__ __align__(16) unsigned char smem[];
    const int HWd = a.TW + 2, HP = (a.TH + 2) * HWd;
    const int HB = c2_halo_bytes(a.TH, a.TW);
    const int M = a.TH * a.TW;
    unsigned char* sW = smem;                            // [4][288][32]
    unsigned char* sX = smem + 36864;                    // [2][HB]
    float* sSc = (float*)(sX + 2 * HB);                  // [32]
    unsigned char* sO = sX + 2 * HB + 128;               // [M][32]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int tyn = (a.H + a.TH - 1) / a.TH, tpi = tyn * a.txn, total = a.N * tpi;

    for (int q = tid; q < 36864 / 16; q += NT) {
        const int r = q >> 1;
        *(U4*)(sW + r * 32 + (((q & 1) ^ ((r >> 2) & 1)) << 4)) = ((const U4*)a.w)[q];
    }
    if (tid < 32) sSc[tid] = a.sc[tid];

    int base[NG];
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        const int i = g * 4 + (lane >> 3);
        const int m = (warp * MW + i) * 8 + (lane & 7);
        const int mm = (i < MW && m < M) ? m : 0;
        base[g] = (mm / a.TW) * HWd + (mm % a.TW);
    }
    // MW = 3: ldmatrix x2 (blocks 0-1, lanes 0-15) + x1 (block 2, lanes 0-7) instead of x4 with the fourth
    // block loaded and discarded: -25% shared memory reads of A
    int base3;
    {
        const int m = (warp * MW + 2) * 8 + (lane & 7);
        const int mm = m < M ? m : 0;
        base3 = (mm / a.TW) * HWd + (mm % a.TW);
    }
    U4 hxA[HPC], hxB[HPC];
    TraceW tw;
    tr_init<TR>(tw);
    // the thread's halo chunks: offset from pixel (ty0 - 1, tx0 - 1) and position
    int hoff[HPC];
    unsigned hpos[HPC];
#pragma unroll
    for (int k = 0; k < HPC; ++k) {
        const int p = tid + k * NT;
        const int hr = p >> 1, hy = hr / HWd, hxx = hr - hy * HWd;
        const bool in = p < 2 * HP;
        hoff[k] = in ? (hy * a.W + hxx) * (PL ? 32 : 128) + (p & 1) * 16 : 0;
        hpos[k] = in ? ((unsigned)hy << 16) | (unsigned)hxx : 0xffffffffu;
    }
    constexpr int PX = PL ? 32 : 128;                   // bytes of one pixel in the bottleneck
    const size_t cstep = PL ? (size_t)a.N * a.H * a.W * 32 : 32;   // stride of a channel block
    auto hload = [&](int tile, int c, U4* hx) {
        if ((DIAG & 2) && (tile != (int)blockIdx.x || c > 1)) return;
        if (DIAG & 8) {
            tile = blockIdx.x;
            c &= 1;
        }
        const int n = tile / tpi, t = tile - n * tpi;
        const int ty0 = (t / a.txn) * a.TH, tx0 = (t - (t / a.txn) * a.txn) * a.TW;
        const unsigned char* xn = a.x + (size_t)n * a.H * a.W * PX
                                  + ((ptrdiff_t)(ty0 - 1) * a.W + (tx0 - 1)) * PX + c * cstep;
#pragma unroll
        for (int k = 0; k < HPC; ++k) {
            const int y = ty0 - 1 + (int)(hpos[k] >> 16), x = tx0 - 1 + (int)(hpos[k] & 0xffffu);
            hx[k] = ((unsigned)y < (unsigned)a.H && (unsigned)x < (unsigned)a.W)
                    ? *(const U4*)(xn + hoff[k]) : U4{0u, 0u, 0u, 0u};
        }
    };
    auto hstore = [&](int buf, const U4* hx) {
#pragma unroll
        for (int k = 0; k < HPC; ++k) {
            const int p = tid + k * NT;
            if (p < 2 * HP) {
                const int hr = p >> 1;
                *(U4*)(sX + buf * HB + hr * 32 + (((p & 1) ^ ((hr >> 2) & 1)) << 4)) = hx[k];
            }
        }
    };
    int tile = blockIdx.x;
    if (tile >= total) return;
    hload(tile, 0, hxA);
    if (PD == 2) hload(tile, 1, hxB);
    __syncthreads();                                    // weights, scales
    hstore(0, hxA);
    __syncthreads();
    tr_mark<TR>(tw, 0);
    int buf = 0;
    const unsigned wb0 = smem_addr(sW);
    for (; tile < total; tile += gridDim.x) {
        const int n = tile / tpi, t = tile - n * tpi;
        const int ty0 = (t / a.txn) * a.TH, tx0 = (t % a.txn) * a.TW;
        int acc[MW][4][2];
#pragma unroll
        for (int i = 0; i < MW; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = 0;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            bool more;
            if (PD == 2) {
                // prefetch of block k + 2 into k's set (even A, odd B), already written
                const int t2 = c < 2 ? tile : tile + gridDim.x;
                if (t2 < total) {
                    if (c & 1) hload(t2, (c + 2) & 3, hxB);
                    else hload(t2, (c + 2) & 3, hxA);
                }
                more = (c < 3 ? tile : tile + gridDim.x) < total;
                buf = c & 1;
            } else {
                const int ntile = c < 3 ? tile : tile + gridDim.x;
                more = ntile < total;
                if (more) hload(ntile, c < 3 ? c + 1 : 0, hxA);
            }
            tr_mark<TR>(tw, 1);
            const unsigned xb = smem_addr(sX + buf * HB);
            const unsigned wb = wb0 + c * 9216;
            auto tapf = [&](int tp) {
                const int toff = (tp / 3) * HWd + (tp % 3);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    unsigned bf[4];
                    {
                        const int row = tp * 32 + lane;
                        ldsm4(wb + row * 32 + ((h ^ ((row >> 2) & 1)) << 4), bf);
                    }
                    if (MW == 3) {
                        unsigned af[3];
                        const int hr = base[0] + toff, hc = base3 + toff;
                        ldsm2(xb + hr * 32 + ((h ^ ((hr >> 2) & 1)) << 4), af);
                        ldsm1(xb + hc * 32 + ((h ^ ((hc >> 2) & 1)) << 4), af[2]);
#pragma unroll
                        for (int j = 0; j < 3; ++j)
#pragma unroll
                            for (int ni = 0; ni < 4; ++ni) {
                                if (DIAG & 1) acc[j][ni][0] ^= (int)(af[j] ^ bf[ni]);
                                else mma_us(acc[j][ni], af[j], bf[ni]);
                            }
                    } else {
#pragma unroll
                    for (int g = 0; g < NG; ++g) {
                        unsigned af[4];
                        const int hr = base[g] + toff;
                        ldsm4(xb + hr * 32 + ((h ^ ((hr >> 2) & 1)) << 4), af);
#pragma unroll
                        for (int j = 0; j < 4; ++j)
                            if (g * 4 + j < MW)
#pragma unroll
                                for (int ni = 0; ni < 4; ++ni) {
                                    if (DIAG & 1) acc[g * 4 + j][ni][0] ^= (int)(af[j] ^ bf[ni]);
                                    else mma_us(acc[g * 4 + j][ni], af[j], bf[ni]);
                                }
                    }
                    }
                }
            };
            if (UNROLL) {
#pragma unroll
                for (int tp = 0; tp < 9; ++tp) tapf(tp);
            } else {
#pragma unroll 1
                for (int tp = 0; tp < 9; ++tp) tapf(tp);
            }
            tr_mark<TR>(tw, 2);
            if (more) {
                if (PD == 2 && !(c & 1)) hstore(buf ^ 1, hxB);
                else hstore(buf ^ 1, hxA);
            }
            tr_mark<TR>(tw, 3);
            __syncthreads();
            tr_mark<TR>(tw, 4);
            buf ^= 1;
        }
#pragma unroll
        for (int i = 0; i < MW; ++i) {
            const int m = (warp * MW + i) * 8 + (lane >> 2);
            if (m >= M) continue;
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int c = ni * 8 + (lane & 3) * 2;
                const float t0 = fminf(fmaxf((float)acc[i][ni][0] * sSc[c], -127.0f), 127.0f);
                const float t1 = fminf(fmaxf((float)acc[i][ni][1] * sSc[c + 1], -127.0f), 127.0f);
                const unsigned q0 = __float_as_uint(t0 + 12582912.0f);
                const unsigned q1 = __float_as_uint(t1 + 12582912.0f);
                *(unsigned short*)(sO + m * 32 + c) = (unsigned short)__byte_perm(q0, q1, 0x0040);
            }
        }
        __syncthreads();
        tr_mark<TR>(tw, 5);
        signed char* on = a.out + (size_t)n * a.H * a.W * 32;
        for (int p = tid; p < ((DIAG & 16) ? 0 : 2 * M); p += NT) {
            const int m = p >> 1;
            const int y = ty0 + m / a.TW, x = tx0 + m % a.TW;
            if (y < a.H && x < a.W)
                *(U4*)(on + ((size_t)(y * a.W + x)) * 32 + (p & 1) * 16) = *(const U4*)(sO + p * 16);
        }
        tr_mark<TR>(tw, 6);
    }
    tr_done<TR>(tw, tr);
}

extern "C" __global__ void __launch_bounds__(256, 1) conv2p_m2(C2Arg a) { conv2p_body<2>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) conv2p_m3(C2Arg a) { conv2p_body<3>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) conv2p_m4(C2Arg a) { conv2p_body<4>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) conv2p_m2u(C2Arg a) { conv2p_body<2, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) conv2p_m3u(C2Arg a) { conv2p_body<3, true>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) conv2p_m4u(C2Arg a) { conv2p_body<4, true>(a); }
// two-stage prefetch
extern "C" __global__ void __launch_bounds__(256, 1) conv2p_m3v(C2Arg a) { conv2p_body<3, true, 2>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) conv2p_m4v(C2Arg a) { conv2p_body<4, true, 2>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) conv2p_m3u_t(C2Arg a, unsigned long long* tr) {
    conv2p_body<3, true, 1, true>(a, tr);
}
extern "C" __global__ void __launch_bounds__(256, 1) conv2p_m3v_t(C2Arg a, unsigned long long* tr) {
    conv2p_body<3, true, 2, true>(a, tr);
}
// with parts switched off (see DIAG of conv2p_body)
extern "C" __global__ void __launch_bounds__(256, 1) conv2p_m3u_d1(C2Arg a) { conv2p_body<3, true, 1, false, 1>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) conv2p_m3u_d2(C2Arg a) { conv2p_body<3, true, 1, false, 2>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) conv2p_m3u_d16(C2Arg a) { conv2p_body<3, true, 1, false, 16>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) conv2p_m3u_d8(C2Arg a) { conv2p_body<3, true, 1, false, 8>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) conv2p_m4u_d8(C2Arg a) { conv2p_body<4, true, 1, false, 8>(a); }
// planar bottleneck
extern "C" __global__ void __launch_bounds__(256, 1) conv2p_m3u_pl(C2Arg a) { conv2p_body<3, true, 1, false, 0, 1>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) conv2p_m4u_pl(C2Arg a) { conv2p_body<4, true, 1, false, 0, 1>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) conv2p_m4u_d1(C2Arg a) { conv2p_body<4, true, 1, false, 1>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) conv2p_m4u_d2(C2Arg a) { conv2p_body<4, true, 1, false, 2>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) conv2p_m4u_d16(C2Arg a) { conv2p_body<4, true, 1, false, 16>(a); }

// --- fixed-shape persistent conv2, halo in two planes ---------------------------
// At the power cap what counts is energy, i.e. instructions and traffic. In the persistent conv2
// (`conv2p_body`) every (tap, half of the 32 channels) does 12 mma and 3 ldmatrix, but the addresses of
// the halo ldmatrix cost ~10 instructions: the XOR swizzle depends on the halo pixel (lane base + tap),
// and TW is an argument. Here TH and TW are template parameters (TW a multiple of 8, H and W multiples
// of the rectangle) and the halo lives in two planes of 16 bytes per pixel (bytes 0-15 and 16-31 of the
// 32-channel block): the 8 rows of an ldmatrix are 8 consecutive pixels of a halo row, 128 contiguous
// bytes, no conflicts without swizzling. The address is the lane base + a constant (buffer, plane,
// tap): no instructions per tap. The second plane is offset by 64 bytes (modulo 128), so the
// quarter-warps writing the halo (even pixels -> plane 0, odd -> plane 1) do not hit the same banks.
// Resident weights in planes [block * 9 + tap][half][32 outputs][16 bytes]. Same arithmetic as
// conv2p_body<MW, true>: bit-identical outputs. PL = 1: the planar bottleneck [4][N*H*W][32] (conv1
// MODE 3).
// TW not a multiple of 8 also works (blocks 1 and 2: 18 x 12, 9 x 18): the m8 blocks straddling a
// rectangle row have some 2-way bank conflicts in the halo ldmatrix (the addresses stay constant).
template <int TH, int TW, int MW, int DIAG = 0, int PL = 0>
DEV void conv2t_body(const C2Arg& a) {
    constexpr int NT = 256;
    constexpr int HWd = TW + 2, HP = (TH + 2) * HWd, M = TH * TW;
    constexpr int PLANE = ((HP * 16 + 127) / 128) * 128 + 64;
    constexpr int HB = 2 * PLANE;
    constexpr int HPC = (2 * HP + NT - 1) / NT;
    static_assert(M <= 8 * MW * 8 && MW >= 2 && MW <= 4 && M * 32 <= HB, "forma non valida");
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* sW = smem;                            // [36][2][32][16]
    unsigned char* sX = smem + 36864;                    // [2][2][PLANE]
    float* sSc = (float*)(sX + 2 * HB);                  // [32]
    unsigned char* sO = sX + HB;                         // [M][32], reuses buffer 1 of sX after c=3
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int txn = a.W / TW, tpi = (a.H / TH) * txn, total = a.N * tpi;
    constexpr int PX = PL ? 32 : 128;                   // bytes of one pixel in the bottleneck
    const size_t cstep = PL ? (size_t)a.N * a.H * a.W * 32 : 32;   // stride of a channel block

    for (int q = tid; q < 36864 / 16; q += NT) {
        const int r = q >> 1;
        *(U4*)(sW + (((r >> 5) * 2 + (q & 1)) * 32 + (r & 31)) * 16) = ((const U4*)a.w)[q];
    }
    if (tid < 32) sSc[tid] = a.sc[tid];

    // ldmatrix bases: x4 (MW = 4: m8 blocks 0-3 = lanes 0-31), x2 + x1 (MW = 3), x2 (MW = 2)
    auto hrow = [&](int i) {                             // halo pixel (tap 0) of the lane
        const int m = (warp * MW + i) * 8 + (lane & 7);
        const int mm = m < M ? m : 0;
        return (mm / TW) * HWd + (mm % TW);
    };
    const unsigned xs = smem_addr(sX);
    const unsigned aA = xs + hrow((MW == 4 ? lane >> 3 : (lane >> 3) & 1)) * 16;
    const unsigned aB = xs + hrow(2) * 16;               // MW = 3: the third m8 block (x1)
    const unsigned wl = smem_addr(sW) + lane * 16;

    // the thread's halo chunks (as in conv2p_body): offset from pixel (ty0 - 1, tx0 - 1), position
    int hoff[HPC];
    unsigned hpos[HPC];
#pragma unroll
    for (int k = 0; k < HPC; ++k) {
        const int p = tid + k * NT;
        const int hr = p >> 1, hy = hr / HWd, hxx = hr - hy * HWd;
        const bool in = p < 2 * HP;
        hoff[k] = in ? (hy * a.W + hxx) * PX + (p & 1) * 16 : 0;
        hpos[k] = in ? ((unsigned)hy << 16) | (unsigned)hxx : 0xffffffffu;
    }
    U4 hx[HPC];
    auto hload = [&](int tile, int c) {
        if ((DIAG & 2) && (tile != (int)blockIdx.x || c > 1)) return;
        const int n = tile / tpi, t = tile - n * tpi;
        const int ty = t / txn;
        const int ty0 = ty * TH, tx0 = (t - ty * txn) * TW;
        const unsigned char* xn = a.x + (size_t)n * a.H * a.W * PX
                                  + ((ptrdiff_t)(ty0 - 1) * a.W + (tx0 - 1)) * PX + c * cstep;
#pragma unroll
        for (int k = 0; k < HPC; ++k) {
            const int y = ty0 - 1 + (int)(hpos[k] >> 16), x = tx0 - 1 + (int)(hpos[k] & 0xffffu);
            hx[k] = ((unsigned)y < (unsigned)a.H && (unsigned)x < (unsigned)a.W)
                    ? *(const U4*)(xn + hoff[k]) : U4{0u, 0u, 0u, 0u};
        }
    };
    auto hstore = [&](int buf) {
#pragma unroll
        for (int k = 0; k < HPC; ++k) {
            const int p = tid + k * NT;
            if (p < 2 * HP)
                *(U4*)(sX + buf * HB + (p & 1) * PLANE + (p >> 1) * 16) = hx[k];
        }
    };
    int tile = blockIdx.x;
    if (tile >= total) return;
    hload(tile, 0);
    __syncthreads();                                    // weights, scales
    hstore(0);
    __syncthreads();
    for (; tile < total; tile += gridDim.x) {
        const int n = tile / tpi, t = tile - n * tpi;
        const int ty = t / txn;
        const int ty0 = ty * TH, tx0 = (t - ty * txn) * TW;
        int acc[MW][4][2];
#pragma unroll
        for (int i = 0; i < MW; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = 0;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            // the next channel block (or the first of the next rectangle) is loaded into registers
            const int ntile = c < 3 ? tile : tile + gridDim.x;
            const bool more = ntile < total;
            if (more) hload(ntile, c < 3 ? c + 1 : 0);
            const int buf = c & 1;
#pragma unroll
            for (int tp = 0; tp < 9; ++tp) {
                const int toff = ((tp / 3) * HWd + (tp % 3)) * 16;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int xo = buf * HB + h * PLANE + toff;
                    unsigned bf[4];
                    ldsm4(wl + ((c * 9 + tp) * 2 + h) * 512, bf);
                    unsigned af[4];
                    if (MW == 4) {
                        ldsm4(aA + xo, af);
                    } else {
                        ldsm2(aA + xo, af);
                        if (MW == 3) ldsm1(aB + xo, af[2]);
                    }
#pragma unroll
                    for (int j = 0; j < MW; ++j)
#pragma unroll
                        for (int ni = 0; ni < 4; ++ni) {
                            if (DIAG & 1) acc[j][ni][0] ^= (int)(af[j] ^ bf[ni]);
                            else mma_us(acc[j][ni], af[j], bf[ni]);
                        }
                }
            }
            if (more) hstore(buf ^ 1);
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < MW; ++i) {
            const int m = (warp * MW + i) * 8 + (lane >> 2);
            if (m >= M) continue;
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int c = ni * 8 + (lane & 3) * 2;
                const float t0 = fminf(fmaxf((float)acc[i][ni][0] * sSc[c], -127.0f), 127.0f);
                const float t1 = fminf(fmaxf((float)acc[i][ni][1] * sSc[c + 1], -127.0f), 127.0f);
                const unsigned q0 = __float_as_uint(t0 + 12582912.0f);
                const unsigned q1 = __float_as_uint(t1 + 12582912.0f);
                *(unsigned short*)(sO + m * 32 + c) = (unsigned short)__byte_perm(q0, q1, 0x0040);
            }
        }
        __syncthreads();
        signed char* on = a.out + (size_t)n * a.H * a.W * 32;
#pragma unroll
        for (int k = 0; k < (2 * M + NT - 1) / NT; ++k) {
            const int p = tid + k * NT;
            if (p < 2 * M) {
                const int m = p >> 1;
                const int y = ty0 + m / TW, x = tx0 + m % TW;
                *(U4*)(on + ((size_t)(y * a.W + x)) * 32 + (p & 1) * 16) = *(const U4*)(sO + p * 16);
            }
        }
        if (tile + gridDim.x < total) __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(256, 1) conv2_t8x24(C2Arg a) { conv2t_body<8, 24, 3>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) conv2_t8x24_pl(C2Arg a) { conv2t_body<8, 24, 3, 0, 1>(a); }
// the shapes of blocks 1 (36 x 36) and 2 (18 x 18)
extern "C" __global__ void __launch_bounds__(256, 1) conv2_t18x12_pl(C2Arg a) { conv2t_body<18, 12, 4, 0, 1>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) conv2_t9x18_pl(C2Arg a) { conv2t_body<9, 18, 3, 0, 1>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) conv2_t16x16_pl(C2Arg a) { conv2t_body<16, 16, 4, 0, 1>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) conv2_t8x32_pl(C2Arg a) { conv2t_body<8, 32, 4, 0, 1>(a); }

// --- "pool-first" transition in fp16 --------------------------------------------------------
// The 1x1 conv and the 2x2 mean are linear: conv(mean of the 4 u) = mean of the 4 convs. Here the mean
// comes first, exactly: s/4 with s = sum of the four u8 (<= 1020) has 2 fractional bits and fits in
// fp16; the int8 weights are exact in fp16; mma.sync m16n8k8 f16 x f16 -> f32. Same result as trans_q
// (sum of the four integer accumulators) up to the roundings of the fp32 accumulation, with a quarter of
// the rows: half the tensor-core time (fp16 runs at half the INT8 rate).
// GEMM row = output pixel; the thread loads the 4 pixels of the square (16 channels each), applies the
// prologue to each and sums. fp16 weights [G][cout][32]; gd[o].x = k_o = ts tws / sigma' (for the mean,
// i.e. 4 times that of trans_q). 8 warps (4 x 2), 128 x 128 per block.
DEV void mma_f16h(float* c, unsigned a0, unsigned a1, unsigned b0) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
        "{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
}

DEV unsigned h2pack(float a, float b) {
    unsigned short ha, hb;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(ha) : "f"(a));
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hb) : "f"(b));
    return (unsigned)ha | ((unsigned)hb << 16);
}

// PH = 1: fp16 prologue (`pro16h`, a.ab = table T*_abh), 4 pixels x 16 channels per thread.
template <int PH, bool TR = false>
DEV void trans_f16_body(const KArg& a, unsigned long long* tr = nullptr) {
    constexpr int BM = 128;
    constexpr int A_BYTES = BM * 64, B_BYTES = 128 * 64;          // 32 fp16 channels per row
    constexpr int STAGE = BM * 144;                               // output: 128 int8 channels + 16
    constexpr int UNION = 2 * (A_BYTES + B_BYTES) > STAGE ? 2 * (A_BYTES + B_BYTES) : STAGE;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* sA = smem;
    unsigned char* sB = smem + 2 * A_BYTES;
    F2* sAB = (F2*)(smem + UNION);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 3, wn = warp >> 2;
    const int n0 = blockIdx.x * 128, m0 = blockIdx.y * BM;
    const int HW = a.HW, W2 = a.W >> 1, HW4 = HW >> 2;
    const int NO = a.rows >> 2;                                   // output pixels
    const int in0g = HW * 32, segg = a.N * HW * 32;
    TraceW tw;
    tr_init<TR>(tw);
    if (PH) {
        for (int i = tid; i < a.G * 16; i += 256) ((U2*)sAB)[i] = ((const U2*)a.ab)[i];
    } else {
        for (int i = tid; i < a.G * 32; i += 256) sAB[i] = ((const F2*)a.ab)[i];
    }

    // the thread loads row ra (output pixel), half ha of the 32 channels, 4 input pixels
    const int ra = tid >> 1, ha = tid & 1;
    const int o = m0 + ra;
    const bool valid = o < NO;
    int off[4] = {0, 0, 0, 0};                                    // offset in the group: pixel*32
    int nimg = 0;
    if (valid) {
        nimg = o / HW4;
        const int p4 = o - nimg * HW4;
        const int oy = p4 / W2, ox = p4 - oy * W2;
#pragma unroll
        for (int k = 0; k < 4; ++k)
            off[k] = ((2 * oy + (k >> 1)) * a.W + 2 * ox + (k & 1)) * 32 + ha * 16;
    }
    const size_t base_in0 = (size_t)nimg * a.G0 * HW * 32, base_seg = (size_t)nimg * HW * 32;
    const int rb = tid >> 1, hb = tid & 1;                         // weights: row rb, 32 B (half)
    const unsigned char* wrow = (const unsigned char*)a.w + ((size_t)(n0 + rb) * 32) * 2 + hb * 32;
    const size_t wg = (size_t)a.cout * 64;                        // stride of a weight group

    float acc[2][8][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    U4 va[4], vb[2];
    auto gload = [&](int g) {
        const signed char* src = g < a.G0 ? a.in0 + base_in0 + (size_t)g * in0g
                                          : a.seg + base_seg + (size_t)(g - a.G0) * segg;
#pragma unroll
        for (int k = 0; k < 4; ++k) va[k] = valid ? *(const U4*)(src + off[k]) : U4{0u, 0u, 0u, 0u};
        vb[0] = *(const U4*)(wrow + g * wg);
        vb[1] = *(const U4*)(wrow + g * wg + 16);
    };
    auto sstore = [&](int buf, int g) {
        // prologue of the 4 pixels, per-channel sum, s/4 in fp16
        U4 u[4];
        if (PH) {
            const U2* ph = (const U2*)sAB + g * 16 + ha * 8;
#pragma unroll
            for (int k = 0; k < 4; ++k) u[k] = pro16h(va[k], ph);
        } else {
            const F2* ab = sAB + g * 32 + ha * 16;
#pragma unroll
            for (int k = 0; k < 4; ++k) u[k] = pro16(va[k], ab);
        }
        unsigned hw[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned w0 = (&u[0].x)[j], w1 = (&u[1].x)[j], w2 = (&u[2].x)[j], w3 = (&u[3].x)[j];
            float sum[4];
#pragma unroll
            for (int b = 0; b < 4; ++b)
                sum[b] = (float)(((w0 >> (8 * b)) & 255u) + ((w1 >> (8 * b)) & 255u)
                                 + ((w2 >> (8 * b)) & 255u) + ((w3 >> (8 * b)) & 255u)) * 0.25f;
            hw[2 * j] = h2pack(sum[0], sum[1]);
            hw[2 * j + 1] = h2pack(sum[2], sum[3]);
        }
        // row ra: 64 B = 4 chunks of 16 B (32 fp16 channels); this thread writes 2 of them (half ha)
        unsigned char* row = sA + buf * A_BYTES + ra * 64;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int piece = ha * 2 + q;
            *(U4*)(row + ((piece ^ ((ra >> 1) & 3)) << 4)) =
                U4{hw[4 * q], hw[4 * q + 1], hw[4 * q + 2], hw[4 * q + 3]};
        }
        unsigned char* wr = sB + buf * B_BYTES + rb * 64;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int piece = hb * 2 + q;
            *(U4*)(wr + ((piece ^ ((rb >> 1) & 3)) << 4)) = vb[q];
        }
    };
    auto compute = [&](int buf) {
        const unsigned aBase = smem_addr(sA + buf * A_BYTES);
        const unsigned bBase = smem_addr(sB + buf * B_BYTES);
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {                          // 4 k8 steps = 32 channels
            unsigned af[2][2];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                // ldmatrix.x4: rows 0-7 and 8-15 of the m16 chunk for the k8 steps ks (x2 is enough: .x4 with the same
                // rows for the two 8-channel chunks is not needed here)
                const int row = wm * 32 + mi * 16 + (lane & 15);
                const int piece = ks >> 1;                        // 16 B = 8 channels: chunk = ks
                (void)piece;
                const int pc = ks;                                // 16 B chunk of the row
                unsigned r[4];
                ldsm4(aBase + row * 64 + ((pc ^ ((row >> 1) & 3)) << 4), r);
                af[mi][0] = r[0];
                af[mi][1] = r[1];
            }
            unsigned bf[8];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int row = wn * 64 + j * 32 + lane;
                ldsm4(bBase + row * 64 + ((ks ^ ((row >> 1) & 3)) << 4), bf + 4 * j);
            }
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                for (int ni = 0; ni < 8; ++ni) mma_f16h(acc[mi][ni], af[mi][0], af[mi][1], bf[ni]);
        }
    };
    gload(0);
    __syncthreads();                                              // sAB
    sstore(0, 0);
    __syncthreads();
    tr_mark<TR>(tw, 0);
    for (int g = 0; g < a.G; ++g) {
        const int buf = g & 1;
        if (g + 1 < a.G) gload(g + 1);
        tr_mark<TR>(tw, 1);
        compute(buf);
        tr_mark<TR>(tw, 2);
        if (g + 1 < a.G) sstore(buf ^ 1, g + 1);
        tr_mark<TR>(tw, 3);
        __syncthreads();
        tr_mark<TR>(tw, 4);
    }
    // epilogue: q = rint(clamp(k_o * acc, +-127)) -> in0 of the next block
    unsigned char* sO = smem;
    const F2* gd = (const F2*)a.gd;
#pragma unroll
    for (int ni = 0; ni < 8; ++ni) {
        const int c = wn * 64 + ni * 8 + (lane & 3) * 2;
        const float k0 = gd[n0 + c].x, k1 = gd[n0 + c + 1].x;
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int r = wm * 32 + mi * 16 + hh * 8 + (lane >> 2);
                const float t0 = fminf(fmaxf(k0 * acc[mi][ni][2 * hh], -127.0f), 127.0f);
                const float t1 = fminf(fmaxf(k1 * acc[mi][ni][2 * hh + 1], -127.0f), 127.0f);
                const unsigned q0 = __float_as_uint(t0 + 12582912.0f);
                const unsigned q1 = __float_as_uint(t1 + 12582912.0f);
                *(unsigned short*)(sO + r * 144 + c) = (unsigned short)__byte_perm(q0, q1, 0x0040);
            }
        }
    }
    __syncthreads();
    tr_mark<TR>(tw, 5);
    for (int i = tid; i < BM * 8; i += 256) {
        const int r = i >> 3, q = i & 7;
        const int oo = m0 + r;
        if (oo >= NO) continue;
        const int n = oo / HW4, pp = oo - n * HW4;
        *(U4*)(a.out + ((size_t)(n * a.G0n + (n0 >> 5) + (q >> 1)) * HW4 + pp) * 32 + (q & 1) * 16) =
            *(const U4*)(sO + r * 144 + q * 16);
    }
    tr_mark<TR>(tw, 6);
    tr_done<TR>(tw, tr);
}

extern "C" __global__ void __launch_bounds__(256, 1) trans_f16(KArg a) { trans_f16_body<0>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) trans_f16h(KArg a) { trans_f16_body<1>(a); }
extern "C" __global__ void __launch_bounds__(256, 1) trans_f16h_t(KArg a, unsigned long long* tr) {
    trans_f16_body<1, true>(a, tr);
}

// --- wide pool-first transition: 16 warps, BN = 256 (t1, t2) or BM = 256 (t0) ----------------
// Like trans_f16h (mean of the 4 pixels before the conv, fp16, mma m16n8k8, fp16 prologue) but with 16
// warps and a thread block computing BN = 256 output channels: in transitions 1 and 2 (cout 256 and
// 512) trans_f16h redid loads, prologue and mean of every pixel for every block of 128 outputs (2 and 4
// times); with the wide block 1 and 2. t0 (cout 128): BM = 256 rows. 32 x 64 tile per warp (8 x 2 or
// 4 x 4). Exact fp16 mean: the prologue leaves 1024 + u (integer u <= 255), -1024 is exact, so is the
// sum of 4 (<= 1020), and so is * 0.25: the same values as trans_f16h (which sums the bytes as
// integers), the same mma sequence for every output: bit-identical result.
// BM = 128: 512 threads = 128 rows x 2 halves x 2 rows of the square (2 pixels per thread, summed with
// the neighbour through shfl.xor 1); BM = 256: 4 pixels per thread. ldmatrix x4 of A = two k8 steps.
DEV void pro4h_u(unsigned x, U2 p01, U2 p23, unsigned& lo, unsigned& hi) {
    const unsigned w = x ^ 0x80808080u;
    lo = __byte_perm(w, 0x64646464u, 0x4140);
    hi = __byte_perm(w, 0x64646464u, 0x4342);
    asm("add.rn.f16x2 %0, %0, %1;" : "+r"(lo) : "r"(0xE480E480u));        // -1152
    asm("add.rn.f16x2 %0, %0, %1;" : "+r"(hi) : "r"(0xE480E480u));
    asm("fma.rn.sat.f16x2 %0, %0, %1, %2;" : "+r"(lo) : "r"(p01.x), "r"(p01.y));
    asm("fma.rn.sat.f16x2 %0, %0, %1, %2;" : "+r"(hi) : "r"(p23.x), "r"(p23.y));
    asm("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(lo) : "r"(0x5BF85BF8u), "r"(0x64006400u));  // 255, 1024
    asm("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(hi) : "r"(0x5BF85BF8u), "r"(0x64006400u));
    asm("sub.rn.f16x2 %0, %0, %1;" : "+r"(lo) : "r"(0x64006400u));        // u, exact
    asm("sub.rn.f16x2 %0, %0, %1;" : "+r"(hi) : "r"(0x64006400u));
}

DEV unsigned hadd2(unsigned a, unsigned b) {
    unsigned r;
    asm("add.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

DEV unsigned hquarter(unsigned a) {
    unsigned r;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(0x34003400u));  // * 0.25
    return r;
}

template <int BM, int BN, bool TR = false>
DEV void trans_w_body(const KArg& a, unsigned long long* tr = nullptr) {
    constexpr int NTH = 512;
    constexpr int WMW = BM / 32, WNW = 16 / WMW, WN = BN / WNW, NI = WN / 8;
    static_assert(WN == 64 && WMW * WNW == 16, "tile 32 x 64 a warp");
    constexpr int PPT = BM / 64;                                  // pixels of the square per thread
    constexpr int BPT = BN / 128;                                 // 16 B weight chunks per thread
    constexpr int A_BYTES = BM * 64, B_BYTES = BN * 64;           // 32 fp16 channels per row
    constexpr int OROW = BN + 16;
    constexpr int STAGE = BM * OROW;
    constexpr int UNION = 2 * (A_BYTES + B_BYTES) > STAGE ? 2 * (A_BYTES + B_BYTES) : STAGE;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* sA = smem;
    unsigned char* sB = smem + 2 * A_BYTES;
    U2* sAB = (U2*)(smem + UNION);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % WMW, wn = warp / WMW;
    const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
    const int HW = a.HW, W2 = a.W >> 1, HW4 = HW >> 2;
    const int NO = a.rows >> 2;
    const int in0g = HW * 32, segg = a.N * HW * 32;
    TraceW tw;
    tr_init<TR>(tw);
    for (int i = tid; i < a.G * 16; i += NTH) sAB[i] = ((const U2*)a.ab)[i];

    // A: row ra (output pixel), half ha (16 channels), row pr of the square (PPT = 2)
    const int ra = PPT == 2 ? tid >> 2 : tid >> 1;
    const int ha = PPT == 2 ? (tid >> 1) & 1 : tid & 1;
    const int pr = PPT == 2 ? tid & 1 : 0;
    const int o = m0 + ra;
    const bool valid = o < NO;
    int off[PPT];
    int nimg;
    {
        const int oo = valid ? o : 0;
        nimg = oo / HW4;
        const int p4 = oo - nimg * HW4;
        const int oy = p4 / W2, ox = p4 - oy * W2;
#pragma unroll
        for (int k = 0; k < PPT; ++k) {
            const int dy = PPT == 2 ? pr : (k >> 1), dx = PPT == 2 ? k : (k & 1);
            off[k] = ((2 * oy + dy) * a.W + 2 * ox + dx) * 32 + ha * 16;
        }
    }
    const size_t base_in0 = (size_t)nimg * a.G0 * HW * 32, base_seg = (size_t)nimg * HW * 32;
    const size_t wg = (size_t)a.cout * 64;                        // stride of a weight group
    const unsigned char* wbase = (const unsigned char*)a.w + (size_t)n0 * 64;

    float acc[2][NI][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    U4 va[PPT], vb[BPT];
    auto gload = [&](int g) {
        const signed char* src = g < a.G0 ? a.in0 + base_in0 + (size_t)g * in0g
                                          : a.seg + base_seg + (size_t)(g - a.G0) * segg;
#pragma unroll
        for (int k = 0; k < PPT; ++k) va[k] = valid ? *(const U4*)(src + off[k]) : U4{0u, 0u, 0u, 0u};
#pragma unroll
        for (int k = 0; k < BPT; ++k) vb[k] = *(const U4*)(wbase + g * wg + (size_t)(tid + k * NTH) * 16);
    };
    auto sstore = [&](int buf, int g) {
        const U2* ph = sAB + g * 16 + ha * 8;
        unsigned s[8];                                            // 16 channels, fp16 pairs
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int k = 0; k < PPT; ++k) {
                unsigned lo, hi;
                pro4h_u((&va[k].x)[j], ph[2 * j], ph[2 * j + 1], lo, hi);
                s[2 * j] = k ? hadd2(s[2 * j], lo) : lo;
                s[2 * j + 1] = k ? hadd2(s[2 * j + 1], hi) : hi;
            }
        }
        unsigned char* row = sA + buf * A_BYTES + ra * 64;
        if (PPT == 2) {
            // pr = 0 writes channels 0-7 of the half, pr = 1 channels 8-15: each sends the other half of its sum to
            // the neighbour and adds the half it receives
            unsigned m[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned send = pr ? s[j] : s[4 + j];
                const unsigned recv = __shfl_xor_sync(0xffffffffu, send, 1);
                m[j] = hquarter(hadd2(pr ? s[4 + j] : s[j], recv));
            }
            const int piece = ha * 2 + pr;
            *(U4*)(row + ((piece ^ ((ra >> 1) & 3)) << 4)) = U4{m[0], m[1], m[2], m[3]};
        } else {
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int piece = ha * 2 + q;
                *(U4*)(row + ((piece ^ ((ra >> 1) & 3)) << 4)) =
                    U4{hquarter(s[4 * q]), hquarter(s[4 * q + 1]), hquarter(s[4 * q + 2]),
                       hquarter(s[4 * q + 3])};
            }
        }
#pragma unroll
        for (int k = 0; k < BPT; ++k) {
            const int q = tid + k * NTH, rb = q >> 2, pc = q & 3;
            *(U4*)(sB + buf * B_BYTES + rb * 64 + ((pc ^ ((rb >> 1) & 3)) << 4)) = vb[k];
        }
    };
    auto compute = [&](int buf) {
        const unsigned aBase = smem_addr(sA + buf * A_BYTES);
        const unsigned bBase = smem_addr(sB + buf * B_BYTES);
#pragma unroll
        for (int kp = 0; kp < 2; ++kp) {
            unsigned af[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                // lanes 0-15: chunk 2 kp (rows 0-7, 8-15), lanes 16-31: chunk 2 kp + 1
                const int row = wm * 32 + mi * 16 + (lane & 15), pc = 2 * kp + (lane >> 4);
                ldsm4(aBase + row * 64 + ((pc ^ ((row >> 1) & 3)) << 4), af[mi]);
            }
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                const int ks = 2 * kp + kk;
                unsigned bf[NI];
#pragma unroll
                for (int j = 0; j < NI / 4; ++j) {
                    const int row = wn * WN + j * 32 + lane;
                    ldsm4(bBase + row * 64 + ((ks ^ ((row >> 1) & 3)) << 4), bf + 4 * j);
                }
#pragma unroll
                for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                    for (int ni = 0; ni < NI; ++ni)
                        mma_f16h(acc[mi][ni], af[mi][2 * kk], af[mi][2 * kk + 1], bf[ni]);
            }
        }
    };
    gload(0);
    __syncthreads();                                              // sAB
    sstore(0, 0);
    __syncthreads();
    tr_mark<TR>(tw, 0);
    for (int g = 0; g < a.G; ++g) {
        const int buf = g & 1;
        if (g + 1 < a.G) gload(g + 1);
        tr_mark<TR>(tw, 1);
        compute(buf);
        tr_mark<TR>(tw, 2);
        if (g + 1 < a.G) sstore(buf ^ 1, g + 1);
        tr_mark<TR>(tw, 3);
        __syncthreads();
        tr_mark<TR>(tw, 4);
    }
    // epilogue: q = rint(clamp(k_o * acc, +-127)) -> in0 of the next block
    unsigned char* sO = smem;
    const F2* gd = (const F2*)a.gd;
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
        const int c = wn * WN + ni * 8 + (lane & 3) * 2;
        const float k0 = gd[n0 + c].x, k1 = gd[n0 + c + 1].x;
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int r = wm * 32 + mi * 16 + hh * 8 + (lane >> 2);
                const float t0 = fminf(fmaxf(k0 * acc[mi][ni][2 * hh], -127.0f), 127.0f);
                const float t1 = fminf(fmaxf(k1 * acc[mi][ni][2 * hh + 1], -127.0f), 127.0f);
                const unsigned q0 = __float_as_uint(t0 + 12582912.0f);
                const unsigned q1 = __float_as_uint(t1 + 12582912.0f);
                *(unsigned short*)(sO + r * OROW + c) = (unsigned short)__byte_perm(q0, q1, 0x0040);
            }
        }
    }
    __syncthreads();
    tr_mark<TR>(tw, 5);
    constexpr int PR = BN / 16;                                   // 16 B chunks per row
    for (int i = tid; i < BM * PR; i += NTH) {
        const int r = i / PR, q = i - r * PR;
        const int oo = m0 + r;
        if (oo >= NO) continue;
        const int n = oo / HW4, pp = oo - n * HW4;
        *(U4*)(a.out + ((size_t)(n * a.G0n + (n0 >> 5) + (q >> 1)) * HW4 + pp) * 32 + (q & 1) * 16) =
            *(const U4*)(sO + r * OROW + q * 16);
    }
    tr_mark<TR>(tw, 6);
    tr_done<TR>(tw, tr);
}

extern "C" __global__ void __launch_bounds__(512, 1) trans_w1(KArg a) { trans_w_body<128, 256>(a); }
extern "C" __global__ void __launch_bounds__(512, 1) trans_w0(KArg a) { trans_w_body<256, 128>(a); }
extern "C" __global__ void __launch_bounds__(512, 1) trans_w1_t(KArg a, unsigned long long* tr) {
    trans_w_body<128, 256, true>(a, tr);
}
extern "C" __global__ void __launch_bounds__(512, 1) trans_w0_t(KArg a, unsigned long long* tr) {
    trans_w_body<256, 128, true>(a, tr);
}

// --- high-occupancy pool-first transition ----------------------------------------------
// Like trans_f16 (mean of the 4 u8 before the conv, fp16, mma m16n8k8), but with 32 warps per SM (2
// blocks of 16, 32 x 32 tile per warp, 32 accumulators) and stages of 16 channels: with 8 warps per SM
// prologue and loads do not overlap the mma. The thread handles 4 channels of a row: 4 pixels x 4
// bytes, prologue (PH = 0 fp32 `pro4`, 1 fp16 `pro4h` with table T*_abh), sum, s/4 in fp16; weights:
// 4 fp16 channels of a row. ldmatrix x4 of A = the two k8 of an m16 chunk.
template <int PH>
DEV void trans_hi_body(const KArg& a) {
    constexpr int BM = 128, NTH = 512;
    constexpr int A_BYTES = BM * 32, B_BYTES = 128 * 32;          // 16 fp16 channels per row
    constexpr int STAGE = BM * 144;
    constexpr int UNION = 2 * (A_BYTES + B_BYTES) > STAGE ? 2 * (A_BYTES + B_BYTES) : STAGE;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* sA = smem;
    unsigned char* sB = smem + 2 * A_BYTES;
    F2* sAB = (F2*)(smem + UNION);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 3, wn = warp >> 2;
    const int n0 = blockIdx.x * 128, m0 = blockIdx.y * BM;
    const int HW = a.HW, W2 = a.W >> 1, HW4 = HW >> 2;
    const int NO = a.rows >> 2;
    const int in0g = HW * 32, segg = a.N * HW * 32;
    if (PH) {
        for (int i = tid; i < a.G * 16; i += NTH) ((U2*)sAB)[i] = ((const U2*)a.ab)[i];
    } else {
        for (int i = tid; i < a.G * 32; i += NTH) sAB[i] = ((const F2*)a.ab)[i];
    }
    // A: row ra (output pixel), 4 channels qa of the half group
    const int ra = tid >> 2, qa = tid & 3;
    int o = m0 + ra;
    if (o >= NO) o = 0;                                           // rows beyond the end: discarded
    const int nimg = o / HW4, p4 = o - nimg * HW4;
    const int oy = p4 / W2, ox = p4 - oy * W2;
    const int pix0 = 2 * oy * a.W + 2 * ox;                       // top-left pixel
    const int oin0 = (nimg * a.G0 * HW + pix0) * 32 + qa * 4;
    const int oseg = (nimg * HW + pix0) * 32 + qa * 4;
    const int dW = a.W * 32;                                      // row below
    // weights: row rb (output channel n0 + rb), 4 fp16 channels qb
    const int rb = tid >> 2, qb = tid & 3;
    const int ow = (n0 + rb) * 64 + qb * 8;                       // bytes, inside a group
    const int wg = a.cout * 64;
    const int offA = ra * 32 + ((((qa >> 1) ^ ((ra >> 2) & 1))) << 4) + (qa & 1) * 8;
    const int offB = rb * 32 + ((((qb >> 1) ^ ((rb >> 2) & 1))) << 4) + (qb & 1) * 8;

    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    // stage s = half group: group s >> 1, channels (s & 1) * 16 + qa * 4 .. +3
    auto gload = [&](int st, unsigned* v, U2& w) {
        const int g = st >> 1, hsel = (st & 1) * 16;
        const signed char* src = g < a.G0 ? a.in0 + (oin0 + g * in0g + hsel)
                                          : a.seg + (oseg + (g - a.G0) * segg + hsel);
        v[0] = *(const unsigned*)src;
        v[1] = *(const unsigned*)(src + 32);
        v[2] = *(const unsigned*)(src + dW);
        v[3] = *(const unsigned*)(src + dW + 32);
        w = *(const U2*)((const unsigned char*)a.w + (ow + g * wg + hsel * 2));
    };
    auto sstore = [&](int buf, int st, const unsigned* v, U2 w) {
        const int g = st >> 1, c0 = (st & 1) * 16 + qa * 4;
        unsigned u[4];
        if (PH) {
            const U2* ph = (const U2*)sAB + g * 16 + (c0 >> 1);
#pragma unroll
            for (int k = 0; k < 4; ++k) u[k] = pro4h(v[k], ph[0], ph[1]);
        } else {
            const F2* ab = sAB + g * 32 + c0;
#pragma unroll
            for (int k = 0; k < 4; ++k) u[k] = pro4(v[k], ab);
        }
        float sm[4];
#pragma unroll
        for (int b = 0; b < 4; ++b)
            sm[b] = (float)(((u[0] >> (8 * b)) & 255u) + ((u[1] >> (8 * b)) & 255u)
                            + ((u[2] >> (8 * b)) & 255u) + ((u[3] >> (8 * b)) & 255u)) * 0.25f;
        *(U2*)(sA + buf * A_BYTES + offA) = U2{h2pack(sm[0], sm[1]), h2pack(sm[2], sm[3])};
        *(U2*)(sB + buf * B_BYTES + offB) = w;
    };
    auto compute = [&](int buf) {
        const unsigned aBase = smem_addr(sA + buf * A_BYTES);
        const unsigned bBase = smem_addr(sB + buf * B_BYTES);
        unsigned af[2][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const int row = wm * 32 + mi * 16 + (lane & 15), pc = lane >> 4;
            ldsm4(aBase + row * 32 + ((pc ^ ((row >> 2) & 1)) << 4), af[mi]);
        }
        unsigned bf[2][4];
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            const int row = wn * 32 + lane;
            ldsm4(bBase + row * 32 + ((ks ^ ((row >> 2) & 1)) << 4), bf[ks]);
        }
#pragma unroll
        for (int ks = 0; ks < 2; ++ks)
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                for (int ni = 0; ni < 4; ++ni)
                    mma_f16h(acc[mi][ni], af[mi][2 * ks], af[mi][2 * ks + 1], bf[ks][ni]);
    };
    const int S = 2 * a.G;
    unsigned va[4], vc[4];
    U2 wa, wc;
    gload(0, va, wa);
    __syncthreads();                                              // sAB
    sstore(0, 0, va, wa);
    gload(S > 1 ? 1 : 0, vc, wc);
    __syncthreads();
    for (int st = 0; st < S; st += 2) {
        gload(st + 2 < S ? st + 2 : S - 1, va, wa);
        compute(0);
        if (st + 1 >= S) break;
        sstore(1, st + 1, vc, wc);
        __syncthreads();
        gload(st + 3 < S ? st + 3 : S - 1, vc, wc);
        compute(1);
        if (st + 2 >= S) break;
        sstore(0, st + 2, va, wa);
        __syncthreads();
    }
    __syncthreads();
    // epilogue: q = rint(clamp(k_o * acc, +-127)) -> in0 of the next block
    unsigned char* sO = smem;
    const F2* gd = (const F2*)a.gd;
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
        const int c = wn * 32 + ni * 8 + (lane & 3) * 2;
        const float k0 = gd[n0 + c].x, k1 = gd[n0 + c + 1].x;
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int r = wm * 32 + mi * 16 + hh * 8 + (lane >> 2);
                const float t0 = fminf(fmaxf(k0 * acc[mi][ni][2 * hh], -127.0f), 127.0f);
                const float t1 = fminf(fmaxf(k1 * acc[mi][ni][2 * hh + 1], -127.0f), 127.0f);
                const unsigned q0 = __float_as_uint(t0 + 12582912.0f);
                const unsigned q1 = __float_as_uint(t1 + 12582912.0f);
                *(unsigned short*)(sO + r * 144 + c) = (unsigned short)__byte_perm(q0, q1, 0x0040);
            }
        }
    }
    __syncthreads();
    for (int i = tid; i < BM * 8; i += NTH) {
        const int r = i >> 3, q = i & 7;
        const int oo = m0 + r;
        if (oo >= NO) continue;
        const int n = oo / HW4, pp = oo - n * HW4;
        *(U4*)(a.out + ((size_t)(n * a.G0n + (n0 >> 5) + (q >> 1)) * HW4 + pp) * 32 + (q & 1) * 16) =
            *(const U4*)(sO + r * 144 + q * 16);
    }
}

extern "C" __global__ void __launch_bounds__(512, 2) trans_h(KArg a) { trans_hi_body<0>(a); }
extern "C" __global__ void __launch_bounds__(512, 2) trans_hh(KArg a) { trans_hi_body<1>(a); }

// --- stem: 7x7 stride-2 conv (fp16 on the tensor cores) + norm0 + ReLU + 3x3/2 max-pool + q ------
// One kernel instead of the fp16 TensorRT engine (conv 2.9 ms + reformatting) and quant_seg with the
// max-pool (stem 6.4 ms per pass). mma.sync m16n8k8 f16 x f16 -> f32 like TensorRT in fp16: no extra
// rounding on the pixels. The thread block is persistent (one grid per SM) and keeps the weights in
// shared memory; for every 9 x 9 rectangle of max-pool outputs it computes the 19 x 19 conv outputs
// needed (1.11x), quantizes them per channel (u8 in [0, 127]: relu(conv + b) / sigma; max commutes with
// a monotonic quantization) and max-pools the bytes (__vmaxu4). Conv positions outside the image are 0:
// after the ReLU that is the same as excluding them from the max.
// K = (channel, ky, kx') with kx' = 0..7: kx' = 0 has zero weight and kx' - 1 = kx, so the pairs
// (kx', kx'+1) with even kx' fall on even input columns (aligned 32-bit words).
// Input x fp16 [N][3][H][W]; weights [64][168] fp16 (norm0 inside, NOT 1/sigma: with small sigmas they
// would overflow fp16); bias [64] (b0), inv [64] = 1/sigma in fp32; output in0 of block 1:
// [N][2][PH*PW][32] (PH = H/4). Used at 288 px; at 512 px the stem stays on TensorRT.
DEV void mma_f16(float* c, unsigned a0, unsigned a1, unsigned b0) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
        "{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
}

#define ST_TP 9                                   // side of the max-pool rectangle
#define ST_CT (2 * ST_TP + 1)                     // side of the conv rectangle (19)
#define ST_CR (ST_CT * ST_CT)                     // conv rows (361)
#define ST_M16 ((ST_CR + 15) / 16)                // chunks of 16 rows (23)
#define ST_PR (4 * ST_TP + 7)                     // input rows (43)
#define ST_PW (2 * ST_TP + 4)                     // words (2 fp16) per input row (22)
#define ST_PWORDS (3 * ST_PR * ST_PW)             // 2838
#define ST_PPT ((ST_PWORDS + 255) / 256)          // words per thread (12)
#define ST_K 168
#define ST_CROW 68                                // bytes of one conv row in smem (64 + 4)
#define ST_SMEM (64 * ST_K * 2 + 512 + ST_PWORDS * 4 + ST_CR * ST_CROW)

// DIAG (measurements only): 1 no max-pool and no writes, 2 no conv epilogue, 4 no load of the next
// rectangle's input, 8 no mma (A and B consumed by an XOR per step).
template <int DIAG, bool TR = false>
DEV void stem_q_body(
    const half_t* __restrict__ x, const half_t* __restrict__ w, const float* __restrict__ bias,
    const float* __restrict__ inv, signed char* __restrict__ out, int N, int H, int W,
    unsigned long long* tr = nullptr)
{
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* sW = smem;                                     // [64][168] fp16
    float* sB = (float*)(smem + 64 * ST_K * 2);                   // [64] bias, [64] 1/sigma
    unsigned* sP = (unsigned*)(smem + 64 * ST_K * 2 + 512);       // [3][43][22] words
    unsigned char* sC = smem + 64 * ST_K * 2 + 512 + ST_PWORDS * 4;   // [361][68]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int CH = H / 2, CW = W / 2, PH = H / 4, PW = W / 4;
    const int tx_n = PW / ST_TP, ty_n = PH / ST_TP, tpi = tx_n * ty_n, total = N * tpi;
    TraceW tw;
    tr_init<TR>(tw);
    for (int i = tid; i < 64 * ST_K / 8; i += 256) ((U4*)sW)[i] = ((const U4*)w)[i];
    if (tid < 64) sB[tid] = bias[tid];
    else if (tid < 128) sB[tid] = inv[tid - 64];

    unsigned pre[ST_PPT];
    auto pload = [&](int tile) {
        if ((DIAG & 4) && tile != (int)blockIdx.x) return;
        const int n = tile / tpi, t = tile - n * tpi;
        const int ty = t / tx_n, tx = t - ty * tx_n;
        const int r0 = 4 * ST_TP * ty - 5, c0 = 4 * ST_TP * tx - 6;
        const half_t* xn = x + (size_t)n * 3 * H * W;
#pragma unroll
        for (int k = 0; k < ST_PPT; ++k) {
            const int i = tid + k * 256;
            pre[k] = 0u;
            if (i < ST_PWORDS) {
                const int ch = i / (ST_PR * ST_PW), rem = i - ch * (ST_PR * ST_PW);
                const int pr = rem / ST_PW, pw = rem - pr * ST_PW;
                const int yy = r0 + pr, xx = c0 + 2 * pw;
                if (yy >= 0 && yy < H && xx >= 0 && xx < W)
                    pre[k] = *(const unsigned*)(xn + ((size_t)ch * H + yy) * W + xx);
            }
        }
    };
    auto pstore = [&]() {
#pragma unroll
        for (int k = 0; k < ST_PPT; ++k) {
            const int i = tid + k * 256;
            if (i < ST_PWORDS) sP[i] = pre[k];
        }
    };
    int tile = blockIdx.x;
    if (tile < total) pload(tile);
    __syncthreads();                                              // weights and bias
    pstore();
    __syncthreads();
    tr_mark<TR>(tw, 0);
    const unsigned wb = smem_addr(sW);
    for (; tile < total; tile += gridDim.x) {
        const int next = tile + gridDim.x;
        if (next < total) pload(next);
        tr_mark<TR>(tw, 1);
        const int n = tile / tpi, t = tile - n * tpi;
        const int ty = t / tx_n, tx = t - ty * tx_n;
        const int cy0 = 2 * ST_TP * ty - 1, cx0 = 2 * ST_TP * tx - 1;
        // conv: the warp does the 16-row chunks warp, warp + 8, warp + 16
        for (int mt = warp; mt < ST_M16; mt += 8) {
            float acc[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
            const int ra = mt * 16 + (lane >> 2), rb = ra + 8;
            const int ra_ = ra < ST_CR ? ra : 0, rb_ = rb < ST_CR ? rb : 0;
            const int ya = ra_ / ST_CT, xa = ra_ - ya * ST_CT;
            const int yb = rb_ / ST_CT, xb = rb_ - yb * ST_CT;
            const int q = lane & 3;
            const int oa = 2 * ya * ST_PW + xa + q, ob = 2 * yb * ST_PW + xb + q;
#pragma unroll 1
            for (int ch = 0; ch < 3; ++ch) {
#pragma unroll
                for (int ky = 0; ky < 7; ++ky) {
                    const int s = ch * 7 + ky;
                    const int rowoff = (ch * ST_PR + ky) * ST_PW;
                    const unsigned a0 = sP[rowoff + oa], a1 = sP[rowoff + ob];
                    unsigned b[8];
                    ldsm4(wb + (lane) * (ST_K * 2) + s * 16, b);
                    ldsm4(wb + (32 + lane) * (ST_K * 2) + s * 16, b + 4);
                    if (DIAG & 8) {
                        acc[0][0] += __int_as_float((int)(a0 ^ a1 ^ b[0] ^ b[1] ^ b[2] ^ b[3] ^ b[4] ^ b[5] ^ b[6] ^ b[7]));
                    } else {
#pragma unroll
                        for (int j = 0; j < 8; ++j) mma_f16(acc[j], a0, a1, b[j]);
                    }
                }
            }
            tr_mark<TR>(tw, 2);
            // epilogue: relu(acc + b) quantized to [0, 127], 0 outside the image
            if (DIAG & 2) {
                float sum = 0.f;
#pragma unroll
                for (int j = 0; j < 8; ++j) sum += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
                if (sum == 12345.f) sC[0] = 1;
                continue;
            }
            const bool va = ra < ST_CR && cy0 + ya >= 0 && cy0 + ya < CH && cx0 + xa >= 0
                            && cx0 + xa < CW;
            const bool vb = rb < ST_CR && cy0 + yb >= 0 && cy0 + yb < CH && cx0 + xb >= 0
                            && cx0 + xb < CW;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int o = j * 8 + q * 2;
                const float b0 = sB[o], b1 = sB[o + 1], i0 = sB[64 + o], i1 = sB[65 + o];
                if (ra < ST_CR) {
                    const unsigned q0 = va ? __float_as_uint(fminf(fmaxf((acc[j][0] + b0) * i0, 0.f), 127.f) + 12582912.0f) : 0u;
                    const unsigned q1 = va ? __float_as_uint(fminf(fmaxf((acc[j][1] + b1) * i1, 0.f), 127.f) + 12582912.0f) : 0u;
                    *(unsigned short*)(sC + ra * ST_CROW + o) = (unsigned short)__byte_perm(q0, q1, 0x0040);
                }
                if (rb < ST_CR) {
                    const unsigned q0 = vb ? __float_as_uint(fminf(fmaxf((acc[j][2] + b0) * i0, 0.f), 127.f) + 12582912.0f) : 0u;
                    const unsigned q1 = vb ? __float_as_uint(fminf(fmaxf((acc[j][3] + b1) * i1, 0.f), 127.f) + 12582912.0f) : 0u;
                    *(unsigned short*)(sC + rb * ST_CROW + o) = (unsigned short)__byte_perm(q0, q1, 0x0040);
                }
            }
            tr_mark<TR>(tw, 3);
        }
        __syncthreads();
        tr_mark<TR>(tw, 4);
        // 3x3/2 max-pool on the bytes and write-out: one task = one output x 16 channels
        for (int k = tid; k < ((DIAG & 1) ? 0 : ST_TP * ST_TP * 4); k += 256) {
            const int pxi = k >> 2, part = k & 3;
            const int pyl = pxi / ST_TP, pxl = pxi - pyl * ST_TP;
            unsigned m[4] = {0u, 0u, 0u, 0u};
#pragma unroll
            for (int dy = 0; dy < 3; ++dy)
#pragma unroll
                for (int dx = 0; dx < 3; ++dx) {
                    const unsigned* c = (const unsigned*)(sC + ((2 * pyl + dy) * ST_CT + 2 * pxl
                                                                + dx) * ST_CROW + part * 16);
#pragma unroll
                    for (int u = 0; u < 4; ++u) m[u] = __vmaxu4(m[u], c[u]);
                }
            const int py = ST_TP * ty + pyl, px = ST_TP * tx + pxl;
            *(U4*)(out + (((size_t)n * 2 + (part >> 1)) * PH * PW + py * PW + px) * 32
                   + (part & 1) * 16) = U4{m[0], m[1], m[2], m[3]};
        }
        tr_mark<TR>(tw, 5);
        if (next < total) pstore();
        tr_mark<TR>(tw, 6);
        __syncthreads();
        tr_mark<TR>(tw, 7);
    }
    tr_done<TR>(tw, tr);
}

// --- INT8 stem: 7x7/2 conv in IMMA (u8 x s8) + norm0 + ReLU + 3x3/2 max-pool + q ----------
// Like stem_q (same 9 x 9 max-pool rectangle, same outputs), but the pixel enters as u8 = rint(255 x)
// (exact for windows without resampling, x = u8 / 255) and the weights are int8 per output channel
// (sw_o = max|w0_o| / 127): acc = sum u8 w8 exact in int32, output q = rint(clamp(acc k_o + c_o, 0,
// 127)) with k_o = sw_o / (255 sigma_o), c_o = b0_o / sigma_o. Half the tensor-core work of stem_q
// (INT8 at twice the fp16 rate). K = (channel, ky, kx') as in stem_q (kx' = 0 zero weight): 21 groups
// of 8 + 8 zeros = 176 = 11 k16 blocks. Fragment A of mma m8n8k16: thread t holds the 4 bytes k =
// 4 (t % 4) .. +3 of row t / 4, i.e. 4 consecutive input columns: aligned to 4 bytes for even output
// columns; for odd ones there is a second copy of the rectangle shifted by 2 bytes. NW warps: warp w
// does the m8 blocks w, w + NW, ... (46 blocks = 368 rows >= 19 x 19) with all 64 channels.
#define SI_K 176
#define SI_RB 44                                   // bytes of one row of the input rectangle
#define SI_PB (3 * ST_PR * SI_RB)                  // 5676: one rectangle (3 channels x 43 rows)
#define SI_CP 5696                                 // stride of the shifted copy: 16 banks later
#define SI_PW (3 * ST_PR * ST_PW)                  // fp16 pairs to read (2838)
#define SI_OFFP (64 * SI_K + 512)                  // weights, k and c
#define SI_OFFC (((SI_OFFP + SI_CP + SI_PB) + 15) / 16 * 16)
#define SI_SMEM (SI_OFFC + ST_CR * ST_CROW)
#define SI_NB 46

DEV unsigned u8pair(unsigned h2) {
    // two fp16 in [0, 1] -> two bytes rint(255 x) (x outside [0, 1] saturated)
    float a, b;
    asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %2; cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
        : "=f"(a), "=f"(b) : "r"(h2));
    const unsigned qa = __float_as_uint(fmaf(__saturatef(a), 255.0f, 12582912.0f));
    const unsigned qb = __float_as_uint(fmaf(__saturatef(b), 255.0f, 12582912.0f));
    return __byte_perm(qa, qb, 0x0040) & 0xffffu;
}

template <int NW>
DEV void stem_i8_body(const half_t* __restrict__ x, const signed char* __restrict__ w,
                      const float* __restrict__ kc, signed char* __restrict__ out,
                      int N, int H, int W) {
    constexpr int NTH = NW * 32;
    constexpr int PPT = (SI_PW + NTH - 1) / NTH;
    constexpr int BPW = (SI_NB + NW - 1) / NW;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* sW = smem;                                     // [64][176] int8
    float* sK = (float*)(smem + 64 * SI_K);                       // [64] k, [64] c
    unsigned char* sP = smem + SI_OFFP;                           // [2][3 * 43][44] u8 (stride SI_CP)
    unsigned char* sC = smem + SI_OFFC;                           // [361][68]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int CH = H / 2, CW = W / 2, PH = H / 4, PW = W / 4;
    const int tx_n = PW / ST_TP, ty_n = PH / ST_TP, tpi = tx_n * ty_n, total = N * tpi;
    for (int i = tid; i < 64 * SI_K / 16; i += NTH) ((U4*)sW)[i] = ((const U4*)w)[i];
    if (tid < 128) sK[tid] = kc[tid];

    unsigned pre[PPT];
    auto pload = [&](int tile) {
        const int n = tile / tpi, t = tile - n * tpi;
        const int ty = t / tx_n, tx = t - ty * tx_n;
        const int r0 = 4 * ST_TP * ty - 5, c0 = 4 * ST_TP * tx - 6;
        const half_t* xn = x + (size_t)n * 3 * H * W;
#pragma unroll
        for (int k = 0; k < PPT; ++k) {
            const int i = tid + k * NTH;
            pre[k] = 0u;
            if (i < SI_PW) {
                const int ch = i / (ST_PR * ST_PW), rem = i - ch * (ST_PR * ST_PW);
                const int pr = rem / ST_PW, pw = rem - pr * ST_PW;
                const int yy = r0 + pr, xx = c0 + 2 * pw;
                if (yy >= 0 && yy < H && xx >= 0 && xx < W)
                    pre[k] = u8pair(*(const unsigned*)(xn + ((size_t)ch * H + yy) * W + xx));
            }
        }
    };
    auto pstore = [&]() {
#pragma unroll
        for (int k = 0; k < PPT; ++k) {
            const int i = tid + k * NTH;
            if (i < SI_PW) {
                const int row = i / ST_PW, pw = i - row * ST_PW;   // row = ch * 43 + pr
                *(unsigned short*)(sP + row * SI_RB + 2 * pw) = (unsigned short)pre[k];
                if (pw > 0)
                    *(unsigned short*)(sP + SI_CP + row * SI_RB + 2 * pw - 2) = (unsigned short)pre[k];
            }
        }
    };
    int tile = blockIdx.x;
    if (tile < total) pload(tile);
    __syncthreads();                                              // weights, k, c
    pstore();
    __syncthreads();
    const unsigned wb = smem_addr(sW);
    // row of fragment A of every m8 block of the warp: offset in the rectangle (the right copy)
    int abase[BPW];
#pragma unroll
    for (int i = 0; i < BPW; ++i) {
        const int px = (warp + i * NW) * 8 + (lane >> 2);
        const int pxx = px < ST_CR ? px : 0;
        const int ya = pxx / ST_CT, xa = pxx - ya * ST_CT;
        abase[i] = (xa & 1) * SI_CP + 2 * ya * SI_RB + 2 * xa - 2 * (xa & 1) + 4 * (lane & 1);
    }
    const int hsel = (lane >> 1) & 1;                             // even or odd group of the k16
    for (; tile < total; tile += gridDim.x) {
        const int next = tile + gridDim.x;
        if (next < total) pload(next);
        const int n = tile / tpi, t = tile - n * tpi;
        const int ty = t / tx_n, tx = t - ty * tx_n;
        const int cy0 = 2 * ST_TP * ty - 1, cx0 = 2 * ST_TP * tx - 1;
        int acc[BPW][8][2];
#pragma unroll
        for (int i = 0; i < BPW; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[i][j][0] = acc[i][j][1] = 0;
#pragma unroll 1
        for (int j = 0; j < SI_K / 16; ++j) {
            unsigned bf[8];
            ldsm4(wb + lane * SI_K + j * 16, bf);
            ldsm4(wb + (32 + lane) * SI_K + j * 16, bf + 4);
            const int g = 2 * j + hsel;
            const int gg = g < 21 ? g : 0;                        // k 168..175: zero weights
            const int goff = ((gg / 7) * ST_PR + (gg % 7)) * SI_RB;
#pragma unroll
            for (int i = 0; i < BPW; ++i) {
                if (warp + i * NW < SI_NB) {
                    const unsigned av = *(const unsigned*)(sP + abase[i] + goff);
#pragma unroll
                    for (int nb = 0; nb < 8; ++nb) mma_us(acc[i][nb], av, bf[nb]);
                }
            }
        }
        // epilogue: q = rint(clamp(k acc + c, 0, 127)), 0 outside the image
#pragma unroll
        for (int i = 0; i < BPW; ++i) {
            const int r = (warp + i * NW) * 8 + (lane >> 2);
            if (warp + i * NW >= SI_NB || r >= ST_CR) continue;
            const int ya = r / ST_CT, xa = r - ya * ST_CT;
            const bool v = cy0 + ya >= 0 && cy0 + ya < CH && cx0 + xa >= 0 && cx0 + xa < CW;
#pragma unroll
            for (int nb = 0; nb < 8; ++nb) {
                const int o = nb * 8 + (lane & 3) * 2;
                const unsigned q0 = v ? __float_as_uint(fminf(fmaxf(fmaf((float)acc[i][nb][0], sK[o], sK[64 + o]), 0.f), 127.f) + 12582912.0f) : 0u;
                const unsigned q1 = v ? __float_as_uint(fminf(fmaxf(fmaf((float)acc[i][nb][1], sK[o + 1], sK[65 + o]), 0.f), 127.f) + 12582912.0f) : 0u;
                *(unsigned short*)(sC + r * ST_CROW + o) = (unsigned short)__byte_perm(q0, q1, 0x0040);
            }
        }
        __syncthreads();
        // 3x3/2 max-pool on the bytes and write-out: one task = one output x 16 channels
        for (int k = tid; k < ST_TP * ST_TP * 4; k += NTH) {
            const int pxi = k >> 2, part = k & 3;
            const int pyl = pxi / ST_TP, pxl = pxi - pyl * ST_TP;
            unsigned m[4] = {0u, 0u, 0u, 0u};
#pragma unroll
            for (int dy = 0; dy < 3; ++dy)
#pragma unroll
                for (int dx = 0; dx < 3; ++dx) {
                    const unsigned* c = (const unsigned*)(sC + ((2 * pyl + dy) * ST_CT + 2 * pxl
                                                                + dx) * ST_CROW + part * 16);
#pragma unroll
                    for (int u = 0; u < 4; ++u) m[u] = __vmaxu4(m[u], c[u]);
                }
            const int py = ST_TP * ty + pyl, px = ST_TP * tx + pxl;
            *(U4*)(out + (((size_t)n * 2 + (part >> 1)) * PH * PW + py * PW + px) * 32
                   + (part & 1) * 16) = U4{m[0], m[1], m[2], m[3]};
        }
        if (next < total) pstore();
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(256, 1) stem_i8(
    const half_t* __restrict__ x, const signed char* __restrict__ w, const float* __restrict__ kc,
    signed char* __restrict__ out, int N, int H, int W) {
    stem_i8_body<8>(x, w, kc, out, N, H, W);
}
extern "C" __global__ void __launch_bounds__(512, 1) stem_i8w(
    const half_t* __restrict__ x, const signed char* __restrict__ w, const float* __restrict__ kc,
    signed char* __restrict__ out, int N, int H, int W) {
    stem_i8_body<16>(x, w, kc, out, N, H, W);
}

// --- stem v2: the same arithmetic as stem_q bit for bit, less work outside the tensor cores ---
// Traced: conv 50%, epilogue 23% (for every chunk 32 reads of bias and 1/sigma and 16 two-byte writes),
// input load 10% (divisions for every word), max-pool 8%. Here: (1) bias and 1/sigma of the lane's 16
// channels in registers; (2) the conv bottleneck in shared memory with the channels in fragment order
// (row r, lane q: 16 bytes = channels 8 j + 2 q, +1 for j = 0..7), one 16-byte write per row; the
// max-pool puts the channels back in order with an exchange among the 4 lanes of the group (shfl.xor)
// before writing; (3) two 16-row chunks per warp together (the weights of every k8 step read once for
// two chunks); (4) the offsets of the thread's input words computed once. 80-byte rows (16 B aligned,
// max-pool reads without bank conflicts).
#define ST2_CROW 80
#define ST2_OFFC (((64 * ST_K * 2 + 512 + ST_PWORDS * 4) + 15) / 16 * 16)   // 16 B aligned
#define ST2_SMEM (ST2_OFFC + ST_CR * ST2_CROW)

DEV unsigned sel4(const unsigned* v, int i) {
    return i == 0 ? v[0] : i == 1 ? v[1] : i == 2 ? v[2] : v[3];
}

// Per-byte maximum of two words with bytes in [0, 127] (`__vmaxu4` is emulated on sm_75, ~6-7
// instructions): t = (a | 0x80) - b per byte never borrows (128 + a - b >= 1) and its top bit is
// (a >= b); prmt with the sign bit in the selector (0x8-0xB) spreads it to the byte. 4 instructions.
// In asm: `__byte_perm` keeps only the low 3 bits of the selector (with the intrinsic the sign bit was
// lost and the max-pool was wrong; the validation gate caught it).
DEV unsigned vmax7(unsigned a, unsigned b) {
    const unsigned t = (a | 0x80808080u) - b;
    unsigned m;
    asm("prmt.b32 %0, %1, %2, 0xBA98;" : "=r"(m) : "r"(t), "r"(0u));
    return (a & m) | (b & ~m);
}

template <int NP>
DEV void stem2_conv(const unsigned* sP, unsigned wb, int lane, const int* mt, float (*acc)[8][4]) {
    const int q = lane & 3;
    int oa[NP], ob[NP];
#pragma unroll
    for (int p = 0; p < NP; ++p) {
        const int ra = mt[p] * 16 + (lane >> 2), rb = ra + 8;
        const int ra_ = ra < ST_CR ? ra : 0, rb_ = rb < ST_CR ? rb : 0;
        const int ya = ra_ / ST_CT, xa = ra_ - ya * ST_CT;
        const int yb = rb_ / ST_CT, xb = rb_ - yb * ST_CT;
        oa[p] = 2 * ya * ST_PW + xa + q;
        ob[p] = 2 * yb * ST_PW + xb + q;
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[p][j][0] = acc[p][j][1] = acc[p][j][2] = acc[p][j][3] = 0.f;
    }
#pragma unroll 1
    for (int ch = 0; ch < 3; ++ch) {
#pragma unroll
        for (int ky = 0; ky < 7; ++ky) {
            const int s = ch * 7 + ky;
            const int rowoff = (ch * ST_PR + ky) * ST_PW;
            unsigned b[8];
            ldsm4(wb + (lane) * (ST_K * 2) + s * 16, b);
            ldsm4(wb + (32 + lane) * (ST_K * 2) + s * 16, b + 4);
#pragma unroll
            for (int p = 0; p < NP; ++p) {
                const unsigned a0 = sP[rowoff + oa[p]], a1 = sP[rowoff + ob[p]];
#pragma unroll
                for (int j = 0; j < 8; ++j) mma_f16(acc[p][j], a0, a1, b[j]);
            }
        }
    }
}

// epilogue of a chunk: relu(acc + b) / sigma quantized to [0, 127] (0 outside the image), rows ra and
// ra + 8, 16 bytes per row in fragment order
DEV void stem2_epi(unsigned char* sC, int mt, int lane, const float (*acc)[4], const float (*bq)[2],
                   const float (*iq)[2], int cy0, int cx0, int CH, int CW) {
    const int q = lane & 3;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = mt * 16 + (lane >> 2) + 8 * h;
        if (r >= ST_CR) continue;
        const int y = r / ST_CT, xx = r - y * ST_CT;
        const bool v = cy0 + y >= 0 && cy0 + y < CH && cx0 + xx >= 0 && cx0 + xx < CW;
        unsigned w[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            unsigned pr[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int j = 2 * k + e;
                const unsigned q0 = __float_as_uint(fminf(fmaxf((acc[j][2 * h] + bq[j][0]) * iq[j][0], 0.f), 127.f) + 12582912.0f);
                const unsigned q1 = __float_as_uint(fminf(fmaxf((acc[j][2 * h + 1] + bq[j][1]) * iq[j][1], 0.f), 127.f) + 12582912.0f);
                pr[e] = __byte_perm(q0, q1, 0x0040) & 0xffffu;
            }
            w[k] = v ? (pr[0] | (pr[1] << 16)) : 0u;
        }
        *(U4*)(sC + r * ST2_CROW + q * 16) = U4{w[0], w[1], w[2], w[3]};
    }
}

template <bool TR>
DEV void stem_q2_body(
    const half_t* __restrict__ x, const half_t* __restrict__ w, const float* __restrict__ bias,
    const float* __restrict__ inv, signed char* __restrict__ out, int N, int H, int W,
    unsigned long long* tr)
{
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* sW = smem;                                     // [64][168] fp16
    float* sB = (float*)(smem + 64 * ST_K * 2);                   // [64] bias, [64] 1/sigma
    unsigned* sP = (unsigned*)(smem + 64 * ST_K * 2 + 512);       // [3][43][22] words
    unsigned char* sC = smem + ST2_OFFC;                          // [361][80]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, q = lane & 3;
    const int CH = H / 2, CW = W / 2, PH = H / 4, PW = W / 4;
    const int tx_n = PW / ST_TP, ty_n = PH / ST_TP, tpi = tx_n * ty_n, total = N * tpi;
    TraceW tw;
    tr_init<TR>(tw);
    for (int i = tid; i < 64 * ST_K / 8; i += 256) ((U4*)sW)[i] = ((const U4*)w)[i];
    if (tid < 64) sB[tid] = bias[tid];
    else if (tid < 128) sB[tid] = inv[tid - 64];
    // the thread's input words: offset (without the rectangle origin) and position
    int poff[ST_PPT];
    unsigned ppos[ST_PPT];                                        // row << 16 | column
#pragma unroll
    for (int k = 0; k < ST_PPT; ++k) {
        const int i = tid + k * 256;
        const int ii = i < ST_PWORDS ? i : 0;
        const int ch = ii / (ST_PR * ST_PW), rem = ii - ch * (ST_PR * ST_PW);
        const int pr = rem / ST_PW, pw = rem - pr * ST_PW;
        poff[k] = ch * H * W + pr * W + 2 * pw;
        ppos[k] = i < ST_PWORDS ? ((unsigned)pr << 16) | (unsigned)(2 * pw) : 0x7fff0000u;  // outside
    }
    unsigned pre[ST_PPT];
    auto pload = [&](int tile) {
        const int n = tile / tpi, t = tile - n * tpi;
        const int ty = t / tx_n, tx = t - ty * tx_n;
        const int r0 = 4 * ST_TP * ty - 5, c0 = 4 * ST_TP * tx - 6;
        const half_t* xb = x + (size_t)n * 3 * H * W + (ptrdiff_t)r0 * W + c0;
#pragma unroll
        for (int k = 0; k < ST_PPT; ++k) {
            const int yy = r0 + (int)(ppos[k] >> 16), xx = c0 + (int)(ppos[k] & 0xffffu);
            pre[k] = ((unsigned)yy < (unsigned)H && (unsigned)xx < (unsigned)W)
                     ? *(const unsigned*)(xb + poff[k]) : 0u;
        }
    };
    auto pstore = [&]() {
#pragma unroll
        for (int k = 0; k < ST_PPT; ++k) {
            const int i = tid + k * 256;
            if (i < ST_PWORDS) sP[i] = pre[k];
        }
    };
    int tile = blockIdx.x;
    if (tile < total) pload(tile);
    __syncthreads();                                              // weights and bias
    float bq[8][2], iq[8][2];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int o = j * 8 + q * 2;
        bq[j][0] = sB[o];
        bq[j][1] = sB[o + 1];
        iq[j][0] = sB[64 + o];
        iq[j][1] = sB[65 + o];
    }
    pstore();
    __syncthreads();
    tr_mark<TR>(tw, 0);
    const unsigned wb = smem_addr(sW);
    for (; tile < total; tile += gridDim.x) {
        const int next = tile + gridDim.x;
        if (next < total) pload(next);
        tr_mark<TR>(tw, 1);
        const int n = tile / tpi, t = tile - n * tpi;
        const int ty = t / tx_n, tx = t - ty * tx_n;
        const int cy0 = 2 * ST_TP * ty - 1, cx0 = 2 * ST_TP * tx - 1;
        // conv: chunks warp and warp + 8 together, then warp + 16 (if present: 23 chunks)
        {
            float acc[2][8][4];
            const int mt[2] = {warp, warp + 8};
            stem2_conv<2>(sP, wb, lane, mt, acc);
            tr_mark<TR>(tw, 2);
            stem2_epi(sC, mt[0], lane, acc[0], bq, iq, cy0, cx0, CH, CW);
            stem2_epi(sC, mt[1], lane, acc[1], bq, iq, cy0, cx0, CH, CW);
            tr_mark<TR>(tw, 3);
        }
        if (warp + 16 < ST_M16) {
            float acc[1][8][4];
            const int mt[1] = {warp + 16};
            stem2_conv<1>(sP, wb, lane, mt, acc);
            tr_mark<TR>(tw, 2);
            stem2_epi(sC, mt[0], lane, acc[0], bq, iq, cy0, cx0, CH, CW);
            tr_mark<TR>(tw, 3);
        }
        __syncthreads();
        tr_mark<TR>(tw, 4);
        // 3x3/2 max-pool on the bytes: task k = output k >> 2, segment q (16 bytes in fragment order: channels
        // 8 j + 2 q, +1); the 4 lanes of the group are the 4 segments of the same output: 4-way exchange (word
        // p of lane q -> lane p) and channels in order
        for (int k0 = warp * 32; k0 < ST_TP * ST_TP * 4; k0 += 256) {
            const int k = k0 + lane;
            const bool act = k < ST_TP * ST_TP * 4;
            const int pxi = act ? k >> 2 : 0;
            const int pyl = pxi / ST_TP, pxl = pxi - pyl * ST_TP;
            unsigned m[4] = {0u, 0u, 0u, 0u};
#pragma unroll
            for (int dy = 0; dy < 3; ++dy)
#pragma unroll
                for (int dx = 0; dx < 3; ++dx) {
                    const U4 c = *(const U4*)(sC + ((2 * pyl + dy) * ST_CT + 2 * pxl + dx) * ST2_CROW
                                              + q * 16);
                    m[0] = vmax7(m[0], c.x);            // bytes in [0, 127] (epilogue)
                    m[1] = vmax7(m[1], c.y);
                    m[2] = vmax7(m[2], c.z);
                    m[3] = vmax7(m[3], c.w);
                }
            unsigned R[4];
            R[0] = sel4(m, q);
            R[1] = __shfl_xor_sync(0xffffffffu, sel4(m, q ^ 1), 1);
            R[2] = __shfl_xor_sync(0xffffffffu, sel4(m, q ^ 2), 2);
            R[3] = __shfl_xor_sync(0xffffffffu, sel4(m, q ^ 3), 3);
            // t[p] = word q of lane p = R[p ^ q]: channels 16 q + 2 p, +1, 16 q + 8 + 2 p, +1
            const unsigned t0 = sel4(R, q), t1 = sel4(R, 1 ^ q), t2 = sel4(R, 2 ^ q), t3 = sel4(R, 3 ^ q);
            if (act) {
                const int py = ST_TP * ty + pyl, px = ST_TP * tx + pxl;
                *(U4*)(out + (((size_t)n * 2 + (q >> 1)) * PH * PW + py * PW + px) * 32 + (q & 1) * 16) =
                    U4{__byte_perm(t0, t1, 0x5410), __byte_perm(t2, t3, 0x5410),
                       __byte_perm(t0, t1, 0x7632), __byte_perm(t2, t3, 0x7632)};
            }
        }
        tr_mark<TR>(tw, 5);
        if (next < total) pstore();
        tr_mark<TR>(tw, 6);
        __syncthreads();
        tr_mark<TR>(tw, 7);
    }
    tr_done<TR>(tw, tr);
}

extern "C" __global__ void __launch_bounds__(256, 1) stem_q2(
    const half_t* __restrict__ x, const half_t* __restrict__ w, const float* __restrict__ bias,
    const float* __restrict__ inv, signed char* __restrict__ out, int N, int H, int W) {
    stem_q2_body<false>(x, w, bias, inv, out, N, H, W, nullptr);
}
extern "C" __global__ void __launch_bounds__(256, 1) stem_q2_t(
    const half_t* __restrict__ x, const half_t* __restrict__ w, const float* __restrict__ bias,
    const float* __restrict__ inv, signed char* __restrict__ out, int N, int H, int W,
    unsigned long long* tr) {
    stem_q2_body<true>(x, w, bias, inv, out, N, H, W, tr);
}

// --- persistent conv1 --------------------------------------------------------------
// Traced: in the conv1 of blocks 0-1 a thread block does 2-15 stages of 32 channels, and start-up
// (first loads from DRAM, prologue table, two barriers), epilogue and write-out weigh 30-70% (block 0
// layer 0: start-up 36%, epilogue 25%, write-out 9%, mma 16%). Here like conv1_u128hs (fp16 prologue,
// 8 warps 4 x 2, 32 x 64 tile per warp, branch-free steady state, u8 NHWC128 bottleneck) but the thread
// block is persistent and loops over the row blocks: (1) the first two stages of the next row block are
// loaded into registers right after this block's last mma, and arrive during the epilogue; (2) the
// epilogue writes straight to global memory: for every m8 block the 4 lanes of a group hold the 64
// channels of the row (16 bytes per lane, channels 8 j + 2 q), a 4-way exchange (shfl.xor, as in
// stem_q2) puts them in order and every lane writes 16 contiguous bytes: no shared memory and no
// barrier for the output.
DEV void quad_t16(const unsigned* w, int q, U4& o) {
    // w[k] = channels (16 k + 2 q, +1, 16 k + 8 + 2 q, +1) of lane q -> o = channels 16 q .. +15
    unsigned R[4];
    R[0] = sel4(w, q);
    R[1] = __shfl_xor_sync(0xffffffffu, sel4(w, q ^ 1), 1);
    R[2] = __shfl_xor_sync(0xffffffffu, sel4(w, q ^ 2), 2);
    R[3] = __shfl_xor_sync(0xffffffffu, sel4(w, q ^ 3), 3);
    const unsigned t0 = sel4(R, q), t1 = sel4(R, 1 ^ q), t2 = sel4(R, 2 ^ q), t3 = sel4(R, 3 ^ q);
    o = U4{__byte_perm(t0, t1, 0x5410), __byte_perm(t2, t3, 0x5410),
           __byte_perm(t0, t1, 0x7632), __byte_perm(t2, t3, 0x7632)};
}

template <bool TR = false>
DEV void conv1p_body(const KArg& a, unsigned long long* tr = nullptr) {
    constexpr int BM = 128, NTH = 256, WARPS_M = 4, WN = 64, NT = 8;
    constexpr int A_BYTES = BM * 32, B_BYTES = 128 * 32;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* sA = smem;
    unsigned char* sB = smem + 2 * A_BYTES;
    U2* sAB = (U2*)(smem + 2 * (A_BYTES + B_BYTES));
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, q = lane & 3;
    const int wm = warp % WARPS_M, wn = warp / WARPS_M;
    const int HW = a.HW, G = a.G;
    const int in0g = HW * 32, segg = a.N * HW * 32;
    const int ntiles = (a.rows + BM - 1) / BM;
    TraceW tw;
    tr_init<TR>(tw);
    for (int i = tid; i < G * 16; i += NTH) sAB[i] = ((const U2*)a.ab)[i];

    const int ra = tid >> 1, ha = tid & 1;
    const int rb = tid >> 1, hb = tid & 1;
    const signed char* wrow = a.w + rb * 32 + hb * 16;           // cout = 128, n0 = 0
    const int wg = a.cout * 32;
    bool valid = false;
    int o_in0 = 0, o_seg = 0;
    auto rows_of = [&](int tile) {
        const int m = tile * BM + ra;
        valid = m < a.rows;
        const int mm = valid ? m : 0;
        const int n = mm / HW, p = mm - n * HW;
        o_in0 = (n * a.G0 * HW + p) * 32 + ha * 16;
        o_seg = (n * HW + p) * 32 + ha * 16;
    };
    U4 va0 = U4{0u, 0u, 0u, 0u}, vb0 = va0, va1 = va0, vb1 = va0;
    auto gload = [&](int g, U4& va, U4& vb) {
        va = valid ? *(const U4*)(g < a.G0 ? a.in0 + o_in0 + g * in0g
                                          : a.seg + o_seg + (g - a.G0) * segg)
                   : U4{0u, 0u, 0u, 0u};
        vb = *(const U4*)(wrow + g * wg);
    };
    auto sstore = [&](int buf, int g, const U4& va, const U4& vb) {
        *(U4*)(sA + buf * A_BYTES + ra * 32 + ((ha ^ ((ra >> 2) & 1)) << 4)) =
            pro16h(va, sAB + g * 16 + ha * 8);
        *(U4*)(sB + buf * B_BYTES + rb * 32 + ((hb ^ ((rb >> 2) & 1)) << 4)) = vb;
    };
    int acc[4][NT][2];
    auto compute = [&](int buf) {
        const unsigned aBase = smem_addr(sA + buf * A_BYTES);
        const unsigned bBase = smem_addr(sB + buf * B_BYTES);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            unsigned af[4], bf[NT];
            {
                const int row = wm * 32 + lane;
                ldsm4(aBase + row * 32 + ((h ^ ((row >> 2) & 1)) << 4), af);
            }
#pragma unroll
            for (int j = 0; j < NT / 4; ++j) {
                const int row = wn * WN + j * 32 + lane;
                ldsm4(bBase + row * 32 + ((h ^ ((row >> 2) & 1)) << 4), bf + 4 * j);
            }
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) mma_us(acc[mi][ni], af[mi], bf[ni]);
        }
    };
    int tile = blockIdx.x;
    if (tile >= ntiles) return;
    rows_of(tile);
    gload(0, va0, vb0);
    if (G > 1) gload(1, va1, vb1);
    __syncthreads();                                             // sAB
    tr_mark<TR>(tw, 0);
    const F2* gd = (const F2*)a.gd;
    for (; tile < ntiles; tile += gridDim.x) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = 0;
        sstore(0, 0, va0, vb0);
        tr_mark<TR>(tw, 3);
        __syncthreads();
        tr_mark<TR>(tw, 4);
        int g0 = 0;
        // steady state: g and g+1 ready (smem, registers), g+2 and g+3 exist
        for (; g0 + 3 < G; g0 += 2) {
            gload(g0 + 2, va0, vb0);
            tr_mark<TR>(tw, 1);
            compute(0);
            tr_mark<TR>(tw, 2);
            sstore(1, g0 + 1, va1, vb1);
            tr_mark<TR>(tw, 3);
            __syncthreads();
            tr_mark<TR>(tw, 4);
            gload(g0 + 3, va1, vb1);
            tr_mark<TR>(tw, 1);
            compute(1);
            tr_mark<TR>(tw, 2);
            sstore(0, g0 + 2, va0, vb0);
            tr_mark<TR>(tw, 3);
            __syncthreads();
            tr_mark<TR>(tw, 4);
        }
        for (int g = g0; g < G; g += 2) {
            if (g + 2 < G) gload(g + 2, va0, vb0);
            tr_mark<TR>(tw, 1);
            compute(0);
            tr_mark<TR>(tw, 2);
            if (g + 1 < G) sstore(1, g + 1, va1, vb1);
            tr_mark<TR>(tw, 3);
            __syncthreads();
            tr_mark<TR>(tw, 4);
            if (g + 1 >= G) break;
            if (g + 3 < G) gload(g + 3, va1, vb1);
            tr_mark<TR>(tw, 1);
            compute(1);
            tr_mark<TR>(tw, 2);
            if (g + 2 < G) sstore(0, g + 2, va0, vb0);
            tr_mark<TR>(tw, 3);
            __syncthreads();
            tr_mark<TR>(tw, 4);
        }
        // the first two stages of the next row block: they arrive during this block's epilogue
        const int m0 = tile * BM;
        const int next = tile + gridDim.x;
        if (next < ntiles) {
            rows_of(next);
            gload(0, va0, vb0);
            if (G > 1) gload(1, va1, vb1);
        }
        tr_mark<TR>(tw, 1);
        // epilogue: u8 bottleneck = rint(255 sat(g acc + d)), the warp's 64 channels per row, 4-lane exchange
        // and 16 contiguous bytes per lane to global memory
        // the bytes of every (mi, ni) packed per mi (16 registers instead of the 64 accumulators)
        unsigned w[4][4];
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int c = wn * WN + ni * 8 + q * 2;
            const F2 p0 = gd[c], p1 = gd[c + 1];
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const unsigned q0 = __float_as_uint(fmaf(
                    __saturatef(fmaf(p0.x, (float)acc[mi][ni][0], p0.y)), 255.0f, 8388608.0f));
                const unsigned q1 = __float_as_uint(fmaf(
                    __saturatef(fmaf(p1.x, (float)acc[mi][ni][1], p1.y)), 255.0f, 8388608.0f));
                const unsigned pr = __byte_perm(q0, q1, 0x0040) & 0xffffu;
                if (ni & 1) w[mi][ni >> 1] |= pr << 16;
                else w[mi][ni >> 1] = pr;
            }
        }
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            U4 o;
            quad_t16(w[mi], q, o);
            const int r = m0 + wm * 32 + mi * 8 + (lane >> 2);
            if (r < a.rows)
                *(U4*)(a.out + (size_t)r * 128 + wn * WN + q * 16) = o;
        }
        tr_mark<TR>(tw, 5);
    }
    tr_done<TR>(tw, tr);
}

extern "C" __global__ void __launch_bounds__(256, 2) conv1_p128(KArg a) { conv1p_body<false>(a); }
extern "C" __global__ void __launch_bounds__(256, 2) conv1_p128_t(KArg a, unsigned long long* tr) {
    conv1p_body<true>(a, tr);
}

extern "C" __global__ void __launch_bounds__(256, 1) stem_q(
    const half_t* __restrict__ x, const half_t* __restrict__ w, const float* __restrict__ bias,
    const float* __restrict__ inv, signed char* __restrict__ out, int N, int H, int W) {
    stem_q_body<0>(x, w, bias, inv, out, N, H, W);
}
#define STEM_DIAG(k) extern "C" __global__ void __launch_bounds__(256, 1) stem_q_d##k(     const half_t* __restrict__ x, const half_t* __restrict__ w, const float* __restrict__ bias,     const float* __restrict__ inv, signed char* __restrict__ out, int N, int H, int W) {     stem_q_body<k>(x, w, bias, inv, out, N, H, W); }
extern "C" __global__ void __launch_bounds__(256, 1) stem_q_t(
    const half_t* __restrict__ x, const half_t* __restrict__ w, const float* __restrict__ bias,
    const float* __restrict__ inv, signed char* __restrict__ out, int N, int H, int W,
    unsigned long long* tr) {
    stem_q_body<0, true>(x, w, bias, inv, out, N, H, W, tr);
}
STEM_DIAG(1)
STEM_DIAG(2)
STEM_DIAG(4)
STEM_DIAG(8)
STEM_DIAG(15)

// --- measurements: INT8 tensor-core peak and SM clock ---------------------------------
// imma_peak: every warp does iters x 8 independent mma.m8n8k16 (1024 MAC each); out[0] so the compiler
// does not remove them. clk: SM cycles and %globaltimer nanoseconds of thread 0 of every block, for the
// real clock under load.
extern "C" __global__ void __launch_bounds__(256) imma_peak(int* out, long long* clk, int iters) {
    int acc[8][2];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = 0;
    const unsigned a = threadIdx.x * 0x01010101u, b = 0x02010201u;
    long long c0 = clock64(), t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) mma_us(acc[j], a + j, b);
    }
    long long c1 = clock64(), t1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
    int s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) s += acc[j][0] + acc[j][1];
    if (s == 0x7fffffff) out[0] = s;
    if (threadIdx.x == 0) {
        clk[2 * blockIdx.x] = c1 - c0;
        clk[2 * blockIdx.x + 1] = t1 - t0;
    }
}

// mma_loop: the inner loop of conv1 on static data in shared memory, without loads or prologue: what
// the tensor cores deliver with the bare loop, at 8 / 16 / 32 warps per SM (the warp-specialised conv1
// with 8 MMA warps sits at ~33% of the peak like the phased one).
// NT: n8 blocks of the warp (8: 32 x 64 tile like conv1_u128; 4: 32 x 32). LDSM = 0: fragments static
// in registers (mma only, with the same dependency between the two channel half-blocks).
// SYNC: one barrier per stage (BK = 32) as in the phased conv1.
template <int NT, bool LDSM, bool SYNC>
DEV void mma_loop_body(int* out, int iters) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int i = tid; i < 512; i += blockDim.x)
        ((U4*)smem)[i] = U4{(unsigned)i, (unsigned)i * 3u, (unsigned)i * 5u, (unsigned)i * 7u};
    __syncthreads();
    constexpr int WN = NT * 8, WARPS_N = 128 / WN;
    const int wm = (warp / WARPS_N) & 3, wn = warp % WARPS_N;
    int acc[4][NT][2];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = 0;
    const unsigned aBase = smem_addr(smem), bBase = aBase + 4096;
    unsigned af[2][4], bf[2][NT];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int i = 0; i < 4; ++i) af[h][i] = lane * 0x01010101u + i + h;
#pragma unroll
        for (int j = 0; j < NT; ++j) bf[h][j] = lane * 0x00010203u + j + h;
    }
    for (int it = 0; it < iters; ++it) {
        if (LDSM) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = wm * 32 + lane;
                ldsm4(aBase + row * 32 + ((h ^ ((row >> 2) & 1)) << 4), af[h]);
#pragma unroll
                for (int j = 0; j < NT / 4; ++j) {
                    const int rb = wn * WN + j * 32 + lane;
                    ldsm4(bBase + rb * 32 + ((h ^ ((rb >> 2) & 1)) << 4), bf[h] + 4 * j);
                }
            }
        }
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) mma_us(acc[mi][ni], af[h][mi], bf[h][ni]);
        if (SYNC) __syncthreads();
    }
    int sum = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) sum += acc[i][j][0] ^ acc[i][j][1];
    if (sum == 0x7fffffff) out[0] = sum;
}
extern "C" __global__ void __launch_bounds__(256) mma_loop(int* out, int iters) {
    mma_loop_body<8, true, false>(out, iters);
}
extern "C" __global__ void __launch_bounds__(256) mma_loop_sync(int* out, int iters) {
    mma_loop_body<8, true, true>(out, iters);
}
extern "C" __global__ void __launch_bounds__(256) mma_loop_reg(int* out, int iters) {
    mma_loop_body<8, false, false>(out, iters);
}
extern "C" __global__ void __launch_bounds__(256) mma_loop4(int* out, int iters) {
    mma_loop_body<4, true, false>(out, iters);
}
extern "C" __global__ void __launch_bounds__(256) mma_loop4_reg(int* out, int iters) {
    mma_loop_body<4, false, false>(out, iters);
}

// conv2_loop: the inner loop of conv2_body (16 x 8 rectangle, MW m8 blocks per warp, 9 taps x 2 halves
// of the 32-channel block: 1 weight ldmatrix, 1 halo ldmatrix, 4 MW mma) on static data in shared
// memory, without loads, writes, barriers or epilogue (without global loads the conv2 gains only 4%,
// but it sits at ~39% of the peak).
template <int MW>
DEV void conv2_loop_body(int* out, int iters) {
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int TH = 16, TW = 8, HWd = TW + 2;
    constexpr int NG = (MW + 3) / 4;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    unsigned char* sX = smem;                            // [180][32]
    unsigned char* sW = smem + 5760;                     // [288][32]
    for (int i = tid; i < (5760 + 9216) / 16; i += blockDim.x)
        ((U4*)smem)[i] = U4{(unsigned)i, (unsigned)i * 3u, (unsigned)i * 5u, (unsigned)i * 7u};
    __syncthreads();
    int base[NG];
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        const int i = g * 4 + (lane >> 3);
        const int m = ((warp & 3) * MW + i) * 8 + (lane & 7);
        const int mm = (i < MW && m < TH * TW) ? m : 0;
        base[g] = (mm / TW) * HWd + (mm % TW);
    }
    int acc[MW][4][2];
#pragma unroll
    for (int i = 0; i < MW; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = 0;
    const unsigned xb = smem_addr(sX), wb = smem_addr(sW);
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int t = 0; t < 9; ++t) {
            const int toff = (t / 3) * HWd + (t % 3);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                unsigned bf[4];
                {
                    const int row = t * 32 + lane;
                    ldsm4(wb + row * 32 + ((h ^ ((row >> 2) & 1)) << 4), bf);
                }
#pragma unroll
                for (int g = 0; g < NG; ++g) {
                    unsigned af[4];
                    const int hr = base[g] + toff;
                    ldsm4(xb + hr * 32 + ((h ^ ((hr >> 2) & 1)) << 4), af);
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (g * 4 + j < MW)
#pragma unroll
                            for (int ni = 0; ni < 4; ++ni) mma_us(acc[g * 4 + j][ni], af[j], bf[ni]);
                }
            }
        }
    }
    int sum = 0;
#pragma unroll
    for (int i = 0; i < MW; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) sum += acc[i][j][0] ^ acc[i][j][1];
    if (sum == 0x7fffffff) out[0] = sum;
}
extern "C" __global__ void __launch_bounds__(128) conv2_loop(int* out, int iters) {
    conv2_loop_body<4>(out, iters);
}

// copy16: the memory bandwidth a simple kernel reaches (16 B per thread, grid-stride).
extern "C" __global__ void __launch_bounds__(256) copy16(const U4* __restrict__ src,
                                                         U4* __restrict__ dst, long long n) {
    for (long long i = blockIdx.x * 256LL + threadIdx.x; i < n; i += (long long)gridDim.x * 256)
        dst[i] = src[i];
}

// l2read: repeated reads of a buffer with ld.global.cg (bypassing L1): small = L2 bandwidth, large =
// read-only DRAM bandwidth.
extern "C" __global__ void __launch_bounds__(256) l2read(const U4* __restrict__ src, int n16,
                                                         int iters, unsigned* out) {
    unsigned acc = 0;
    for (int it = 0; it < iters; ++it)
        for (int i = blockIdx.x * 256 + threadIdx.x; i < n16; i += gridDim.x * 256) {
            unsigned a, b, c, d;
            asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(src + i));
            acc ^= a ^ b ^ c ^ d;
        }
    if (acc == 0x12345678u) out[0] = acc;
}

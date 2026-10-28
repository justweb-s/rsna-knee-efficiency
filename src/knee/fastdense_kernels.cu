// CUDA kernels of `knee.fastdense`, compiled with NVRTC (`knee.cudart`) for sm_75.
//
// Why: on the T4 Triton does not emit mma instructions (tl.dot -> fp32 FMA), and cuBLAS cannot do the
// prologue. Here the conv1 (1x1, cin -> 128) of every layer of a DenseNet block runs on the fp16
// tensor cores (mma.sync m16n8k8, fp32 accumulation), with BatchNorm + ReLU of the input applied while
// loading (prologue, like `gemm_q` of `dense8s_kernels.cu` in INT8):
//
// c1f_*  forward: U = relu(X a + c) W^T (X: the block buffer, rows = pixels, stride sx; W:
//        [128][cin]), U in fp16 [rows][128] and, in the epilogue, the per-column sums of (u - sh) and
//        (u - sh)^2 over the values ALREADY rounded to fp16 (the norm2 statistics), one partial per
//        block of rows;
// c1b    backward, for a slice of 64 input channels and a slice of rows: dY = dU W (GEMM 1),
//        gm = dY [x a + c > 0], G += a gm (the block's gradient buffer), the BN reductions (sum of
//        gm and of gm xhat) and y = relu(x a + c) in shared memory, then dW += dU^T y (GEMM 2): the
//        input X is read once, dY and y never go to global memory.
//
// No includes: fp16 and vector types by hand (like `dense8s_kernels.cu`).

typedef unsigned short half_t;

struct alignas(16) U4 { unsigned int x, y, z, w; };
struct alignas(16) F4 { float x, y, z, w; };
struct alignas(8) F2 { float x, y; };

#define DEV __device__ __forceinline__

extern __shared__ __align__(16) unsigned char smem[];       // dynamic, for all kernels

DEV unsigned smem_addr(const void* p) {
    unsigned r;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p));
    return r;
}

DEV void ldsm4(unsigned addr, unsigned* r) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

DEV void ldsm4t(unsigned addr, unsigned* r) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

// d (16x8, f32) += a (16x8, f16, row) b (8x8, f16, col)
DEV void mma16(float* d, unsigned a0, unsigned a1, unsigned b) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
        "{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(b));
}

DEV float h2f(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

DEV unsigned short f2h(float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

DEV float lo2f(unsigned v) { return h2f((unsigned short)(v & 0xffffu)); }
DEV float hi2f(unsigned v) { return h2f((unsigned short)(v >> 16)); }
DEV unsigned pack2(unsigned short lo, unsigned short hi) { return (unsigned)lo | ((unsigned)hi << 16); }

// relu(x a + c) on 2 fp16 channels (computed in fp32, one rounding to fp16 like eager BN + ReLU)
DEV unsigned pro2(unsigned x, float a0, float c0, float a1, float c1) {
    return pack2(f2h(fmaxf(fmaf(lo2f(x), a0, c0), 0.0f)), f2h(fmaxf(fmaf(hi2f(x), a1, c1), 0.0f)));
}

// ---------------------------------------------------------------------------------------------
// conv1 forward
// ---------------------------------------------------------------------------------------------

struct C1F {
    const half_t* X;            // block buffer [rows][sx], columns [0, cin) are read
    const half_t* W;            // [128][cin]
    const float* A;             // prologue: a = gamma rstd [cin]
    const float* C;             //          c = beta - mean a [cin]
    const float* sh;            // [128] constant subtracted before the sums (norm2 running mean)
    half_t* U;                  // [rows][128]
    float* part;                // [gridDim.x][256]: sums (128) and sums of squares (128)
    int rows, cin, sx;
};

// Block of BM rows x 128 columns, k in slices of 32 channels (64 bytes per row). 8 warps: WARPS_M x
// WARPS_N, every warp 32 rows x WN columns (2 x NT m16n8 tiles). Shared memory: two stages of A
// [BM][32] and B [128][32] with swizzled 16-byte chunks (chunk ^ ((row >> 1) & 3): the 8 rows of an
// ldmatrix fall into 8 different bank groups); the prologue is applied between the register and
// shared memory. One stage in flight in registers while the other is computed.
template <int BM>
DEV void c1f_body(const C1F& p) {
    constexpr int WARPS_M = BM / 32;
    constexpr int WARPS_N = 8 / WARPS_M;
    constexpr int WN = 128 / WARPS_N;
    constexpr int NT = WN / 8;
    constexpr int A_H = BM * 32, B_H = 128 * 32;   // halves per stage
    constexpr int ACH = BM / 64;                   // 16-byte chunks of A per thread and stage
    half_t* sA = (half_t*)smem;
    half_t* sB = sA + 2 * A_H;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % WARPS_M, wn = warp / WARPS_M;
    const int m0 = blockIdx.x * BM;
    const int KT = p.cin >> 5;

    U4 ra[ACH], rb[2];
    auto gload = [&](int kt) {
        const int k0 = kt * 32;
#pragma unroll
        for (int i = 0; i < ACH; ++i) {
            const int q = tid + i * 256, r = q >> 2, ch = q & 3;
            const int m = m0 + r;
            ra[i] = m < p.rows ? *(const U4*)(p.X + (size_t)m * p.sx + k0 + ch * 8) : U4{0u, 0u, 0u, 0u};
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int q = tid + i * 256, n = q >> 2, ch = q & 3;
            rb[i] = *(const U4*)(p.W + (size_t)n * p.cin + k0 + ch * 8);
        }
    };
    auto sstore = [&](int buf, int kt) {
        const int k0 = kt * 32;
#pragma unroll
        for (int i = 0; i < ACH; ++i) {
            const int q = tid + i * 256, r = q >> 2, ch = q & 3;
            const int kc = k0 + ch * 8;
            const F4 a0 = *(const F4*)(p.A + kc), a1 = *(const F4*)(p.A + kc + 4);
            const F4 c0 = *(const F4*)(p.C + kc), c1 = *(const F4*)(p.C + kc + 4);
            U4 v = U4{pro2(ra[i].x, a0.x, c0.x, a0.y, c0.y), pro2(ra[i].y, a0.z, c0.z, a0.w, c0.w),
                      pro2(ra[i].z, a1.x, c1.x, a1.y, c1.y), pro2(ra[i].w, a1.z, c1.z, a1.w, c1.w)};
            if (m0 + r >= p.rows) v = U4{0u, 0u, 0u, 0u};
            *(U4*)(sA + buf * A_H + r * 32 + ((ch ^ ((r >> 1) & 3)) << 3)) = v;
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int q = tid + i * 256, n = q >> 2, ch = q & 3;
            *(U4*)(sB + buf * B_H + n * 32 + ((ch ^ ((n >> 1) & 3)) << 3)) = rb[i];
        }
    };

    float acc[2][NT][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    auto compute = [&](int buf) {
        const unsigned aBase = smem_addr(sA + buf * A_H), bBase = smem_addr(sB + buf * B_H);
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            unsigned af[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int r = wm * 32 + mi * 16 + (lane & 15);
                const int ch = kk * 2 + (lane >> 4);
                ldsm4(aBase + (r * 32 + ((ch ^ ((r >> 1) & 3)) << 3)) * 2, af[mi]);
            }
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                unsigned bf[NT];
                const int ch = kk * 2 + s;
#pragma unroll
                for (int j = 0; j < NT / 4; ++j) {
                    const int n = wn * WN + j * 32 + lane;
                    ldsm4(bBase + (n * 32 + ((ch ^ ((n >> 1) & 3)) << 3)) * 2, bf + 4 * j);
                }
#pragma unroll
                for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni)
                        mma16(acc[mi][ni], af[mi][2 * s], af[mi][2 * s + 1], bf[ni]);
            }
        }
    };

    gload(0);
    sstore(0, 0);
    __syncthreads();
    for (int kt = 0; kt < KT; ++kt) {
        if (kt + 1 < KT) gload(kt + 1);
        compute(kt & 1);
        if (kt + 1 < KT) sstore((kt + 1) & 1, kt + 1);
        __syncthreads();
    }

    // epilogue: fp16 in shared memory ([BM][128], chunk ^ (row & 7)) and per-column sums
    half_t* sO = (half_t*)smem;
    float s1[NT][2], s2[NT][2];
#pragma unroll
    for (int ni = 0; ni < NT; ++ni) {
        const int c = wn * WN + ni * 8 + (lane & 3) * 2;
        const float h0 = p.sh[c], h1 = p.sh[c + 1];
        s1[ni][0] = s1[ni][1] = s2[ni][0] = s2[ni][1] = 0.0f;
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const int r = wm * 32 + mi * 16 + (lane >> 2) + hr * 8;
                const unsigned short u0 = f2h(acc[mi][ni][2 * hr]), u1 = f2h(acc[mi][ni][2 * hr + 1]);
                *(unsigned*)(sO + r * 128 + (((c >> 3) ^ (r & 7)) << 3) + (c & 7)) = pack2(u0, u1);
                if (m0 + r < p.rows) {
                    const float d0 = h2f(u0) - h0, d1 = h2f(u1) - h1;
                    s1[ni][0] += d0;
                    s1[ni][1] += d1;
                    s2[ni][0] += d0 * d0;
                    s2[ni][1] += d1 * d1;
                }
            }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) {
                s1[ni][j] += __shfl_xor_sync(0xffffffffu, s1[ni][j], o);
                s2[ni][j] += __shfl_xor_sync(0xffffffffu, s2[ni][j], o);
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < BM * 16 / 256; ++i) {
        const int q = tid + i * 256, r = q >> 4, ch = q & 15;
        if (m0 + r < p.rows)
            *(U4*)(p.U + (size_t)(m0 + r) * 128 + ch * 8) = *(const U4*)(sO + r * 128 + ((ch ^ (r & 7)) << 3));
    }
    __syncthreads();
    float* red = (float*)smem;                     // [WARPS_M][256]
    if (lane < 4) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni)
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int c = wn * WN + ni * 8 + lane * 2 + j;
                red[wm * 256 + c] = s1[ni][j];
                red[wm * 256 + 128 + c] = s2[ni][j];
            }
    }
    __syncthreads();
    {
        float t = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS_M; ++w) t += red[w * 256 + tid];
        p.part[(size_t)blockIdx.x * 256 + tid] = t;
    }
}

extern "C" __global__ void __launch_bounds__(256, 2) c1f_128(C1F p) { c1f_body<128>(p); }
extern "C" __global__ void __launch_bounds__(256, 2) c1f_64(C1F p) { c1f_body<64>(p); }

// ---------------------------------------------------------------------------------------------
// conv1 backward
// ---------------------------------------------------------------------------------------------

struct C1B {
    const half_t* dU;           // [rows][128]
    const half_t* X;            // [rows][sx], columns [0, cin)
    const half_t* W;            // [128][cin]
    const float* A;             // a = gamma rstd [cin]
    const float* C;             // c = beta - mean a [cin]
    const float* MU;            // batch mean [cin]
    const float* RS;            // rstd [cin]
    half_t* G;                  // [rows][sg]: G[:, :cin] += a gm
    float* partR;               // [split][2][cin]: sum of gm, sum of gm xhat
    float* partW;               // [split][128][cin]: dW
    int rows, cin, sx, sg, split;
    // Fused BN2 (c1b_b, c1b_pb): instead of dU, dV (gradient of the relu(bn2) output) and U, and
    // dU = a2 (gm - k1 - uhat k2), gm = dV [u a2 + c2 > 0], uhat = (u - mu2) rs2, computed while the dU
    // block goes to shared memory (no dU in global memory)
    const half_t* dV;           // [rows][128]
    const half_t* U2;           // [rows][128]
    const float* bn2;           // [6][128]: a2, c2, mu2, rs2, k1 = R1/P, k2 = R2/P
};

// Program (slice of 64 channels = blockIdx.x, slice of rows = blockIdx.y: blocks of 64 rows
// t = blockIdx.y, blockIdx.y + split, ...). Shared memory: sW [128][64] (the slice weights, once),
// sDU [64][128] (the dU block), sY [64][64] (y = relu(x a + c) of the block), rows in 16-byte chunks
// swizzled with (row & 7). 8 warps:
// - GEMM 1 (dY = dU W, 64 x 64, k = 128): warps 4 x 2, each 16 rows x 32 channels (4 tiles);
// - epilogue in the fragment registers: x and G (channel pairs) from global memory;
// - GEMM 2 (dW += dU^T y, 128 x 64, k = 64 rows): warps 4 x 2, each 32 x 32 (2 x 4 tiles), fp32
//   accumulators for the whole lifetime of the program.
template <bool PF, bool BN2F>
DEV void c1b_body(const C1B& p) {
    half_t* sW = (half_t*)smem;                    // [128][64]
    half_t* sDU = sW + 128 * 64;                   // [64][128]
    half_t* sY = sDU + 64 * 128;                   // [64][64]
    float* sP = (float*)(sY + 64 * 64);            // [4][64]: a, c, mu, rs
    float* sB2 = sP + 256;                         // [6][128]: fused BN2

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 3, wn = warp >> 2;
    const int c0 = blockIdx.x * 64;
    const int ntiles = (p.rows + 63) >> 6;

    // the slice weights (columns beyond cin set to zero) and the BN parameters
    for (int q = tid; q < 128 * 8; q += 256) {
        const int o = q >> 3, ch = q & 7;
        const int c = c0 + ch * 8;
        const U4 v = c < p.cin ? *(const U4*)(p.W + (size_t)o * p.cin + c) : U4{0u, 0u, 0u, 0u};
        *(U4*)(sW + o * 64 + ((ch ^ (o & 7)) << 3)) = v;
    }
    if (tid < 64) {
        const int c = c0 + tid;
        const bool ok = c < p.cin;
        sP[tid] = ok ? p.A[c] : 0.0f;
        sP[64 + tid] = ok ? p.C[c] : 0.0f;
        sP[128 + tid] = ok ? p.MU[c] : 0.0f;
        sP[192 + tid] = ok ? p.RS[c] : 0.0f;
    }
    if (BN2F)
        for (int i = tid; i < 6 * 128; i += 256) sB2[i] = p.bn2[i];

    float accw[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) accw[i][j][0] = accw[i][j][1] = accw[i][j][2] = accw[i][j][3] = 0.0f;
    float r1[4][2], r2[4][2];
#pragma unroll
    for (int i = 0; i < 4; ++i) r1[i][0] = r1[i][1] = r2[i][0] = r2[i][1] = 0.0f;

    // dU of block t in registers: 64 x 128 halves = 1024 chunks, 4 per thread (with BN2F: dV and U, and
    // dU is computed on the way to shared memory)
    U4 rdu[4], ru[4];
    int tcur = 0;
    auto gload = [&](int t) {
        tcur = t;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q = tid + i * 256, r = q >> 4, ch = q & 15;
            const int m = t * 64 + r;
            const bool ok = m < p.rows;
            const size_t off = (size_t)m * 128 + ch * 8;
            rdu[i] = ok ? *(const U4*)((BN2F ? p.dV : p.dU) + off) : U4{0u, 0u, 0u, 0u};
            if (BN2F) ru[i] = ok ? *(const U4*)(p.U2 + off) : U4{0u, 0u, 0u, 0u};
        }
    };
    auto bn2w = [&](unsigned dv, unsigned u, int k) -> unsigned {   // channels k, k + 1
        float o[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float uu = j ? hi2f(u) : lo2f(u), dd = j ? hi2f(dv) : lo2f(dv);
            const float a2 = sB2[k + j], c2 = sB2[128 + k + j];
            const float gm = fmaf(uu, a2, c2) > 0.0f ? dd : 0.0f;
            const float uh = (uu - sB2[256 + k + j]) * sB2[384 + k + j];
            o[j] = a2 * (gm - sB2[512 + k + j] - uh * sB2[640 + k + j]);
        }
        return pack2(f2h(o[0]), f2h(o[1]));
    };
    auto sstore = [&]() {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q = tid + i * 256, r = q >> 4, ch = q & 15;
            U4 v = rdu[i];
            if (BN2F) {
                const int k = ch * 8;
                v = U4{bn2w(rdu[i].x, ru[i].x, k), bn2w(rdu[i].y, ru[i].y, k + 2),
                       bn2w(rdu[i].z, ru[i].z, k + 4), bn2w(rdu[i].w, ru[i].w, k + 6)};
                if (tcur * 64 + r >= p.rows) v = U4{0u, 0u, 0u, 0u};
            }
            *(U4*)(sDU + r * 128 + ((ch ^ (r & 7)) << 3)) = v;
        }
    };

    const unsigned wBase = smem_addr(sW), duBase = smem_addr(sDU), yBase = smem_addr(sY);
    // x and G of the fragment: rows r and r + 8, 4 tiles of 8 channels; with PF those of the next block
    // are fetched during the current block (global memory latency hidden)
    const int rbase = wm * 16 + (lane >> 2);
    unsigned xv[4][2], gv[4][2], xn[4][2], gn[4][2];
    auto xgload = [&](int t, unsigned (&xx)[4][2], unsigned (&gg)[4][2]) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int c = c0 + wn * 32 + ni * 8 + (lane & 3) * 2;
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const int m = t * 64 + rbase + hr * 8;
                const bool ok = m < p.rows && c < p.cin;
                xx[ni][hr] = ok ? *(const unsigned*)(p.X + (size_t)m * p.sx + c) : 0u;
                gg[ni][hr] = ok ? *(const unsigned*)(p.G + (size_t)m * p.sg + c) : 0u;
            }
        }
    };
    int t = blockIdx.y;
    if (t < ntiles) {
        gload(t);
        if (PF) xgload(t, xv, gv);
    }
    for (; t < ntiles; t += p.split) {
        __syncthreads();                           // sDU, sY free (GEMM 2 of the previous block)
        sstore();
        __syncthreads();
        if (t + p.split < ntiles) {
            gload(t + p.split);
            if (PF) xgload(t + p.split, xn, gn);
        }
        if (!PF) xgload(t, xv, gv);

        // GEMM 1: dY (16 rows x 32 channels of the warp) = dU (rows, k = o) W (k = o, channels)
        float dy[4][4];
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) dy[ni][0] = dy[ni][1] = dy[ni][2] = dy[ni][3] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            unsigned af[4];
            {
                const int r = wm * 16 + (lane & 15);
                const int ch = kk * 2 + (lane >> 4);
                ldsm4(duBase + (r * 128 + ((ch ^ (r & 7)) << 3)) * 2, af);
            }
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                unsigned bf[4];
                const int o = kk * 16 + s * 8 + (lane & 7);
                const int ch = wn * 4 + (lane >> 3);
                ldsm4t(wBase + (o * 64 + ((ch ^ (o & 7)) << 3)) * 2, bf);
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) mma16(dy[ni], af[2 * s], af[2 * s + 1], bf[ni]);
            }
        }

        // epilogue: gm, G += a gm, reductions, y into sY
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int cl = wn * 32 + ni * 8 + (lane & 3) * 2;
            const int c = c0 + cl;
            float a2[2], cc[2], mu[2], rs[2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                a2[j] = sP[cl + j];
                cc[j] = sP[64 + cl + j];
                mu[j] = sP[128 + cl + j];
                rs[j] = sP[192 + cl + j];
            }
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const int rl = rbase + hr * 8;
                const int m = t * 64 + rl;
                const bool ok = m < p.rows && c < p.cin;
                const float x0 = lo2f(xv[ni][hr]), x1 = hi2f(xv[ni][hr]);
                const float z0 = fmaf(x0, a2[0], cc[0]), z1 = fmaf(x1, a2[1], cc[1]);
                const bool p0 = ok && z0 > 0.0f, p1 = ok && z1 > 0.0f;
                const float gm0 = p0 ? dy[ni][2 * hr] : 0.0f, gm1 = p1 ? dy[ni][2 * hr + 1] : 0.0f;
                r1[ni][0] += gm0;
                r1[ni][1] += gm1;
                r2[ni][0] += gm0 * ((x0 - mu[0]) * rs[0]);
                r2[ni][1] += gm1 * ((x1 - mu[1]) * rs[1]);
                if (ok) {
                    const float g0 = lo2f(gv[ni][hr]) + a2[0] * gm0, g1 = hi2f(gv[ni][hr]) + a2[1] * gm1;
                    *(unsigned*)(p.G + (size_t)m * p.sg + c) = pack2(f2h(g0), f2h(g1));
                }
                *(unsigned*)(sY + rl * 64 + (((cl >> 3) ^ (rl & 7)) << 3) + (cl & 7)) =
                    pack2(f2h(p0 ? z0 : 0.0f), f2h(p1 ? z1 : 0.0f));
            }
        }
        __syncthreads();                           // sY complete

        // GEMM 2: dW (32 o x 32 channels of the warp) += dU^T (o, k = rows) y (k = rows, channels)
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            unsigned af[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int kr = kk * 16 + (lane & 7) + ((lane >> 4) << 3);
                const int ch = (wm * 32 + mi * 16) / 8 + ((lane >> 3) & 1);
                ldsm4t(duBase + (kr * 128 + ((ch ^ (kr & 7)) << 3)) * 2, af[mi]);
            }
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                unsigned bf[4];
                const int kr = kk * 16 + s * 8 + (lane & 7);
                const int ch = wn * 4 + (lane >> 3);
                ldsm4t(yBase + (kr * 64 + ((ch ^ (kr & 7)) << 3)) * 2, bf);
#pragma unroll
                for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                    for (int ni = 0; ni < 4; ++ni)
                        mma16(accw[mi][ni], af[mi][2 * s], af[mi][2 * s + 1], bf[ni]);
            }
        }
        if (PF) {
#pragma unroll
            for (int ni = 0; ni < 4; ++ni)
#pragma unroll
                for (int hr = 0; hr < 2; ++hr) {
                    xv[ni][hr] = xn[ni][hr];
                    gv[ni][hr] = gn[ni][hr];
                }
        }
    }

    // reductions: across the 8 rows of the fragment (lane >> 2), then across the 4 row warps
#pragma unroll
    for (int ni = 0; ni < 4; ++ni)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) {
                r1[ni][j] += __shfl_xor_sync(0xffffffffu, r1[ni][j], o);
                r2[ni][j] += __shfl_xor_sync(0xffffffffu, r2[ni][j], o);
            }
    __syncthreads();
    float* red = (float*)sY;                       // [4][128]: r1 (64), r2 (64) per row warp
    if (lane < 4) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int cl = wn * 32 + ni * 8 + lane * 2 + j;
                red[wm * 128 + cl] = r1[ni][j];
                red[wm * 128 + 64 + cl] = r2[ni][j];
            }
    }
    __syncthreads();
    if (tid < 128) {
        const int cl = tid & 63, which = tid >> 6;
        const int c = c0 + cl;
        if (c < p.cin) {
            const float v = red[which * 64 + cl] + red[128 + which * 64 + cl] +
                            red[256 + which * 64 + cl] + red[384 + which * 64 + cl];
            p.partR[((size_t)blockIdx.y * 2 + which) * p.cin + c] = v;
        }
    }
    // dW of the slice: o = rows of the fragment, channels = columns
    float* pw = p.partW + (size_t)blockIdx.y * 128 * p.cin;
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int c = c0 + wn * 32 + ni * 8 + (lane & 3) * 2;
            if (c < p.cin) {
#pragma unroll
                for (int hr = 0; hr < 2; ++hr) {
                    const int o = wm * 32 + mi * 16 + (lane >> 2) + hr * 8;
                    *(F2*)(pw + (size_t)o * p.cin + c) = F2{accw[mi][ni][2 * hr], accw[mi][ni][2 * hr + 1]};
                }
            }
        }
}

extern "C" __global__ void __launch_bounds__(256, 1) c1b(C1B p) { c1b_body<false, false>(p); }
extern "C" __global__ void __launch_bounds__(256, 1) c1b_p(C1B p) { c1b_body<true, false>(p); }
extern "C" __global__ void __launch_bounds__(256, 1) c1b_b(C1B p) { c1b_body<false, true>(p); }
extern "C" __global__ void __launch_bounds__(256, 1) c1b_pb(C1B p) { c1b_body<true, true>(p); }

// ---------------------------------------------------------------------------------------------
// conv1 backward, two programs per SM (c1b2)
// ---------------------------------------------------------------------------------------------
// c1b takes ~99 ms per step, ~2x the DRAM limit: one program per SM (220 registers, 41 KB of shared
// memory), phases in sequence (GEMM 1, epilogue, GEMM 2) separated by barriers and nothing to overlap.
// Here blocks of 32 rows: sDU [32][128], sY [32][64], sW [128][64] = 29 KB, <= 128 registers -> two
// programs per SM, the phases of one cover the barriers of the other. Same arithmetic as c1b (same
// sums per element).
// - GEMM 1 (dY = dU W, 32 x 64, k = 128): warps 2 (rows, 16) x 4 (channels, 16), 1 x 2 tiles;
// - GEMM 2 (dW += dU^T y, 128 x 64, k = 32 rows): warps 4 (o, 32) x 2 (channels, 32), 2 x 4 tiles.
extern "C" __global__ void __launch_bounds__(256, 2) c1b2(C1B p) {
    half_t* sW = (half_t*)smem;                    // [128][64]
    half_t* sDU = sW + 128 * 64;                   // [32][128]
    half_t* sY = sDU + 32 * 128;                   // [32][64]
    float* sP = (float*)(sY + 32 * 64);            // [4][64]: a, c, mu, rs

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int c0 = blockIdx.x * 64;
    const int ntiles = (p.rows + 31) >> 5;

    for (int q = tid; q < 128 * 8; q += 256) {
        const int o = q >> 3, ch = q & 7;
        const int c = c0 + ch * 8;
        const U4 v = c < p.cin ? *(const U4*)(p.W + (size_t)o * p.cin + c) : U4{0u, 0u, 0u, 0u};
        *(U4*)(sW + o * 64 + ((ch ^ (o & 7)) << 3)) = v;
    }
    if (tid < 64) {
        const int c = c0 + tid;
        const bool ok = c < p.cin;
        sP[tid] = ok ? p.A[c] : 0.0f;
        sP[64 + tid] = ok ? p.C[c] : 0.0f;
        sP[128 + tid] = ok ? p.MU[c] : 0.0f;
        sP[192 + tid] = ok ? p.RS[c] : 0.0f;
    }

    // GEMM 1 / epilogue: warp (wm1 rows, wn1 channels); GEMM 2: warp (wm2 o, wn2 channels)
    const int wm1 = warp & 1, wn1 = warp >> 1;
    const int wm2 = warp & 3, wn2 = warp >> 2;
    float accw[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) accw[i][j][0] = accw[i][j][1] = accw[i][j][2] = accw[i][j][3] = 0.0f;
    float r1[2][2], r2[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i) r1[i][0] = r1[i][1] = r2[i][0] = r2[i][1] = 0.0f;

    // dU of the block: 32 x 128 halves = 512 chunks, 2 per thread
    U4 rdu[2];
    auto gload = [&](int t) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int q = tid + i * 256, r = q >> 4, ch = q & 15;
            const int m = t * 32 + r;
            rdu[i] = m < p.rows ? *(const U4*)(p.dU + (size_t)m * 128 + ch * 8) : U4{0u, 0u, 0u, 0u};
        }
    };
    auto sstore = [&]() {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int q = tid + i * 256, r = q >> 4, ch = q & 15;
            *(U4*)(sDU + r * 128 + ((ch ^ (r & 7)) << 3)) = rdu[i];
        }
    };

    const unsigned wBase = smem_addr(sW), duBase = smem_addr(sDU), yBase = smem_addr(sY);
    const int rbase = wm1 * 16 + (lane >> 2);
    int t = blockIdx.y;
    if (t < ntiles) gload(t);
    for (; t < ntiles; t += p.split) {
        __syncthreads();
        sstore();
        __syncthreads();
        if (t + p.split < ntiles) gload(t + p.split);

        unsigned xv[2][2], gv[2][2];
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            const int c = c0 + wn1 * 16 + ni * 8 + (lane & 3) * 2;
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const int m = t * 32 + rbase + hr * 8;
                const bool ok = m < p.rows && c < p.cin;
                xv[ni][hr] = ok ? *(const unsigned*)(p.X + (size_t)m * p.sx + c) : 0u;
                gv[ni][hr] = ok ? *(const unsigned*)(p.G + (size_t)m * p.sg + c) : 0u;
            }
        }

        // GEMM 1
        float dy[2][4];
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) dy[ni][0] = dy[ni][1] = dy[ni][2] = dy[ni][3] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            unsigned af[4], bf[4];
            {
                const int r = wm1 * 16 + (lane & 15);
                const int ch = kk * 2 + (lane >> 4);
                ldsm4(duBase + (r * 128 + ((ch ^ (r & 7)) << 3)) * 2, af);
            }
            {
                // matrices: (o 0-7, n8 0), (o 0-7, n8 1), (o 8-15, n8 0), (o 8-15, n8 1)
                const int o = kk * 16 + (lane & 7) + ((lane >> 4) << 3);
                const int ch = wn1 * 2 + ((lane >> 3) & 1);
                ldsm4t(wBase + (o * 64 + ((ch ^ (o & 7)) << 3)) * 2, bf);
            }
            mma16(dy[0], af[0], af[1], bf[0]);
            mma16(dy[1], af[0], af[1], bf[1]);
            mma16(dy[0], af[2], af[3], bf[2]);
            mma16(dy[1], af[2], af[3], bf[3]);
        }

        // epilogue
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            const int cl = wn1 * 16 + ni * 8 + (lane & 3) * 2;
            const int c = c0 + cl;
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const int rl = rbase + hr * 8;
                const int m = t * 32 + rl;
                const bool ok = m < p.rows && c < p.cin;
                const float x0 = lo2f(xv[ni][hr]), x1 = hi2f(xv[ni][hr]);
                const float a0 = sP[cl], a1 = sP[cl + 1];
                const float z0 = fmaf(x0, a0, sP[64 + cl]), z1 = fmaf(x1, a1, sP[64 + cl + 1]);
                const bool p0 = ok && z0 > 0.0f, p1 = ok && z1 > 0.0f;
                const float gm0 = p0 ? dy[ni][2 * hr] : 0.0f, gm1 = p1 ? dy[ni][2 * hr + 1] : 0.0f;
                r1[ni][0] += gm0;
                r1[ni][1] += gm1;
                r2[ni][0] += gm0 * ((x0 - sP[128 + cl]) * sP[192 + cl]);
                r2[ni][1] += gm1 * ((x1 - sP[128 + cl + 1]) * sP[192 + cl + 1]);
                if (ok) {
                    const float g0 = lo2f(gv[ni][hr]) + a0 * gm0, g1 = hi2f(gv[ni][hr]) + a1 * gm1;
                    *(unsigned*)(p.G + (size_t)m * p.sg + c) = pack2(f2h(g0), f2h(g1));
                }
                *(unsigned*)(sY + rl * 64 + (((cl >> 3) ^ (rl & 7)) << 3) + (cl & 7)) =
                    pack2(f2h(p0 ? z0 : 0.0f), f2h(p1 ? z1 : 0.0f));
            }
        }
        __syncthreads();

        // GEMM 2: k = 32 rows
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            unsigned af[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int kr = kk * 16 + (lane & 7) + ((lane >> 4) << 3);
                const int ch = (wm2 * 32 + mi * 16) / 8 + ((lane >> 3) & 1);
                ldsm4t(duBase + (kr * 128 + ((ch ^ (kr & 7)) << 3)) * 2, af[mi]);
            }
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                unsigned bf[4];
                const int kr = kk * 16 + s * 8 + (lane & 7);
                const int ch = wn2 * 4 + (lane >> 3);
                ldsm4t(yBase + (kr * 64 + ((ch ^ (kr & 7)) << 3)) * 2, bf);
#pragma unroll
                for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                    for (int ni = 0; ni < 4; ++ni)
                        mma16(accw[mi][ni], af[mi][2 * s], af[mi][2 * s + 1], bf[ni]);
            }
        }
    }

    // reductions: 8 rows of the fragment, then the 2 row warps (wm1)
#pragma unroll
    for (int ni = 0; ni < 2; ++ni)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) {
                r1[ni][j] += __shfl_xor_sync(0xffffffffu, r1[ni][j], o);
                r2[ni][j] += __shfl_xor_sync(0xffffffffu, r2[ni][j], o);
            }
    __syncthreads();
    float* red = (float*)sY;                       // [2][128]: r1 (64), r2 (64) per row warp
    if (lane < 4) {
#pragma unroll
        for (int ni = 0; ni < 2; ++ni)
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int cl = wn1 * 16 + ni * 8 + lane * 2 + j;
                red[wm1 * 128 + cl] = r1[ni][j];
                red[wm1 * 128 + 64 + cl] = r2[ni][j];
            }
    }
    __syncthreads();
    if (tid < 128) {
        const int cl = tid & 63, which = tid >> 6;
        const int c = c0 + cl;
        if (c < p.cin)
            p.partR[((size_t)blockIdx.y * 2 + which) * p.cin + c] =
                red[which * 64 + cl] + red[128 + which * 64 + cl];
    }
    float* pw = p.partW + (size_t)blockIdx.y * 128 * p.cin;
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int c = c0 + wn2 * 32 + ni * 8 + (lane & 3) * 2;
            if (c < p.cin) {
#pragma unroll
                for (int hr = 0; hr < 2; ++hr) {
                    const int o = wm2 * 32 + mi * 16 + (lane >> 2) + hr * 8;
                    *(F2*)(pw + (size_t)o * p.cin + c) = F2{accw[mi][ni][2 * hr], accw[mi][ni][2 * hr + 1]};
                }
            }
        }
}

// ---------------------------------------------------------------------------------------------
// conv1 backward with x and G staged through shared memory (c1b3)
// ---------------------------------------------------------------------------------------------
// In c1b x and G are read (and G written) at 4 bytes per thread in the fragment layout: 8 rows x 16
// bytes per warp instruction, 32-byte sectors half used, ~1,500 L1 transactions per block of 64 rows.
// Here the x and G blocks [64][64] arrive with coalesced 16-byte loads (one block ahead in registers,
// like dU), the epilogue reads them from shared memory and writes G back in place, and G returns to
// global memory in 16-byte chunks. 57 KB of shared memory (sW 16, sDU 16, sY 8, sX 8, sG 8, sP 1): one
// program per SM. The default backward kernel.
// ABL (micro-benchmark only, wrong results): 1 no G read and written, 2 no X read, 3 no epilogue (only
// y = x), 4 no GEMM 2, 5 no GEMM 1
template <int ABL>
DEV void c1b3_body(const C1B& p) {
    half_t* sW = (half_t*)smem;                    // [128][64]
    half_t* sDU = sW + 128 * 64;                   // [64][128]
    half_t* sY = sDU + 64 * 128;                   // [64][64]
    half_t* sX = sY + 64 * 64;                     // [64][64]
    half_t* sG = sX + 64 * 64;                     // [64][64]
    float* sP = (float*)(sG + 64 * 64);            // [4][64]: a, c, mu, rs

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 3, wn = warp >> 2;
    const int c0 = blockIdx.x * 64;
    const int ntiles = (p.rows + 63) >> 6;
    const bool full_cols = c0 + 64 <= p.cin;      // cin a multiple of 32: the last slice may have 32

    for (int q = tid; q < 128 * 8; q += 256) {
        const int o = q >> 3, ch = q & 7;
        const int c = c0 + ch * 8;
        const U4 v = c < p.cin ? *(const U4*)(p.W + (size_t)o * p.cin + c) : U4{0u, 0u, 0u, 0u};
        *(U4*)(sW + o * 64 + ((ch ^ (o & 7)) << 3)) = v;
    }
    if (tid < 64) {
        const int c = c0 + tid;
        const bool ok = c < p.cin;
        sP[tid] = ok ? p.A[c] : 0.0f;
        sP[64 + tid] = ok ? p.C[c] : 0.0f;
        sP[128 + tid] = ok ? p.MU[c] : 0.0f;
        sP[192 + tid] = ok ? p.RS[c] : 0.0f;
    }

    float accw[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) accw[i][j][0] = accw[i][j][1] = accw[i][j][2] = accw[i][j][3] = 0.0f;
    float r1[4][2], r2[4][2];
#pragma unroll
    for (int i = 0; i < 4; ++i) r1[i][0] = r1[i][1] = r2[i][0] = r2[i][1] = 0.0f;

    // block t in registers: dU 64 x 128 (4 chunks per thread), x and G 64 x 64 (2 + 2)
    U4 rdu[4], rx[2], rg[2];
    auto gload = [&](int t) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q = tid + i * 256, r = q >> 4, ch = q & 15;
            const int m = t * 64 + r;
            rdu[i] = m < p.rows ? *(const U4*)(p.dU + (size_t)m * 128 + ch * 8) : U4{0u, 0u, 0u, 0u};
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int q = tid + i * 256, r = q >> 3, ch = q & 7;
            const int m = t * 64 + r;
            const int c = c0 + ch * 8;
            const bool ok = m < p.rows && c < p.cin;
            rx[i] = (ok && ABL != 2) ? *(const U4*)(p.X + (size_t)m * p.sx + c) : U4{0u, 0u, 0u, 0u};
            rg[i] = (ok && ABL != 1) ? *(const U4*)(p.G + (size_t)m * p.sg + c) : U4{0u, 0u, 0u, 0u};
        }
    };
    auto sstore = [&]() {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q = tid + i * 256, r = q >> 4, ch = q & 15;
            *(U4*)(sDU + r * 128 + ((ch ^ (r & 7)) << 3)) = rdu[i];
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int q = tid + i * 256, r = q >> 3, ch = q & 7;
            *(U4*)(sX + r * 64 + ((ch ^ (r & 7)) << 3)) = rx[i];
            *(U4*)(sG + r * 64 + ((ch ^ (r & 7)) << 3)) = rg[i];
        }
    };
    auto gstore = [&](int t) {                     // updated G: sG -> global memory
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int q = tid + i * 256, r = q >> 3, ch = q & 7;
            const int m = t * 64 + r;
            const int c = c0 + ch * 8;
            if (ABL != 1 && m < p.rows && (full_cols || c < p.cin))
                *(U4*)(p.G + (size_t)m * p.sg + c) = *(const U4*)(sG + r * 64 + ((ch ^ (r & 7)) << 3));
        }
    };

    const unsigned wBase = smem_addr(sW), duBase = smem_addr(sDU), yBase = smem_addr(sY);
    const int rbase = wm * 16 + (lane >> 2);
    int t = blockIdx.y;
    if (t < ntiles) gload(t);
    for (; t < ntiles; t += p.split) {
        __syncthreads();                           // smem free (GEMM 2 and the G write before)
        sstore();
        __syncthreads();
        if (t + p.split < ntiles) gload(t + p.split);

        // GEMM 1: dY (16 rows x 32 channels of the warp)
        float dy[4][4];
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) dy[ni][0] = dy[ni][1] = dy[ni][2] = dy[ni][3] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < (ABL == 5 ? 0 : 8); ++kk) {
            unsigned af[4];
            {
                const int r = wm * 16 + (lane & 15);
                const int ch = kk * 2 + (lane >> 4);
                ldsm4(duBase + (r * 128 + ((ch ^ (r & 7)) << 3)) * 2, af);
            }
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                unsigned bf[4];
                const int o = kk * 16 + s * 8 + (lane & 7);
                const int ch = wn * 4 + (lane >> 3);
                ldsm4t(wBase + (o * 64 + ((ch ^ (o & 7)) << 3)) * 2, bf);
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) mma16(dy[ni], af[2 * s], af[2 * s + 1], bf[ni]);
            }
        }

        // epilogue: x and G from shared memory, G updated in place, y into sY
#pragma unroll
        for (int ni = 0; ni < (ABL == 3 ? 0 : 4); ++ni) {
            const int cl = wn * 32 + ni * 8 + (lane & 3) * 2;
            const int c = c0 + cl;
            const float a0 = sP[cl], a1 = sP[cl + 1];
            const float cc0 = sP[64 + cl], cc1 = sP[64 + cl + 1];
            const float mu0 = sP[128 + cl], mu1 = sP[128 + cl + 1];
            const float rs0 = sP[192 + cl], rs1 = sP[192 + cl + 1];
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const int rl = rbase + hr * 8;
                const int m = t * 64 + rl;
                const bool ok = m < p.rows && c < p.cin;
                const int off = rl * 64 + (((cl >> 3) ^ (rl & 7)) << 3) + (cl & 7);
                const unsigned xw = *(const unsigned*)(sX + off);
                const float x0 = lo2f(xw), x1 = hi2f(xw);
                const float z0 = fmaf(x0, a0, cc0), z1 = fmaf(x1, a1, cc1);
                const bool p0 = ok && z0 > 0.0f, p1 = ok && z1 > 0.0f;
                const float gm0 = p0 ? dy[ni][2 * hr] : 0.0f, gm1 = p1 ? dy[ni][2 * hr + 1] : 0.0f;
                r1[ni][0] += gm0;
                r1[ni][1] += gm1;
                r2[ni][0] += gm0 * ((x0 - mu0) * rs0);
                r2[ni][1] += gm1 * ((x1 - mu1) * rs1);
                const unsigned gw = *(const unsigned*)(sG + off);
                *(unsigned*)(sG + off) = pack2(f2h(lo2f(gw) + a0 * gm0), f2h(hi2f(gw) + a1 * gm1));
                *(unsigned*)(sY + off) = pack2(f2h(p0 ? z0 : 0.0f), f2h(p1 ? z1 : 0.0f));
            }
        }
        __syncthreads();                           // sY and sG complete

        gstore(t);

        // GEMM 2: dW (32 o x 32 channels of the warp) += dU^T y
#pragma unroll
        for (int kk = 0; kk < (ABL == 4 ? 0 : 4); ++kk) {
            unsigned af[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int kr = kk * 16 + (lane & 7) + ((lane >> 4) << 3);
                const int ch = (wm * 32 + mi * 16) / 8 + ((lane >> 3) & 1);
                ldsm4t(duBase + (kr * 128 + ((ch ^ (kr & 7)) << 3)) * 2, af[mi]);
            }
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                unsigned bf[4];
                const int kr = kk * 16 + s * 8 + (lane & 7);
                const int ch = wn * 4 + (lane >> 3);
                ldsm4t(yBase + (kr * 64 + ((ch ^ (kr & 7)) << 3)) * 2, bf);
#pragma unroll
                for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                    for (int ni = 0; ni < 4; ++ni)
                        mma16(accw[mi][ni], af[mi][2 * s], af[mi][2 * s + 1], bf[ni]);
            }
        }
    }

#pragma unroll
    for (int ni = 0; ni < 4; ++ni)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) {
                r1[ni][j] += __shfl_xor_sync(0xffffffffu, r1[ni][j], o);
                r2[ni][j] += __shfl_xor_sync(0xffffffffu, r2[ni][j], o);
            }
    __syncthreads();
    float* red = (float*)sY;                       // [4][128]
    if (lane < 4) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int cl = wn * 32 + ni * 8 + lane * 2 + j;
                red[wm * 128 + cl] = r1[ni][j];
                red[wm * 128 + 64 + cl] = r2[ni][j];
            }
    }
    __syncthreads();
    if (tid < 128) {
        const int cl = tid & 63, which = tid >> 6;
        const int c = c0 + cl;
        if (c < p.cin) {
            const float v = red[which * 64 + cl] + red[128 + which * 64 + cl] +
                            red[256 + which * 64 + cl] + red[384 + which * 64 + cl];
            p.partR[((size_t)blockIdx.y * 2 + which) * p.cin + c] = v;
        }
    }
    float* pw = p.partW + (size_t)blockIdx.y * 128 * p.cin;
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int c = c0 + wn * 32 + ni * 8 + (lane & 3) * 2;
            if (c < p.cin) {
#pragma unroll
                for (int hr = 0; hr < 2; ++hr) {
                    const int o = wm * 32 + mi * 16 + (lane >> 2) + hr * 8;
                    *(F2*)(pw + (size_t)o * p.cin + c) = F2{accw[mi][ni][2 * hr], accw[mi][ni][2 * hr + 1]};
                }
            }
        }
}

extern "C" __global__ void __launch_bounds__(256, 1) c1b3(C1B p) { c1b3_body<0>(p); }
extern "C" __global__ void __launch_bounds__(256, 1) c1b3_a1(C1B p) { c1b3_body<1>(p); }
extern "C" __global__ void __launch_bounds__(256, 1) c1b3_a2(C1B p) { c1b3_body<2>(p); }
extern "C" __global__ void __launch_bounds__(256, 1) c1b3_a3(C1B p) { c1b3_body<3>(p); }
extern "C" __global__ void __launch_bounds__(256, 1) c1b3_a4(C1B p) { c1b3_body<4>(p); }
extern "C" __global__ void __launch_bounds__(256, 1) c1b3_a5(C1B p) { c1b3_body<5>(p); }

// ---------------------------------------------------------------------------------------------
// 3x3 bottleneck conv (128 -> 32, padding 1): forward (c3f) and weight gradient (c3w)
// ---------------------------------------------------------------------------------------------
// On cuDNN the 3x3 conv costs ~105 ms per step (fprop 34, dgrad 22, wgrad 46 ms: 8-15 TFLOPS on the
// power-capped T4), plus ~45 ms of elementwise passes around it (V = relu(bn2(U)) written and re-read,
// segment statistics, copy into the buffer). Implicit GEMM on the pixels in n, h, w order: every tap
// (dy, dx) of the 3x3 window is the same table of rows shifted by dy W + dx. Shared memory holds the
// halo of rows [p0 - W - 1, p0 + BM + W + 1) of a block of BM pixels; ldmatrix takes one address per
// row, so the shift costs one address per lane, and taps falling outside the image (borders, the
// neighbouring image, beyond the last row) point to a row of zeros: the conv's padding of 1 without
// copies. Rows of 64 bytes (32 channels), 16-byte chunks swizzled with (row >> 1) & 3: 8 consecutive
// rows of an ldmatrix in 8 different bank groups, for any shift.

#define C3_WMAX 72          // maximum width (block 1 at 288 px): fixes the halo registers;
                            // beyond it, `CudaOps` stays on cuDNN

DEV void ldsm2(unsigned addr, unsigned* r) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

DEV void ldsm1(unsigned addr, unsigned* r) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(r[0]) : "r"(addr));
}

template <int NT> struct LdsmN;
template <> struct LdsmN<4> { static DEV void ld(unsigned a, unsigned* r) { ldsm4(a, r); } };
template <> struct LdsmN<2> { static DEV void ld(unsigned a, unsigned* r) { ldsm2(a, r); } };
template <> struct LdsmN<1> { static DEV void ld(unsigned a, unsigned* r) { ldsm1(a, r); } };

// bit t (t = (dy + 1) 3 + dx + 1) set if pixel p + (dy, dx) is inside p's image
DEV unsigned tap_mask(int p, int rows, int H, int W) {
    if (p >= rows) return 0u;
    const int hw = p % (H * W);
    const int h = hw / W, w = hw - h * W;
    unsigned m = 0u;
#pragma unroll
    for (int t = 0; t < 9; ++t) {
        const int dy = t / 3 - 1, dx = t % 3 - 1;
        if (h + dy >= 0 && h + dy < H && w + dx >= 0 && w + dx < W) m |= 1u << t;
    }
    return m;
}

// index (in halves) of 16-byte chunk `ch` of row `row` in a table with 64-byte rows
DEV int swz64(int row, int ch) { return row * 32 + ((ch ^ ((row >> 1) & 3)) << 3); }

struct C3F {
    const half_t* U;            // [rows][128]: conv1 output (before norm2)
    const half_t* Wk;           // [9][32][128]: Wk[t][o][c] = conv2.weight[o][c][t / 3][t % 3]
    const float* A;             // [128] a = gamma2 rstd2
    const float* C;             // [128] c = beta2 - mean2 a
    const float* sh;            // [32] constant subtracted before the sums (segment statistics)
    half_t* V;                  // [rows][128]: relu(U a + c) for the backward, or 0 (eval)
    half_t* X;                  // block buffer [rows][sx]: columns [c0, c0 + 32)
    float* part;                // [gridDim.x][64]: sums (32) and sums of squares (32)
    int rows, H, W, sx, c0;
};

// Forward: S = conv3x3(relu(U a + c)), BM pixels x 32 outputs per program, k = 9 taps x 128 channels in
// slices of 32. Per slice: the halo of relu(U a + c) (prologue in fp32 like eager BN, one rounding to
// fp16) and the weights [9][32 o][32 c] in shared memory, the next slice in flight in registers. The
// central rows of the halo also go to V. 8 warps: WARPS_M x WARPS_N, every warp 32 pixels x WN outputs.
// Epilogue: S in fp16 into the block buffer and the per-column sums of the rounded values (like c1f).
// ABL (micro-benchmark only): 1 no prologue (raw U in shared memory, no V), 2 no tensor cores (loads,
// prologue and epilogue only)
template <int BM, int ABL = 0>
DEV void c3f_body(const C3F& p) {
    constexpr int WARPS_M = BM / 32, WARPS_N = 8 / WARPS_M, WN = 32 / WARPS_N, NT = WN / 8;
    constexpr int HIT = (4 * (BM + 2 * C3_WMAX + 2) + 255) / 256;   // halo chunks per thread
    const int R = BM + 2 * p.W + 2;                                   // halo rows
    half_t* sH = (half_t*)smem;                                       // [R + 1][32], row R = zeros
    half_t* sW = sH + (R + 1) * 32;                                   // [9][32][32]
    float* sA = (float*)(sW + 9 * 32 * 32);                           // [128]
    float* sC = sA + 128;                                             // [128]
    unsigned short* sM = (unsigned short*)(sC + 128);                 // [BM]

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % WARPS_M, wn = warp / WARPS_M;
    const int p0 = blockIdx.x * BM;
    const int r0 = p0 - p.W - 1;                                      // global row of halo row 0

    for (int i = tid; i < BM; i += 256) sM[i] = (unsigned short)tap_mask(p0 + i, p.rows, p.H, p.W);
    if (tid < 128) {
        sA[tid] = p.A[tid];
        sC[tid] = p.C[tid];
    }
    if (tid < 4) *(U4*)(sH + R * 32 + tid * 8) = U4{0u, 0u, 0u, 0u};

    U4 rh[HIT], rw[5];
    auto gload = [&](int kc) {
#pragma unroll
        for (int i = 0; i < HIT; ++i) {
            const int q = tid + i * 256, r = q >> 2, j = q & 3;
            const int g = r0 + r;
            rh[i] = (r < R && g >= 0 && g < p.rows)
                        ? *(const U4*)(p.U + (size_t)g * 128 + kc * 32 + j * 8) : U4{0u, 0u, 0u, 0u};
        }
#pragma unroll
        for (int i = 0; i < 5; ++i) {
            const int q = tid + i * 256;
            if (q < 9 * 32 * 4) {
                const int t = q >> 7, o = (q >> 2) & 31, j = q & 3;
                rw[i] = *(const U4*)(p.Wk + (size_t)(t * 32 + o) * 128 + kc * 32 + j * 8);
            }
        }
    };
    auto sstore = [&](int kc) {
#pragma unroll
        for (int i = 0; i < HIT; ++i) {
            const int q = tid + i * 256, r = q >> 2, j = q & 3;
            if (r < R) {
                const int g = r0 + r;
                const int kch = kc * 32 + j * 8;
                U4 v = rh[i];
                if (ABL != 1) {
                    const F4 a0 = *(const F4*)(sA + kch), a1 = *(const F4*)(sA + kch + 4);
                    const F4 c0 = *(const F4*)(sC + kch), c1 = *(const F4*)(sC + kch + 4);
                    v = U4{pro2(rh[i].x, a0.x, c0.x, a0.y, c0.y), pro2(rh[i].y, a0.z, c0.z, a0.w, c0.w),
                           pro2(rh[i].z, a1.x, c1.x, a1.y, c1.y), pro2(rh[i].w, a1.z, c1.z, a1.w, c1.w)};
                }
                if (g < 0 || g >= p.rows) v = U4{0u, 0u, 0u, 0u};
                *(U4*)(sH + swz64(r, j)) = v;
                if (ABL != 1 && p.V != nullptr && g >= p0 && g < p0 + BM && g < p.rows)
                    *(U4*)(p.V + (size_t)g * 128 + kch) = v;
            }
        }
#pragma unroll
        for (int i = 0; i < 5; ++i) {
            const int q = tid + i * 256;
            if (q < 9 * 32 * 4) {
                const int t = q >> 7, o = (q >> 2) & 31, j = q & 3;
                *(U4*)(sW + swz64(t * 32 + o, j)) = rw[i];
            }
        }
    };

    float acc[2][NT][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    gload(0);
    __syncthreads();                               // sA, sC, sM, row of zeros
    sstore(0);
    __syncthreads();

    // the lane's two A rows (one per m16 tile) in the block and their valid taps
    int arow[2];
    unsigned amsk[2];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        arow[mi] = wm * 32 + mi * 16 + (lane & 15);
        amsk[mi] = sM[arow[mi]];
    }
    const unsigned hBase = smem_addr(sH), wBase = smem_addr(sW);
    const int ob = wn * WN + (lane & (WN - 1));    // the lane's sW row (output) in the ldmatrix

    for (int kc = 0; kc < 4; ++kc) {
        if (kc + 1 < 4) gload(kc + 1);
#pragma unroll
        for (int t = 0; t < (ABL == 2 ? 0 : 9); ++t) {
            const int off = (t / 3) * p.W + (t % 3);     // (dy + 1) W + (dx + 1)
            int hr[2];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) hr[mi] = ((amsk[mi] >> t) & 1u) ? arow[mi] + off : R;
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                unsigned af[2][4];
#pragma unroll
                for (int mi = 0; mi < 2; ++mi)
                    ldsm4(hBase + swz64(hr[mi], kk * 2 + (lane >> 4)) * 2, af[mi]);
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    unsigned bf[NT];
                    LdsmN<NT>::ld(wBase + swz64(t * 32 + ob, kk * 2 + s) * 2, bf);
#pragma unroll
                    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                        for (int ni = 0; ni < NT; ++ni)
                            mma16(acc[mi][ni], af[mi][2 * s], af[mi][2 * s + 1], bf[ni]);
                }
            }
        }
        __syncthreads();
        if (kc + 1 < 4) {
            sstore(kc + 1);
            __syncthreads();
        }
    }

    // epilogue: S in fp16 into sO [BM][32] (over the halo) and per-column sums
    half_t* sO = sH;
    float s1[NT][2], s2[NT][2];
#pragma unroll
    for (int ni = 0; ni < NT; ++ni) {
        const int c = wn * WN + ni * 8 + (lane & 3) * 2;
        const float h0 = p.sh[c], h1 = p.sh[c + 1];
        s1[ni][0] = s1[ni][1] = s2[ni][0] = s2[ni][1] = 0.0f;
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const int r = wm * 32 + mi * 16 + (lane >> 2) + hr * 8;
                const unsigned short u0 = f2h(acc[mi][ni][2 * hr]), u1 = f2h(acc[mi][ni][2 * hr + 1]);
                *(unsigned*)(sO + swz64(r, c >> 3) + (c & 7)) = pack2(u0, u1);
                if (p0 + r < p.rows) {
                    const float d0 = h2f(u0) - h0, d1 = h2f(u1) - h1;
                    s1[ni][0] += d0;
                    s1[ni][1] += d1;
                    s2[ni][0] += d0 * d0;
                    s2[ni][1] += d1 * d1;
                }
            }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) {
                s1[ni][j] += __shfl_xor_sync(0xffffffffu, s1[ni][j], o);
                s2[ni][j] += __shfl_xor_sync(0xffffffffu, s2[ni][j], o);
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < BM * 4 / 256; ++i) {
        const int q = tid + i * 256, r = q >> 2, j = q & 3;
        if (p0 + r < p.rows)
            *(U4*)(p.X + (size_t)(p0 + r) * p.sx + p.c0 + j * 8) = *(const U4*)(sO + swz64(r, j));
    }
    __syncthreads();
    float* red = (float*)smem;                     // [WARPS_M][64]
    if (lane < 4) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni)
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int c = wn * WN + ni * 8 + lane * 2 + j;
                red[wm * 64 + c] = s1[ni][j];
                red[wm * 64 + 32 + c] = s2[ni][j];
            }
    }
    __syncthreads();
    if (tid < 64) {
        float t = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS_M; ++w) t += red[w * 64 + tid];
        p.part[(size_t)blockIdx.x * 64 + tid] = t;
    }
}

extern "C" __global__ void __launch_bounds__(256, 1) c3f_256(C3F p) { c3f_body<256>(p); }
extern "C" __global__ void __launch_bounds__(256, 1) c3f_256_a1(C3F p) { c3f_body<256, 1>(p); }
extern "C" __global__ void __launch_bounds__(256, 1) c3f_256_a2(C3F p) { c3f_body<256, 2>(p); }
extern "C" __global__ void __launch_bounds__(256, 1) c3f_128(C3F p) { c3f_body<128>(p); }
extern "C" __global__ void __launch_bounds__(256, 2) c3f_64(C3F p) { c3f_body<64>(p); }

struct C3W {
    const half_t* dS;           // [rows][32]: gradient of the conv output (contiguous)
    const half_t* V;            // [rows][128]: conv input, relu(bn2(U))
    float* partW;               // [split][9][32][128]: the program's partial dW[t][o][c]
    int rows, H, W, split;
};

// Weight gradient: dW[o, c, t] = sum_p dS[p - (dy, dx), o] V[p, c] (p and p - (dy, dx) inside the
// image). Program = slice of 32 channels of V (blockIdx.x) x blocks of 128 pixels in turn (blockIdx.y,
// + split, ...), fp32 accumulators for the whole lifetime of the program. 9 warps, one per tap: each
// 32 o x 32 c (2 x 4 m16n8 tiles) with k = the pixels; fragment A (dS transposed) comes from the dS
// halo shifted by its tap, B (V) from the central rows, the same for all warps. The next block in
// flight in registers.
extern "C" __global__ void __launch_bounds__(288, 2) c3w(C3W p) {
    constexpr int BM = 128;
    constexpr int HIT = (4 * (BM + 2 * C3_WMAX + 2) + 287) / 288;
    constexpr int VIT = (4 * BM + 287) / 288;
    const int R = BM + 2 * p.W + 2;
    half_t* sD = (half_t*)smem;                    // [R + 1][32]: dS halo, row R = zeros
    half_t* sV = sD + (R + 1) * 32;                // [BM][32]
    unsigned short* sM = (unsigned short*)(sV + BM * 32);   // [BM]

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;   // warp = tap
    const int kc = blockIdx.x;
    const int ntiles = (p.rows + BM - 1) / BM;
    const int dy = warp / 3 - 1, dx = warp % 3 - 1;
    const int offB = (1 - dy) * p.W + (1 - dx);    // halo row of p - (dy, dx), minus p - p0
    const int tb = 8 - warp;                       // tap_mask bit: p - (dy, dx) inside the image
    if (tid < 4) *(U4*)(sD + R * 32 + tid * 8) = U4{0u, 0u, 0u, 0u};

    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    U4 rd[HIT], rv[VIT];
    auto gload = [&](int t) {
        const int r0 = t * BM - p.W - 1;
#pragma unroll
        for (int i = 0; i < HIT; ++i) {
            const int q = tid + i * 288, r = q >> 2, j = q & 3;
            const int g = r0 + r;
            rd[i] = (r < R && g >= 0 && g < p.rows)
                        ? *(const U4*)(p.dS + (size_t)g * 32 + j * 8) : U4{0u, 0u, 0u, 0u};
        }
#pragma unroll
        for (int i = 0; i < VIT; ++i) {
            const int q = tid + i * 288, r = q >> 2, j = q & 3;
            const int g = t * BM + r;
            rv[i] = (q < 4 * BM && g < p.rows)
                        ? *(const U4*)(p.V + (size_t)g * 128 + kc * 32 + j * 8) : U4{0u, 0u, 0u, 0u};
        }
    };
    auto sstore = [&](int t) {
#pragma unroll
        for (int i = 0; i < HIT; ++i) {
            const int q = tid + i * 288, r = q >> 2, j = q & 3;
            if (r < R) *(U4*)(sD + swz64(r, j)) = rd[i];
        }
#pragma unroll
        for (int i = 0; i < VIT; ++i) {
            const int q = tid + i * 288, r = q >> 2, j = q & 3;
            if (q < 4 * BM) *(U4*)(sV + swz64(r, j)) = rv[i];
        }
        for (int i = tid; i < BM; i += 288) sM[i] = (unsigned short)tap_mask(t * BM + i, p.rows, p.H, p.W);
    };

    const unsigned dBase = smem_addr(sD), vBase = smem_addr(sV);
    int t = blockIdx.y;
    if (t < ntiles) gload(t);
    for (; t < ntiles; t += p.split) {
        __syncthreads();                           // smem free
        sstore(t);
        __syncthreads();
        if (t + p.split < ntiles) gload(t + p.split);
#pragma unroll 2
        for (int kk = 0; kk < BM / 16; ++kk) {
            unsigned af[2][4], bf[2][4];
            const int px = kk * 16 + (lane & 7) + ((lane >> 4) << 3);
            const int row = ((sM[px] >> tb) & 1u) ? px + offB : R;
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
                ldsm4t(dBase + swz64(row, mi * 2 + ((lane >> 3) & 1)) * 2, af[mi]);
#pragma unroll
            for (int s = 0; s < 2; ++s)
                ldsm4t(vBase + swz64(kk * 16 + s * 8 + (lane & 7), lane >> 3) * 2, bf[s]);
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                for (int ni = 0; ni < 4; ++ni)
#pragma unroll
                    for (int s = 0; s < 2; ++s)
                        mma16(acc[mi][ni], af[mi][2 * s], af[mi][2 * s + 1], bf[s][ni]);
        }
    }

    float* pw = p.partW + ((size_t)blockIdx.y * 9 + warp) * 32 * 128;
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int c = kc * 32 + ni * 8 + (lane & 3) * 2;
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const int o = mi * 16 + (lane >> 2) + hr * 8;
                *(F2*)(pw + o * 128 + c) = F2{acc[mi][ni][2 * hr], acc[mi][ni][2 * hr + 1]};
            }
        }
}


// ---------------------------------------------------------------------------------------------
// stem backward: norm0 (BN in training + ReLU) and 3x3 / 2 max-pool (padding 1)
// ---------------------------------------------------------------------------------------------
// The Triton `_stem_pool_bwd` takes 14 ms per step (two gather passes over the 4 candidate windows of
// every pixel, per-element masks and indices). Here:
// - stem_red: the BN reductions (sum of gm and of gm xhat over the input pixels) are sums over the
//   WINDOWS: the gm of a pixel is the sum of the dS of the windows of which it is the maximum (with
//   output > 0), and xhat is that of the pixel of the maximum, i.e. of YM (Y at the tap of the
//   maximum, saved in the forward). Only dS, S and YM are read, a quarter of the pixels;
// - stem_bwd: dY = a (gm - k1 - xhat k2) for every pixel, gm gathered from the windows covering it:
//   even rows and columns one, odd ones two (1, 2 or 4 windows), 8 channels per thread.
struct STB {
    const half_t* Y;            // [n*H*W][64]: conv0 output
    const half_t* dS;           // [M2][64]: gradient of the stem output (contiguous)
    const half_t* S;            // [M2][64]: stem output, relu of the maximum
    const half_t* YM;           // [M2][64]: Y at the tap of the maximum
    const unsigned char* AM;    // [M2][64]: the tap of the maximum (0..8, dy 3 + dx)
    const float* tab;           // [5][64]: a, mu, rs, k1 = R1 / M, k2 = R2 / M
    half_t* dY;                 // [n*H*W][64]
    float* part;                // [gridDim.x][128]: R1 (64), R2 (64)
    int M2, H, W, Ho, Wo;
};

DEV bool hpos(unsigned short v) { return v != 0 && !(v & 0x8000u); }
DEV unsigned short hsel(const U4& u, int k) {
    const unsigned w = k < 2 ? u.x : k < 4 ? u.y : k < 6 ? u.z : u.w;
    return (unsigned short)((k & 1) ? (w >> 16) : (w & 0xffffu));
}

extern "C" __global__ void __launch_bounds__(256) stem_red(STB p) {
    __shared__ float red[8][128];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = tid & 7, l = tid >> 3;           // 8 groups of 8 channels x 32 pixels
    float mu[8], rs[8], r1[8], r2[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        mu[k] = p.tab[64 + g * 8 + k];
        rs[k] = p.tab[128 + g * 8 + k];
        r1[k] = r2[k] = 0.0f;
    }
    for (int q = blockIdx.x * 32 + l; q < p.M2; q += gridDim.x * 32) {
        const U4 ds = *(const U4*)(p.dS + (size_t)q * 64 + g * 8);
        const U4 s = *(const U4*)(p.S + (size_t)q * 64 + g * 8);
        const U4 ym = *(const U4*)(p.YM + (size_t)q * 64 + g * 8);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (hpos(hsel(s, k))) {
                const float d = h2f(hsel(ds, k));
                r1[k] += d;
                r2[k] += d * ((h2f(hsel(ym, k)) - mu[k]) * rs[k]);
            }
        }
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        r1[k] += __shfl_xor_sync(0xffffffffu, r1[k], 8);
        r1[k] += __shfl_xor_sync(0xffffffffu, r1[k], 16);
        r2[k] += __shfl_xor_sync(0xffffffffu, r2[k], 8);
        r2[k] += __shfl_xor_sync(0xffffffffu, r2[k], 16);
    }
    if (lane < 8) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            red[warp][g * 8 + k] = r1[k];
            red[warp][64 + g * 8 + k] = r2[k];
        }
    }
    __syncthreads();
    if (tid < 128) {
        float t = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) t += red[w][tid];
        p.part[(size_t)blockIdx.x * 128 + tid] = t;
    }
}

extern "C" __global__ void __launch_bounds__(128) stem_bwd(STB p) {
    const int tid = threadIdx.x, g = tid & 7, wl = tid >> 3;   // 16 pixels x 8 channel groups
    const int row = blockIdx.y;                                 // n H + h
    const int n = row / p.H, h = row - n * p.H;
    const int w = blockIdx.x * 16 + wl;
    if (w >= p.W) return;
    int hos[2], dys[2], wos[2], dxs[2];
    int nh = 1, nw = 1;
    hos[0] = h >> 1;
    dys[0] = (h & 1) ? 2 : 1;
    hos[1] = (h >> 1) + 1;
    dys[1] = 0;
    if ((h & 1) && hos[1] < p.Ho) nh = 2;
    wos[0] = w >> 1;
    dxs[0] = (w & 1) ? 2 : 1;
    wos[1] = (w >> 1) + 1;
    dxs[1] = 0;
    if ((w & 1) && wos[1] < p.Wo) nw = 2;
    float gm[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) gm[k] = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            if (i < nh && j < nw) {
                const size_t q = ((size_t)n * p.Ho + hos[i]) * p.Wo + wos[j];
                const unsigned key = (unsigned)(dys[i] * 3 + dxs[j]);
                const uint2 am = *(const uint2*)(p.AM + q * 64 + g * 8);
                const U4 s = *(const U4*)(p.S + q * 64 + g * 8);
                const U4 ds = *(const U4*)(p.dS + q * 64 + g * 8);
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const unsigned ak = ((k < 4 ? am.x : am.y) >> ((k & 3) * 8)) & 0xffu;
                    if (ak == key && hpos(hsel(s, k))) gm[k] += h2f(hsel(ds, k));
                }
            }
        }
    const size_t pix = (size_t)row * p.W + w;
    const U4 y = *(const U4*)(p.Y + pix * 64 + g * 8);
    unsigned short o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int c = g * 8 + k;
        const float xh = (h2f(hsel(y, k)) - p.tab[64 + c]) * p.tab[128 + c];
        o[k] = f2h(p.tab[c] * (gm[k] - p.tab[192 + c] - xh * p.tab[256 + c]));
    }
    *(U4*)(p.dY + pix * 64 + g * 8) = U4{pack2(o[0], o[1]), pack2(o[2], o[3]), pack2(o[4], o[5]),
                                         pack2(o[6], o[7])};
}

// ---------------------------------------------------------------------------------------------
// 3x3 bottleneck conv, input gradient with the BN2 backward (c3d) and BN2 application (bn2_apply)
// ---------------------------------------------------------------------------------------------
// c3d: dV[p, c] = sum over taps and outputs of dS[p - (dy, dx), o] W[o, c, t] (implicit GEMM like
// c3f, k = 9 taps x 32 outputs: the whole dS halo fits in shared memory, no k slices) and, in the
// epilogue, the BN2 backward up to the reductions: gm = dV [U a + c > 0] (dV in fp32, never written),
// the per-channel sums of gm and of gm uhat (uhat = (U - mu) rs) and G = a gm in fp16. Replaces
// cuDNN's dgrad, the `_bn_bwd_reduce` pass and dV in memory.
// Program = slice of 32 channels (blockIdx.x) x blocks of 256 pixels in turn (persistent): the slice
// weights [9][32 c][32 o] are loaded once, the next block (dS halo and U of the slice) in flight in
// registers. 8 warps, each 32 pixels x 32 channels.
// bn2_apply: dU = G + alpha + beta U (the BN corrections, `fastdense.split_coeffs`).
struct C3D {
    const half_t* dS;           // [rows][32]: gradient of the conv output (contiguous)
    const half_t* Wd;           // [9][128][32]: Wd[t][c][o] = conv2.weight[o][c][t / 3][t % 3]
    const half_t* U;            // [rows][128]: norm2 input
    const float* tab;           // [4][128]: a, c, mu, rs of norm2
    half_t* G;                  // [rows][128]: a gm
    float* part;                // [split][256]: sum of gm (128), sum of gm uhat (128)
    int rows, H, W, split;
};

extern "C" __global__ void __launch_bounds__(256, 1) c3d(C3D p) {
    constexpr int BM = 256;
    constexpr int HIT = (4 * (BM + 2 * C3_WMAX + 2) + 255) / 256;
    constexpr int UIT = 4 * BM / 256;
    const int R = BM + 2 * p.W + 2;
    half_t* sW = (half_t*)smem;                    // [9][32 c][32 o]
    half_t* sD = sW + 9 * 32 * 32;                 // [R + 1][32 o], row R = zeros
    half_t* sU = sD + (R + 1) * 32;                // [BM][32 c]: U, then a gm
    float* sP = (float*)(sU + BM * 32);            // [4][32]: a, c, mu, rs of the slice
    unsigned short* sM = (unsigned short*)(sP + 128);   // [BM]

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int kc = blockIdx.x;
    const int ntiles = (p.rows + BM - 1) / BM;

    for (int q = tid; q < 9 * 32 * 4; q += 256) {
        const int t = q >> 7, c = (q >> 2) & 31, j = q & 3;
        *(U4*)(sW + swz64(t * 32 + c, j)) =
            *(const U4*)(p.Wd + ((size_t)t * 128 + kc * 32 + c) * 32 + j * 8);
    }
    if (tid < 128) sP[tid] = p.tab[(tid >> 5) * 128 + kc * 32 + (tid & 31)];
    if (tid < 4) *(U4*)(sD + R * 32 + tid * 8) = U4{0u, 0u, 0u, 0u};

    float r1[4][2], r2[4][2];
#pragma unroll
    for (int i = 0; i < 4; ++i) r1[i][0] = r1[i][1] = r2[i][0] = r2[i][1] = 0.0f;

    U4 rd[HIT], ru[UIT];
    auto gload = [&](int t) {
        const int r0 = t * BM - p.W - 1;
#pragma unroll
        for (int i = 0; i < HIT; ++i) {
            const int q = tid + i * 256, r = q >> 2, j = q & 3;
            const int g = r0 + r;
            rd[i] = (r < R && g >= 0 && g < p.rows)
                        ? *(const U4*)(p.dS + (size_t)g * 32 + j * 8) : U4{0u, 0u, 0u, 0u};
        }
#pragma unroll
        for (int i = 0; i < UIT; ++i) {
            const int q = tid + i * 256, r = q >> 2, j = q & 3;
            const int g = t * BM + r;
            ru[i] = g < p.rows ? *(const U4*)(p.U + (size_t)g * 128 + kc * 32 + j * 8)
                               : U4{0u, 0u, 0u, 0u};
        }
    };
    auto sstore = [&](int t) {
#pragma unroll
        for (int i = 0; i < HIT; ++i) {
            const int q = tid + i * 256, r = q >> 2, j = q & 3;
            if (r < R) *(U4*)(sD + swz64(r, j)) = rd[i];
        }
#pragma unroll
        for (int i = 0; i < UIT; ++i) {
            const int q = tid + i * 256, r = q >> 2, j = q & 3;
            *(U4*)(sU + swz64(r, j)) = ru[i];
        }
        sM[tid] = (unsigned short)tap_mask(t * BM + tid, p.rows, p.H, p.W);
    };

    const unsigned dBase = smem_addr(sD), wBase = smem_addr(sW);
    int t = blockIdx.y;
    if (t < ntiles) gload(t);
    for (; t < ntiles; t += p.split) {
        __syncthreads();                           // smem free (G of the previous block written)
        sstore(t);
        __syncthreads();
        if (t + p.split < ntiles) gload(t + p.split);

        float acc[2][4][4];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
        int arow[2];
        unsigned amsk[2];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            arow[mi] = warp * 32 + mi * 16 + (lane & 15);
            amsk[mi] = sM[arow[mi]];
        }
#pragma unroll
        for (int tp = 0; tp < 9; ++tp) {
            const int offB = (2 - tp / 3) * p.W + (2 - tp % 3);   // (1 - dy) W + (1 - dx)
            int hr[2];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) hr[mi] = ((amsk[mi] >> (8 - tp)) & 1u) ? arow[mi] + offB : R;
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                unsigned af[2][4];
#pragma unroll
                for (int mi = 0; mi < 2; ++mi)
                    ldsm4(dBase + swz64(hr[mi], kk * 2 + (lane >> 4)) * 2, af[mi]);
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    unsigned bf[4];
                    ldsm4(wBase + swz64(tp * 32 + lane, kk * 2 + s) * 2, bf);
#pragma unroll
                    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                        for (int ni = 0; ni < 4; ++ni)
                            mma16(acc[mi][ni], af[mi][2 * s], af[mi][2 * s + 1], bf[ni]);
                }
            }
        }

        // epilogue: gm, the BN sums and a gm in place of U in sU
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int c = ni * 8 + (lane & 3) * 2;
            const float a0 = sP[c], a1 = sP[c + 1], cc0 = sP[32 + c], cc1 = sP[32 + c + 1];
            const float mu0 = sP[64 + c], mu1 = sP[64 + c + 1], rs0 = sP[96 + c], rs1 = sP[96 + c + 1];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                for (int hr = 0; hr < 2; ++hr) {
                    const int r = warp * 32 + mi * 16 + (lane >> 2) + hr * 8;
                    const bool ok = t * BM + r < p.rows;
                    const int off = swz64(r, c >> 3) + (c & 7);
                    const unsigned uw = *(const unsigned*)(sU + off);
                    const float u0 = lo2f(uw), u1 = hi2f(uw);
                    const bool p0 = ok && fmaf(u0, a0, cc0) > 0.0f, p1 = ok && fmaf(u1, a1, cc1) > 0.0f;
                    const float gm0 = p0 ? acc[mi][ni][2 * hr] : 0.0f, gm1 = p1 ? acc[mi][ni][2 * hr + 1] : 0.0f;
                    r1[ni][0] += gm0;
                    r1[ni][1] += gm1;
                    r2[ni][0] += gm0 * ((u0 - mu0) * rs0);
                    r2[ni][1] += gm1 * ((u1 - mu1) * rs1);
                    *(unsigned*)(sU + off) = pack2(f2h(a0 * gm0), f2h(a1 * gm1));
                }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < UIT; ++i) {
            const int q = tid + i * 256, r = q >> 2, j = q & 3;
            const int g = t * BM + r;
            if (g < p.rows) *(U4*)(p.G + (size_t)g * 128 + kc * 32 + j * 8) = *(const U4*)(sU + swz64(r, j));
        }
    }

#pragma unroll
    for (int ni = 0; ni < 4; ++ni)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) {
                r1[ni][j] += __shfl_xor_sync(0xffffffffu, r1[ni][j], o);
                r2[ni][j] += __shfl_xor_sync(0xffffffffu, r2[ni][j], o);
            }
    __syncthreads();
    float* red = (float*)sU;                       // [8][64]
    if (lane < 4) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int c = ni * 8 + lane * 2 + j;
                red[warp * 64 + c] = r1[ni][j];
                red[warp * 64 + 32 + c] = r2[ni][j];
            }
    }
    __syncthreads();
    if (tid < 64) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) v += red[w * 64 + tid];
        const int which = tid >> 5, c = tid & 31;
        p.part[(size_t)blockIdx.y * 256 + which * 128 + kc * 32 + c] = v;
    }
}

struct BNA {
    const half_t* G;            // [rows][128]: a gm
    const half_t* U;            // [rows][128]
    const float* tab;           // [2][128]: alpha, beta
    half_t* dU;                 // [rows][128]
    int rows;
};

// dU = G + alpha + beta U, 8 channels per thread (16 bytes)
extern "C" __global__ void __launch_bounds__(256) bn2_apply(BNA p) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= p.rows * 16) return;
    const int c0 = (i & 15) * 8;
    const U4 g = ((const U4*)p.G)[i], u = ((const U4*)p.U)[i];
    const F4 al0 = *(const F4*)(p.tab + c0), al1 = *(const F4*)(p.tab + c0 + 4);
    const F4 be0 = *(const F4*)(p.tab + 128 + c0), be1 = *(const F4*)(p.tab + 128 + c0 + 4);
    const float al[8] = {al0.x, al0.y, al0.z, al0.w, al1.x, al1.y, al1.z, al1.w};
    const float be[8] = {be0.x, be0.y, be0.z, be0.w, be1.x, be1.y, be1.z, be1.w};
    const unsigned gw[4] = {g.x, g.y, g.z, g.w}, uw[4] = {u.x, u.y, u.z, u.w};
    unsigned o[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float d0 = lo2f(gw[k]) + al[2 * k] + be[2 * k] * lo2f(uw[k]);
        const float d1 = hi2f(gw[k]) + al[2 * k + 1] + be[2 * k + 1] * hi2f(uw[k]);
        o[k] = pack2(f2h(d0), f2h(d1));
    }
    ((U4*)p.dU)[i] = U4{o[0], o[1], o[2], o[3]};
}

// ---------------------------------------------------------------------------------------------
// the tiny launches fused (`CudaOps(fuse_small=True)`): fp16 weights once per step (wprep), per-column
// sums of the partials with the closing step (colsum), weight gradient of the 3x3 (wsum)
// ---------------------------------------------------------------------------------------------
// ~3,500 launches per step, ~23 ms of tiny kernels (casts and permutes of the weights at every layer,
// torch sums of the partials, the BN coefficients in 3-4 torch operations per BatchNorm) and as much
// CPU work.
//
// wprep: all conv weights (fp32, their strides) into an fp16 arena in the kernels' layouts: [o][c]
// (1x1), [t][o][c] (c3f) and [t][c][o] (c3d) for the 3x3. One global index per element; the job is
// found by bisection in the job table.
struct WJob {
    const float* src;           // [O][C][KH][KW] with strides s0..s3
    half_t* dst;
    long long s0, s1, s2, s3;
    long long start;            // first global index of the job
    int O, C, T, layout;        // layout 0: [o][c] (T = 1), 1: [t][o][c], 2: [t][c][o]
};

struct WP {
    const WJob* jobs;
    long long total;
    int njobs;
};

extern "C" __global__ void __launch_bounds__(256) wprep(WP p) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= p.total) return;
    int lo = 0, hi = p.njobs - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (p.jobs[mid].start <= i) lo = mid;
        else hi = mid - 1;
    }
    const WJob& J = p.jobs[lo];
    const long long e = i - J.start;
    int o, c, t;
    if (J.layout == 0) {
        o = (int)(e / J.C);
        c = (int)(e - (long long)o * J.C);
        t = 0;
    } else {
        const int oc = J.O * J.C;
        t = (int)(e / oc);
        const int r = (int)(e - (long long)t * oc);
        if (J.layout == 1) {
            o = r / J.C;
            c = r - o * J.C;
        } else {
            c = r / J.O;
            o = r - c * J.O;
        }
    }
    const int kh = t / 3, kw = t - kh * 3;
    J.dst[e] = f2h(J.src[o * J.s0 + c * J.s1 + kh * J.s2 + kw * J.s3]);
}

// colsum: per-column sums of two column ranges [lo1, lo1 + C) and [lo2, lo2 + C) of the R rows of
// part, in row slices (blockIdx.y), and the closing step in the block that finishes last (fixed
// summation order: deterministic). Closing modes:
//   0: training BatchNorm from the sums shifted by sh -> tab [4][C] (a, c, mean, rstd) and the
//      running averages (as F.batch_norm: corrected variance, momentum, num_batches_tracked);
//   1: the sums into out1[0:C], out2[0:C] (segment statistics into those of the block);
//   2: BN2 backward: db = s1 into out1, dg = s2 into out2, and alpha, beta (`split_coeffs`) into
//      ab [2][C] from the norm2 tab.
struct CS {
    const float* part;
    const float* part2;         // if not 0, the second range starts here (s1 and s2 separate tensors)
    float* scr;                 // [gridDim.y][2][C]
    unsigned* cnt;              // [gridDim.x], zero between launches
    const float* sh;
    const float* g;
    const float* b;
    float* rm;
    float* rv;
    long long* nbt;
    float* tab;                 // mode 0: output; mode 2: input
    float* out1;
    float* out2;
    float* ab;
    int R, ld, lo1, lo2, C, mode, upd_nbt;
    float inv_n, n_corr, eps, mom;
};

extern "C" __global__ void __launch_bounds__(128) colsum(CS p) {
    __shared__ int last;
    const int c = blockIdx.x * 128 + threadIdx.x;
    const bool ok = c < p.C;
    float a1 = 0.0f, a2 = 0.0f;
    if (ok) {
        for (int r = blockIdx.y; r < p.R; r += gridDim.y) {
            a1 += p.part[(size_t)r * p.ld + p.lo1 + c];
            a2 += (p.part2 ? p.part2 : p.part)[(size_t)r * p.ld + p.lo2 + c];
        }
        p.scr[((size_t)blockIdx.y * 2) * p.C + c] = a1;
        p.scr[((size_t)blockIdx.y * 2 + 1) * p.C + c] = a2;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(p.cnt + blockIdx.x, 1u) == gridDim.y - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (ok) {
        float s1 = 0.0f, s2 = 0.0f;
        for (int k = 0; k < (int)gridDim.y; ++k) {
            s1 += __ldcg(p.scr + ((size_t)k * 2) * p.C + c);
            s2 += __ldcg(p.scr + ((size_t)k * 2 + 1) * p.C + c);
        }
        if (p.mode == 0) {
            const float sh = p.sh[c];               // before rm: sh may be rm
            const float m1 = s1 * p.inv_n;
            const float var = fmaxf(s2 * p.inv_n - m1 * m1, 0.0f);
            const float mu = sh + m1;
            const float rs = 1.0f / sqrtf(var + p.eps);
            const float a = p.g[c] * rs;
            p.tab[c] = a;
            p.tab[p.C + c] = p.b[c] - mu * a;
            p.tab[2 * p.C + c] = mu;
            p.tab[3 * p.C + c] = rs;
            p.rm[c] = p.rm[c] * (1.0f - p.mom) + mu * p.mom;
            p.rv[c] = p.rv[c] * (1.0f - p.mom) + var * p.n_corr * p.mom;
            if (p.upd_nbt && c == 0) p.nbt[0] += 1;
        } else if (p.mode == 1) {
            p.out1[c] = s1;
            p.out2[c] = s2;
        } else {
            p.out1[c] = s1;
            p.out2[c] = s2;
            const float a = p.tab[c], mu = p.tab[2 * p.C + c], rs = p.tab[3 * p.C + c];
            const float beta = -a * rs * s2 * p.inv_n;
            p.ab[c] = -a * s1 * p.inv_n - beta * mu;
            p.ab[p.C + c] = beta;
        }
    }
    if (threadIdx.x == 0) p.cnt[blockIdx.x] = 0u;
}

// wsum: dW[o][c][t] = sum_s partW[s][t][o][c] (c3w, 32 x 128 x 9 in fp32)
struct WS {
    const float* partW;         // [S][9][32][128]
    float* dW;                  // [32][128][9]
    int S;
};

extern "C" __global__ void __launch_bounds__(256) wsum(WS p) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= 32 * 128 * 9) return;
    const int o = i / 1152, r = i - o * 1152, c = r / 9, t = r - c * 9;
    float s = 0.0f;
    for (int k = 0; k < p.S; ++k) s += p.partW[(((size_t)k * 9 + t) * 32 + o) * 128 + c];
    p.dW[i] = s;
}

// ---------------------------------------------------------------------------------------------
// persistent conv1 forward (c1fp)
// ---------------------------------------------------------------------------------------------
// c1f (one program per block of rows) in block 1: 3.4 ms against ~0.6 of tensor core and ~1.1 of DRAM
// time: 5 k slices per program, the initial load and the epilogue of every program exposed. Here one
// program per SM (x 2) takes the row blocks in turn (blockIdx.x, + gridDim.x, ...): during the last
// slice of a block the first slice of the next one is already arriving (in registers), and the
// epilogue (U to memory, sums) covers it. The per-column sums are accumulated in the program (thread
// tid = column tid of the 256): one partial per program. Same arithmetic as c1f.
template <int BM>
DEV void c1fp_body(const C1F& p) {
    constexpr int WARPS_M = BM / 32;
    constexpr int WARPS_N = 8 / WARPS_M;
    constexpr int WN = 128 / WARPS_N;
    constexpr int NT = WN / 8;
    constexpr int A_H = BM * 32, B_H = 128 * 32;
    constexpr int ACH = BM / 64;
    half_t* sA = (half_t*)smem;
    half_t* sB = sA + 2 * A_H;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % WARPS_M, wn = warp / WARPS_M;
    const int KT = p.cin >> 5;
    const int ntiles = (p.rows + BM - 1) / BM;

    U4 ra[ACH], rb[2];
    auto gload = [&](int t, int kt) {
        const int k0 = kt * 32, m0 = t * BM;
#pragma unroll
        for (int i = 0; i < ACH; ++i) {
            const int q = tid + i * 256, r = q >> 2, ch = q & 3;
            const int m = m0 + r;
            ra[i] = m < p.rows ? *(const U4*)(p.X + (size_t)m * p.sx + k0 + ch * 8) : U4{0u, 0u, 0u, 0u};
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int q = tid + i * 256, n = q >> 2, ch = q & 3;
            rb[i] = *(const U4*)(p.W + (size_t)n * p.cin + k0 + ch * 8);
        }
    };
    auto sstore = [&](int buf, int t, int kt) {
        const int k0 = kt * 32, m0 = t * BM;
#pragma unroll
        for (int i = 0; i < ACH; ++i) {
            const int q = tid + i * 256, r = q >> 2, ch = q & 3;
            const int kc = k0 + ch * 8;
            const F4 a0 = *(const F4*)(p.A + kc), a1 = *(const F4*)(p.A + kc + 4);
            const F4 c0 = *(const F4*)(p.C + kc), c1 = *(const F4*)(p.C + kc + 4);
            U4 v = U4{pro2(ra[i].x, a0.x, c0.x, a0.y, c0.y), pro2(ra[i].y, a0.z, c0.z, a0.w, c0.w),
                      pro2(ra[i].z, a1.x, c1.x, a1.y, c1.y), pro2(ra[i].w, a1.z, c1.z, a1.w, c1.w)};
            if (m0 + r >= p.rows) v = U4{0u, 0u, 0u, 0u};
            *(U4*)(sA + buf * A_H + r * 32 + ((ch ^ ((r >> 1) & 3)) << 3)) = v;
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int q = tid + i * 256, n = q >> 2, ch = q & 3;
            *(U4*)(sB + buf * B_H + n * 32 + ((ch ^ ((n >> 1) & 3)) << 3)) = rb[i];
        }
    };

    float acc[2][NT][4];
    auto compute = [&](int buf) {
        const unsigned aBase = smem_addr(sA + buf * A_H), bBase = smem_addr(sB + buf * B_H);
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            unsigned af[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int r = wm * 32 + mi * 16 + (lane & 15);
                const int ch = kk * 2 + (lane >> 4);
                ldsm4(aBase + (r * 32 + ((ch ^ ((r >> 1) & 3)) << 3)) * 2, af[mi]);
            }
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                unsigned bf[NT];
                const int ch = kk * 2 + s;
#pragma unroll
                for (int j = 0; j < NT / 4; ++j) {
                    const int n = wn * WN + j * 32 + lane;
                    ldsm4(bBase + (n * 32 + ((ch ^ ((n >> 1) & 3)) << 3)) * 2, bf + 4 * j);
                }
#pragma unroll
                for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni)
                        mma16(acc[mi][ni], af[mi][2 * s], af[mi][2 * s + 1], bf[ni]);
            }
        }
    };

    float colacc = 0.0f;                           // column tid (s1 | s2) over the program's blocks
    int t = blockIdx.x;
    if (t < ntiles) gload(t, 0);
    for (; t < ntiles; t += gridDim.x) {
        const int m0 = t * BM;
        sstore(0, t, 0);
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
        for (int kt = 0; kt < KT; ++kt) {
            const bool last = kt + 1 == KT;
            if (!last) gload(t, kt + 1);
            else if (t + (int)gridDim.x < ntiles) gload(t + gridDim.x, 0);   // the next block
            compute(kt & 1);
            if (!last) sstore((kt + 1) & 1, t, kt + 1);
            __syncthreads();
        }

        // epilogue as in c1f
        half_t* sO = (half_t*)smem;
        float s1[NT][2], s2[NT][2];
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int c = wn * WN + ni * 8 + (lane & 3) * 2;
            const float h0 = p.sh[c], h1 = p.sh[c + 1];
            s1[ni][0] = s1[ni][1] = s2[ni][0] = s2[ni][1] = 0.0f;
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                for (int hr = 0; hr < 2; ++hr) {
                    const int r = wm * 32 + mi * 16 + (lane >> 2) + hr * 8;
                    const unsigned short u0 = f2h(acc[mi][ni][2 * hr]), u1 = f2h(acc[mi][ni][2 * hr + 1]);
                    *(unsigned*)(sO + r * 128 + (((c >> 3) ^ (r & 7)) << 3) + (c & 7)) = pack2(u0, u1);
                    if (m0 + r < p.rows) {
                        const float d0 = h2f(u0) - h0, d1 = h2f(u1) - h1;
                        s1[ni][0] += d0;
                        s1[ni][1] += d1;
                        s2[ni][0] += d0 * d0;
                        s2[ni][1] += d1 * d1;
                    }
                }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
#pragma unroll
                for (int o = 4; o < 32; o <<= 1) {
                    s1[ni][j] += __shfl_xor_sync(0xffffffffu, s1[ni][j], o);
                    s2[ni][j] += __shfl_xor_sync(0xffffffffu, s2[ni][j], o);
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < BM * 16 / 256; ++i) {
            const int q = tid + i * 256, r = q >> 4, ch = q & 15;
            if (m0 + r < p.rows)
                *(U4*)(p.U + (size_t)(m0 + r) * 128 + ch * 8) = *(const U4*)(sO + r * 128 + ((ch ^ (r & 7)) << 3));
        }
        __syncthreads();
        float* red = (float*)smem;                 // [WARPS_M][256]
        if (lane < 4) {
#pragma unroll
            for (int ni = 0; ni < NT; ++ni)
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int c = wn * WN + ni * 8 + lane * 2 + j;
                    red[wm * 256 + c] = s1[ni][j];
                    red[wm * 256 + 128 + c] = s2[ni][j];
                }
        }
        __syncthreads();
#pragma unroll
        for (int w = 0; w < WARPS_M; ++w) colacc += red[w * 256 + tid];
        __syncthreads();                           // smem free for the next block
    }
    p.part[(size_t)blockIdx.x * 256 + tid] = colacc;
}

extern "C" __global__ void __launch_bounds__(256, 2) c1fp_128(C1F p) { c1fp_body<128>(p); }
extern "C" __global__ void __launch_bounds__(256, 2) c1fp_64(C1F p) { c1fp_body<64>(p); }
extern "C" __global__ void __launch_bounds__(256, 1) c1fp1_128(C1F p) { c1fp_body<128>(p); }

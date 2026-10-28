"""float64 -> uint8 normalisation and histogram-based order statistics on the GPU.

No torch/Triton/NVRTC import at module level, above the fork of the workers: they are loaded only
when a CUDA tensor arrives. Contents:
1. `normalize(flat, lo, scale, bad, out=None)`: a Triton (or NVRTC) pass that applies
   `(x - lo) * scale` in IEEE float64 without FMA, clamps to [0, 255] and casts to uint8.
2. `hist_to_uint8(x, k1, k2, gamma, out=None)`: exact selection of the order statistics
   `k1, k1+1, k2, k2+1` through a 3-level radix histogram (12 + 10 + 10 bits) + `_lerp` +
   float64 -> uint8 normalisation, in 4 NVRTC CUDA kernels over all stacks of the study at once,
   without allocating intermediate tensors in DRAM (3 reads of `x` + 1 `uint8` write instead of
   ~27 passes and ~70 PyTorch launches per study).
3. `gather_u8_f16(v_u8, slice_idx, out)`: gathers the slices of the window budget and converts
   `uint8 -> float16 * (1/255)` in a single kernel through a shared 256-entry LUT.
"""
from __future__ import annotations

import ctypes
import importlib
import logging
import threading
import time


_LOG = logging.getLogger(__name__)
_LOCK = threading.Lock()
_BLOCK = 256
_STATE = {"tried": False, "kernel": None, "torch": None, "triton": None, "error": None}
_DEVICES: dict = {}
_NVRTC_DEVICES: dict = {}


_NVRTC_SRC = r"""
struct alignas(16) F4 { float x, y, z, w; };
struct alignas(8) U2 { unsigned int x, y; };

#define MAX_R 16
#define MAX_CAND 131072
#define OFF_HIST1 0
#define OFF_HIST2 65536
#define OFF_MIN_U 98304
#define OFF_MAX_U 131072
#define OFF_MIN_NXT 163840
#define OFF_CAND_CNT 163872
#define OFF_HAS_NAN 163904
#define OFF_META1 163920
#define OFF_PREV_NXT 164048
#define OFF_CAND 164352
#define WS_WORDS (OFF_CAND + MAX_R * 2 * MAX_CAND)

__device__ __forceinline__ unsigned int order_key(float x) {
    int b = __float_as_int(x);
    int k = (b < 0) ? (b ^ 0x7fffffff) : b;
    return (unsigned int)k ^ 0x80000000u;
}

__device__ __forceinline__ float key_to_float(unsigned int u) {
    int k = (int)(u ^ 0x80000000u);
    int b = (k < 0) ? (k ^ 0x7fffffff) : k;
    return __int_as_float(b);
}

__device__ __forceinline__ bool is_nan_f(float x) {
    return (__float_as_uint(x) & 0x7fffffffu) > 0x7f800000u;
}

__device__ __forceinline__ bool is_finite_d(double x) {
    return (__double2hiint(x) & 0x7ff00000) != 0x7ff00000;
}

__device__ __forceinline__ unsigned int block_prefix_sum(unsigned int tsum, unsigned int* s_warp) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    unsigned int wsum = tsum;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        unsigned int up = __shfl_up_sync(0xffffffffu, wsum, d);
        if (lane >= d) wsum += up;
    }
    if (lane == 31) s_warp[warp] = wsum;
    __syncthreads();
    if (warp == 0) {
        unsigned int bsum = (lane < 8) ? s_warp[lane] : 0u;
#pragma unroll
        for (int d = 1; d < 8; d <<= 1) {
            unsigned int up = __shfl_up_sync(0xffffffffu, bsum, d);
            if (lane >= d) bsum += up;
        }
        if (lane < 8) s_warp[lane] = bsum - s_warp[lane];
    }
    __syncthreads();
    return s_warp[warp] + (wsum - tsum);
}

extern "C" __global__ void __launch_bounds__(256) hist_init(int R, unsigned int* __restrict__ ws) {
    const int r = blockIdx.x, tid = threadIdx.x;
    if (r >= R) return;
    unsigned int* h1 = ws + OFF_HIST1 + r * 4096;
    for (int i = tid; i < 4096; i += 256) h1[i] = 0u;
    unsigned int* h2 = ws + OFF_HIST2 + r * 2048;
    unsigned int* mnu = ws + OFF_MIN_U + r * 2048;
    unsigned int* mxu = ws + OFF_MAX_U + r * 2048;
    for (int i = tid; i < 2048; i += 256) {
        h2[i] = 0u;
        mnu[i] = 0xffffffffu;
        mxu[i] = 0u;
    }
    if (tid < 2) {
        (ws + OFF_MIN_NXT)[r * 2 + tid] = 0xffffffffu;
        (ws + OFF_CAND_CNT)[r * 2 + tid] = 0u;
    }
    if (tid == 0) (ws + OFF_HAS_NAN)[r] = 0u;
}

extern "C" __global__ void __launch_bounds__(256) hist_pass1(
    const float* __restrict__ flat, int n, unsigned int* __restrict__ ws)
{
    __shared__ unsigned int s_hist[4096];
    __shared__ unsigned int s_nan;
    const int r = blockIdx.y, tid = threadIdx.x, lane = tid & 31;
    for (int i = tid; i < 4096; i += 256) s_hist[i] = 0u;
    if (tid == 0) s_nan = 0u;
    __syncthreads();

    const float* row = flat + (unsigned long long)r * n;
    unsigned int my_nan = 0u;
    auto add_b1 = [&](float x) {
        if (is_nan_f(x)) my_nan = 1u;
        unsigned int b1 = order_key(x) >> 20;
        unsigned int peers = __match_any_sync(0xffffffffu, b1);
        if (lane == __ffs(peers) - 1) atomicAdd(&s_hist[b1], __popc(peers));
    };

    const int n4 = (n % 4 == 0) ? (n >> 2) : 0;
    const int step4 = gridDim.x * 256;
    for (int base = blockIdx.x * 256; base < n4; base += step4) {
        int i4 = base + tid;
        if (base + 256 <= n4) {
            F4 v = ((const F4*)row)[i4];
            add_b1(v.x);
            add_b1(v.y);
            add_b1(v.z);
            add_b1(v.w);
        } else if (i4 < n4) {
            F4 v = ((const F4*)row)[i4];
            if (is_nan_f(v.x) || is_nan_f(v.y) || is_nan_f(v.z) || is_nan_f(v.w)) my_nan = 1u;
            atomicAdd(&s_hist[order_key(v.x) >> 20], 1u);
            atomicAdd(&s_hist[order_key(v.y) >> 20], 1u);
            atomicAdd(&s_hist[order_key(v.z) >> 20], 1u);
            atomicAdd(&s_hist[order_key(v.w) >> 20], 1u);
        }
    }
    for (int i = (n4 << 2) + blockIdx.x * 256 + tid; i < n; i += step4) {
        float x = row[i];
        if (is_nan_f(x)) my_nan = 1u;
        atomicAdd(&s_hist[order_key(x) >> 20], 1u);
    }
    if (my_nan) s_nan = 1u;
    __syncthreads();

    unsigned int* g_hist1 = ws + OFF_HIST1 + r * 4096;
    for (int i = tid; i < 4096; i += 256) {
        unsigned int c = s_hist[i];
        if (c) atomicAdd(&g_hist1[i], c);
    }
    if (tid == 0 && s_nan) (ws + OFF_HAS_NAN)[r] = 1u;
}

extern "C" __global__ void __launch_bounds__(256) hist_pass2(
    const float* __restrict__ flat, int n, int k1, int k2, unsigned int* __restrict__ ws)
{
    __shared__ unsigned int s_hist2[2][1024];
    __shared__ unsigned int s_min_u[2][1024];
    __shared__ unsigned int s_max_u[2][1024];
    __shared__ unsigned int s_min_nxt[2];
    __shared__ unsigned int s_b1[2], s_r1[2], s_c1[2], s_b1_nxt[2];
    __shared__ unsigned int s_warp[8];

    const int r = blockIdx.y, tid = threadIdx.x, lane = tid & 31;
    for (int i = tid; i < 1024; i += 256) {
        s_hist2[0][i] = 0u;
        s_hist2[1][i] = 0u;
        s_min_u[0][i] = 0xffffffffu;
        s_min_u[1][i] = 0xffffffffu;
        s_max_u[0][i] = 0u;
        s_max_u[1][i] = 0u;
    }
    if (tid < 2) s_min_nxt[tid] = 0xffffffffu;

    const unsigned int* g_hist1 = ws + OFF_HIST1 + r * 4096;
    unsigned int vals[16];
    unsigned int tsum = 0u;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        vals[j] = g_hist1[tid * 16 + j];
        tsum += vals[j];
    }
    unsigned int cur = block_prefix_sum(tsum, s_warp);
    const unsigned int uk1 = (unsigned int)k1, uk1_n = uk1 + 1u;
    const unsigned int uk2 = (unsigned int)k2, uk2_n = uk2 + 1u;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        unsigned int b = tid * 16 + j;
        unsigned int c = vals[j];
        unsigned int nxt = cur + c;
        if (cur <= uk1 && uk1 < nxt) { s_b1[0] = b; s_r1[0] = uk1 - cur; s_c1[0] = c; }
        if (cur <= uk1_n && uk1_n < nxt) { s_b1_nxt[0] = b; }
        if (cur <= uk2 && uk2 < nxt) { s_b1[1] = b; s_r1[1] = uk2 - cur; s_c1[1] = c; }
        if (cur <= uk2_n && uk2_n < nxt) { s_b1_nxt[1] = b; }
        cur = nxt;
    }
    __syncthreads();

    if (blockIdx.x == 0 && tid < 2) {
        unsigned int* m1 = ws + OFF_META1 + (r * 2 + tid) * 4;
        m1[0] = s_b1[tid];
        m1[1] = s_r1[tid];
        m1[2] = s_c1[tid];
        m1[3] = s_b1_nxt[tid];
    }

    const unsigned int b1_0 = s_b1[0], c1_0 = s_c1[0], bn_0 = s_b1_nxt[0];
    const unsigned int b1_1 = s_b1[1], c1_1 = s_c1[1], bn_1 = s_b1_nxt[1];
    const bool same_b1 = (b1_1 == b1_0);
    unsigned int* g_cand_cnt = ws + OFF_CAND_CNT;
    unsigned int* g_cand = ws + OFF_CAND;
    const float* row = flat + (unsigned long long)r * n;

    auto proc = [&](unsigned int u, bool valid) {
        unsigned int top12 = valid ? (u >> 20) : 0xffffffffu;
        unsigned int m0 = __ballot_sync(0xffffffffu, top12 == b1_0);
        if (m0 && top12 == b1_0) {
            unsigned int mid10 = (u >> 10) & 1023u;
            unsigned int mp = __match_any_sync(m0, mid10);
            if (lane == __ffs(mp) - 1) atomicAdd(&s_hist2[0][mid10], __popc(mp));
            unsigned int up = __match_any_sync(m0, u);
            if (lane == __ffs(up) - 1) {
                atomicMin(&s_min_u[0][mid10], u);
                atomicMax(&s_max_u[0][mid10], u);
            }
            if (c1_0 <= MAX_CAND) {
                int first = __ffs(m0) - 1;
                unsigned int base_pos = 0u;
                if (lane == first) base_pos = atomicAdd(&g_cand_cnt[r * 2 + 0], __popc(m0));
                base_pos = __shfl_sync(m0, base_pos, first);
                unsigned int my_off = __popc(m0 & ((1u << lane) - 1u));
                if (base_pos + my_off < MAX_CAND)
                    g_cand[(unsigned long long)(r * 2 + 0) * MAX_CAND + base_pos + my_off] = u;
            }
        }
        if (bn_0 != b1_0 && top12 == bn_0) atomicMin(&s_min_nxt[0], u);
        if (!same_b1) {
            unsigned int m1 = __ballot_sync(0xffffffffu, top12 == b1_1);
            if (m1 && top12 == b1_1) {
                unsigned int mid10 = (u >> 10) & 1023u;
                unsigned int mp = __match_any_sync(m1, mid10);
                if (lane == __ffs(mp) - 1) atomicAdd(&s_hist2[1][mid10], __popc(mp));
                unsigned int up = __match_any_sync(m1, u);
                if (lane == __ffs(up) - 1) {
                    atomicMin(&s_min_u[1][mid10], u);
                    atomicMax(&s_max_u[1][mid10], u);
                }
                if (c1_1 <= MAX_CAND) {
                    int first = __ffs(m1) - 1;
                    unsigned int base_pos = 0u;
                    if (lane == first) base_pos = atomicAdd(&g_cand_cnt[r * 2 + 1], __popc(m1));
                    base_pos = __shfl_sync(m1, base_pos, first);
                    unsigned int my_off = __popc(m1 & ((1u << lane) - 1u));
                    if (base_pos + my_off < MAX_CAND)
                        g_cand[(unsigned long long)(r * 2 + 1) * MAX_CAND + base_pos + my_off] = u;
                }
            }
        }
        if (bn_1 != b1_1 && top12 == bn_1) atomicMin(&s_min_nxt[1], u);
    };

    const int n4 = (n % 4 == 0) ? (n >> 2) : 0;
    const int step4 = gridDim.x * 256;
    for (int base = blockIdx.x * 256; base < n4; base += step4) {
        int i4 = base + tid;
        bool ok = i4 < n4;
        F4 v = ok ? ((const F4*)row)[i4] : F4{0.f, 0.f, 0.f, 0.f};
        proc(order_key(v.x), ok);
        proc(order_key(v.y), ok);
        proc(order_key(v.z), ok);
        proc(order_key(v.w), ok);
    }
    for (int base = (n4 << 2) + blockIdx.x * 256; base < n; base += step4) {
        int i = base + tid;
        bool ok = i < n;
        proc(ok ? order_key(row[i]) : 0u, ok);
    }
    __syncthreads();

    unsigned int* g_hist2 = ws + OFF_HIST2 + r * 2048;
    unsigned int* g_min_u = ws + OFF_MIN_U + r * 2048;
    unsigned int* g_max_u = ws + OFF_MAX_U + r * 2048;
    for (int i = tid; i < 1024; i += 256) {
        unsigned int c0 = s_hist2[0][i];
        if (c0) {
            atomicAdd(&g_hist2[i], c0);
            atomicMin(&g_min_u[i], s_min_u[0][i]);
            atomicMax(&g_max_u[i], s_max_u[0][i]);
            if (same_b1) {
                atomicAdd(&g_hist2[1024 + i], c0);
                atomicMin(&g_min_u[1024 + i], s_min_u[0][i]);
                atomicMax(&g_max_u[1024 + i], s_max_u[0][i]);
            }
        }
        if (!same_b1) {
            unsigned int c1 = s_hist2[1][i];
            if (c1) {
                atomicAdd(&g_hist2[1024 + i], c1);
                atomicMin(&g_min_u[1024 + i], s_min_u[1][i]);
                atomicMax(&g_max_u[1024 + i], s_max_u[1][i]);
            }
        }
    }
    if (tid < 2 && s_min_nxt[tid] != 0xffffffffu) {
        atomicMin(&(ws + OFF_MIN_NXT)[r * 2 + tid], s_min_nxt[tid]);
    }
}

extern "C" __global__ void __launch_bounds__(256) hist_pass3(
    const float* __restrict__ flat, int n, unsigned int* __restrict__ ws)
{
    __shared__ unsigned int s_hist3[1024];
    __shared__ unsigned int s_warp[8];
    __shared__ unsigned int s_b2, s_r2, s_c2, s_b2_nxt;
    __shared__ unsigned int s_u_prev, s_u_nxt;

    const int q = blockIdx.x, r = blockIdx.y, tid = threadIdx.x;
    const unsigned int* m1 = ws + OFF_META1 + (r * 2 + q) * 4;
    const unsigned int b1 = m1[0], r1 = m1[1], c1 = m1[2], b1_nxt = m1[3];
    const unsigned int b1_0 = (ws + OFF_META1 + (r * 2 + 0) * 4)[0];

    const unsigned int* g_hist2 = ws + OFF_HIST2 + (r * 2 + q) * 1024;
    unsigned int vals[4];
    unsigned int tsum = 0u;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        vals[j] = g_hist2[tid * 4 + j];
        tsum += vals[j];
    }
    unsigned int cur = block_prefix_sum(tsum, s_warp);
    const unsigned int r1_n = r1 + 1u;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        unsigned int b = tid * 4 + j;
        unsigned int c = vals[j];
        unsigned int nxt = cur + c;
        if (cur <= r1 && r1 < nxt) { s_b2 = b; s_r2 = r1 - cur; s_c2 = c; }
        if (b1_nxt == b1 && cur <= r1_n && r1_n < nxt) { s_b2_nxt = b; }
        cur = nxt;
    }
    __syncthreads();

    const unsigned int b2 = s_b2, r2 = s_r2, c2 = s_c2, b2_nxt = s_b2_nxt;
    const unsigned int* g_min_u = ws + OFF_MIN_U + (r * 2 + q) * 1024;
    const unsigned int* g_max_u = ws + OFF_MAX_U + (r * 2 + q) * 1024;
    const unsigned int u_min = g_min_u[b2], u_max = g_max_u[b2];

    const bool need_prev = !(u_min == u_max || r2 == 0u || r2 + 1u == c2);
    const bool need_nxt = (b1_nxt == b1 && b2_nxt == b2) && !(u_min == u_max || r2 + 2u == c2);

    if (tid == 0) {
        if (!need_prev) s_u_prev = (r2 == 0u || u_min == u_max) ? u_min : u_max;
        if (!need_nxt) {
            if (b1_nxt != b1) s_u_nxt = (ws + OFF_MIN_NXT)[r * 2 + q];
            else if (b2_nxt != b2) s_u_nxt = g_min_u[b2_nxt];
            else s_u_nxt = u_max;
        }
    }

    if (need_prev || need_nxt) {
        for (int i = tid; i < 1024; i += 256) s_hist3[i] = 0u;
        __syncthreads();
        const unsigned int target22 = (b1 << 10) | b2;
        if (c1 <= MAX_CAND) {
            int cq = (q == 1 && b1 == b1_0) ? 0 : q;
            const unsigned int* cand = ws + OFF_CAND + (unsigned long long)(r * 2 + cq) * MAX_CAND;
            for (int i = tid; i < (int)c1; i += 256) {
                unsigned int u = cand[i];
                if ((u >> 10) == target22) atomicAdd(&s_hist3[u & 1023u], 1u);
            }
        } else {
            const float* row = flat + (unsigned long long)r * n;
            for (int i = tid; i < n; i += 256) {
                unsigned int u = order_key(row[i]);
                if ((u >> 10) == target22) atomicAdd(&s_hist3[u & 1023u], 1u);
            }
        }
        __syncthreads();
        unsigned int v3[4];
        unsigned int ts3 = 0u;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v3[j] = s_hist3[tid * 4 + j];
            ts3 += v3[j];
        }
        unsigned int cur3 = block_prefix_sum(ts3, s_warp);
        const unsigned int r2_n = r2 + 1u;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned int b3 = tid * 4 + j;
            unsigned int c3 = v3[j];
            unsigned int nxt3 = cur3 + c3;
            if (need_prev && cur3 <= r2 && r2 < nxt3) s_u_prev = (target22 << 10) | b3;
            if (need_nxt && cur3 <= r2_n && r2_n < nxt3) s_u_nxt = (target22 << 10) | b3;
            cur3 = nxt3;
        }
    }
    __syncthreads();
    if (tid == 0) {
        unsigned int* pn = ws + OFF_PREV_NXT + r * 4 + q * 2;
        pn[0] = s_u_prev;
        pn[1] = s_u_nxt;
    }
}

__device__ __forceinline__ unsigned int norm1_u8(float xf, double lo, double scale) {
    double y = __dmul_rn(__dsub_rn((double)xf, lo), scale);
    if (!(y == y) || y <= 0.0) return 0u;
    if (y >= 255.0) return 255u;
    return (unsigned int)(int)y;
}

extern "C" __global__ void __launch_bounds__(256) hist_pass4_norm(
    const float* __restrict__ flat, int n, double t0, double t1, double omt0, double omt1,
    const unsigned int* __restrict__ ws, unsigned char* __restrict__ out)
{
    __shared__ double s_lo, s_scale;
    __shared__ unsigned int s_bad;

    const int r = blockIdx.y, tid = threadIdx.x;
    if (tid == 0) {
        const unsigned int* pn = ws + OFF_PREV_NXT + r * 4;
        float p0 = key_to_float(pn[0]), n0 = key_to_float(pn[1]);
        float p1 = key_to_float(pn[2]), n1 = key_to_float(pn[3]);
        double d0 = (double)__fsub_rn(n0, p0);
        double d1 = (double)__fsub_rn(n1, p1);
        double lo = (t0 >= 0.5) ? __dsub_rn((double)n0, __dmul_rn(d0, omt0))
                                : __dadd_rn((double)p0, __dmul_rn(d0, t0));
        double hi = (t1 >= 0.5) ? __dsub_rn((double)n1, __dmul_rn(d1, omt1))
                                : __dadd_rn((double)p1, __dmul_rn(d1, t1));
        bool bad = ((ws + OFF_HAS_NAN)[r] != 0u) || !is_finite_d(lo) || !is_finite_d(hi) || (hi <= lo);
        s_bad = bad ? 1u : 0u;
        s_lo = lo;
        s_scale = bad ? 0.0 : __ddiv_rn(255.0, __dsub_rn(hi, lo));
    }
    __syncthreads();

    const float* row = flat + (unsigned long long)r * n;
    unsigned char* dst = out + (unsigned long long)r * n;
    const int n4 = (n % 4 == 0) ? (n >> 2) : 0;
    const int step4 = gridDim.x * 256;

    if (s_bad) {
        for (int i4 = blockIdx.x * 256 + tid; i4 < n4; i4 += step4) {
            ((unsigned int*)dst)[i4] = 0u;
        }
        for (int i = (n4 << 2) + blockIdx.x * 256 + tid; i < n; i += step4) {
            dst[i] = 0;
        }
        return;
    }

    const double lo = s_lo, scale = s_scale;
    for (int i4 = blockIdx.x * 256 + tid; i4 < n4; i4 += step4) {
        F4 v = ((const F4*)row)[i4];
        unsigned int b0 = norm1_u8(v.x, lo, scale);
        unsigned int b1 = norm1_u8(v.y, lo, scale);
        unsigned int b2 = norm1_u8(v.z, lo, scale);
        unsigned int b3 = norm1_u8(v.w, lo, scale);
        ((unsigned int*)dst)[i4] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
    for (int i = (n4 << 2) + blockIdx.x * 256 + tid; i < n; i += step4) {
        dst[i] = (unsigned char)norm1_u8(row[i], lo, scale);
    }
}

extern "C" __global__ void __launch_bounds__(256) gather_u8_to_f16(
    const unsigned char* __restrict__ v_u8,
    const long long* __restrict__ slice_idx,
    const unsigned short* __restrict__ lut,
    unsigned short* __restrict__ out,
    int B, int SK, int M, int HW)
{
    __shared__ unsigned short s_lut[256];
    const int tid = threadIdx.x;
    s_lut[tid] = lut[tid];
    __syncthreads();
    const int bm = blockIdx.y;
    const int b = bm / M, m = bm - b * M;
    const int s = (int)slice_idx[m];
    const unsigned char* src = v_u8 + ((unsigned long long)b * SK + s) * HW;
    unsigned short* dst = out + (unsigned long long)bm * HW;
    const int hw4 = HW >> 2;
    const int step4 = gridDim.x * 256;
    for (int i4 = blockIdx.x * 256 + tid; i4 < hw4; i4 += step4) {
        unsigned int w4 = ((const unsigned int*)src)[i4];
        U2 o;
        o.x = (unsigned int)s_lut[w4 & 0xffu] | ((unsigned int)s_lut[(w4 >> 8) & 0xffu] << 16);
        o.y = (unsigned int)s_lut[(w4 >> 16) & 0xffu] | ((unsigned int)s_lut[w4 >> 24] << 16);
        ((U2*)dst)[i4] = o;
    }
}
"""


def _kernel_impl(X, LO, SCALE, BAD, OUT, N, SX0, SX1, SL, SS, SB, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = col < N
    x = tl.load(X + row * SX0 + col * SX1, mask=active, other=0).to(tl.float64)
    low = tl.load(LO + row * SL)
    scale = tl.load(SCALE + row * SS)
    bad = tl.load(BAD + row * SB)
    y = (x - low) * scale
    # NaN has an undefined float->int cast in Triton: make it zero before the cast.
    # Bad rows go to zero even if lo/scale or the pixels are NaN/Inf.
    y = tl.where((y == y) & ~bad, y, 0.0)
    y = tl.minimum(tl.maximum(y, 0.0), 255.0)
    tl.store(OUT + row * N + col, y.to(tl.uint8), mask=active)


def _load() -> bool:
    """Lazy import and decoration; the caller's lock protects the first use across streams/threads."""
    if not _STATE["tried"]:
        _STATE["tried"] = True
        try:
            torch = importlib.import_module("torch")
            triton = importlib.import_module("triton")
            language = importlib.import_module("triton.language")
            # The JIT resolves the function globals when it compiles, not when the module is imported.
            globals()["tl"] = language
            _STATE.update(torch=torch, triton=triton, kernel=triton.jit(_kernel_impl))
        except Exception as exc:
            _STATE["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
            _LOG.warning("Triton normalisation not available: %s", _STATE["error"])
    return _STATE["kernel"] is not None


def _launch(flat, lo, scale, bad, out=None):
    torch, triton = _STATE["torch"], _STATE["triton"]
    rows, n = flat.shape
    if out is None:
        out = torch.empty((rows, n), dtype=torch.uint8, device=flat.device)
    else:
        out = out.reshape(rows, n)
    if rows and n:
        with torch.cuda.device(flat.device):
            _STATE["kernel"][(triton.cdiv(n, _BLOCK), rows)](
                flat, lo, scale, bad, out, n, *flat.stride(), lo.stride(0), scale.stride(0), bad.stride(0),
                BLOCK=_BLOCK, num_warps=4, enable_fp_fusion=False,
            )
    return out


def _check(device) -> bool:
    """A comparison on the device and a single synchronous boolean read, on first use per GPU."""
    torch = _STATE["torch"]
    lows = [0.0, -1234.56789012345, 0.1, 1e6 + 0.12345, 1e-40, -1e300, float("nan"),
            0.0, 0.0, 0.0, 0.0, 1e308, -1e308]
    scales = [1.0, 0.123456789, 3.7, 255.0 / 4095.0, 1e40, 1e300, 1.0,
              float("inf"), 0.0, float("nan"), -7.9, 1e308, 1e-308]
    invalid = [False] * len(lows)
    invalid[6], invalid[9] = True, True
    with torch.cuda.device(device):
        # Views with stride 2, like the caller's lo = lerp[:, 0:1].
        lo = torch.tensor([[v, 0.0] for v in lows], device=device, dtype=torch.float64)[:, :1]
        scale = torch.tensor([[v, 0.0] for v in scales], device=device, dtype=torch.float64)[:, :1]
        bad = torch.tensor(invalid, device=device, dtype=torch.bool).reshape(-1, 1)
        levels = torch.arange(1, 256, device=device, dtype=torch.float64).unsqueeze(0)
        at = (lo + levels / scale).float()
        inf = torch.full_like(at, float("inf"))
        extremes = [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1.0, -1.0,
                    255.0, 256.0, -256.0, torch.finfo(torch.float32).max, -torch.finfo(torch.float32).max,
                    2.0 ** -149, -(2.0 ** -149)]
        extra = torch.tensor(extremes, device=device, dtype=torch.float32).expand(len(lows), -1)
        # 1025 pixels: the last masked block is part of the comparison too.
        fill = torch.linspace(-65536.0, 65536.0, 1025 - 3 * 255 - len(extremes), device=device).expand(len(lows), -1)
        flat = torch.cat((torch.nextafter(at, -inf), at, torch.nextafter(at, inf), extra, fill), dim=1)
        ref = flat.double()
        ref.sub_(lo).mul_(scale).clamp_(0, 255)
        ref = ref.to(torch.uint8).masked_fill_(bad, 0)
        got = _launch(flat, lo, scale, bad)
        return bool(torch.equal(got, ref))


def _disable(state: dict, exc) -> None:
    state["enabled"] = False
    state["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    _LOG.warning("Triton normalisation disabled: %s", state["error"])


def normalize(flat, lo, scale, bad, out=None):
    """(R,n) CUDA float32 -> uint8, or None to use the usual path.

    lo/scale: (R,1) float64; bad: (R,1) bool, same device as flat. Supports strided views without
    copies. A failure in import, compilation, verification or launch disables the path (per device,
    when possible) and stays visible in status(). After the first comparison it reads no tensors on
    the CPU and does not synchronise the stream.
    """
    if not getattr(flat, "is_cuda", False):
        return None
    with _LOCK:
        if not _load():
            return None
    torch = _STATE["torch"]
    args = (flat, lo, scale, bad)
    if (not all(isinstance(x, torch.Tensor) and x.layout == torch.strided for x in args)
            or flat.ndim != 2 or flat.dtype != torch.float32
            or lo.dtype != torch.float64 or scale.dtype != torch.float64 or bad.dtype != torch.bool
            or any(x.device != flat.device or x.shape != (flat.shape[0], 1) for x in args[1:])):
        return None
    if (out is not None and (not isinstance(out, torch.Tensor) or not out.is_contiguous()
                             or out.dtype != torch.uint8 or out.device != flat.device
                             or out.numel() != flat.numel())):
        out = None
    key = str(flat.device)
    state = _DEVICES.get(key)
    if state is None:
        with _LOCK:
            state = _DEVICES.get(key)
            if state is None:
                state = {"checked": False, "enabled": False, "error": None, "compile_s": None}
                started = time.perf_counter()
                try:
                    if not _check(flat.device):
                        raise RuntimeError("uint8 check against torch.float64 not bit-identical")
                    state.update(checked=True, enabled=True)
                except Exception as exc:
                    _disable(state, exc)
                state["compile_s"] = round(time.perf_counter() - started, 4)
                _DEVICES[key] = state
    if not state["enabled"]:
        return None
    try:
        return _launch(flat, lo, scale, bad, out=out)
    except Exception as exc:
        _disable(state, exc)
        return None


_WS_WORDS = 164352 + 16 * 2 * 131072


def _launch_nvrtc_hist(dev_state: dict, flat, k1: int, k2: int, gamma, out=None, sample=None):
    import torch
    R, n = flat.shape
    if out is None:
        out = torch.empty((R, n), dtype=torch.uint8, device=flat.device)
    else:
        out = out.reshape(R, n)
    ws = dev_state["ws"]
    k_init, k_p1, k_p2, k_p3, k_p4 = dev_state["kernels"]
    st = torch.cuda.current_stream(flat.device).cuda_stream
    P = ctypes.c_void_p
    D = ctypes.c_double
    t0, t1 = float(gamma[0]), float(gamma[1])
    omt0, omt1 = float(1.0 - gamma[0]), float(1.0 - gamma[1])
    x_ptr = P(flat.data_ptr())
    stat = flat if sample is None else sample
    stat_ptr = P(stat.data_ptr())
    stat_n = int(stat.shape[1])
    ws_ptr = P(ws.data_ptr())
    out_ptr = P(out.data_ptr())
    k_init.launch((R, 1), (256,), [int(R), ws_ptr], st)
    k_p1.launch((32, R), (256,), [stat_ptr, stat_n, ws_ptr], st)
    k_p2.launch((32, R), (256,), [stat_ptr, stat_n, int(k1), int(k2), ws_ptr], st)
    k_p3.launch((2, R), (256,), [stat_ptr, stat_n, ws_ptr], st)
    k_p4.launch((64, R), (256,), [x_ptr, int(n), D(t0), D(t1), D(omt0), D(omt1), ws_ptr, out_ptr], st)
    return out


def _precompiled_nvrtc(arch: str):
    """An offline CUBIN only if architecture, CUDA source and binary all match."""
    import hashlib
    import json
    import os
    from pathlib import Path

    directory = os.environ.get("KNEE_UINT8_CUBIN_DIR")
    if not directory:
        return None
    try:
        root = Path(directory)
        meta = json.loads((root / "fused-compileinfo.json").read_text(encoding="utf-8"))
        image = (root / "fused.cubin").read_bytes()
        if (meta["arch"] != arch or meta["source_sha256"] != hashlib.sha256(_NVRTC_SRC.encode()).hexdigest()
                or meta["cubin_sha256"] != hashlib.sha256(image).hexdigest() or not image.startswith(b"\x7fELF")):
            raise ValueError("uint8 CUBIN differs from source, architecture or manifest")
        return image
    except (OSError, ValueError, KeyError) as exc:
        _LOG.warning("offline uint8 CUBIN ignored: %s", exc)
        return None


def _init_nvrtc(device) -> dict:
    import numpy as np
    import torch
    from .. import cudart
    from .torch_uint8 import CLIP_PCT, _ranks

    dev = torch.device(device)
    idx = dev.index or 0
    state = {"checked": False, "enabled": False, "error": None, "compile_s": None,
             "hist_calls": 0, "gather_calls": 0, "sample_strides": []}
    t0 = time.perf_counter()
    try:
        with torch.cuda.device(dev):
            cc = torch.cuda.get_device_capability(dev)
            arch = f"sm_{cc[0]}{cc[1]}"
            cubin = _precompiled_nvrtc(arch)
            state["image_source"] = "precompiled" if cubin is not None else "nvrtc"
            if cubin is None:
                cubin = cudart.compile_source(_NVRTC_SRC, arch=arch, name="fused_uint8_nvrtc.cu")
            try:
                mod = cudart.Module(cubin, idx)
            except cudart.CudaError as exc:
                if state["image_source"] != "precompiled":
                    raise
                state["cubin_error"] = str(exc)[:300]
                cubin = cudart.compile_source(_NVRTC_SRC, arch=arch, name="fused_uint8_nvrtc.cu")
                mod = cudart.Module(cubin, idx)
                state["image_source"] = "nvrtc"
            kernels = (
                mod.function("hist_init"),
                mod.function("hist_pass1"),
                mod.function("hist_pass2"),
                mod.function("hist_pass3"),
                mod.function("hist_pass4_norm"),
            )
            k_gather = mod.function("gather_u8_to_f16")
            ws = torch.empty(_WS_WORDS, dtype=torch.int32, device=dev)
            lut = torch.arange(256, dtype=torch.uint8, device=dev).half().mul_(1.0 / 255.0).view(torch.uint16)
            state.update(mod=mod, kernels=kernels, k_gather=k_gather, ws=ws, lut=lut)

            # Bit-exact self-check of `hist_to_uint8` against `to_uint8_torch` on 8 cases.
            rng = np.random.default_rng(42)
            n_test = 4099
            rows = [
                rng.gamma(2.0, 150.0, size=n_test).astype(np.float32),
                rng.normal(-500.0, 80.0, size=n_test).astype(np.float32),
                rng.integers(0, 30, size=n_test).astype(np.float32),
                np.full(n_test, 7.0, dtype=np.float32),
                np.where(rng.random(n_test) < 0.992, 0.0, 9.0).astype(np.float32),
                np.zeros(n_test, dtype=np.float32),
                np.where(rng.random(n_test) < 0.005, np.nan, rng.normal(0, 1, n_test)).astype(np.float32),
                np.where(rng.random(n_test) < 0.02, np.inf, rng.normal(0, 1, n_test)).astype(np.float32),
            ]
            rows[0][: n_test // 3] = 0.0
            x_test = torch.from_numpy(np.stack(rows)).to(dev).reshape(len(rows), 1, 1, n_test)

            def reference(stride):
                # The lock of the first NVRTC use is already held: calling _normalize_rows would re-enter the same
                # lock through normalize(). The small, independent NumPy reference avoids both the deadlock and
                # comparing the kernel with itself.
                with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                    limits = np.percentile(np.stack(rows)[:, ::stride], CLIP_PCT, axis=1)
                    expected = []
                    for row, low, high in zip(rows, limits[0], limits[1]):
                        bad = not (np.isfinite(low) and np.isfinite(high)) or high <= low
                        values = np.zeros(row.shape, np.uint8) if bad else \
                            np.clip((row.astype(np.float64) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
                        expected.append(values)
                return torch.from_numpy(np.stack(expected)).to(dev)

            k1, k2, gamma = _ranks(n_test)
            ref = reference(1)
            got = _launch_nvrtc_hist(state, x_test.reshape(len(rows), n_test), k1, k2, gamma)
            if not bool(torch.equal(got, ref)):
                raise RuntimeError("NVRTC hist_to_uint8 check not bit-identical")
            # The kernel uses percentiles on the stride-7 subsample and normalises all pixels.
            # The same NumPy reference also checks rows not aligned to float4.
            sample = x_test.reshape(len(rows), n_test)[:, ::7].contiguous()
            sk1, sk2, sgamma = _ranks(sample.shape[1])
            want = reference(7)
            got = _launch_nvrtc_hist(state, x_test.reshape(len(rows), n_test), sk1, sk2, sgamma,
                                     sample=sample)
            if not bool(torch.equal(got, want)):
                raise RuntimeError("NVRTC stride-7 percentile check not bit-identical")

            # Bit-exact self-check of `gather_u8_to_f16`.
            u8_t = torch.randint(0, 256, (2, 12, 64, 64), dtype=torch.uint8, device=dev)
            sidx_t = torch.tensor([0, 3, 5, 11, 2, 7], dtype=torch.int64, device=dev)
            ref_g = u8_t.index_select(1, sidx_t).half().mul_(1.0 / 255.0)
            got_g = torch.empty_like(ref_g)
            P = ctypes.c_void_p
            st = torch.cuda.current_stream(dev).cuda_stream
            k_gather.launch(
                (4, 2 * int(sidx_t.numel())), (256,),
                [P(u8_t.data_ptr()), P(sidx_t.data_ptr()), P(lut.data_ptr()),
                 P(got_g.data_ptr()), 2, 12, int(sidx_t.numel()), 64 * 64], st,
            )
            if not bool(torch.equal(got_g, ref_g)):
                raise RuntimeError("NVRTC gather_u8_to_f16 check not bit-identical")
            state.update(checked=True, enabled=True)
    except Exception as exc:
        state["enabled"] = False
        state["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        _LOG.warning("NVRTC uint8 kernel disabled: %s", state["error"])
    state["compile_s"] = round(time.perf_counter() - t0, 4)
    return state


def _get_nvrtc(device) -> dict:
    key = str(device)
    state = _NVRTC_DEVICES.get(key)
    if state is None:
        with _LOCK:
            state = _NVRTC_DEVICES.get(key)
            if state is None:
                state = _init_nvrtc(device)
                _NVRTC_DEVICES[key] = state
    return state


def warmup(device) -> dict:
    """Prepares the actual backend; the Triton fallback stays lazy when NVRTC is active.

    Uses the same NVRTC self-check as the first study. It sends no dummy histograms: workspace, LUT
    and kernels are initialised by _get_nvrtc without incrementing hist_calls/gather_calls. If NVRTC
    is not available, it repeats the previous normalize warm-up. The lock belongs to
    _get_nvrtc/normalize: it must not be acquired again here.
    """
    started = time.perf_counter()
    try:
        state = _get_nvrtc(device)
    except Exception as exc:
        state = {"enabled": False, "checked": False,
                 "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
    enabled = bool(state.get("enabled", False))
    backend = "nvrtc"
    if not enabled:
        torch = _get_torch()
        result = normalize(torch.zeros((1, 16), device=device),
                           torch.zeros((1, 1), device=device, dtype=torch.float64),
                           torch.ones((1, 1), device=device, dtype=torch.float64),
                           torch.zeros((1, 1), device=device, dtype=torch.bool))
        backend = "triton" if result is not None else "unavailable"
    return {"device": str(device), "backend": backend,
            "elapsed_s": round(time.perf_counter() - started, 4),
            "nvrtc_enabled": enabled, "nvrtc_checked": bool(state.get("checked", False)),
            "nvrtc_error": state.get("error")}


def hist_to_uint8(x, k1: int, k2: int, gamma, out=None, sample_stride: int = 1):
    """(R, K, H, W) contiguous CUDA float32 -> uint8 in 4 NVRTC launches, or None for the usual path."""
    if not getattr(x, "is_cuda", False) or not x.is_contiguous() or x.dtype != _get_torch().float32:
        return None
    R = x.shape[0]
    n = x.numel() // max(R, 1)
    if not isinstance(sample_stride, int) or sample_stride < 1:
        return None
    stat_n = len(range(0, n, sample_stride))
    if R < 1 or R > 16 or n < 4 or n >= (1 << 24) or not (0 <= k1 < k2 < stat_n - 1):
        return None
    state = _get_nvrtc(x.device)
    if not state["enabled"]:
        return None
    try:
        flat = x.reshape(R, n)
        sample = flat[:, ::sample_stride].contiguous() if sample_stride > 1 else None
        res = _launch_nvrtc_hist(state, flat, k1, k2, gamma, out=out, sample=sample)
        state["hist_calls"] += 1
        if sample_stride not in state["sample_strides"]:
            state["sample_strides"].append(sample_stride)
        return res.reshape(x.shape)
    except Exception as exc:
        state["enabled"] = False
        state["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        return None


def gather_u8_f16(v_u8, slice_idx, out) -> bool:
    """Gathers `slice_idx` from `v_u8` `(B, S, K, H, W)` uint8 into `out` `(B*nw, depth, H, W)` fp16."""
    torch = _get_torch()
    if (not getattr(v_u8, "is_cuda", False) or not v_u8.is_contiguous() or v_u8.dtype != torch.uint8
            or not slice_idx.is_contiguous() or slice_idx.dtype != torch.int64
            or not out.is_contiguous() or out.dtype != torch.float16
            or v_u8.device != out.device or slice_idx.device != out.device):
        return False
    b, s, k, h, w = v_u8.shape
    hw = h * w
    m = int(slice_idx.numel())
    if hw % 4 != 0 or out.numel() != b * m * hw or b * m == 0:
        return False
    state = _get_nvrtc(v_u8.device)
    if not state["enabled"]:
        return False
    try:
        P = ctypes.c_void_p
        st = torch.cuda.current_stream(v_u8.device).cuda_stream
        state["k_gather"].launch(
            (min(32, max(1, (hw // 4 + 255) // 256)), b * m), (256,),
            [P(v_u8.data_ptr()), P(slice_idx.data_ptr()), P(state["lut"].data_ptr()),
             P(out.data_ptr()), int(b), int(s * k), int(m), int(hw)], st,
        )
        state["gather_calls"] += 1
        return True
    except Exception as exc:
        state["enabled"] = False
        state["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        return False


def _get_torch():
    if _STATE["torch"] is None:
        _STATE["torch"] = importlib.import_module("torch")
    return _STATE["torch"]


def status() -> dict:
    """Serialisable telemetry, without imports or CUDA accesses; compile_s includes the self-check."""
    devices = {key: dict(value) for key, value in _DEVICES.items()}
    nvrtc_devs = {
        key: {k: v for k, v in val.items() if k in ("checked", "enabled", "error", "compile_s",
                                                  "hist_calls", "gather_calls", "sample_strides",
                                                  "image_source", "cubin_error")}
        for key, val in _NVRTC_DEVICES.items()
    }
    used = list(devices.values()) + list(nvrtc_devs.values())
    return {"checked": bool(used) and all(s["checked"] for s in used),
            "enabled": any(s["enabled"] for s in used),
            "nvrtc_enabled": any(s["enabled"] for s in nvrtc_devs.values()),
            "nvrtc_devices": nvrtc_devs,
            "error": _STATE["error"] or next((s["error"] for s in devices.values() if s["error"]), None),
            "compile_s": sum(s["compile_s"] or 0 for s in devices.values()) if devices else None,
            "devices": devices}

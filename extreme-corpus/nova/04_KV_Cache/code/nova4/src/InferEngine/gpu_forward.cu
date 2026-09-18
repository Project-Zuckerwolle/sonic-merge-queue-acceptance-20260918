// gpu_forward.cu — On-GPU inkrementeller Forward (siehe gpu_forward.h).
#include "InferEngine/gpu_forward.h"
#include "InferEngine/real_inference.h"   // Int3Weights

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace nova::infer {

namespace {

// x[k] = dequant(token_embd[token]) — Zeile `token`, Länge K=hidden.
__global__ void embed_gather_kernel(const signed char* __restrict__ U, const float* __restrict__ scales,
                                    int token, int K, int ng, int group, float* __restrict__ x) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    x[k] = (float)U[(size_t)token * K + k] * scales[(size_t)token * ng + k / group];
}

// out = rmsnorm(x)·g. Ein Block; ss-Reduktion in Shared (float, matcht CPU auf ~1e-6).
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ g,
                               int H, float eps, float* __restrict__ out) {
    __shared__ float red[256];
    const int tid = threadIdx.x, nt = blockDim.x;
    float part = 0.0f;
    for (int i = tid; i < H; i += nt) part += x[i] * x[i];
    red[tid] = part; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    const float inv = rsqrtf(red[0] / H + eps);
    for (int i = tid; i < H; i += nt) out[i] = x[i] * inv * g[i];
}

// RoPE (interleaved) auf v[nHeads*hd] @pos — bit-gleich zu real_inference::rope_at.
__global__ void rope_kernel(float* __restrict__ v, int pos, int nHeads, int hd, float theta) {
    const int half = hd / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // über nHeads*half
    if (idx >= nHeads * half) return;
    const int h = idx / half, i = idx % half;
    float* p = v + (size_t)h * hd;
    const float freq = powf(theta, -2.0f * i / hd);
    const float ang = pos * freq, c = cosf(ang), sn = sinf(ang);
    const int a = 2 * i, b = 2 * i + 1;
    const float va = p[a], vb = p[b];
    p[a] = va * c - vb * sn;
    p[b] = va * sn + vb * c;
}

// GQA kausale Attention, FLASH-Style (Online-Softmax): ein Block je Q-Head, O(head_dim) Shared statt
// O(T) — funktioniert für beliebig langen Kontext (80k+). Numerisch exakt zur Batch-Softmax.
__global__ void attn_kernel(const float* __restrict__ q, const float* __restrict__ kc,
                            const float* __restrict__ vc, int nH, int nKV, int hd, int T,
                            float scale, float* __restrict__ ctx) {
    const int h = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const int grp = nH / nKV, kvh = h / grp;
    const float* qv = q + (size_t)h * hd;
    __shared__ float acc[128], red[128], s_corr, s_p, s_l;
    float m = -1e30f;                              // thread-0 maßgeblich
    for (int i = tid; i < hd; i += nt) acc[i] = 0.0f;
    if (tid == 0) s_l = 0.0f;
    __syncthreads();
    for (int t = 0; t < T; ++t) {
        const float* kk = kc + ((size_t)t * nKV + kvh) * hd;
        float part = 0.0f; for (int i = tid; i < hd; i += nt) part += qv[i] * kk[i];
        red[tid] = part; __syncthreads();
        for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
        if (tid == 0) { float score = red[0] * scale; float mn = fmaxf(m, score);
                        s_corr = expf(m - mn); s_p = expf(score - mn); m = mn; s_l = s_l * s_corr + s_p; }
        __syncthreads();
        const float* vv = vc + ((size_t)t * nKV + kvh) * hd;
        for (int i = tid; i < hd; i += nt) acc[i] = acc[i] * s_corr + s_p * vv[i];
        __syncthreads();
    }
    float* out = ctx + (size_t)h * hd;
    for (int i = tid; i < hd; i += nt) out[i] = acc[i] / s_l;
}

__global__ void silu_mul_kernel(float* __restrict__ g, const float* __restrict__ u, int F) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= F) return;
    const float x = g[i];
    g[i] = (x / (1.0f + expf(-x))) * u[i];
}

__global__ void add_kernel(float* __restrict__ x, const float* __restrict__ t, int H) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H) x[i] += t[i];
}

// argmax über logits[V] — ein Block, Ties: niedrigster Index (matcht CPU argmax_ptr).
__global__ void argmax_kernel(const float* __restrict__ v, int n, int* __restrict__ out) {
    __shared__ float sval[256]; __shared__ int sidx[256];
    const int tid = threadIdx.x, nt = blockDim.x;
    float best = -1e30f; int bi = 0;
    for (int i = tid; i < n; i += nt) { float x = v[i]; if (x > best) { best = x; bi = i; } }
    sval[tid] = best; sidx[tid] = bi; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sval[tid + s] > sval[tid] || (sval[tid + s] == sval[tid] && sidx[tid + s] < sidx[tid])) {
                sval[tid] = sval[tid + s]; sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) *out = sidx[0];
}

float* dev_upload(const std::vector<float>& h) {
    float* d = nullptr; cudaMalloc(reinterpret_cast<void**>(&d), h.size() * sizeof(float));
    cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

// --- Batched Kernels ([feature][N] column-major), gestreamter 24B-Verify. Ein Block/Spalte bzw.
//     ein Thread/(Spalte,Element). Numerik bit-gleich zu forward_batch (CPU): FP32, gleiche Reduktion.
__global__ void rmsnorm_batch_kernel(const float* __restrict__ src, const float* __restrict__ g,
                                     int H, int N, float eps, float* __restrict__ dst) {
    const int n = blockIdx.x; if (n >= N) return;             // ein Block je Spalte
    __shared__ float red[256];
    const int tid = threadIdx.x, nt = blockDim.x;
    float part = 0.0f;
    for (int i = tid; i < H; i += nt) { float v = src[(size_t)i * N + n]; part += v * v; }
    red[tid] = part; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    const float inv = rsqrtf(red[0] / H + eps);
    for (int i = tid; i < H; i += nt) dst[(size_t)i * N + n] = src[(size_t)i * N + n] * inv * g[i];
}
// Tree (Phase 6A): pos_arr != nullptr -> RoPE-Position je Knoten = Baum-Tiefe (statt start_pos+n).
__global__ void rope_batch_kernel(float* __restrict__ Vm, int N, int start_pos, int nHeads, int hd, float theta,
                                  const int* __restrict__ pos_arr = nullptr) {
    const int half = hd / 2;
    const long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)N * nHeads * half) return;
    const int i = int(idx % half);
    const int h = int((idx / half) % nHeads);
    const int n = int(idx / ((long long)half * nHeads));
    const int pos = pos_arr ? pos_arr[n] : start_pos + n;
    const float freq = powf(theta, -2.0f * i / hd);
    const float ang = pos * freq, c = cosf(ang), sn = sinf(ang);
    const int a = 2 * i, b = 2 * i + 1;
    const float va = Vm[(size_t)(h * hd + a) * N + n], vb = Vm[(size_t)(h * hd + b) * N + n];
    Vm[(size_t)(h * hd + a) * N + n] = va * c - vb * sn;
    Vm[(size_t)(h * hd + b) * N + n] = va * sn + vb * c;
}
__global__ void kv_append_batch_kernel(const float* __restrict__ Kc, const float* __restrict__ Vc,
                                       int N, int start_pos, int kvd,
                                       float* __restrict__ kcache, float* __restrict__ vcache) {
    const long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)N * kvd) return;
    const int i = int(idx % kvd), n = int(idx / kvd);
    const size_t dst = (size_t)(start_pos + n) * kvd + i;
    kcache[dst] = Kc[(size_t)i * N + n];
    vcache[dst] = Vc[(size_t)i * N + n];
}
// FLASH-Style batched Attention (Online-Softmax), O(head_dim) Shared -> beliebig langer Kontext.
// Tree (Phase 6A): mask != nullptr -> Spalte n sieht unter den Batch-Slots (t>=start_pos) nur ihre
// Ahnen (mask[n*N + (t-start_pos)]==1). Prefix-Keys (t<start_pos) immer sichtbar. Der Masken-Check ist
// block-uniform (h,n block-konstant) -> alle Threads springen gleich -> kein __syncthreads-Deadlock.
__global__ void attn_batch_kernel(const float* __restrict__ Q, const float* __restrict__ kcache,
                                  const float* __restrict__ vcache, int nH, int nKV, int hd, int N,
                                  int start_pos, int kvd, float scale, float* __restrict__ Ctx,
                                  const unsigned char* __restrict__ mask = nullptr) {
    const int h = blockIdx.x, n = blockIdx.y;                 // Block = (Q-Head, Spalte)
    const int tid = threadIdx.x, nt = blockDim.x;
    const int grp = nH / nKV, kvh = h / grp;
    const int T = start_pos + n + 1;                          // kausal: Spalte n sieht 0..start_pos+n
    __shared__ float acc[128], red[128], s_corr, s_p, s_l;
    float m = -1e30f;
    for (int i = tid; i < hd; i += nt) acc[i] = 0.0f;
    if (tid == 0) s_l = 0.0f;
    __syncthreads();
    for (int t = 0; t < T; ++t) {
        if (mask && t >= start_pos && !mask[(size_t)n * N + (t - start_pos)]) continue;  // Nicht-Ahne -> überspringen
        const float* kk = kcache + (size_t)t * kvd + (size_t)kvh * hd;
        float part = 0.0f; for (int i = tid; i < hd; i += nt) part += Q[(size_t)(h * hd + i) * N + n] * kk[i];
        red[tid] = part; __syncthreads();
        for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
        if (tid == 0) { float score = red[0] * scale; float mn = fmaxf(m, score);
                        s_corr = expf(m - mn); s_p = expf(score - mn); m = mn; s_l = s_l * s_corr + s_p; }
        __syncthreads();
        const float* vv = vcache + (size_t)t * kvd + (size_t)kvh * hd;
        for (int i = tid; i < hd; i += nt) acc[i] = acc[i] * s_corr + s_p * vv[i];
        __syncthreads();
    }
    for (int i = tid; i < hd; i += nt) Ctx[(size_t)(h * hd + i) * N + n] = acc[i] / s_l;
}
__global__ void silu_mul_batch_kernel(float* __restrict__ G, const float* __restrict__ U, long long total) {
    const long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const float x = G[idx]; G[idx] = (x / (1.0f + expf(-x))) * U[idx];
}
__global__ void add_batch_kernel(float* __restrict__ X, const float* __restrict__ t, long long total) {
    const long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < total) X[idx] += t[idx];
}

// --- Rotation: X_head <- R · X_head je (Spalte, Head). Orthogonale R verteilt K-Ausreißer über die
//     Kanäle (QuaRot/PolarQuant-Prinzip), sodass per-Block-Quant funktioniert. Q·K bleibt erhalten,
//     wenn Q UND K gleich rotiert werden. X column-major [nHeads*hd][N]. In-place (Input erst nach sh).
__global__ void rotate_heads_kernel(float* __restrict__ X, const float* __restrict__ R,
                                    int N, int nHeads, int hd) {
    extern __shared__ float sh[];
    const int n = blockIdx.x, h = blockIdx.y;
    const int base = h * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) sh[i] = X[(size_t)(base + i) * N + n];
    __syncthreads();
    for (int o = threadIdx.x; o < hd; o += blockDim.x) {
        const float* Rr = R + (size_t)o * hd;
        float acc = 0.0f;
        for (int i = 0; i < hd; ++i) acc += Rr[i] * sh[i];
        X[(size_t)(base + o) * N + n] = acc;
    }
}

// --- KV-Quantisierung (affine Min/Max je Head/Token, bits∈{2,3,4}). Kc: [kvd][N] column-major.
//     Ein Thread je (Spalte n, KV-Head). Head-Bytebereich ist ausgerichtet (hd*bits % 8 == 0 für
//     hd=128,bits=2/3/4), daher keine Byte-Races zwischen Heads. ~14×/‑10×/‑8× kleiner als FP32.
__global__ void kv_quant_kernel(const float* __restrict__ Kc, int N, int start_pos, int nKV, int hd,
                                int kvd, int bits, int qjl, unsigned char* __restrict__ codes,
                                unsigned short* __restrict__ meta) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * nKV) return;
    const int kvh = idx % nKV, n = idx / nKV;
    const int pos = start_pos + n;
    const int base = kvh * hd;
    const int levels = (1 << bits) - 1;
    // qjl: symmetrischer 3-Bit-Codebuch (2-Bit {-1.5,-0.5,0.5,1.5}·s + 1-Bit QJL-Sign = 8 Level
    // {±0.25,±0.75,±1.25,±1.75}·s, s=amax/1.5). Sonst: affine Min/Max.
    float scale, zero, inv;
    if (qjl) {
        float amax = 0.0f;
        for (int i = 0; i < hd; ++i) amax = fmaxf(amax, fabsf(Kc[(size_t)(base + i) * N + n]));
        scale = amax > 0.0f ? amax / 1.5f : 1e-8f; zero = 0.0f; inv = 1.0f / (0.5f * scale);
    } else {
        float mn = 1e30f, mx = -1e30f;
        for (int i = 0; i < hd; ++i) { float v = Kc[(size_t)(base + i) * N + n]; mn = fminf(mn, v); mx = fmaxf(mx, v); }
        scale = (mx - mn) / levels; if (scale <= 0.0f) scale = 1e-8f; zero = mn; inv = 1.0f / scale;
    }
    meta[((size_t)pos * nKV + kvh) * 2 + 0] = __half_as_ushort(__float2half(scale));  // BITS speichern
    meta[((size_t)pos * nKV + kvh) * 2 + 1] = __half_as_ushort(__float2half(zero));
    const size_t tok_bytes = (size_t)kvd * bits / 8;
    unsigned char* hrow = codes + (size_t)pos * tok_bytes + (size_t)base * bits / 8;
    unsigned acc = 0; int accbits = 0, ob = 0;
    for (int i = 0; i < hd; ++i) {
        const float v = Kc[(size_t)(base + i) * N + n];
        int q = qjl ? (int)lrintf(v * inv + 3.5f) : (int)lrintf((v - zero) * inv);
        if (q < 0) q = 0; if (q > levels) q = levels;
        acc |= (unsigned)(q & levels) << accbits; accbits += bits;
        while (accbits >= 8) { hrow[ob++] = (unsigned char)(acc & 0xFF); acc >>= 8; accbits -= 8; }
    }
    if (accbits > 0) hrow[ob] = (unsigned char)(acc & 0xFF);
}
// Dequant -> FP32-Scratch [T][kvd] (Layout wie d_kcache_), ein Thread je (Position t, KV-Head).
__global__ void kv_dequant_kernel(const unsigned char* __restrict__ codes, const unsigned short* __restrict__ meta,
                                  int T, int nKV, int hd, int kvd, int bits, int qjl, float* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * nKV) return;
    const int kvh = idx % nKV, t = idx / nKV;
    const int base = kvh * hd;
    const int levels = (1 << bits) - 1;
    const float scale = __half2float(__ushort_as_half(meta[((size_t)t * nKV + kvh) * 2 + 0]));
    const float zero  = __half2float(__ushort_as_half(meta[((size_t)t * nKV + kvh) * 2 + 1]));
    const size_t tok_bytes = (size_t)kvd * bits / 8;
    const unsigned char* hrow = codes + (size_t)t * tok_bytes + (size_t)base * bits / 8;
    unsigned acc = 0; int accbits = 0, ib = 0;
    for (int i = 0; i < hd; ++i) {
        while (accbits < bits) { acc |= (unsigned)hrow[ib++] << accbits; accbits += 8; }
        const int q = acc & levels; acc >>= bits; accbits -= bits;
        out[(size_t)t * kvd + base + i] = qjl ? (q - 3.5f) * (0.5f * scale) : (q * scale + zero);
    }
}

}  // namespace

GpuForward::~GpuForward() { free(); }

bool GpuForward::init(const Int3Weights& w, int max_seq, std::string* err, bool streamed, int kv_bits,
                      bool want_batch) {
    w_ = &w; cfg_ = w.cfg; max_seq_ = max_seq; seq_ = 0; streamed_ = streamed; kv_bits_ = kv_bits;
    // Phase 6A-Wide: der residente Draft (streamed=false) braucht für den batched Baum-Forward dieselben
    // batched Puffer + Embedding-Scratch wie der 24B. want_batch schaltet sie zu (nur wenn Tree aktiv).
    const bool batch_bufs = streamed_ || want_batch;
    if (!stream_) { cudaStream_t s = nullptr; cudaStreamCreate(&s); stream_ = s; }   // Phase 3.2: Compute-Stream
    const int H = cfg_.hidden, qd = cfg_.q_dim(), kvd = cfg_.kv_dim(), F = cfg_.ffn, V = cfg_.vocab;
    if (batch_bufs) h_embed_.assign(H, 0.0f);   // Host-Scratch fürs Embedding (dequant_row -> H2D)

    // 1. Gewichts-Zeiger cachen. Resident-Modus (3B): einmal entpacken (int3_residentize). Streamed-
    //    Modus (24B): NICHT residentisieren — die Gewichte bleiben in W.packed und streamen pro Layer.
    auto need = [&](const std::string& n) -> const Int3Matrix* {
        const Int3Matrix* m = w.mat(n);
        if (m && !streamed_) int3_residentize(*m);
        return m;
    };
    gw_ = GpuForwardWeights{};
    gw_.embd = need("token_embd");
    gw_.out  = need("output");
    for (int l = 0; l < cfg_.n_layers; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        gw_.q.push_back(need(b + "attn_q"));   gw_.k.push_back(need(b + "attn_k"));
        gw_.v.push_back(need(b + "attn_v"));   gw_.o.push_back(need(b + "attn_output"));
        gw_.gate.push_back(need(b + "ffn_gate")); gw_.up.push_back(need(b + "ffn_up"));
        gw_.down.push_back(need(b + "ffn_down"));
    }
    if (!gw_.embd || !gw_.out) { if (err) *err = "token_embd/output fehlt"; return false; }
    // Streamed-Embedding (24B): token_embd ist NICHT resident -> Embedding via Host-dequant_row + H2D.
    // Dafür muss token_embd FP32-Skalen haben (kein BLOCK3) — im Loader so erzwungen (token_embd bleibt INT3).

    // 2. Norms hochladen.
    d_attn_norm_.clear(); d_ffn_norm_.clear();
    for (int l = 0; l < cfg_.n_layers; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        const auto* an = w.norm(b + "attn_norm"); const auto* fn = w.norm(b + "ffn_norm");
        if (!an || !fn) { if (err) *err = "norm fehlt in " + b; return false; }
        d_attn_norm_.push_back(dev_upload(*an)); d_ffn_norm_.push_back(dev_upload(*fn));
    }
    const auto* on = w.norm("output_norm");
    if (!on) { if (err) *err = "output_norm fehlt"; return false; }
    d_out_norm_ = dev_upload(*on);

    // 3. Aktivierungs-Puffer + KV allozieren.
    auto alloc = [](float** p, size_t n) { cudaMalloc(reinterpret_cast<void**>(p), n * sizeof(float)); };
    alloc(&d_x, H); alloc(&d_xn, H); alloc(&d_q, qd); alloc(&d_k, kvd); alloc(&d_v, kvd);
    alloc(&d_ctx, qd); alloc(&d_tmp, H); alloc(&d_g, F); alloc(&d_u, F); alloc(&d_logits, V);
    cudaMalloc(reinterpret_cast<void**>(&d_arg), sizeof(int));
    if (batch_bufs) {   // batched Puffer [feature][MAX_BATCH] für den Spec-Verify (24B) bzw. Baum-Draft (3B)
        const int B = MAX_BATCH;
        alloc(&d_xb, (size_t)H * B); alloc(&d_xnb, (size_t)H * B); alloc(&d_qb, (size_t)qd * B);
        alloc(&d_kb, (size_t)kvd * B); alloc(&d_vb, (size_t)kvd * B); alloc(&d_ctxb, (size_t)qd * B);
        alloc(&d_tmpb, (size_t)H * B); alloc(&d_gb, (size_t)F * B); alloc(&d_ub, (size_t)F * B);
        alloc(&d_logitsb, (size_t)V * B);
        h_embed_b_.assign((size_t)H * B, 0.0f);
        cudaMalloc(reinterpret_cast<void**>(&d_treepos_),  (size_t)B * sizeof(int));       // Phase 6A
        cudaMalloc(reinterpret_cast<void**>(&d_treemask_), (size_t)B * B);
        hidtap_[0] = cfg_.n_layers / 4; hidtap_[1] = cfg_.n_layers / 2;                    // Phase 6B (low/mid, reserve)
        hidtap_[2] = cfg_.n_layers - 1;   // TOP-Layer-Feature (klassisches EAGLE-1-Feature vor der finalen Norm)
        for (int k = 0; k < 3; ++k) alloc(&d_hid_[k], (size_t)H * B);
    }
    if (kv_bits_ == 0) {   // FP32-KV (token-identisch)
        d_kcache_.assign(cfg_.n_layers, nullptr); d_vcache_.assign(cfg_.n_layers, nullptr);
        for (int l = 0; l < cfg_.n_layers; ++l) {
            cudaMalloc(reinterpret_cast<void**>(&d_kcache_[l]), (size_t)max_seq_ * kvd * sizeof(float));
            cudaMalloc(reinterpret_cast<void**>(&d_vcache_[l]), (size_t)max_seq_ * kvd * sizeof(float));
        }
    } else {               // quantisierter KV: packed codes + meta je Layer, gemeinsamer FP32-Dequant-Scratch
        const int nKV = cfg_.n_kv_heads;
        const int sbits = (kv_bits_ == 2) ? 3 : kv_bits_;   // 2 = 2-Bit+QJL -> 3-Bit-Storage (symmetrisch)
        const size_t tok_bytes = (size_t)kvd * sbits / 8;
        d_kcode_.assign(cfg_.n_layers, nullptr); d_vcode_.assign(cfg_.n_layers, nullptr);
        d_kmeta_.assign(cfg_.n_layers, nullptr); d_vmeta_.assign(cfg_.n_layers, nullptr);
        for (int l = 0; l < cfg_.n_layers; ++l) {
            cudaMalloc(reinterpret_cast<void**>(&d_kcode_[l]), (size_t)max_seq_ * tok_bytes);
            cudaMalloc(reinterpret_cast<void**>(&d_vcode_[l]), (size_t)max_seq_ * tok_bytes);
            cudaMalloc(reinterpret_cast<void**>(&d_kmeta_[l]), (size_t)max_seq_ * nKV * 2 * sizeof(unsigned short));
            cudaMalloc(reinterpret_cast<void**>(&d_vmeta_[l]), (size_t)max_seq_ * nKV * 2 * sizeof(unsigned short));
        }
        cudaMalloc(reinterpret_cast<void**>(&d_kdq_), (size_t)max_seq_ * kvd * sizeof(float));
        cudaMalloc(reinterpret_cast<void**>(&d_vdq_), (size_t)max_seq_ * kvd * sizeof(float));
        // Orthogonale Hadamard-Rotation R[o][i] = (-1)^popcount(o&i)/sqrt(hd) (hd=128=2^7). Verteilt
        // K-Ausreißer über die Kanäle -> per-Block-Quant funktioniert (QuaRot/PolarQuant-Prinzip).
        const int hd = cfg_.head_dim;
        std::vector<float> R((size_t)hd * hd, 0.0f);
        const bool pow2 = hd > 0 && (hd & (hd - 1)) == 0;
        if (pow2) {   // Sylvester-Hadamard (orthogonal nur für 2er-Potenz; hd=128 real)
            const float invs = 1.0f / std::sqrt((float)hd);
            for (int o = 0; o < hd; ++o)
                for (int i = 0; i < hd; ++i) {
                    unsigned x = (unsigned)(o & i); int p = 0; while (x) { p ^= 1; x &= x - 1; }
                    R[(size_t)o * hd + i] = p ? -invs : invs;
                }
        } else {      // Fallback Identität (keine Rotation) für Nicht-2er-Potenz-head_dim (Testmodell)
            for (int o = 0; o < hd; ++o) R[(size_t)o * hd + o] = 1.0f;
        }
        d_rot_ = dev_upload(R);
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { if (err) *err = std::string("cudaMalloc: ") + cudaGetErrorString(e); return false; }
    ready_ = true;
    return true;
}

void GpuForward::reset() { seq_ = 0; }
void GpuForward::trim(int len) { if (len < seq_) seq_ = len; }

// Gemeinsamer per-Layer-Forward -> d_logits. Resident (3B): fused_int3_gemv_dev. Streamed (24B):
// fused_int3_gemv_streamed_dev (Gewicht streamt H2D pro Layer, Aktivierungen bleiben resident).
void GpuForward::run_layers(int token_id, int pos) {
    const int H = cfg_.hidden, nH = cfg_.n_heads, nKV = cfg_.n_kv_heads, hd = cfg_.head_dim;
    const int kvd = cfg_.kv_dim(), F = cfg_.ffn;
    const float eps = cfg_.rms_eps, theta = cfg_.rope_theta, scale = 1.0f / sqrtf((float)hd);
    const int TB = 256;
    cudaStream_t st = (cudaStream_t)stream_;   // Phase 3.2: alles auf dem Compute-Stream, nicht Stream 0
    auto grid = [&](int n) { return (n + TB - 1) / TB; };
    auto gemv_w = [&](const Int3Matrix& W, const float* xd, float* yd) {
        if (streamed_) fused_int3_gemv_streamed_dev(W, xd, yd, st);   // streamt Gewicht, Device-x/y
        else           fused_int3_gemv_dev(W, xd, yd, st);            // resident
    };

    // Embedding. Streamed: token_embd nicht resident -> Host-dequant_row + 1× H2D. Resident: Kernel.
    if (streamed_) {
        dequant_row(*gw_.embd, token_id, h_embed_.data());
        cudaMemcpyAsync(d_x, h_embed_.data(), (size_t)H * sizeof(float), cudaMemcpyHostToDevice, st);
    } else {
        const Int3Matrix* E = gw_.embd; const int ng = E->groups_per_row();
        embed_gather_kernel<<<grid(H), TB, 0, st>>>((const signed char*)E->d_unpacked, (const float*)E->d_scales,
                                             token_id, E->K, ng, E->group, d_x);
    }
    for (int l = 0; l < cfg_.n_layers; ++l) {
        // Attention.
        rmsnorm_kernel<<<1, TB, 0, st>>>(d_x, d_attn_norm_[l], H, eps, d_xn);
        gemv_w(*gw_.q[l], d_xn, d_q);
        gemv_w(*gw_.k[l], d_xn, d_k);
        gemv_w(*gw_.v[l], d_xn, d_v);
        rope_kernel<<<grid(nH * hd / 2), TB, 0, st>>>(d_q, pos, nH, hd, theta);
        rope_kernel<<<grid(nKV * hd / 2), TB, 0, st>>>(d_k, pos, nKV, hd, theta);
        if (kv_bits_ != 0) {   // Q und K gleich rotieren (N=1)
            rotate_heads_kernel<<<dim3(1, nH),  hd, hd * sizeof(float), st>>>(d_q, d_rot_, 1, nH,  hd);
            rotate_heads_kernel<<<dim3(1, nKV), hd, hd * sizeof(float), st>>>(d_k, d_rot_, 1, nKV, hd);
        }
        const int T = pos + 1;
        const float *Ksrc, *Vsrc;
        if (kv_bits_ == 0) {
            cudaMemcpyAsync(d_kcache_[l] + (size_t)pos * kvd, d_k, kvd * sizeof(float), cudaMemcpyDeviceToDevice, st);
            cudaMemcpyAsync(d_vcache_[l] + (size_t)pos * kvd, d_v, kvd * sizeof(float), cudaMemcpyDeviceToDevice, st);
            Ksrc = d_kcache_[l]; Vsrc = d_vcache_[l];
        } else {   // ein Token quantisieren (N=1), Layer dequantisieren
            const int sbits = (kv_bits_ == 2) ? 3 : kv_bits_, qjl = (kv_bits_ == 2) ? 1 : 0;
            kv_quant_kernel<<<grid(nKV), TB, 0, st>>>(d_k, 1, pos, nKV, hd, kvd, sbits, qjl, d_kcode_[l], d_kmeta_[l]);
            kv_quant_kernel<<<grid(nKV), TB, 0, st>>>(d_v, 1, pos, nKV, hd, kvd, sbits, qjl, d_vcode_[l], d_vmeta_[l]);
            kv_dequant_kernel<<<grid((long long)T * nKV), TB, 0, st>>>(d_kcode_[l], d_kmeta_[l], T, nKV, hd, kvd, sbits, qjl, d_kdq_);
            kv_dequant_kernel<<<grid((long long)T * nKV), TB, 0, st>>>(d_vcode_[l], d_vmeta_[l], T, nKV, hd, kvd, sbits, qjl, d_vdq_);
            Ksrc = d_kdq_; Vsrc = d_vdq_;
        }
        attn_kernel<<<nH, 128, 0, st>>>(d_q, Ksrc, Vsrc, nH, nKV, hd, T, scale, d_ctx);   // flash: kein O(T)-Shared
        gemv_w(*gw_.o[l], d_ctx, d_tmp);
        add_kernel<<<grid(H), TB, 0, st>>>(d_x, d_tmp, H);
        // FFN (SwiGLU).
        rmsnorm_kernel<<<1, TB, 0, st>>>(d_x, d_ffn_norm_[l], H, eps, d_xn);
        gemv_w(*gw_.gate[l], d_xn, d_g);
        gemv_w(*gw_.up[l], d_xn, d_u);
        silu_mul_kernel<<<grid(F), TB, 0, st>>>(d_g, d_u, F);
        gemv_w(*gw_.down[l], d_g, d_tmp);
        add_kernel<<<grid(H), TB, 0, st>>>(d_x, d_tmp, H);
    }
    rmsnorm_kernel<<<1, TB, 0, st>>>(d_x, d_out_norm_, H, eps, d_xn);
    gemv_w(*gw_.out, d_xn, d_logits);
}

int GpuForward::step_argmax(int token_id, int pos, float* host_logits) {
    const int V = cfg_.vocab;
    cudaStream_t st = (cudaStream_t)stream_;
    run_layers(token_id, pos);
    argmax_kernel<<<1, 256, 0, st>>>(d_logits, V, d_arg);
    if (host_logits) cudaMemcpyAsync(host_logits, d_logits, (size_t)V * sizeof(float), cudaMemcpyDeviceToHost, st);  // nur Validierung
    int h_arg = 0;
    cudaMemcpyAsync(&h_arg, d_arg, sizeof(int), cudaMemcpyDeviceToHost, st);   // einziger D2H/Token
    cudaStreamSynchronize(st);                                                 // Host-Ergebnis fertig
    seq_ = pos + 1;
    return h_arg;
}

// Wie step_argmax, aber liefert die vollen V Logits (für Spec-Verify des gestreamten 24B).
void GpuForward::step_logits(int token_id, int pos, float* host_logits) {
    const int V = cfg_.vocab;
    cudaStream_t st = (cudaStream_t)stream_;
    run_layers(token_id, pos);
    cudaMemcpyAsync(host_logits, d_logits, (size_t)V * sizeof(float), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);   // host_logits wird vom Aufrufer konsumiert
    seq_ = pos + 1;
}

// Phase 6A: Forward + Top-2-Tokens (für Baum-Draft: greedy + 2. Wahl). Host-Top-2 über die Logits
// (V-Scan, ~0,1 ms; Ties = niedrigster Index, matcht argmax_ptr). Nur der Draft (resident) nutzt das.
void GpuForward::step_top2(int token_id, int pos, int out2[2]) {
    const int V = cfg_.vocab;
    cudaStream_t st = (cudaStream_t)stream_;
    run_layers(token_id, pos);
    std::vector<float> logits((size_t)V);
    cudaMemcpyAsync(logits.data(), d_logits, (size_t)V * sizeof(float), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    int b0 = 0; float v0 = logits[0];
    for (int i = 1; i < V; ++i) if (logits[i] > v0) { v0 = logits[i]; b0 = i; }
    int b1 = -1; float v1 = -1e30f;
    for (int i = 0; i < V; ++i) if (i != b0 && logits[i] > v1) { v1 = logits[i]; b1 = i; }
    out2[0] = b0; out2[1] = (b1 < 0 ? b0 : b1);
    seq_ = pos + 1;
}

// Phase 6A: KV-Block-Copy [from..from+len-1] -> [to..to+len-1] je Layer (Tree-Accept: Gewinner-Kette
// kompaktieren). Nicht-überlappend (Aufrufer: from-to=K>=len). Handhabt kv_bits 0 (FP32) und >0 (codes+meta).
// Phase 6B EAGLE: batched Forward + 3 Layer-Hidden nach hid_host [3][H][N] (column-major je Tap).
void GpuForward::forward_batch_hidden(const int* tokens, int N, int start_pos, float* host_out, float* hid_host) {
    if (N > MAX_BATCH) N = MAX_BATCH;
    cudaStream_t st = (cudaStream_t)stream_;
    const int H = cfg_.hidden;
    const bool prev = hid_capture_; hid_capture_ = true;
    forward_batch_gpu(tokens, N, start_pos, host_out);      // taps -> d_hid_[k] im Layer-Loop
    hid_capture_ = prev;
    for (int k = 0; k < 3; ++k)
        cudaMemcpyAsync(hid_host + (size_t)k * H * N, d_hid_[k], (size_t)H * N * sizeof(float), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
}

void GpuForward::gather_kv_block(int from, int to, int len) {
    if (from == to || len <= 0) return;
    cudaStream_t st = (cudaStream_t)stream_;
    const int kvd = cfg_.kv_dim(), nKV = cfg_.n_kv_heads;
    if (kv_bits_ == 0) {
        for (int l = 0; l < cfg_.n_layers; ++l) {
            cudaMemcpyAsync(d_kcache_[l] + (size_t)to * kvd, d_kcache_[l] + (size_t)from * kvd, (size_t)len * kvd * sizeof(float), cudaMemcpyDeviceToDevice, st);
            cudaMemcpyAsync(d_vcache_[l] + (size_t)to * kvd, d_vcache_[l] + (size_t)from * kvd, (size_t)len * kvd * sizeof(float), cudaMemcpyDeviceToDevice, st);
        }
    } else {
        const int sbits = (kv_bits_ == 2) ? 3 : kv_bits_;
        const size_t tok_bytes = (size_t)kvd * sbits / 8;
        for (int l = 0; l < cfg_.n_layers; ++l) {
            cudaMemcpyAsync(d_kcode_[l] + (size_t)to * tok_bytes, d_kcode_[l] + (size_t)from * tok_bytes, (size_t)len * tok_bytes, cudaMemcpyDeviceToDevice, st);
            cudaMemcpyAsync(d_vcode_[l] + (size_t)to * tok_bytes, d_vcode_[l] + (size_t)from * tok_bytes, (size_t)len * tok_bytes, cudaMemcpyDeviceToDevice, st);
            cudaMemcpyAsync(d_kmeta_[l] + (size_t)to * nKV * 2, d_kmeta_[l] + (size_t)from * nKV * 2, (size_t)len * nKV * 2 * sizeof(unsigned short), cudaMemcpyDeviceToDevice, st);
            cudaMemcpyAsync(d_vmeta_[l] + (size_t)to * nKV * 2, d_vmeta_[l] + (size_t)from * nKV * 2, (size_t)len * nKV * 2 * sizeof(unsigned short), cudaMemcpyDeviceToDevice, st);
        }
    }
}

// Batched per-Layer-Forward über N Spalten [feature][N] -> d_logitsb [V][N]. Jedes Gewicht streamt
// EINMAL pro Layer (fused_int3_gemm_streamed_dev), wird auf alle N Spalten angewandt (Amortisierung).
void GpuForward::run_layers_batch(const int* tokens, int N, int start_pos,
                                  const int* d_pos, const unsigned char* d_mask) {
    const int H = cfg_.hidden, nH = cfg_.n_heads, nKV = cfg_.n_kv_heads, hd = cfg_.head_dim;
    const int kvd = cfg_.kv_dim(), F = cfg_.ffn;
    const float eps = cfg_.rms_eps, theta = cfg_.rope_theta, scale = 1.0f / sqrtf((float)hd);
    const int TB = 256;
    cudaStream_t st = (cudaStream_t)stream_;   // Phase 3.2: Compute-Stream
    auto grid = [&](long long n) { return (int)((n + TB - 1) / TB); };
    // Embedding: N Spalten via Host-dequant_row -> [H][N] -> 1× H2D.
    for (int n = 0; n < N; ++n) {
        dequant_row(*gw_.embd, tokens[n], h_embed_.data());
        for (int i = 0; i < H; ++i) h_embed_b_[(size_t)i * N + n] = h_embed_[i];
    }
    cudaMemcpyAsync(d_xb, h_embed_b_.data(), (size_t)H * N * sizeof(float), cudaMemcpyHostToDevice, st);
    for (int l = 0; l < cfg_.n_layers; ++l) {
        rmsnorm_batch_kernel<<<N, TB, 0, st>>>(d_xb, d_attn_norm_[l], H, N, eps, d_xnb);
        fused_int3_gemm_streamed_dev(*gw_.q[l], d_xnb, N, d_qb, st);
        fused_int3_gemm_streamed_dev(*gw_.k[l], d_xnb, N, d_kb, st);
        fused_int3_gemm_streamed_dev(*gw_.v[l], d_xnb, N, d_vb, st);
        rope_batch_kernel<<<grid((long long)N * nH * (hd / 2)), TB, 0, st>>>(d_qb, N, start_pos, nH, hd, theta, d_pos);
        rope_batch_kernel<<<grid((long long)N * nKV * (hd / 2)), TB, 0, st>>>(d_kb, N, start_pos, nKV, hd, theta, d_pos);
        if (kv_bits_ != 0) {   // Q und K gleich rotieren (Q·K bleibt erhalten); K wird rotiert-quantisiert
            rotate_heads_kernel<<<dim3(N, nH),  hd, hd * sizeof(float), st>>>(d_qb, d_rot_, N, nH,  hd);
            rotate_heads_kernel<<<dim3(N, nKV), hd, hd * sizeof(float), st>>>(d_kb, d_rot_, N, nKV, hd);
        }
        const float *Ksrc, *Vsrc;
        if (kv_bits_ == 0) {
            kv_append_batch_kernel<<<grid((long long)N * kvd), TB, 0, st>>>(d_kb, d_vb, N, start_pos, kvd,
                                                                    d_kcache_[l], d_vcache_[l]);
            Ksrc = d_kcache_[l]; Vsrc = d_vcache_[l];
        } else {   // quantisiert schreiben, dann Layer just-in-time nach FP32-Scratch dequantisieren
            const int sbits = (kv_bits_ == 2) ? 3 : kv_bits_, qjl = (kv_bits_ == 2) ? 1 : 0;
            kv_quant_kernel<<<grid((long long)N * nKV), TB, 0, st>>>(d_kb, N, start_pos, nKV, hd, kvd, sbits, qjl, d_kcode_[l], d_kmeta_[l]);
            kv_quant_kernel<<<grid((long long)N * nKV), TB, 0, st>>>(d_vb, N, start_pos, nKV, hd, kvd, sbits, qjl, d_vcode_[l], d_vmeta_[l]);
            const int T = start_pos + N;
            kv_dequant_kernel<<<grid((long long)T * nKV), TB, 0, st>>>(d_kcode_[l], d_kmeta_[l], T, nKV, hd, kvd, sbits, qjl, d_kdq_);
            kv_dequant_kernel<<<grid((long long)T * nKV), TB, 0, st>>>(d_vcode_[l], d_vmeta_[l], T, nKV, hd, kvd, sbits, qjl, d_vdq_);
            Ksrc = d_kdq_; Vsrc = d_vdq_;
        }
        dim3 ablk(nH, N);
        attn_batch_kernel<<<ablk, 128, 0, st>>>(d_qb, Ksrc, Vsrc,   // flash: O(head_dim)-Shared, langer Kontext
                                         nH, nKV, hd, N, start_pos, kvd, scale, d_ctxb, d_mask);
        fused_int3_gemm_streamed_dev(*gw_.o[l], d_ctxb, N, d_tmpb, st);
        add_batch_kernel<<<grid((long long)H * N), TB, 0, st>>>(d_xb, d_tmpb, (long long)H * N);
        rmsnorm_batch_kernel<<<N, TB, 0, st>>>(d_xb, d_ffn_norm_[l], H, N, eps, d_xnb);
        fused_int3_gemm_streamed_dev(*gw_.gate[l], d_xnb, N, d_gb, st);
        fused_int3_gemm_streamed_dev(*gw_.up[l], d_xnb, N, d_ub, st);
        silu_mul_batch_kernel<<<grid((long long)F * N), TB, 0, st>>>(d_gb, d_ub, (long long)F * N);
        fused_int3_gemm_streamed_dev(*gw_.down[l], d_gb, N, d_tmpb, st);
        add_batch_kernel<<<grid((long long)H * N), TB, 0, st>>>(d_xb, d_tmpb, (long long)H * N);
        if (hid_capture_) {   // Phase 6B: Layer-l-Hidden abgreifen, wenn l ein Tap ist
            for (int k = 0; k < 3; ++k)
                if (l == hidtap_[k] && d_hid_[k])
                    cudaMemcpyAsync(d_hid_[k], d_xb, (size_t)H * N * sizeof(float), cudaMemcpyDeviceToDevice, st);
        }
    }
    rmsnorm_batch_kernel<<<N, TB, 0, st>>>(d_xb, d_out_norm_, H, N, eps, d_xnb);
    fused_int3_gemm_streamed_dev(*gw_.out, d_xnb, N, d_logitsb, st);
}

void GpuForward::forward_batch_gpu(const int* tokens, int N, int start_pos, float* host_out,
                                   const int* tree_pos, const unsigned char* tree_mask) {
    if (N > MAX_BATCH) N = MAX_BATCH;                         // gemm_i8 acc[32]
    const int V = cfg_.vocab;
    cudaStream_t st = (cudaStream_t)stream_;
    // Phase 6A: Baum-Positionen + Ahnen-Maske hochladen (klein), sonst linear (nullptr).
    const int*           dp = nullptr;
    const unsigned char* dm = nullptr;
    if (tree_pos && d_treepos_)  { cudaMemcpyAsync(d_treepos_,  tree_pos,  (size_t)N * sizeof(int), cudaMemcpyHostToDevice, st); dp = d_treepos_; }
    if (tree_mask && d_treemask_){ cudaMemcpyAsync(d_treemask_, tree_mask, (size_t)N * N,           cudaMemcpyHostToDevice, st); dm = d_treemask_; }
    run_layers_batch(tokens, N, start_pos, dp, dm);
    std::vector<float> tmp((size_t)V * N);                    // d_logitsb [V][N] -> host [N][V]
    cudaMemcpyAsync(tmp.data(), d_logitsb, (size_t)V * N * sizeof(float), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);                                // host-Readback fertig
    for (int n = 0; n < N; ++n)
        for (int vv = 0; vv < V; ++vv) host_out[(size_t)n * V + vv] = tmp[(size_t)vv * N + n];
    seq_ = start_pos + N;
}

void GpuForward::free() {
    auto fr = [](float*& p) { if (p) { cudaFree(p); p = nullptr; } };
    fr(d_x); fr(d_xn); fr(d_q); fr(d_k); fr(d_v); fr(d_ctx); fr(d_tmp); fr(d_g); fr(d_u); fr(d_logits);
    fr(d_xb); fr(d_xnb); fr(d_qb); fr(d_kb); fr(d_vb); fr(d_ctxb); fr(d_tmpb); fr(d_gb); fr(d_ub); fr(d_logitsb);
    if (d_arg) { cudaFree(d_arg); d_arg = nullptr; }
    if (d_treepos_)  { cudaFree(d_treepos_);  d_treepos_ = nullptr; }    // Phase 6A
    if (d_treemask_) { cudaFree(d_treemask_); d_treemask_ = nullptr; }
    for (int k = 0; k < 3; ++k) if (d_hid_[k]) { cudaFree(d_hid_[k]); d_hid_[k] = nullptr; }   // Phase 6B
    for (auto& p : d_attn_norm_) fr(p);
    for (auto& p : d_ffn_norm_) fr(p);
    fr(d_out_norm_);
    for (auto& p : d_kcache_) fr(p);
    for (auto& p : d_vcache_) fr(p);
    auto fru = [](unsigned char*& p) { if (p) { cudaFree(p); p = nullptr; } };
    auto frh = [](unsigned short*& p) { if (p) { cudaFree(p); p = nullptr; } };
    for (auto& p : d_kcode_) fru(p); for (auto& p : d_vcode_) fru(p);
    for (auto& p : d_kmeta_) frh(p); for (auto& p : d_vmeta_) frh(p);
    fr(d_kdq_); fr(d_vdq_); fr(d_rot_);
    d_attn_norm_.clear(); d_ffn_norm_.clear(); d_kcache_.clear(); d_vcache_.clear();
    d_kcode_.clear(); d_vcode_.clear(); d_kmeta_.clear(); d_vmeta_.clear();
    if (stream_) { cudaStreamDestroy((cudaStream_t)stream_); stream_ = nullptr; }   // Phase 3.2
    ready_ = false;
}

}  // namespace nova::infer

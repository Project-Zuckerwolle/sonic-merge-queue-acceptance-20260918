// fused_gemm.cu — CUDA INT3-Dequant-GEMV/GEMM, koalescierter Zwei-Pass (Kernel-Upgrade ③).
//
// Der alte fused Kernel entpackte die 3-Bit-Werte byte-straddelnd INLINE im Matmul (dget3, 3 Byte-
// Reads/Wert, ein Thread je Zeile) — vollständig unkoalesciert, ~7× langsamer als der reine Transfer.
// Neu: zwei koalescierte Pässe.
//   Pass 1  unpack_int3_kernel : INT3 → INT8 (q-4) in einen VRAM-Scratch, ein Thread je Element
//                                (koalescierte Writes, byte-lokale Reads).
//   Pass 2  gemv_i8 / gemm_i8  : ein WARP je Output-Zeile, 32 Lanes lesen U[row*K + k] koalesciert
//                                über K und reduzieren per __shfl_down.
// Ist die Matrix VRAM-resident (d_unpacked gesetzt — das permanente 3B-Draft), entfallen H2D + Pass 1.
// Numerik bit-nah zur Host-Referenz fused_int3_gemv (INT8 exakt, float-Skalen·x, Baum-Reduktion):
// Cosine>0,99, argmax stabil (test_fused_gemm + RealInference-A/A2/D).
#include "InferEngine/fused_gemm.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdlib>

namespace nova::infer {

namespace {

// Phase 4 (C1 Umbau): NOVA_PINNED_STREAM. Die gestreamten 24B-Host-Puffer (packed + Skalen) werden einmalig
// via cudaHostRegister in-place gepinnt -> H2D nutzt DMA (pinned). GEMESSEN (2026-07-11): pageable 4,22 GB/s ->
// pinned 5,74 GB/s (527 Matrizen ok, 156 fail am Windows-Lock-Limit ~7,25 GB) = +25% Transfer = +9-20% tok/s,
// LOSSLESS. Deshalb DEFAULT AN (der Verify ist ~90% des Steps, Transfer ~88% des Verify). NOVA_PINNED_STREAM=0
// schaltet ab. Schlägt ein Pin fehl (Aggregat-Limit), bleibt der Puffer pageable -> graceful degradation.
const bool g_pinned_stream = []() {
    const char* e = std::getenv("NOVA_PINNED_STREAM");
    return !e || (e[0] && e[0] != '0');   // Default AN (nur =0 schaltet ab)
}();
// Perf Phase 3: NOVA_STREAM_OVERLAP. Doppelpuffer-Scratch + dedizierter Copy-Stream: der H2D+Unpack
// des NÄCHSTEN Gewichts überlappt den Compute (GEMM+Attention+Norm) des aktuellen — der PCIe-Link
// bleibt busy statt zwischen den synchronen Copies zu idlen. Setzt gepinnte Host-Puffer voraus
// (cudaMemcpyAsync ist nur pinned wirklich asynchron). Default aus -> alter synchroner Pfad.
const bool g_stream_overlap = []() {
    const char* e = std::getenv("NOVA_STREAM_OVERLAP");
    return e && e[0] && e[0] != '0';
}();
// Phase 3: Pin-Buchhaltung — cudaHostRegister-Ergebnis PRÜFEN statt blind verwerfen. So ist
// belegbar, ob alle gestreamten Puffer wirklich pinned sind (sonst fällt H2D pageable-langsam zurück).
// U1-Befund (gemessen): die 156/683 Fails sind KEIN Page-Sharing (Dedup half nicht, already=0), sondern
// err=2 (cudaErrorMemoryAllocation / OOM) auf den GROSSEN Gewicht-Puffern — das Pinnen der vollen ~9,4 GB
// Gewichte überschreitet das Windows-Lock-Limit (~5-6 GB). GETESTET + VERWORFEN: das Prozess-Working-Set-
// Minimum via SetProcessWorkingSetSizeEx(HARDWS_MIN, 12 GB) anzuheben macht es SCHLECHTER (ok=133/fail=550)
// und bricht den Decode (0 Tokens) — das Hard-Min reserviert physische Seiten, um die cudaHostRegister dann
// konkurriert; Working-Set ist für MmProbeAndLockPages der falsche Hebel. All-pinned bräuchte die Gewichte
// direkt in cudaHostAlloc-Speicher (Loader-Änderung) = die moderate U1≡U2-Arbeit. try_pin bleibt best-effort.
long g_pin_ok = 0, g_pin_fail = 0, g_pin_already = 0;
inline void try_pin(const void* p, size_t n) {
    if (!p || !n) return;
    const cudaError_t e = cudaHostRegister(const_cast<void*>(p), n, cudaHostRegisterDefault);
    if (e == cudaSuccess) { ++g_pin_ok; }
    else if (e == cudaErrorHostMemoryAlreadyRegistered) { ++g_pin_already; }
    else { ++g_pin_fail; }   // Rest-OOM (Lock-Limit): bleibt pageable (graceful degradation)
    cudaGetLastError();      // sticky error klären
}

// --- Phase 2 FMT_BLOCK3: zweistufige Skalen -> FP32-Effektivskalen [M*ng] expandieren ----------
// Ein Thread je (Zeile,Gruppe). Effektive Skala = superscale/63 * subscale_q6. Läuft einmal je
// Matrix im Streaming-Pfad (vernachlässigbar gegen den Transfer); die Pass-2-Kernel bleiben unverändert.
__global__ void expand_block3_scales_kernel(const unsigned short* __restrict__ super,
                                            const unsigned char* __restrict__ sub,
                                            int M, int ng, int nsb, float* __restrict__ out) {
    const long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= (long long)M * ng) return;
    const int i  = int(idx / ng);
    const int g  = int(idx % ng);
    const int sb = g >> 3;          // 8 Gruppen je Superblock
    const int j  = g & 7;
    const float supf = __half2float(__ushort_as_half(super[(size_t)i * nsb + sb]));
    const size_t bit = (((size_t)i * nsb + sb) * 6) * 8 + (size_t)j * 6;
    unsigned int v = 0;
#pragma unroll
    for (int b = 0; b < 6; ++b) { const size_t q = bit + b; if ((sub[q >> 3] >> (q & 7)) & 1u) v |= (1u << b); }
    out[idx] = supf / 63.0f * float(v);
}

// --- Pass 1: INT3 (byte-straddelnd) -> INT8 (q-4, -4..3), ein Thread je Element ----------------
__global__ void unpack_int3_kernel(const unsigned char* __restrict__ packed,
                                   long long total, signed char* __restrict__ out) {
    const long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const unsigned long long bp = (unsigned long long)idx * 3;
    int v = 0;
    for (int b = 0; b < 3; ++b) {
        const unsigned long long q = bp + b;
        if ((packed[q >> 3] >> (q & 7)) & 1u) v |= (1 << b);
    }
    out[idx] = (signed char)(v - 4);   // -4..3
}

// Phase 3: generalisiertes Unpack für den residenten Draft (bits 3..6). Wert v-offset (offset=2^(bits-1))
// passt in INT8; der gemv_i8_kernel (INT8·scale) bleibt unverändert. bits=3,offset=4 == unpack_int3_kernel.
__global__ void unpack_nbit_kernel(const unsigned char* __restrict__ packed, long long total,
                                   int bits, int offset, signed char* __restrict__ out) {
    const long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const unsigned long long bp = (unsigned long long)idx * (unsigned)bits;
    int v = 0;
    for (int b = 0; b < bits; ++b) {
        const unsigned long long q = bp + b;
        if ((packed[q >> 3] >> (q & 7)) & 1u) v |= (1 << b);
    }
    out[idx] = (signed char)(v - offset);
}

// --- Pass 2 GEMV: y[i] = Σ_k U[i*K+k] · scale[i][k/group] · x[k], ein Warp je Zeile -------------
__global__ void gemv_i8_kernel(const signed char* __restrict__ U, const float* __restrict__ scales,
                               const float* __restrict__ x, int M, int K, int group, int ng,
                               float* __restrict__ y) {
    const int gid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = gid >> 5;
    const int lane = gid & 31;
    if (warp >= M) return;
    const signed char* row = U + (size_t)warp * K;
    const float* srow = scales + (size_t)warp * ng;
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32)          // Lanes lesen zusammenhängende k -> koalesciert
        acc += (float)row[k] * srow[k / group] * x[k];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[warp] = acc;
}

// --- Pass 2 GEMM: Y[i*N+n] = Σ_k U[i*K+k]·scale·X[k*N+n], ein Warp je Zeile, ≤32 Spalten je Block -
// Phase 4: N-Kachelung über blockIdx.y (je Block eine 32-Spalten-Kachel) hebt den acc[32]-Cap ->
// beliebiges N (Prefill-Chunks, Tree-Breite). Für N≤32 == altes Verhalten (grid.y=1). Numerik identisch.
__global__ void gemm_i8_kernel(const signed char* __restrict__ U, const float* __restrict__ scales,
                               const float* __restrict__ X, int M, int K, int N, int group, int ng,
                               float* __restrict__ Y) {
    const int gid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = gid >> 5;
    const int lane = gid & 31;
    if (warp >= M) return;
    const int n0   = blockIdx.y * 32;                       // Start-Spalte dieser Kachel
    const int ncol = (N - n0) < 32 ? (N - n0) : 32;         // ≤32 Spalten je Block
    if (ncol <= 0) return;
    const signed char* row = U + (size_t)warp * K;
    const float* srow = scales + (size_t)warp * ng;
    float acc[32];
    for (int n = 0; n < ncol; ++n) acc[n] = 0.0f;
    for (int k = lane; k < K; k += 32) {
        const float qs = (float)row[k] * srow[k / group];
        const float* xr = X + (size_t)k * N + n0;
        for (int n = 0; n < ncol; ++n) acc[n] += qs * xr[n];
    }
    for (int n = 0; n < ncol; ++n) {
        float a = acc[n];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
        if (lane == 0) Y[(size_t)warp * N + n0 + n] = a;
    }
}

// ================================================================================================
// Phase 4a (NOVA_GEMM_FAST): schneller INT8-GEMM via __dp4a + Shared-Memory-Weight-Reuse.
// Der alte gemm_i8_kernel castet INT8→float und macht FP32-FMA; die N-Kachelung (grid.y) liest jede
// Gewichtszeile ⌈N/32⌉× NEU aus dem VRAM (bei N=512: 16×) — das ist der Batched-Compute-Flaschenhals
// (Perf-Log: 260 GFLOP/s, <1% des INT8-Peaks). Neu:
//   (1) INT8-Aktivierungs-Quantizer: X[K][N] (FP32) → X8[N][K] (INT8, K-contig) + act_scale[N]
//       (pro Spalte symmetrisch amax/127). Transponiert zugleich, damit dp4a 4 K-contige Bytes liest.
//   (2) Tiled dp4a-GEMM: ein Block rechnet BM×BN Ausgaben, lädt je K-Kachel (BK=group=32) eine
//       Gewicht- UND Aktivierungs-Kachel EINMAL in Shared-Mem und reused sie über die ganze Kachel →
//       kein 16×-Neulesen. Inner-Loop: __dp4a(int8x4,int8x4)→int32 je Gruppe, dann ×weight_scale[m][g].
//       Epilog: ×act_scale[n]. INT8-Aktivierungen sind verlustbehaftet → cosine>0,99 (nicht bit-exakt),
//       daher flag-gated & regime-getrennt (Prefill/Dump/Tree; greedy-exakter Decode-Verify bleibt FP32).
const bool g_gemm_fast = []() {
    const char* e = std::getenv("NOVA_GEMM_FAST");
    return e && e[0] && e[0] != '0';
}();
// Regime-Trennung: der dp4a-Pfad ist INT8-Akt-verlustbehaftet (cosine 0,99998) → NUR für großes N
// (Prefill/Dump/Tree). Der greedy-exakte Decode-Verify (N=K≤8) MUSS FP32 bleiben, sonst kippen einzelne
// argmax-Tokens (A/A2/D nicht mehr token-identisch). NOVA_GEMM_FAST_MINN (Default 64) ist die Schwelle,
// ab der der gestreamte Forward auf den Fast-Kernel schaltet. (fused_int3_gemm_cuda für den Gate-Test
// nutzt den Fast-Pfad schwellenlos, damit die Korrektheit bei jedem N geprüft wird.)
const int g_gemm_fast_minn = []() {
    const char* e = std::getenv("NOVA_GEMM_FAST_MINN");
    const int v = e ? std::atoi(e) : 64;
    return v > 0 ? v : 64;
}();

// (1) Quantizer: ein Block je Spalte n; amax-Reduktion dann Quantisierung. X[k*N+n] → X8[n*K+k].
__global__ void quantize_act_kernel(const float* __restrict__ X, int K, int N,
                                    signed char* __restrict__ X8, float* __restrict__ act_scale) {
    const int n = blockIdx.x;
    if (n >= N) return;
    __shared__ float sh[256];
    float amax = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) { const float v = fabsf(X[(size_t)k * N + n]); if (v > amax) amax = v; }
    sh[threadIdx.x] = amax; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (threadIdx.x < s) { if (sh[threadIdx.x + s] > sh[threadIdx.x]) sh[threadIdx.x] = sh[threadIdx.x + s]; } __syncthreads(); }
    const float scale = sh[0] > 0.0f ? sh[0] / 127.0f : 1.0f;
    if (threadIdx.x == 0) act_scale[n] = scale;
    const float inv = 1.0f / scale;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        int q = __float2int_rn(X[(size_t)k * N + n] * inv);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        X8[(size_t)n * K + k] = (signed char)q;
    }
}

// (2) Tiled dp4a-GEMM. BM×BN Ausgabe-Kachel je Block, BK=32 (=group). blockDim = 16×16 = 256 Threads,
// jeder Thread rechnet TM×TN = 2×2 Ausgaben. U8[M][K] (K-contig), X8[N][K] (K-contig), scale[M][ng].
#define GF_BM 32
#define GF_BN 32
#define GF_BK 32
#define GF_TM 2
#define GF_TN 2
__global__ void gemm_dp4a_kernel(const signed char* __restrict__ U8, const float* __restrict__ wscale,
                                 const signed char* __restrict__ X8, const float* __restrict__ act_scale,
                                 int M, int K, int N, int ng, float* __restrict__ Y) {
    __shared__ signed char Us[GF_BM][GF_BK];   // 32×32 = 1 KB
    __shared__ signed char Xs[GF_BN][GF_BK];   // 32×32 = 1 KB
    const int row0 = blockIdx.y * GF_BM;
    const int col0 = blockIdx.x * GF_BN;
    const int tx = threadIdx.x, ty = threadIdx.y;   // 0..15
    const int tid = ty * blockDim.x + tx;           // 0..255
    float facc[GF_TM][GF_TN];
    #pragma unroll
    for (int i = 0; i < GF_TM; ++i) for (int j = 0; j < GF_TN; ++j) facc[i][j] = 0.0f;
    const int ngroups = K / GF_BK;                  // K ist Vielfaches von 32 (Fast-Pfad-Vorbedingung)
    for (int g = 0; g < ngroups; ++g) {
        const int k0 = g * GF_BK;
        // Kacheln laden: 1024 Bytes je Tile, 256 Threads → 4 Elemente/Thread.
        #pragma unroll
        for (int e = 0; e < (GF_BM * GF_BK) / 256; ++e) {
            const int idx = tid + e * 256;
            const int r = idx / GF_BK, c = idx % GF_BK;
            const int gr = row0 + r;
            Us[r][c] = (gr < M) ? U8[(size_t)gr * K + k0 + c] : (signed char)0;
        }
        #pragma unroll
        for (int e = 0; e < (GF_BN * GF_BK) / 256; ++e) {
            const int idx = tid + e * 256;
            const int r = idx / GF_BK, c = idx % GF_BK;
            const int gc = col0 + r;
            Xs[r][c] = (gc < N) ? X8[(size_t)gc * K + k0 + c] : (signed char)0;
        }
        __syncthreads();
        // dp4a je (mi,ni)-Mikro-Ausgabe über die 32 k dieser Gruppe (8 dp4a).
        #pragma unroll
        for (int i = 0; i < GF_TM; ++i) {
            const int r = ty * GF_TM + i;
            #pragma unroll
            for (int j = 0; j < GF_TN; ++j) {
                const int c = tx * GF_TN + j;
                int iacc = 0;
                #pragma unroll
                for (int kk = 0; kk < GF_BK; kk += 4) {
                    const int uu = *reinterpret_cast<const int*>(&Us[r][kk]);
                    const int xx = *reinterpret_cast<const int*>(&Xs[c][kk]);
                    iacc = __dp4a(uu, xx, iacc);
                }
                const int gr = row0 + r;
                if (gr < M) facc[i][j] += (float)iacc * wscale[(size_t)gr * ng + g];
            }
        }
        __syncthreads();
    }
    // Epilog: ×act_scale[n], schreiben.
    #pragma unroll
    for (int i = 0; i < GF_TM; ++i) {
        const int gr = row0 + ty * GF_TM + i;
        if (gr >= M) continue;
        #pragma unroll
        for (int j = 0; j < GF_TN; ++j) {
            const int gc = col0 + tx * GF_TN + j;
            if (gc < N) Y[(size_t)gr * N + gc] = facc[i][j] * act_scale[gc];
        }
    }
}

// Persistente, bei Bedarf wachsende Device-Puffer — vermeiden cudaMalloc/Free pro Aufruf
// (~1000/Token). Aufrufe erfolgen seriell aus forward_step/-batch, daher kein Lock.
unsigned char* g_packed = nullptr;   size_t g_packed_cap = 0;   // H2D gepackt (Streaming-24B)
signed char*   g_unpacked = nullptr; size_t g_unpacked_cap = 0; // INT8-Scratch (Streaming-24B)
float* g_scales = nullptr; size_t g_scales_cap = 0;
unsigned short* g_super = nullptr; size_t g_super_cap = 0;      // FMT_BLOCK3: FP16-Superskalen (H2D)
unsigned char*  g_sub   = nullptr; size_t g_sub_cap   = 0;      // FMT_BLOCK3: 6-Bit-Subskalen (H2D)
float* g_x = nullptr; size_t g_x_cap = 0;
float* g_y = nullptr; size_t g_y_cap = 0;
float* g_X = nullptr; size_t g_X_cap = 0;
float* g_Y = nullptr; size_t g_Y_cap = 0;
signed char* g_X8 = nullptr; size_t g_X8_cap = 0;        // Phase 4a: INT8-quantisierte Aktivierung [N][K]
float* g_actscale = nullptr; size_t g_actscale_cap = 0;  // Phase 4a: act_scale[N]
template <class T> void ensure(T*& p, size_t& cap, size_t need) {
    if (need > cap) { if (p) cudaFree(p); cudaMalloc(reinterpret_cast<void**>(&p), need); cap = need; }
}

// Phase 4a Launcher: quantisiert Xdev[K][N] → g_X8[N][K] + act_scale, dann tiled dp4a-GEMM → Ydev[M][N].
// Vorbedingung: group==32 && K%32==0 (24B-FFN erfüllt beides). Numerik: cosine>0,99 vs FP32-GEMV.
void gemm_fast_launch(const signed char* U8, const float* wscale, const float* Xdev,
                      int M, int K, int N, int ng, float* Ydev, cudaStream_t s) {
    ensure(g_X8, g_X8_cap, (size_t)N * K);
    ensure(g_actscale, g_actscale_cap, (size_t)N * sizeof(float));
    quantize_act_kernel<<<N, 256, 0, s>>>(Xdev, K, N, g_X8, g_actscale);
    dim3 grid((N + GF_BN - 1) / GF_BN, (M + GF_BM - 1) / GF_BM);
    dim3 block(GF_BN / GF_TN, GF_BM / GF_TM);   // 16×16 = 256
    gemm_dp4a_kernel<<<grid, block, 0, s>>>(U8, wscale, g_X8, g_actscale, M, K, N, ng, Ydev);
}

// --- Instrumentierung (Messplan Phase 2): Event-Timing je Stufe, Standard AUS -------------------
bool        g_instr_on = false;
GemmInstr   g_instr;
cudaEvent_t g_ev0 = nullptr, g_ev1 = nullptr;
inline void  instr_begin() { if (g_instr_on) { if (!g_ev0) { cudaEventCreate(&g_ev0); cudaEventCreate(&g_ev1); } cudaEventRecord(g_ev0, 0); } }
inline float instr_end()   { cudaEventRecord(g_ev1, 0); cudaEventSynchronize(g_ev1); float ms = 0; cudaEventElapsedTime(&ms, g_ev0, g_ev1); return ms; }

// Liefert die INT8-/Skalen-Device-Zeiger für W: resident direkt, sonst H2D packed + Pass-1-Entpacken.
void resolve_device(const Int3Matrix& W, const signed char*& U, const float*& scales) {
    if (W.d_unpacked) { U = (const signed char*)W.d_unpacked; scales = (const float*)W.d_scales; return; }
    const long long total = (long long)W.M * W.K;
    const int ng = W.groups_per_row();
    // Phase 4: einmalig die Host-Puffer pinnen (in-place). Danach laufen alle H2D dieser Matrix pinned.
    if (g_pinned_stream && !W.host_pinned) {
        W.host_pinned = true;   // nur ein Versuch, auch bei Teilfehler
        try_pin(W.packed.data(), W.packed.size());
        if (W.fmt == SCALE_BLOCK3) {
            try_pin(W.superscales.data(), W.superscales.size() * sizeof(unsigned short));
            try_pin(W.subscales.data(), W.subscales.size());
        } else {
            try_pin(W.scales.data(), W.scales.size() * sizeof(float));
        }
    }
    ensure(g_packed, g_packed_cap, W.packed.size());
    ensure(g_unpacked, g_unpacked_cap, (size_t)total);
    ensure(g_scales, g_scales_cap, (size_t)W.M * ng * sizeof(float));
    // Gewichte (bit-identisch in beiden Formaten): H2D packed.
    instr_begin();
    cudaMemcpy(g_packed, W.packed.data(), W.packed.size(), cudaMemcpyHostToDevice);
    if (g_instr_on) { g_instr.h2d_packed_ms += instr_end(); g_instr.h2d_packed_bytes += double(W.packed.size()); }
    // Skalen: FMT_BLOCK3 -> kompakt H2D + Expand-Kernel; sonst FP32 direkt.
    if (W.fmt == SCALE_BLOCK3) {
        const int nsb = W.superblocks_per_row();
        ensure(g_super, g_super_cap, W.superscales.size() * sizeof(unsigned short));
        ensure(g_sub,   g_sub_cap,   W.subscales.size());
        instr_begin();
        cudaMemcpy(g_super, W.superscales.data(), W.superscales.size() * sizeof(unsigned short), cudaMemcpyHostToDevice);
        cudaMemcpy(g_sub,   W.subscales.data(),   W.subscales.size(),                            cudaMemcpyHostToDevice);
        if (g_instr_on) {
            g_instr.h2d_scales_ms += instr_end();
            g_instr.h2d_scales_bytes += double(W.superscales.size() * sizeof(unsigned short) + W.subscales.size());
        }
        const long long ns = (long long)W.M * ng;
        expand_block3_scales_kernel<<<(ns + 255) / 256, 256>>>(g_super, g_sub, W.M, ng, nsb, g_scales);
    } else {
        instr_begin();
        cudaMemcpy(g_scales, W.scales.data(), W.scales.size() * sizeof(float), cudaMemcpyHostToDevice);
        if (g_instr_on) { g_instr.h2d_scales_ms += instr_end(); g_instr.h2d_scales_bytes += double(W.scales.size() * sizeof(float)); }
    }
    const int t = 256;
    instr_begin();
    // block3 ist 3-Bit (bestehender Pfad). nbit-Matrizen (z.B. 4-Bit Mixed-Attention) MÜSSEN mit
    // W.bits entpackt werden — vorher hart unpack_int3_kernel => 4-Bit-Gewichte wurden als 3-Bit
    // gelesen => Garbage. (Der residente Pfad machte das via unpack_nbit_kernel bereits richtig.)
    if (W.fmt == SCALE_BLOCK3) {
        unpack_int3_kernel<<<(total + t - 1) / t, t>>>(g_packed, total, g_unpacked);
    } else {
        const int bits = W.bits > 0 ? W.bits : 3;
        unpack_nbit_kernel<<<(total + t - 1) / t, t>>>(g_packed, total, bits, 1 << (bits - 1), g_unpacked);
    }
    if (g_instr_on) { g_instr.unpack_ms += instr_end(); ++g_instr.unpack_calls; }
    U = g_unpacked; scales = g_scales;
}

// --- Phase 3.3: Doppelpuffer-Pipeline (NOVA_STREAM_OVERLAP) --------------------------------------
// Zwei rotierende Scratch-Slots + separater Copy-Stream. Der Compute-Stream (aus GpuForward, NICHT
// mehr Null-Stream nach Phase 3.2) wartet per Event auf das Gewicht; der Copy-Stream wartet per Event
// auf die vorige GEMM des Slots. Weil Compute auf einem echten Stream läuft, überlappt H2D(M+1) auf
// dem Copy-Stream echt mit GEMM+Attn+Norm(M) auf dem Compute-Stream. Nur mit gepinnten Puffern async.
const bool g_stream_debug = []() {
    const char* e = std::getenv("NOVA_STREAM_DEBUG");
    return e && e[0] && e[0] != '0';
}();
inline void dbg_check(const char* where) {
    if (!g_stream_debug) return;
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) std::fprintf(stderr, "[STREAM_DEBUG] %s: %s\n", where, cudaGetErrorString(e));
}
struct StreamSlot {
    unsigned char* packed = nullptr;   size_t packed_cap = 0;
    signed char*   unpacked = nullptr; size_t unpacked_cap = 0;
    float*         scales = nullptr;   size_t scales_cap = 0;
    unsigned short* super = nullptr;   size_t super_cap = 0;
    unsigned char*  sub = nullptr;     size_t sub_cap = 0;
    cudaEvent_t    copy_done = nullptr, gemm_done = nullptr;
    bool           gemm_pending = false;
};
StreamSlot   g_slot[2];
int          g_slot_cur = 0;
cudaStream_t g_copy_stream = nullptr;
bool         g_pipe_init = false;
void pipe_init() {
    if (g_pipe_init) return;
    cudaStreamCreate(&g_copy_stream);
    for (int i = 0; i < 2; ++i) {
        cudaEventCreateWithFlags(&g_slot[i].copy_done, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&g_slot[i].gemm_done, cudaEventDisableTiming);
    }
    g_pipe_init = true;
}
// Streamt W async in einen rotierenden Slot; `compute` (echter Stream) wartet aufs Gewicht. Slot-Index zurück.
int resolve_device_pipe(const Int3Matrix& W, const signed char*& U, const float*& scales, cudaStream_t compute) {
    pipe_init();
    const long long total = (long long)W.M * W.K;
    const int ng = W.groups_per_row();
    if (g_pinned_stream && !W.host_pinned) {
        W.host_pinned = true;
        try_pin(W.packed.data(), W.packed.size());
        if (W.fmt == SCALE_BLOCK3) {
            try_pin(W.superscales.data(), W.superscales.size() * sizeof(unsigned short));
            try_pin(W.subscales.data(), W.subscales.size());
        } else {
            try_pin(W.scales.data(), W.scales.size() * sizeof(float));
        }
    }
    const int slot = g_slot_cur; g_slot_cur ^= 1;
    StreamSlot& S = g_slot[slot];
    if (S.gemm_pending) cudaStreamWaitEvent(g_copy_stream, S.gemm_done, 0);   // Puffer erst frei nach voriger GEMM
    ensure(S.packed,   S.packed_cap,   W.packed.size());
    ensure(S.unpacked, S.unpacked_cap, (size_t)total);
    ensure(S.scales,   S.scales_cap,   (size_t)W.M * ng * sizeof(float));
    cudaMemcpyAsync(S.packed, W.packed.data(), W.packed.size(), cudaMemcpyHostToDevice, g_copy_stream);
    if (W.fmt == SCALE_BLOCK3) {
        const int nsb = W.superblocks_per_row();
        ensure(S.super, S.super_cap, W.superscales.size() * sizeof(unsigned short));
        ensure(S.sub,   S.sub_cap,   W.subscales.size());
        cudaMemcpyAsync(S.super, W.superscales.data(), W.superscales.size() * sizeof(unsigned short),
                        cudaMemcpyHostToDevice, g_copy_stream);
        cudaMemcpyAsync(S.sub,   W.subscales.data(),   W.subscales.size(), cudaMemcpyHostToDevice, g_copy_stream);
        const long long ns = (long long)W.M * ng;
        expand_block3_scales_kernel<<<(ns + 255) / 256, 256, 0, g_copy_stream>>>(S.super, S.sub, W.M, ng, nsb, S.scales);
    } else {
        cudaMemcpyAsync(S.scales, W.scales.data(), W.scales.size() * sizeof(float),
                        cudaMemcpyHostToDevice, g_copy_stream);
    }
    const int t = 256;
    if (W.fmt == SCALE_BLOCK3) {
        unpack_int3_kernel<<<(total + t - 1) / t, t, 0, g_copy_stream>>>(S.packed, total, S.unpacked);
    } else {
        const int bits = W.bits > 0 ? W.bits : 3;
        unpack_nbit_kernel<<<(total + t - 1) / t, t, 0, g_copy_stream>>>(S.packed, total, bits, 1 << (bits - 1), S.unpacked);
    }
    cudaEventRecord(S.copy_done, g_copy_stream);
    cudaStreamWaitEvent(compute, S.copy_done, 0);   // GEMM (compute) wartet auf sein Gewicht
    U = S.unpacked; scales = S.scales;
    return slot;
}

}  // namespace

void fused_int3_gemv_cuda(const Int3Matrix& W, const float* x, float* y) {
    const int ng = W.groups_per_row();
    ensure(g_x, g_x_cap, (size_t)W.K * sizeof(float));
    ensure(g_y, g_y_cap, (size_t)W.M * sizeof(float));
    instr_begin();
    cudaMemcpy(g_x, x, (size_t)W.K * sizeof(float), cudaMemcpyHostToDevice);
    if (g_instr_on) { g_instr.h2d_x_ms += instr_end(); g_instr.h2d_x_bytes += double((size_t)W.K * sizeof(float)); }
    const signed char* U; const float* scales;
    resolve_device(W, U, scales);
    const int threads = 128, warps_per_block = threads / 32;
    const int blocks = (W.M + warps_per_block - 1) / warps_per_block;
    instr_begin();
    gemv_i8_kernel<<<blocks, threads>>>(U, scales, g_x, W.M, W.K, W.group, ng, g_y);
    if (g_instr_on) { g_instr.kernel_ms += instr_end(); ++g_instr.kernel_calls; }
    instr_begin();
    cudaMemcpy(y, g_y, (size_t)W.M * sizeof(float), cudaMemcpyDeviceToHost);   // synchronisiert
    if (g_instr_on) { g_instr.d2h_ms += instr_end(); g_instr.d2h_bytes += double((size_t)W.M * sizeof(float)); ++g_instr.calls; }
}

void fused_int3_gemm_cuda(const Int3Matrix& W, const float* X, int N, float* Y) {
    if (N > 4096) { fused_int3_gemm(W, X, N, Y); return; }   // Host-Fallback nur für Extrem-N (Kernel kachelt N sonst)
    const int ng = W.groups_per_row();
    ensure(g_X, g_X_cap, (size_t)W.K * N * sizeof(float));
    ensure(g_Y, g_Y_cap, (size_t)W.M * N * sizeof(float));
    instr_begin();
    cudaMemcpy(g_X, X, (size_t)W.K * N * sizeof(float), cudaMemcpyHostToDevice);
    if (g_instr_on) { g_instr.h2d_x_ms += instr_end(); g_instr.h2d_x_bytes += double((size_t)W.K * N * sizeof(float)); }
    const signed char* U; const float* scales;
    resolve_device(W, U, scales);
    const int threads = 128, warps_per_block = threads / 32;
    const int blocks = (W.M + warps_per_block - 1) / warps_per_block;
    instr_begin();
    if (g_gemm_fast && W.group == 32 && (W.K % 32) == 0) {
        gemm_fast_launch(U, scales, g_X, W.M, W.K, N, ng, g_Y, 0);   // Phase 4a dp4a-Pfad
    } else {
        gemm_i8_kernel<<<dim3(blocks, (N + 31) / 32), threads>>>(U, scales, g_X, W.M, W.K, N, W.group, ng, g_Y);
    }
    if (g_instr_on) { g_instr.kernel_ms += instr_end(); ++g_instr.kernel_calls; }
    instr_begin();
    cudaMemcpy(Y, g_Y, (size_t)W.M * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (g_instr_on) { g_instr.d2h_ms += instr_end(); g_instr.d2h_bytes += double((size_t)W.M * N * sizeof(float)); ++g_instr.calls; }
}

// --- VRAM-Residenz (3B-Draft): einmal entpacken + Skalen laden, danach nur noch Pass 2 ---------
void int3_residentize(const Int3Matrix& W) {
    if (W.d_unpacked) return;                       // idempotent
    const long long total = (long long)W.M * W.K;
    unsigned char* dpk = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&dpk), W.packed.size());
    cudaMemcpy(dpk, W.packed.data(), W.packed.size(), cudaMemcpyHostToDevice);
    signed char* du = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&du), (size_t)total);
    const int t = 256;
    const int bits = W.bits > 0 ? W.bits : 3;
    unpack_nbit_kernel<<<(total + t - 1) / t, t>>>(dpk, total, bits, 1 << (bits - 1), du);  // Phase 3: bits 3..6
    cudaDeviceSynchronize();
    cudaFree(dpk);                                  // gepackt nach dem Entpacken nicht mehr nötig
    float* dsc = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&dsc), W.scales.size() * sizeof(float));
    cudaMemcpy(dsc, W.scales.data(), W.scales.size() * sizeof(float), cudaMemcpyHostToDevice);
    W.d_unpacked = du; W.d_scales = dsc;
}

void int3_free_resident(const Int3Matrix& W) {
    if (W.d_unpacked) { cudaFree(W.d_unpacked); W.d_unpacked = nullptr; }
    if (W.d_scales)   { cudaFree(W.d_scales);   W.d_scales = nullptr; }
}

// GEMV auf reinen Device-Zeigern (W resident, x/y im VRAM) — kein Memcpy, für den on-GPU-Forward.
void fused_int3_gemv_dev(const Int3Matrix& W, const float* x_dev, float* y_dev, void* stream) {
    const int ng = W.groups_per_row();
    const int threads = 128, warps_per_block = threads / 32;
    const int blocks = (W.M + warps_per_block - 1) / warps_per_block;
    gemv_i8_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (const signed char*)W.d_unpacked, (const float*)W.d_scales, x_dev, W.M, W.K, W.group, ng, y_dev);
}

// --- Streamed-device (on-GPU-Forward des GESTREAMTEN 24B) ----------------------------------------
// Streamt+entpackt das INT3-Gewicht H2D (Layer-Streaming BLEIBT — W nicht resident), rechnet dann
// auf DEVICE-Aktivierungen x_dev/y_dev — KEIN Aktivierungs-Round-Trip (kein H2D x / D2H y). Der
// bisherige fused_int3_gemv_cuda pingpongte die Aktivierung pro Matmul; das entfällt hier komplett.
// Die H2D-Transfer-Instrumentierung steckt in resolve_device (unverändert gezählt).
void fused_int3_gemv_streamed_dev(const Int3Matrix& W, const float* x_dev, float* y_dev, void* stream) {
    const int ng = W.groups_per_row();
    const cudaStream_t cs = (cudaStream_t)stream;
    const int threads = 128, warps_per_block = threads / 32;
    const int blocks = (W.M + warps_per_block - 1) / warps_per_block;
    const signed char* U; const float* scales;
    if (g_stream_overlap && !g_instr_on && cs) {        // Phase 3.3: Overlap (Compute-Stream ist echt, kein Null)
        const int slot = resolve_device_pipe(W, U, scales, cs);
        gemv_i8_kernel<<<blocks, threads, 0, cs>>>(U, scales, x_dev, W.M, W.K, W.group, ng, y_dev);
        cudaEventRecord(g_slot[slot].gemm_done, cs); g_slot[slot].gemm_pending = true;
        dbg_check("gemv_streamed");
    } else {
        resolve_device(W, U, scales);                   // alter synchroner Pfad (instrumentiert / kein Stream)
        gemv_i8_kernel<<<blocks, threads, 0, cs>>>(U, scales, x_dev, W.M, W.K, W.group, ng, y_dev);
    }
}
void fused_int3_gemm_streamed_dev(const Int3Matrix& W, const float* X_dev, int N, float* Y_dev, void* stream) {
    const int ng = W.groups_per_row();
    const cudaStream_t cs = (cudaStream_t)stream;
    const int threads = 128, warps_per_block = threads / 32;
    const int blocks = (W.M + warps_per_block - 1) / warps_per_block;
    // Phase 4a: Fast-dp4a nur bei großem N (Prefill/Dump/Tree); Decode-Verify (N klein) bleibt FP32-exakt.
    const bool use_fast = g_gemm_fast && N >= g_gemm_fast_minn && W.group == 32 && (W.K % 32) == 0;
    const signed char* U; const float* scales;
    if (g_stream_overlap && !g_instr_on && cs) {        // Phase 3.3: Overlap
        const int slot = resolve_device_pipe(W, U, scales, cs);
        if (use_fast) gemm_fast_launch(U, scales, X_dev, W.M, W.K, N, ng, Y_dev, cs);
        else gemm_i8_kernel<<<dim3(blocks, (N + 31) / 32), threads, 0, cs>>>(U, scales, X_dev, W.M, W.K, N, W.group, ng, Y_dev);
        cudaEventRecord(g_slot[slot].gemm_done, cs); g_slot[slot].gemm_pending = true;
        dbg_check("gemm_streamed");
    } else {
        resolve_device(W, U, scales);
        if (use_fast) gemm_fast_launch(U, scales, X_dev, W.M, W.K, N, ng, Y_dev, cs);
        else gemm_i8_kernel<<<dim3(blocks, (N + 31) / 32), threads, 0, cs>>>(U, scales, X_dev, W.M, W.K, N, W.group, ng, Y_dev);
    }
}

// --- Instrumentierungs-API (Messplan Phase 2) --------------------------------------------------
void      fused_instr_enable(bool on) { g_instr_on = on; }
bool      fused_instr_enabled()       { return g_instr_on; }
void      fused_instr_reset()         { g_instr = GemmInstr{}; }
GemmInstr fused_instr_snapshot()      { return g_instr; }

// Phase 3: Pin-Buchhaltung + Overlap-Status (für Log/Diagnose).
void fused_pin_stats(long& ok, long& fail, long& already) { ok = g_pin_ok; fail = g_pin_fail; already = g_pin_already; }
bool fused_stream_overlap_enabled() { return g_stream_overlap; }

}  // namespace nova::infer

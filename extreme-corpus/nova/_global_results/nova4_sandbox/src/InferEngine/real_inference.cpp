// real_inference.cpp — Implementierung der INT3-RAM-Engine (C1).
#include "InferEngine/real_inference.h"

#include "ModelStore/gguf_reader.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace nova::infer {

// Fertigstellung Phase 1: Config-Override für die NOVA_*-Flags. nova_main füttert die config.json-
// Werte via real_inference_set_flag VOR load() ein; die ss_*-Helfer lesen Config -> Env -> Default.
// So wirkt der schnelle Pfad als Produktions-Default, bleibt aber per Env (c1_measure) übersteuerbar.
static std::map<std::string, std::string> g_flag_cfg;
void real_inference_set_flag(const std::string& key, const std::string& value) { g_flag_cfg[key] = value; }
static const char* flag_val(const char* name) {
    auto it = g_flag_cfg.find(name);
    if (it != g_flag_cfg.end()) return it->second.c_str();
    return std::getenv(name);
}

// Phase 1 (C1 Umbau): NOVA_SPEC_SINGLE_STREAM. Wenn gesetzt, verifiziert spec_step das
// 24B in EINEM Stream/Schritt statt zwei (forward_batch + separater forward_step). Der
// akzeptierte Token wird an den nächsten forward_batch vorangestellt, KV via trim gerollt.
// Default aus -> alter Pfad unverändert. Bit-identische Ausgabe (Test D).
static bool ss_single_stream() {
    static const bool on = []() {
        const char* e = flag_val("NOVA_SPEC_SINGLE_STREAM");
        return e && e[0] && e[0] != '0';
    }();
    return on;
}

// Phase 2 (C1 Umbau): NOVA_FMT_BLOCK3. Wenn gesetzt, wird das gestreamte 24B mit zweistufigen
// Skalen (FP16-Super + 6-Bit-Sub) gepackt statt FP32 -> ~19% weniger Transfer. Gewichte identisch.
// Nur für das 24B-Streaming; das residente 3B bleibt FP32 (residentize nutzt FP32-Skalen).
static bool ss_fmt_block3() {
    static const bool on = []() {
        const char* e = flag_val("NOVA_FMT_BLOCK3");
        return e && e[0] && e[0] != '0';
    }();
    return on;
}

// Phase 3 (C1 Umbau): NOVA_DRAFT_BITS (3..6, Default 3) = Quantisierungs-Bit-Breite des residenten 3B.
// Höhere Präzision -> potenziell höhere Accept-Rate. Weights bleiben INT8-resident (Wert passt), nur
// die Skalen-/Level-Granularität ändert sich -> kein Kernel-Umbau. NOVA_DRAFT_KV (Default 8192) = KV-Reserve.
static int ss_draft_bits() {
    static const int b = []() {
        const char* e = flag_val("NOVA_DRAFT_BITS");
        int v = e ? std::atoi(e) : 3;
        return v < 3 ? 3 : (v > 6 ? 6 : v);
    }();
    return b;
}
// Fertigstellung: NOVA_TGT_BITS (3..6, Default 3) = Quant-Bit-Breite des gestreamten 24B.
// 3 -> block3 (transfer-effizient). 4+ -> nbit (bessere Qualität; INT3 war zu aggressiv für
// zuverlässige Instruct-Befolgung, Design §4.1). Der streamed-Forward unpackt W.bits generisch.
static int ss_tgt_bits() {
    static const int b = []() {
        const char* e = flag_val("NOVA_TGT_BITS");
        int v = e ? std::atoi(e) : 3;
        return v < 3 ? 3 : (v > 6 ? 6 : v);
    }();
    return b;
}
// Mixed-Precision (§4.1): Attention-Layer des 24B mit dieser Bit-Breite (Default 4), FFN mit
// tgt_bits (3, block3). Attention ist klein (~+0.3 GB, passt in 16 GB) und bestimmt die
// Instruct-/Kontext-Befolgung — INT3-Attention war zu grob. 0/<=tgt_bits => Mixed aus.
static int ss_tgt_attn_bits() {
    static const int b = []() {
        const char* e = flag_val("NOVA_TGT_ATTN_BITS");
        int v = e ? std::atoi(e) : 4;   // Default 4: Attention INT4 (Mixed §4.1) für Instruct-Qualität;
                                        // streamed-nbit-Unpack ist gefixt. FFN bleibt INT3-block3.
        return v < 0 ? 0 : (v > 6 ? 6 : v);
    }();
    return b;
}
static int ss_draft_kv() {
    static const int k = []() {
        const char* e = flag_val("NOVA_DRAFT_KV");
        int v = e ? std::atoi(e) : 8192;
        return v < 256 ? 256 : (v > 8192 ? 8192 : v);
    }();
    return k;
}

// Foundation (On-GPU-Forward): NOVA_GPU_FWD24. Wenn gesetzt, läuft der 24B-Forward komplett on-GPU
// (Aktivierungen+KV resident, Gewichte weiter gestreamt) -> CPU aus dem Loop. NOVA_TGT_KV = residente
// 24B-KV-Reserve (FP32, ~320 KB/Token). Default aus -> alter CPU-Glue-Pfad unverändert.
static bool ss_gpu_fwd24() {
    static const bool on = []() {
        const char* e = flag_val("NOVA_GPU_FWD24");
        return e && e[0] && e[0] != '0';
    }();
    return on;
}
static int ss_tgt_kv() {
    static const int k = []() {
        const char* e = flag_val("NOVA_TGT_KV");
        int v = e ? std::atoi(e) : 2048;
        return v < 64 ? 64 : (v > 131072 ? 131072 : v);
    }();
    return k;
}
// NOVA_KV_BITS: residente KV-Quantisierung im on-GPU-Forward. 0 = FP32 (token-identisch). 2/3/4 =
// affine Min/Max-Quantisierung je Head/Token (~14×/‑10×/‑8× kleiner) -> viel mehr Kontext/VRAM.
// Verlustbehaftet (nicht token-identisch) — bewusst in Kauf genommen, um Ressourcen zu maximieren.
static int ss_kv_bits() {
    static const int b = []() {
        const char* e = flag_val("NOVA_KV_BITS");
        int v = e ? std::atoi(e) : 0;
        if (v != 2 && v != 3 && v != 4) return 0;
        return v;
    }();
    return b;
}
// Perf Phase 2: NOVA_PREFIX_CACHE. Token-Level Prefix-KV-Reuse (vLLM-APC-Stil): der stabile
// System-Prefix (persona/identity/hot/…) wird nur EINMAL prefillt und über begin()-Aufrufe im
// residenten KV gehalten; nur der gemeinsame-Prefix-Suffix wird neu prefillt statt prefix+dynamic
// jedes Mal komplett. Spart ceil(N_stabil/MB) 24B-Streams pro Turn. Default aus -> alter Pfad
// (jeder begin() re-prefillt alles). Greedy-exakt: reused-KV[0..L-1] ist deterministische Funktion
// der identischen Prefix-Tokens, also byte-identisch zum vollen Prefill.
static bool ss_prefix_cache() {
    static const bool on = []() {
        const char* e = flag_val("NOVA_PREFIX_CACHE");
        return e && e[0] && e[0] != '0';
    }();
    return on;
}
// Perf Phase 6: NOVA_ADAPTIVE_K. K je Verify an die beobachtete Accept-Länge anpassen. Messung:
// bei accept ~1,7 kostet K=8 sechs verworfene Draft-Forwards+Syncs/Verify -> K=3 ist ~+22% tok/s.
// Adaptiv (EMA) statt fix, damit high-accept-Prompts weiterhin größere K (mehr Amortisierung) nutzen.
// Default aus -> spec_k_ fix (alter Pfad).
static bool ss_adaptive_k() {
    static const bool on = []() {
        const char* e = flag_val("NOVA_ADAPTIVE_K");
        return e && e[0] && e[0] != '0';
    }();
    return on;
}
// Perf Phase 5: NOVA_PREFILL_CHUNK. Prefill batcht MB Tokens je 24B-Stream. Größer -> weniger Streams
// (ceil(N/MB)) -> schnelleres erstes Token bei langen Kontexten. Cap = GpuForward::MAX_BATCH (256).
// Default 256 (war 32 -> ~8× weniger Prefill-Streams). Der N-gekachelte gemm_i8 trägt N>32.
static int ss_prefill_chunk() {
    static const int mb = []() {
        const char* e = flag_val("NOVA_PREFILL_CHUNK");
        int v = e ? std::atoi(e) : 256;
        return v < 1 ? 1 : (v > 256 ? 256 : v);
    }();
    return mb;
}
// Perf Phase 6A: NOVA_SPEC_TREE. Baum-Verify (m=2 Draft-Ketten: greedy + 1 Verzweigung) in EINEM
// 24B-Stream — hedged die 3B↔24B-Divergenz (Target-Greedy oft die 3B-2.-Wahl). Greedy-exakt. Nur mit
// Single-Stream + on-GPU-Draft. Default aus -> linearer Pfad.
static bool ss_spec_tree() {
    static const bool on = []() {
        const char* e = flag_val("NOVA_SPEC_TREE");
        return e && e[0] && e[0] != '0';
    }();
    return on;
}
// Phase 0 (Tree-Replan): NOVA_SPEC_RANK — Coverage messen: der Rang des 24B-Greedy-Tokens im sortierten
// 3B-Dist an jeder committeten Position. Entscheidet, ob ein 3B-Baum accept überhaupt heben KANN
// (coverage(m)=P(24B-Token in 3B-Top-m)). Gemessen im linearen Pfad, stderr-Histogramm.
static bool ss_spec_rank() {
    static const bool on = []() {
        const char* e = flag_val("NOVA_SPEC_RANK");
        return e && e[0] && e[0] != '0';
    }();
    return on;
}
static long g_rk_tot = 0, g_rk_cov[6] = {0, 0, 0, 0, 0, 0};
static long g_rk_dtot = 0, g_rk_dcov[6] = {0, 0, 0, 0, 0, 0};
static const int RK_M[6] = {1, 2, 4, 8, 16, 32};
static void rank_record(int rank, bool divergence) {
    ++g_rk_tot; for (int k = 0; k < 6; ++k) if (rank < RK_M[k]) ++g_rk_cov[k];
    if (divergence) { ++g_rk_dtot; for (int k = 0; k < 6; ++k) if (rank < RK_M[k]) ++g_rk_dcov[k]; }
    if (g_rk_tot % 40 == 0) {
        auto P = [](long a, long b) { return b ? double(a) / double(b) : 0.0; };
        std::fprintf(stderr, "[RANK] all N=%ld cov 1/2/4/8/16/32 = %.2f %.2f %.2f %.2f %.2f %.2f | DIVERGENZ N=%ld cov = %.2f %.2f %.2f %.2f %.2f %.2f\n",
            g_rk_tot, P(g_rk_cov[0], g_rk_tot), P(g_rk_cov[1], g_rk_tot), P(g_rk_cov[2], g_rk_tot), P(g_rk_cov[3], g_rk_tot), P(g_rk_cov[4], g_rk_tot), P(g_rk_cov[5], g_rk_tot),
            g_rk_dtot, P(g_rk_dcov[0], g_rk_dtot), P(g_rk_dcov[1], g_rk_dtot), P(g_rk_dcov[2], g_rk_dtot), P(g_rk_dcov[3], g_rk_dtot), P(g_rk_dcov[4], g_rk_dtot), P(g_rk_dcov[5], g_rk_dtot));
    }
}

#ifdef NOVA_HAVE_CUDA
static bool g_use_gpu = true;   // unter CUDA: Matmuls auf die GPU (fused_int3_gemv_cuda)
#else
static bool g_use_gpu = false;
#endif
void real_inference_use_gpu(bool on) { g_use_gpu = on; }
bool real_inference_gpu() { return g_use_gpu; }

// Load-Stage-Timing (Messplan Phase 1). Global, da load_int3_from_gguf eine freie Funktion ist.
static LoadTiming g_load_timing;
LoadTiming last_load_timing() { return g_load_timing; }
using clk_ld = std::chrono::steady_clock;
static double ms_since(const clk_ld::time_point& t0) {
    return std::chrono::duration<double, std::milli>(clk_ld::now() - t0).count();
}

namespace {

// Ein Zeilenblock des fused INT3-Dequant-GEMV (bit-identisch zu fused_int3_gemv).
void gemv_rows(const Int3Matrix& W, const float* x, float* y, int r0, int r1) {
    const int ng = W.groups_per_row();
    for (int i = r0; i < r1; ++i) {
        double acc = 0.0;
        const size_t row_bit = size_t(i) * W.K * 3;
        for (int k = 0; k < W.K; ++k) {
            const size_t bp = row_bit + size_t(k) * 3; uint8_t u = 0;
            for (int b = 0; b < 3; ++b)
                if (W.packed[(bp + b) / 8] >> ((bp + b) % 8) & 1u) u |= uint8_t(1u << b);
            acc += double(int(u) - 4) * double(W.scales[size_t(i) * ng + k / W.group]) * double(x[k]);
        }
        y[i] = float(acc);
    }
}

// Multithreaded GEMV: y = dequant(W)·x, Zeilen gleichmäßig über ALLE Kerne.
void gemv_mt(const Int3Matrix& W, const float* x, float* y) {
    static const unsigned HW = std::max(1u, std::thread::hardware_concurrency());
    const int M = W.M;
    const unsigned nt = std::min<unsigned>(HW, unsigned(std::max(1, M / 64)));
    if (nt <= 1) { gemv_rows(W, x, y, 0, M); return; }
    std::vector<std::thread> ts; ts.reserve(nt);
    const int chunk = (M + int(nt) - 1) / int(nt);
    for (unsigned t = 0; t < nt; ++t) {
        const int r0 = int(t) * chunk, r1 = std::min(M, r0 + chunk);
        if (r0 >= r1) break;
        ts.emplace_back(gemv_rows, std::cref(W), x, y, r0, r1);
    }
    for (auto& th : ts) th.join();
}

// INT3-Packen eines Zeilenblocks [r0,r1) (Numerik identisch zu pack_int3).
void pack_rows(const float* Wf, int M, int K, int group, int ng, int r0, int r1,
               std::vector<uint8_t>& packed, std::vector<float>& scales) {
    for (int i = r0; i < r1; ++i) {
        for (int gpi = 0; gpi < ng; ++gpi) {
            const int k0 = gpi * group, k1 = std::min(k0 + group, K);
            float amax = 0.0f;
            for (int k = k0; k < k1; ++k) amax = std::max(amax, std::fabs(Wf[size_t(i) * K + k]));
            const float scale = amax > 0.0f ? amax / 4.0f : 1.0f;
            scales[size_t(i) * ng + gpi] = scale;
            for (int k = k0; k < k1; ++k) {
                int q = int(std::lround(Wf[size_t(i) * K + k] / scale));
                q = std::min(3, std::max(-4, q));
                const uint8_t uu = uint8_t(q + 4);
                const size_t bp = (size_t(i) * K + k) * 3;
                for (int b = 0; b < 3; ++b)
                    if ((uu >> b) & 1u) packed[(bp + b) / 8] |= uint8_t(1u << ((bp + b) % 8));
            }
        }
    }
}

// Parallel-Version von pack_int3: verteilt die M Zeilen gleichmäßig auf alle Kerne.
// Bit-identisch zu fused_gemm_host.cpp::pack_int3 (Zeilen sind unabhängig).
Int3Matrix pack_int3_mt(const float* Wf, int M, int K, int group) {
    static const unsigned HW = std::max(1u, std::thread::hardware_concurrency());
    Int3Matrix out; out.M = M; out.K = K; out.group = group;
    const int ng = out.groups_per_row();
    out.scales.assign(size_t(M) * ng, 0.0f);
    out.packed.assign((size_t(M) * K * 3 + 7) / 8, 0);   // 0-Init: Threads setzen nur eigene Bits
    const unsigned nt = std::min<unsigned>(HW, unsigned(std::max(1, M / 16)));
    if (nt <= 1) { pack_rows(Wf, M, K, group, ng, 0, M, out.packed, out.scales); return out; }
    // Byte-Grenzen: Zeilen an 8er-Vielfachen splitten, damit keine zwei Threads dasselbe
    // Byte anfassen (M*K*3 pro Zeile; Zeile beginnt bei i*K*3 Bit -> auf 8 ausrichten).
    std::vector<std::thread> ts; ts.reserve(nt);
    const int chunk = ((M + int(nt) - 1) / int(nt) + 7) & ~7;  // Vielfaches von 8 Zeilen
    for (int r0 = 0; r0 < M; r0 += chunk) {
        const int r1 = std::min(M, r0 + chunk);
        ts.emplace_back(pack_rows, Wf, M, K, group, ng, r0, r1,
                        std::ref(out.packed), std::ref(out.scales));
    }
    for (auto& th : ts) th.join();
    return out;
}

// Phase 2 FMT_BLOCK3: packt Zeilen [r0,r1) — Gewichte bit-identisch zu pack_rows, Skalen zweistufig.
void pack_rows_block3(const float* Wf, int M, int K, int group, int ng, int nsb, int r0, int r1,
                      std::vector<uint8_t>& packed, std::vector<uint16_t>& super,
                      std::vector<uint8_t>& sub) {
    std::vector<float> gs(ng);
    for (int i = r0; i < r1; ++i) {
        for (int gpi = 0; gpi < ng; ++gpi) {
            const int k0 = gpi * group, k1 = std::min(k0 + group, K);
            float amax = 0.0f;
            for (int k = k0; k < k1; ++k) amax = std::max(amax, std::fabs(Wf[size_t(i) * K + k]));
            const float scale = amax > 0.0f ? amax / 4.0f : 0.0f;   // Null-Gruppe -> 0
            gs[gpi] = scale;
            const float qs = scale > 0.0f ? scale : 1.0f;
            for (int k = k0; k < k1; ++k) {
                int q = int(std::lround(Wf[size_t(i) * K + k] / qs));
                q = std::min(3, std::max(-4, q));
                const uint8_t uu = uint8_t(q + 4);
                const size_t bp = (size_t(i) * K + k) * 3;
                for (int b = 0; b < 3; ++b)
                    if ((uu >> b) & 1u) packed[(bp + b) / 8] |= uint8_t(1u << ((bp + b) % 8));
            }
        }
        for (int sb = 0; sb < nsb; ++sb) {
            const int g0 = sb * 8, g1 = std::min(g0 + 8, ng);
            float smax = 0.0f;
            for (int g = g0; g < g1; ++g) smax = std::max(smax, gs[g]);
            super[size_t(i) * nsb + sb] = b3_f32_to_f16(smax);
            const float inv = smax > 0.0f ? 63.0f / smax : 0.0f;
            for (int g = g0; g < g1; ++g) {
                int q6 = int(std::lround(gs[g] * inv));
                if (q6 < 0) q6 = 0; if (q6 > 63) q6 = 63;
                if (gs[g] > 0.0f && q6 == 0) q6 = 1;
                b3_put6(sub, (size_t(i) * nsb + sb) * 6, g - g0, uint8_t(q6));
            }
        }
    }
}

// Parallel-Block3-Packer (Gewichte bit-identisch zu pack_int3_mt; nur Skalen-Kodierung anders).
Int3Matrix pack_block3_mt(const float* Wf, int M, int K, int group) {
    static const unsigned HW = std::max(1u, std::thread::hardware_concurrency());
    Int3Matrix out; out.M = M; out.K = K; out.group = group; out.fmt = SCALE_BLOCK3;
    const int ng = out.groups_per_row();
    const int nsb = out.superblocks_per_row();
    out.packed.assign((size_t(M) * K * 3 + 7) / 8, 0);
    out.superscales.assign(size_t(M) * nsb, 0);
    out.subscales.assign(size_t(M) * nsb * 6, 0);
    const unsigned nt = std::min<unsigned>(HW, unsigned(std::max(1, M / 16)));
    if (nt <= 1) { pack_rows_block3(Wf, M, K, group, ng, nsb, 0, M, out.packed, out.superscales, out.subscales); return out; }
    std::vector<std::thread> ts; ts.reserve(nt);
    const int chunk = ((M + int(nt) - 1) / int(nt) + 7) & ~7;   // Vielfaches von 8 Zeilen (Byte-Grenzen)
    for (int r0 = 0; r0 < M; r0 += chunk) {
        const int r1 = std::min(M, r0 + chunk);
        ts.emplace_back(pack_rows_block3, Wf, M, K, group, ng, nsb, r0, r1,
                        std::ref(out.packed), std::ref(out.superscales), std::ref(out.subscales));
    }
    for (auto& th : ts) th.join();
    return out;
}

// Phase 3: N-Bit-Packen eines Zeilenblocks (3..6 Bit). Levels -(2^(b-1))..(2^(b-1))-1, scale=amax/2^(b-1).
// Für das residente 3B-Draft: unpack_nbit_kernel entpackt zu INT8 (Wert-offset passt), gemv unverändert.
void pack_rows_nbit(const float* Wf, int M, int K, int group, int ng, int bits, int r0, int r1,
                    std::vector<uint8_t>& packed, std::vector<float>& scales) {
    const int lvl = 1 << (bits - 1);
    for (int i = r0; i < r1; ++i)
        for (int gpi = 0; gpi < ng; ++gpi) {
            const int k0 = gpi * group, k1 = std::min(k0 + group, K);
            float amax = 0.0f;
            for (int k = k0; k < k1; ++k) amax = std::max(amax, std::fabs(Wf[size_t(i) * K + k]));
            const float scale = amax > 0.0f ? amax / float(lvl) : 1.0f;
            scales[size_t(i) * ng + gpi] = scale;
            for (int k = k0; k < k1; ++k) {
                int q = int(std::lround(Wf[size_t(i) * K + k] / scale));
                q = std::min(lvl - 1, std::max(-lvl, q));
                const uint32_t u = uint32_t(q + lvl);   // 0..2^bits-1
                const size_t bp = (size_t(i) * K + k) * bits;
                for (int b = 0; b < bits; ++b)
                    if ((u >> b) & 1u) packed[(bp + b) / 8] |= uint8_t(1u << ((bp + b) % 8));
            }
        }
}

// Parallel-N-Bit-Packer (nur 3B-Draft). group/scales wie INT3; nur Bit-Breite + Level-Zahl variieren.
Int3Matrix pack_nbit_mt(const float* Wf, int M, int K, int group, int bits) {
    static const unsigned HW = std::max(1u, std::thread::hardware_concurrency());
    Int3Matrix out; out.M = M; out.K = K; out.group = group; out.bits = bits; out.fmt = SCALE_FP32;
    const int ng = out.groups_per_row();
    out.scales.assign(size_t(M) * ng, 0.0f);
    out.packed.assign((size_t(M) * K * bits + 7) / 8, 0);
    const unsigned nt = std::min<unsigned>(HW, unsigned(std::max(1, M / 16)));
    if (nt <= 1) { pack_rows_nbit(Wf, M, K, group, ng, bits, 0, M, out.packed, out.scales); return out; }
    std::vector<std::thread> ts; ts.reserve(nt);
    const int chunk = ((M + int(nt) - 1) / int(nt) + 7) & ~7;   // Vielfaches von 8 Zeilen (Byte-Grenzen)
    for (int r0 = 0; r0 < M; r0 += chunk) {
        const int r1 = std::min(M, r0 + chunk);
        ts.emplace_back(pack_rows_nbit, Wf, M, K, group, ng, bits, r0, r1,
                        std::ref(out.packed), std::ref(out.scales));
    }
    for (auto& th : ts) th.join();
    return out;
}

// --- Basis-Kernels (bit-identisch zur CPU-Referenz mistral3_forward.cpp) ------
void rmsnorm(const float* x, const std::vector<float>& g, int H, float eps, float* out) {
    double ss = 0; for (int i = 0; i < H; ++i) ss += double(x[i]) * x[i];
    const float inv = float(1.0 / std::sqrt(ss / H + eps));
    for (int i = 0; i < H; ++i) out[i] = x[i] * inv * g[i];
}
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// RoPE für EINEN Vektor v[nHeads*hd] an absoluter Position pos.
void rope_at(float* v, int pos, int nHeads, int hd, float theta, bool interleaved) {
    const int half = hd / 2;
    for (int h = 0; h < nHeads; ++h) {
        float* p = v + size_t(h) * hd;
        for (int i = 0; i < half; ++i) {
            const float freq = std::pow(theta, -2.0f * i / hd);
            const float ang = pos * freq, c = std::cos(ang), sn = std::sin(ang);
            const int a = interleaved ? 2 * i : i;
            const int b = interleaved ? 2 * i + 1 : i + half;
            const float va = p[a], vb = p[b];
            p[a] = va * c - vb * sn;
            p[b] = va * sn + vb * c;
        }
    }
}

// y = dequant(Wq) · x (INT3 fused, cosine>0.99 validiert). Fehlt Wq -> y=0.
inline void gemv(const Int3Weights& w, const std::string& name, const float* x, float* y, int outdim) {
    const Int3Matrix* W = w.mat(name);
    if (!W) { for (int i = 0; i < outdim; ++i) y[i] = 0.0f; return; }
#ifdef NOVA_HAVE_CUDA
    if (g_use_gpu) { fused_int3_gemv_cuda(*W, x, y); return; }  // Matmul auf der RTX 3080
#endif
    gemv_mt(*W, x, y);   // CPU-Fallback: multithreaded, bit-identisch
}

// Batched: Y[M×N] = dequant(W)·X[K×N]. GPU wenn verfügbar, sonst Host.
// HINWEIS (Phase 4a): fused_int3_gemm_cuda ist SCHWELLENLOS — bei NOVA_GEMM_FAST=1 nutzt es den lossy
// dp4a-Pfad für JEDES N. Dieser CPU-Glue-`forward_batch`-Pfad ist NUR der Fallback ohne NOVA_GPU_FWD24;
// die Produktion (NOVA_GPU_FWD24=1) verifiziert Decode über GpuForward::forward_batch_gpu (streamed_dev,
// N≥`NOVA_GEMM_FAST_MINN` → sonst FP32). INVARIANTE: `NOVA_GEMM_FAST=1` NUR mit `NOVA_GPU_FWD24=1` fahren,
// sonst wäre der greedy-Decode-Verify hier lossy. (Default NOVA_GEMM_FAST=aus → immer exakt.)
inline void gemm(const Int3Weights& w, const std::string& name, const float* X, int N, float* Y, int outdim) {
    const Int3Matrix* W = w.mat(name);
    if (!W) { for (size_t i = 0; i < size_t(outdim) * N; ++i) Y[i] = 0.0f; return; }
#ifdef NOVA_HAVE_CUDA
    if (g_use_gpu) { fused_int3_gemm_cuda(*W, X, N, Y); return; }
#endif
    fused_int3_gemm(*W, X, N, Y);
}

inline int argmax_ptr(const float* v, int n) {
    int b = 0; for (int i = 1; i < n; ++i) if (v[i] > v[b]) b = i; return b;
}

}  // namespace

void dequant_row(const Int3Matrix& W, int o, float* out) {
    const int ng = W.groups_per_row();
    const size_t row_bit = size_t(o) * W.K * 3;
    for (int k = 0; k < W.K; ++k) {
        // 3 Bit little-endian entpacken (wie get3 in fused_gemm_host.cpp).
        const size_t bp = row_bit + size_t(k) * 3; uint8_t u = 0;
        for (int b = 0; b < 3; ++b)
            if (W.packed[(bp + b) / 8] >> ((bp + b) % 8) & 1u) u |= uint8_t(1u << b);
        const int q = int(u) - 4;
        out[k] = float(q) * W.scales[size_t(o) * ng + k / W.group];
    }
}

size_t Int3Weights::approx_bytes() const {
    size_t b = 0;
    for (const auto& kv : mats)  b += kv.second.packed.size() + kv.second.scales.size() * sizeof(float);
    for (const auto& kv : norms) b += kv.second.size() * sizeof(float);
    return b;
}

std::vector<float> forward_step(const Int3Weights& w, int token_id, int pos, KvCacheF32& kv) {
    const Mistral3Config& c = w.cfg;
    const int H = c.hidden, nH = c.n_heads, nKV = c.n_kv_heads, hd = c.head_dim;
    const int qd = c.q_dim(), kvd = c.kv_dim(), F = c.ffn, V = c.vocab, grp = c.kv_group();
    const float eps = c.rms_eps, theta = c.rope_theta, scale = 1.0f / std::sqrt(float(hd));
    if (int(kv.k.size()) != c.n_layers) kv.reset(c.n_layers);

    // Embedding-Zeile (token_embd: M=vocab, K=hidden).
    std::vector<float> x(H, 0.0f);
    if (const Int3Matrix* emb = w.mat("token_embd")) dequant_row(*emb, token_id, x.data());

    std::vector<float> xn(H), q(qd), knew(kvd), vnew(kvd), ctx(qd), tmpH(H), g(F), u(F);

    for (int l = 0; l < c.n_layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        // --- Attention ---
        if (const auto* an = w.norm(p + "attn_norm")) rmsnorm(x.data(), *an, H, eps, xn.data());
        gemv(w, p + "attn_q", xn.data(), q.data(), qd);
        gemv(w, p + "attn_k", xn.data(), knew.data(), kvd);
        gemv(w, p + "attn_v", xn.data(), vnew.data(), kvd);
        rope_at(q.data(), pos, nH, hd, theta, w.interleaved);
        rope_at(knew.data(), pos, nKV, hd, theta, w.interleaved);
        // KV anhängen (keys nach RoPE).
        kv.k[l].insert(kv.k[l].end(), knew.begin(), knew.end());
        kv.v[l].insert(kv.v[l].end(), vnew.begin(), vnew.end());
        const int T = pos + 1;
        // GQA kausale Attention über die gecachten T Positionen.
        for (int h = 0; h < nH; ++h) {
            const int kvh = h / grp;
            const float* qv = &q[size_t(h) * hd];
            std::vector<float> sc(T); float mx = -1e30f;
            for (int t = 0; t < T; ++t) {
                const float* kk = &kv.k[l][(size_t(t) * nKV + kvh) * hd];
                double d = 0; for (int i = 0; i < hd; ++i) d += double(qv[i]) * kk[i];
                sc[t] = float(d) * scale; if (sc[t] > mx) mx = sc[t];
            }
            double sum = 0; for (int t = 0; t < T; ++t) { sc[t] = std::exp(sc[t] - mx); sum += sc[t]; }
            float* out = &ctx[size_t(h) * hd];
            for (int i = 0; i < hd; ++i) out[i] = 0.0f;
            for (int t = 0; t < T; ++t) {
                const float ww = float(sc[t] / sum);
                const float* vv = &kv.v[l][(size_t(t) * nKV + kvh) * hd];
                for (int i = 0; i < hd; ++i) out[i] += ww * vv[i];
            }
        }
        gemv(w, p + "attn_output", ctx.data(), tmpH.data(), H);
        for (int i = 0; i < H; ++i) x[i] += tmpH[i];
        // --- FFN (SwiGLU) ---
        if (const auto* fn = w.norm(p + "ffn_norm")) rmsnorm(x.data(), *fn, H, eps, xn.data());
        gemv(w, p + "ffn_gate", xn.data(), g.data(), F);
        gemv(w, p + "ffn_up", xn.data(), u.data(), F);
        for (int i = 0; i < F; ++i) g[i] = silu(g[i]) * u[i];
        gemv(w, p + "ffn_down", g.data(), tmpH.data(), H);
        for (int i = 0; i < H; ++i) x[i] += tmpH[i];
    }

    // Finaler Norm + Logits.
    std::vector<float> xf(H);
    if (const auto* on = w.norm("output_norm")) rmsnorm(x.data(), *on, H, eps, xf.data());
    std::vector<float> logits(V, 0.0f);
    gemv(w, "output", xf.data(), logits.data(), V);
    kv.seq_len = pos + 1;
    return logits;
}

void kv_trim(KvCacheF32& kv, int len, const Mistral3Config& cfg) {
    const int kvd = cfg.kv_dim();
    for (int l = 0; l < int(kv.k.size()); ++l) {
        kv.k[l].resize(size_t(len) * kvd);
        kv.v[l].resize(size_t(len) * kvd);
    }
    kv.seq_len = len;
}

std::vector<float> forward_batch(const Int3Weights& w, const int* tokens, int N,
                                 int start_pos, KvCacheF32& kv) {
    const Mistral3Config& c = w.cfg;
    const int H = c.hidden, nH = c.n_heads, nKV = c.n_kv_heads, hd = c.head_dim;
    const int qd = c.q_dim(), kvd = c.kv_dim(), F = c.ffn, V = c.vocab, grp = c.kv_group();
    const float eps = c.rms_eps, theta = c.rope_theta, scale = 1.0f / std::sqrt(float(hd));
    if (int(kv.k.size()) != c.n_layers) kv.reset(c.n_layers);

    // Aktivierungen als [feature][N] (spaltenweise Positionen): A[f*N + n].
    std::vector<float> X(size_t(H) * N, 0.0f);
    { std::vector<float> row(H);
      for (int n = 0; n < N; ++n)
        if (const Int3Matrix* emb = w.mat("token_embd")) {
            dequant_row(*emb, tokens[n], row.data());
            for (int h = 0; h < H; ++h) X[size_t(h) * N + n] = row[h];
        } }
    std::vector<float> Xn(size_t(H) * N), Q(size_t(qd) * N), Kc(size_t(kvd) * N), Vc(size_t(kvd) * N),
                       Ctx(size_t(qd) * N), Th(size_t(H) * N), G(size_t(F) * N), Uu(size_t(F) * N);

    auto rmsnorm_cols = [&](const std::vector<float>& src, const std::vector<float>& g, std::vector<float>& dst) {
        for (int n = 0; n < N; ++n) {
            double ss = 0; for (int h = 0; h < H; ++h) { float v = src[size_t(h) * N + n]; ss += double(v) * v; }
            float inv = float(1.0 / std::sqrt(ss / H + eps));
            for (int h = 0; h < H; ++h) dst[size_t(h) * N + n] = src[size_t(h) * N + n] * inv * g[h];
        } };

    for (int l = 0; l < c.n_layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        if (const auto* an = w.norm(p + "attn_norm")) rmsnorm_cols(X, *an, Xn);
        gemm(w, p + "attn_q", Xn.data(), N, Q.data(), qd);
        gemm(w, p + "attn_k", Xn.data(), N, Kc.data(), kvd);
        gemm(w, p + "attn_v", Xn.data(), N, Vc.data(), kvd);
        // RoPE je Spalte (absolute Position start_pos+n); Keys/Values an KV anhängen.
        std::vector<float> qc(qd), kc(kvd), vc(kvd);
        for (int n = 0; n < N; ++n) {
            const int pos = start_pos + n;
            for (int i = 0; i < qd; ++i) qc[i] = Q[size_t(i) * N + n];
            rope_at(qc.data(), pos, nH, hd, theta, w.interleaved);
            for (int i = 0; i < qd; ++i) Q[size_t(i) * N + n] = qc[i];
            for (int i = 0; i < kvd; ++i) kc[i] = Kc[size_t(i) * N + n];
            rope_at(kc.data(), pos, nKV, hd, theta, w.interleaved);
            kv.k[l].insert(kv.k[l].end(), kc.begin(), kc.end());
            for (int i = 0; i < kvd; ++i) vc[i] = Vc[size_t(i) * N + n];
            kv.v[l].insert(kv.v[l].end(), vc.begin(), vc.end());
        }
        // Kausale GQA-Attention je Spalte n (attendiert KV[0..start_pos+n]).
        for (int n = 0; n < N; ++n) {
            const int T = start_pos + n + 1;
            for (int h = 0; h < nH; ++h) {
                const int kvh = h / grp;
                std::vector<float> sc(T); float mx = -1e30f;
                for (int t = 0; t < T; ++t) {
                    const float* kk = &kv.k[l][(size_t(t) * nKV + kvh) * hd];
                    double d = 0; for (int i = 0; i < hd; ++i) d += double(Q[size_t(h * hd + i) * N + n]) * kk[i];
                    sc[t] = float(d) * scale; if (sc[t] > mx) mx = sc[t];
                }
                double sm = 0; for (int t = 0; t < T; ++t) { sc[t] = std::exp(sc[t] - mx); sm += sc[t]; }
                for (int i = 0; i < hd; ++i) {
                    float o = 0;
                    for (int t = 0; t < T; ++t) o += float(sc[t] / sm) * kv.v[l][(size_t(t) * nKV + kvh) * hd + i];
                    Ctx[size_t(h * hd + i) * N + n] = o;
                }
            }
        }
        gemm(w, p + "attn_output", Ctx.data(), N, Th.data(), H);
        for (size_t idx = 0; idx < size_t(H) * N; ++idx) X[idx] += Th[idx];
        if (const auto* fn = w.norm(p + "ffn_norm")) rmsnorm_cols(X, *fn, Xn);
        gemm(w, p + "ffn_gate", Xn.data(), N, G.data(), F);
        gemm(w, p + "ffn_up", Xn.data(), N, Uu.data(), F);
        for (size_t idx = 0; idx < size_t(F) * N; ++idx) G[idx] = silu(G[idx]) * Uu[idx];
        gemm(w, p + "ffn_down", G.data(), N, Th.data(), H);
        for (size_t idx = 0; idx < size_t(H) * N; ++idx) X[idx] += Th[idx];
    }
    std::vector<float> Xf(size_t(H) * N);
    if (const auto* on = w.norm("output_norm")) rmsnorm_cols(X, *on, Xf);
    std::vector<float> Ylog(size_t(V) * N, 0.0f);
    gemm(w, "output", Xf.data(), N, Ylog.data(), V);
    kv.seq_len = start_pos + N;
    // Umsortieren [V][N] -> [N][V] (Zeile n = Logits an Position start_pos+n).
    std::vector<float> out(size_t(N) * V);
    for (int n = 0; n < N; ++n)
        for (int vv = 0; vv < V; ++vv) out[size_t(n) * V + vv] = Ylog[size_t(vv) * N + n];
    return out;
}

// ===== U7: .nv3w Startup-Cache (Design §5, Roadmap U7) =====================================
// Serialisiert die FERTIG-gepackten Int3Weights (INT3-packed + Skalen + Norms) einmalig nach dem ersten
// GGUF-Requant. Folgende Loads lesen die ~9,4 GB direkt von SSD (~19s) statt zu dequantisieren+requantisieren
// (~211s). Flag NOVA_NV_CACHE (Default AUS → Produktionspfad unverändert). Validierung: gguf-Größe+mtime +
// cfg/group/bits müssen matchen (sonst stale → GGUF-Pfad). Bit-identisch, da exakt die gepackten Bytes.
namespace {
constexpr uint32_t NV3W_MAGIC = 0x57334E56u;  // "NV3W"
constexpr uint32_t NV3W_VER   = 1u;
struct GgufMeta { uint64_t size = 0; int64_t mtime = 0; };
GgufMeta gguf_meta(const std::string& path) {
    GgufMeta m; std::error_code ec;
    m.size = (uint64_t)std::filesystem::file_size(path, ec);
    m.mtime = (int64_t)std::filesystem::last_write_time(path, ec).time_since_epoch().count();
    return m;
}
template <class T> void nv_wr(std::ofstream& o, const T& v) { o.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
template <class T> bool nv_rd(std::ifstream& i, T& v) { return (bool)i.read(reinterpret_cast<char*>(&v), sizeof(T)); }
template <class V> void nv_wr_vec(std::ofstream& o, const V& v) {
    uint64_t n = v.size(); o.write(reinterpret_cast<const char*>(&n), 8);
    if (n) o.write(reinterpret_cast<const char*>(v.data()), n * sizeof(typename V::value_type));
}
template <class V> bool nv_rd_vec(std::ifstream& i, V& v) {
    uint64_t n = 0; if (!i.read(reinterpret_cast<char*>(&n), 8)) return false;
    v.resize(n); if (n && !i.read(reinterpret_cast<char*>(v.data()), n * sizeof(typename V::value_type))) return false;
    return true;
}
void nv_wr_str(std::ofstream& o, const std::string& s) { uint32_t n = (uint32_t)s.size(); nv_wr(o, n); if (n) o.write(s.data(), n); }
bool nv_rd_str(std::ifstream& i, std::string& s) { uint32_t n = 0; if (!nv_rd(i, n)) return false; s.resize(n); if (n && !i.read(&s[0], n)) return false; return true; }
std::string nv_cache_path(const std::string& gguf, int group, int bits, int attn_bits, bool block3) {
    if (const char* e = flag_val("NOVA_NV_CACHE_PATH")) return e;
    char suf[80]; std::snprintf(suf, sizeof(suf), ".g%d.b%d.a%d.%s.nv3w", group, bits, attn_bits, block3 ? "b3" : "fp");
    return gguf + suf;
}
bool nv_cache_enabled() { const char* e = flag_val("NOVA_NV_CACHE"); return e && e[0] && e[0] != '0'; }
bool nv_cache_save(const std::string& path, const Int3Weights& out, const GgufMeta& gm, const Mistral3Config& cfg,
                   int group, int bits, int attn_bits, bool block3) {
    std::ofstream o(path, std::ios::binary | std::ios::trunc);
    if (!o) return false;
    nv_wr(o, NV3W_MAGIC); nv_wr(o, NV3W_VER); nv_wr(o, gm.size); nv_wr(o, gm.mtime);
    int32_t nl = cfg.n_layers, hd = cfg.hidden, vc = cfg.vocab; nv_wr(o, nl); nv_wr(o, hd); nv_wr(o, vc);
    int32_t g = group, b = bits, a = attn_bits; uint8_t b3 = block3 ? 1 : 0, il = out.interleaved ? 1 : 0;
    nv_wr(o, g); nv_wr(o, b); nv_wr(o, a); nv_wr(o, b3); nv_wr(o, il);
    uint32_t nmats = (uint32_t)out.mats.size(), nnorms = (uint32_t)out.norms.size();
    nv_wr(o, nmats); nv_wr(o, nnorms);
    for (const auto& kv : out.mats) {
        nv_wr_str(o, kv.first); const Int3Matrix& m = kv.second;
        int32_t M = m.M, K = m.K, gr = m.group, bi = m.bits, fm = m.fmt;
        nv_wr(o, M); nv_wr(o, K); nv_wr(o, gr); nv_wr(o, bi); nv_wr(o, fm);
        nv_wr_vec(o, m.packed); nv_wr_vec(o, m.scales); nv_wr_vec(o, m.superscales); nv_wr_vec(o, m.subscales);
    }
    for (const auto& kv : out.norms) { nv_wr_str(o, kv.first); nv_wr_vec(o, kv.second); }
    uint32_t trailer = NV3W_MAGIC; nv_wr(o, trailer);   // Vollständigkeits-Marker
    o.flush(); return (bool)o;
}
bool nv_cache_load(const std::string& path, Int3Weights& out, const GgufMeta& gm, const Mistral3Config& cfg,
                   int group, int bits, int attn_bits, bool block3) {
    std::ifstream i(path, std::ios::binary);
    if (!i) return false;
    uint32_t magic = 0, ver = 0; if (!nv_rd(i, magic) || !nv_rd(i, ver) || magic != NV3W_MAGIC || ver != NV3W_VER) return false;
    uint64_t sz = 0; int64_t mt = 0; nv_rd(i, sz); nv_rd(i, mt);
    if (sz != gm.size || mt != gm.mtime) return false;                       // Quelle geändert → stale
    int32_t nl = 0, hd = 0, vc = 0; nv_rd(i, nl); nv_rd(i, hd); nv_rd(i, vc);
    if (nl != cfg.n_layers || hd != cfg.hidden || vc != cfg.vocab) return false;
    int32_t g = 0, b = 0, a = 0; uint8_t b3 = 0, il = 0; nv_rd(i, g); nv_rd(i, b); nv_rd(i, a); nv_rd(i, b3); nv_rd(i, il);
    if (g != group || b != bits || a != attn_bits || b3 != (block3 ? 1 : 0)) return false;
    out.cfg = cfg; out.interleaved = (il != 0); out.mats.clear(); out.norms.clear();
    uint32_t nmats = 0, nnorms = 0; if (!nv_rd(i, nmats) || !nv_rd(i, nnorms)) return false;
    for (uint32_t k = 0; k < nmats; ++k) {
        std::string key; if (!nv_rd_str(i, key)) return false;
        Int3Matrix m; int32_t M, K, gr, bi, fm;
        if (!nv_rd(i, M) || !nv_rd(i, K) || !nv_rd(i, gr) || !nv_rd(i, bi) || !nv_rd(i, fm)) return false;
        m.M = M; m.K = K; m.group = gr; m.bits = bi; m.fmt = fm;
        if (!nv_rd_vec(i, m.packed) || !nv_rd_vec(i, m.scales) || !nv_rd_vec(i, m.superscales) || !nv_rd_vec(i, m.subscales)) return false;
        out.mats[key] = std::move(m);
    }
    for (uint32_t k = 0; k < nnorms; ++k) {
        std::string key; if (!nv_rd_str(i, key)) return false;
        std::vector<float> v; if (!nv_rd_vec(i, v)) return false;
        out.norms[key] = std::move(v);
    }
    uint32_t trailer = 0; if (!nv_rd(i, trailer) || trailer != NV3W_MAGIC) return false;   // vollständig geschrieben?
    return true;
}
}  // namespace

// --- GGUF -> INT3-Gewichte laden ----------------------------------------------
bool load_int3_from_gguf(const std::string& gguf_path, const Mistral3Config& cfg,
                         Int3Weights& out, int group, bool rope_interleaved,
                         std::string* err,
                         const std::function<void(const std::string&, int, int)>& progress,
                         bool block3, int bits, int attn_bits) {
    g_load_timing = LoadTiming{};                         // Messplan Phase 1: Stage-Timer zurücksetzen
    // U7: .nv3w-Cache-Schnellpfad (NOVA_NV_CACHE). Trifft der Cache (gültig) → kein Dequant/Requant.
    const bool cache_on = nv_cache_enabled();
    const std::string cache_path = cache_on ? nv_cache_path(gguf_path, group, bits, attn_bits, block3) : std::string();
    if (cache_on) {
        const GgufMeta gm = gguf_meta(gguf_path);
        auto t_c = clk_ld::now();
        if (nv_cache_load(cache_path, out, gm, cfg, group, bits, attn_bits, block3)) {
            g_load_timing.read_ms = ms_since(t_c);
            std::fprintf(stderr, "[real] .nv3w-Cache geladen (%.1f s, kein Requant): %s\n",
                         g_load_timing.read_ms / 1e3, cache_path.c_str());
            return true;
        }
        std::fprintf(stderr, "[real] .nv3w-Cache fehlt/stale → GGUF-Pfad + Cache schreiben.\n");
    }
    auto t_open = clk_ld::now();
    modelstore::GgufReader g;
    if (!g.open(gguf_path, err)) return false;
    g_load_timing.open_ms = ms_since(t_open);
    out.cfg = cfg; out.interleaved = rope_interleaved;
    out.mats.clear(); out.norms.clear();

    // Namen-Index.
    std::map<std::string, const modelstore::GgufTensorInfo*> byname;
    for (const auto& t : g.tensors()) byname[t.name] = &t;

    auto load_mat = [&](const std::string& gname, const std::string& key) -> bool {
        auto it = byname.find(gname);
        if (it == byname.end()) { if (err) *err = "Tensor fehlt: " + gname; return false; }
        const auto& t = *it->second;
        auto t_r = clk_ld::now();
        std::vector<float> f; if (!g.read_tensor_f32(t, f, err)) return false;
        g_load_timing.read_ms += ms_since(t_r); ++g_load_timing.tensors;
        // GGUF-dims [ne0=in, ne1=out]; read_tensor_f32 liefert row-major [out][in].
        const int K = int(t.dims.empty() ? 1 : t.dims[0]);
        const int M = int(t.dims.size() > 1 ? t.dims[1] : 1);
        auto t_p = clk_ld::now();
        // FMT_BLOCK3 nur für gestreamte Matmul-Gewichte. token_embd wird per dequant_row (Host-
        // Embedding-Gather) gelesen, das FP32-scales[] braucht -> token_embd bleibt INT3/FP32.
        // Mixed-Precision (Design §4.1): Attention-Layer höher quantisiert (attn_bits) als die
        // FFN — die Attention bestimmt Instruct-/Kontext-Befolgung, ist aber klein (~0.3 GB extra),
        // die große FFN bleibt bei `bits`/block3. Der Streamed-Forward unpackt W.bits pro Matrix.
        const bool is_attn = key.find(".attn_") != std::string::npos;
        const int  this_bits = (is_attn && attn_bits > bits) ? attn_bits : bits;
        const bool use_b3 = block3 && (key != "token_embd") && !(is_attn && attn_bits > bits);
        out.mats[key] = use_b3          ? pack_block3_mt(f.data(), M, K, group)        // Phase 2 FMT_BLOCK3
                      : (this_bits != 3) ? pack_nbit_mt(f.data(), M, K, group, this_bits)  // N-Bit (Draft/Attn)
                                         : pack_int3_mt(f.data(), M, K, group);         // alle CPU-Kerne
        g_load_timing.pack_ms += ms_since(t_p);
        return true;
    };
    auto load_norm = [&](const std::string& gname, const std::string& key) -> bool {
        auto it = byname.find(gname);
        if (it == byname.end()) { if (err) *err = "Norm fehlt: " + gname; return false; }
        auto t_r = clk_ld::now();
        std::vector<float> f; if (!g.read_tensor_f32(*it->second, f, err)) return false;
        g_load_timing.read_ms += ms_since(t_r); ++g_load_timing.tensors;
        out.norms[key] = std::move(f);
        return true;
    };

    const int per_layer = 9;
    const int total = cfg.n_layers * per_layer + 3;  // + token_embd, output_norm, output
    int done = 0;
    auto tick = [&](const std::string& n) { if (progress) progress(n, ++done, total); else ++done; };

    if (!load_mat("token_embd.weight", "token_embd")) return false; tick("token_embd");
    for (int l = 0; l < cfg.n_layers; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        if (!load_norm(b + "attn_norm.weight",   b + "attn_norm"))   return false; tick(b + "attn_norm");
        if (!load_mat (b + "attn_q.weight",      b + "attn_q"))      return false; tick(b + "attn_q");
        if (!load_mat (b + "attn_k.weight",      b + "attn_k"))      return false; tick(b + "attn_k");
        if (!load_mat (b + "attn_v.weight",      b + "attn_v"))      return false; tick(b + "attn_v");
        if (!load_mat (b + "attn_output.weight", b + "attn_output")) return false; tick(b + "attn_output");
        if (!load_norm(b + "ffn_norm.weight",    b + "ffn_norm"))    return false; tick(b + "ffn_norm");
        if (!load_mat (b + "ffn_gate.weight",    b + "ffn_gate"))    return false; tick(b + "ffn_gate");
        if (!load_mat (b + "ffn_up.weight",      b + "ffn_up"))      return false; tick(b + "ffn_up");
        if (!load_mat (b + "ffn_down.weight",    b + "ffn_down"))    return false; tick(b + "ffn_down");
    }
    if (!load_norm("output_norm.weight", "output_norm")) return false; tick("output_norm");
    // output.weight ist optional: bei tied embeddings (kein eigener Tensor) == token_embd.
    if (byname.count("output.weight")) { if (!load_mat("output.weight", "output")) return false; }
    else { out.mats["output"] = out.mats["token_embd"]; }
    tick("output");
    // U7: Cache nach dem ersten Requant schreiben (best-effort; Fehler ist nicht fatal).
    if (cache_on) {
        auto t_w = clk_ld::now();
        const GgufMeta gm = gguf_meta(gguf_path);
        if (nv_cache_save(cache_path, out, gm, cfg, group, bits, attn_bits, block3))
            std::fprintf(stderr, "[real] .nv3w-Cache geschrieben (%.1f s): %s\n", ms_since(t_w) / 1e3, cache_path.c_str());
        else
            std::fprintf(stderr, "[real] .nv3w-Cache schreiben fehlgeschlagen (ignoriert): %s\n", cache_path.c_str());
    }
    return true;
}

// --- IInference ---------------------------------------------------------------
bool RealInference::load(const std::string& gguf_path, const Mistral3Config& cfg,
                         const std::string& tekken_dir, Options opt, std::string* err) {
    opt_ = opt;
    auto prog = [](const std::string& n, int i, int t) {
        if (i % 40 == 0 || i == t) std::fprintf(stderr, "[real] Gewichte %d/%d (%s)\n", i, t, n.c_str());
    };
    const int tbits = ss_tgt_bits();
    const int abits = ss_tgt_attn_bits();            // Mixed: Attention höher als FFN
    const bool b3 = ss_fmt_block3() && tbits == 3;   // block3 nur für 3-Bit-FFN; Attention ggf. nbit
    if (!load_int3_from_gguf(gguf_path, cfg, w_, 32, true, err, prog, b3, tbits, abits)) return false;
    if (!tok_.load(tekken_dir, err)) return false;
    loaded_ = true;
    std::fprintf(stderr, "[real] Modell geladen: ~%.1f GB FFN-INT%d%s / Attn-INT%d im RAM, Tokenizer bereit.\n",
                 double(w_.approx_bytes()) / 1e9, tbits, b3 ? "+BLOCK3" : "",
                 (abits > tbits ? abits : tbits));
#ifdef NOVA_HAVE_CUDA
    // Foundation: 24B-Forward on-GPU (Gewichte gestreamt, Aktivierungen+KV resident) — CPU aus dem Loop.
    if (g_use_gpu && ss_gpu_fwd24()) {
        std::string te;
        if (tgt_gpu_.init(w_, ss_tgt_kv(), &te, /*streamed=*/true, ss_kv_bits())) {
            tgt_gpu_active_ = true;
            std::fprintf(stderr, "[target] 24B on-GPU-Forward aktiv (Gewichte gestreamt, Akt.+KV resident, KV bis %d, kv_bits=%d).\n", ss_tgt_kv(), ss_kv_bits());
        } else {
            std::fprintf(stderr, "[target] GPU-Forward init fehlgeschlagen (%s) -> CPU-Glue.\n", te.c_str());
        }
    }
#endif
    return true;
}

bool RealInference::load_draft(const std::string& gguf_path, const Mistral3Config& cfg, std::string* err) {
    auto prog = [](const std::string&, int i, int t) {
        if (i % 40 == 0 || i == t) std::fprintf(stderr, "[draft] Gewichte %d/%d\n", i, t);
    };
    const int dbits = ss_draft_bits();
    if (!load_int3_from_gguf(gguf_path, cfg, draft_w_, 32, true, err, prog, false, dbits)) return false;
    draft_cfg_ = cfg; has_draft_ = true;
    std::fprintf(stderr, "[draft] 3B geladen: ~%.1f GB INT%d im RAM (Spec-Decoding aktiv, K=%d).\n",
                 double(draft_w_.approx_bytes()) / 1e9, dbits, spec_k_);
#ifdef NOVA_HAVE_CUDA
    // User-Entscheidung: das 3B-Draft ist PERMANENT VRAM-resident (nie streamen). GpuForward
    // residentisiert die Gewichte (INT3→INT8) UND hält Aktivierungen+KV im VRAM → der Draft-Forward
    // läuft komplett on-GPU, ohne die ~1456 Per-Matmul-Roundtrips/Spec-Schritt.
    if (g_use_gpu) {
        std::string de;
        const int dkv = ss_draft_kv();
        if (draft_gpu_.init(draft_w_, /*max_seq=*/dkv, &de, /*streamed=*/false, /*kv_bits=*/0,
                            /*want_batch=*/ss_spec_tree())) {   // Phase 6A-Wide: Baum-Draft braucht batched Puffer
            draft_gpu_active_ = true;
            std::fprintf(stderr, "[draft] 3B on-GPU-Forward aktiv (VRAM-resident, INT%d, KV bis %d, kein Streaming).\n",
                         dbits, dkv);
        } else {
            std::fprintf(stderr, "[draft] GpuForward init fehlgeschlagen (%s) -> CPU-Draft-Fallback.\n", de.c_str());
        }
    }
#endif
    return true;
}

// Target-Forward-Wrapper: on-GPU (tgt_gpu_, Gewichte gestreamt + Aktivierungen/KV resident, CPU aus
// dem Loop) wenn NOVA_GPU_FWD24 aktiv, sonst der bisherige CPU-Glue-Pfad (forward_step/-batch/kv_trim).
std::vector<float> RealInference::tgt_forward_step(int tok, int pos) {
#ifdef NOVA_HAVE_CUDA
    if (tgt_gpu_active_) { std::vector<float> o(w_.cfg.vocab); tgt_gpu_.step_logits(tok, pos, o.data()); return o; }
#endif
    return forward_step(w_, tok, pos, kv_);
}
std::vector<float> RealInference::tgt_forward_batch(const int* toks, int N, int start_pos) {
#ifdef NOVA_HAVE_CUDA
    if (tgt_gpu_active_) {
        std::vector<float> o(size_t(N) * w_.cfg.vocab);
        tgt_gpu_.forward_batch_gpu(toks, N, start_pos, o.data());
        return o;
    }
#endif
    return forward_batch(w_, toks, N, start_pos, kv_);
}
void RealInference::tgt_kv_trim(int len) {
#ifdef NOVA_HAVE_CUDA
    if (tgt_gpu_active_) { tgt_gpu_.trim(len); return; }
#endif
    kv_trim(kv_, len, w_.cfg);
}

// Phase 6B EAGLE: Trainingsdaten-Dump. Korpus (length-prefixed UTF-8 Texte) -> gepackte 512-Token-Fenster
// durch das 24B (forward_batch_hidden) -> je Position {Top-Layer-Hidden bf16, Input-Token, 24B-argmax}.
// Gepackte Fenster (start_pos=0, KV reset je Fenster) = volle Stream-Auslastung; die Hidden sind echte
// 24B-Berechnungen. EAGLE-1-Kontrakt: Kopf sagt f_s aus (embed(t_s), f_{s-1}) voraus + Token t_{s+1}=argmax.
// EGL2-Format: Header "EGL2"|H|V|count(int64), dann count Records {H×bf16 Top-Hidden, int32 in_tok, int32 label}.
int RealInference::dump_eagle(const std::string& corpus_path, const std::string& out_path) {
#ifdef NOVA_HAVE_CUDA
    if (g_use_gpu && ss_gpu_fwd24() && !tgt_gpu_active_ && loaded_) {
        std::string te; if (tgt_gpu_.init(w_, ss_tgt_kv(), &te, /*streamed=*/true, ss_kv_bits())) tgt_gpu_active_ = true;
    }
    if (!tgt_gpu_active_) { std::fprintf(stderr, "[eagle_dump] kein on-GPU-Target (NOVA_GPU_FWD24?)\n"); return 1; }
    if (!tok_.ready())    { std::fprintf(stderr, "[eagle_dump] kein Tokenizer\n"); return 2; }
    const int H = w_.cfg.hidden, V = w_.cfg.vocab;
    int W = GpuForward::MAX_BATCH;   // Fenstergröße: NOVA_EAGLE_WIN (Durchsatz-Optimum ~128, compute-bound)
    { const char* e = std::getenv("NOVA_EAGLE_WIN"); if (e) { int w = std::atoi(e); if (w >= 1 && w <= GpuForward::MAX_BATCH) W = w; } }
    std::fprintf(stderr, "[eagle_dump] Fenster W=%d\n", W);
    std::ifstream in(corpus_path, std::ios::binary);
    if (!in) { std::fprintf(stderr, "[eagle_dump] corpus nicht lesbar: %s\n", corpus_path.c_str()); return 3; }
    // Akkumulieren über mehrere Läufe: NOVA_EAGLE_APPEND hängt Records an ein bestehendes EGL2 an
    // (Header wird NICHT neu geschrieben; train_eagle.py leitet die Record-Anzahl aus der Dateigröße ab).
    // NOVA_EAGLE_SKIP überspringt die ersten N Korpus-Samples → jeder Lauf verarbeitet einen anderen
    // Korpus-Abschnitt statt dieselben Tokens doppelt zu dumpen.
    const bool want_append = []() { const char* e = std::getenv("NOVA_EAGLE_APPEND"); return e && e[0] && e[0] != '0'; }();
    const long skip_samples = []() { const char* e = std::getenv("NOVA_EAGLE_SKIP"); return e ? std::atol(e) : 0L; }();
    // Format: NOVA_EAGLE_EGL3 = alle 3 Taps (low/mid/high). NOVA_EAGLE_INT8 = int8-Hidden (halbe Disk, F4):
    // je Tap [bf16 scale][H int8], symmetrisch amax/127. Magic EGL2/EGL3 (bf16) bzw. EGQ2/EGQ3 (int8).
    const bool egl3  = []() { const char* e = std::getenv("NOVA_EAGLE_EGL3"); return e && e[0] && e[0] != '0'; }();
    const bool quant = []() { const char* e = std::getenv("NOVA_EAGLE_INT8"); return e && e[0] && e[0] != '0'; }();
    const int  ntaps = egl3 ? 3 : 1;
    const char* magic = quant ? (egl3 ? "EGQ3" : "EGQ2") : (egl3 ? "EGL3" : "EGL2");
    const long long base = 20;                                   // magic(4)+H(4)+V(4)+count(8)
    const long long rec  = quant ? (long long)ntaps * (H + 2) + 8 : (long long)ntaps * H * 2 + 8;
    // F1 Self-healing Append: Datei vor Append auf Record-Grenze truncaten (Torn-Tail eines gekillten Laufs
    // verwerfen) + Magic prüfen → verhindert die Misalignment-Korruption, die 546k Records zerstört hatte.
    bool do_append = false;
    if (want_append) {
        std::error_code ec; long long fsz = (long long)std::filesystem::file_size(out_path, ec);
        if (!ec && fsz >= base) {
            char m[4] = {0}; { std::ifstream chk(out_path, std::ios::binary); chk.read(m, 4); }
            if (std::memcmp(m, magic, 4) != 0) {
                std::fprintf(stderr, "[eagle_dump] APPEND-Format-Mismatch (%.4s != %s) → Abbruch\n", m, magic); return 5;
            }
            const long long clean = base + ((fsz - base) / rec) * rec;
            if (clean != fsz) {
                std::filesystem::resize_file(out_path, (uintmax_t)clean, ec);
                std::fprintf(stderr, "[eagle_dump] self-heal: %lld→%lld B (Torn-Tail %lld B verworfen)\n", fsz, clean, fsz - clean);
            }
            do_append = true;
        }
    }
    std::fstream out;
    if (do_append) out.open(out_path, std::ios::binary | std::ios::in | std::ios::out);
    else           out.open(out_path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out) { std::fprintf(stderr, "[eagle_dump] out nicht schreibbar: %s\n", out_path.c_str()); return 4; }
    int64_t count = 0;
    if (do_append) {
        std::error_code ec; long long fsz = (long long)std::filesystem::file_size(out_path, ec);
        count = (fsz - base) / rec;                              // Record-Zahl = Wahrheit aus Dateigröße
        out.seekp(0, std::ios::end);
    } else {
        out.write(magic, 4);
        int32_t h32 = H, v32 = V; out.write((const char*)&h32, 4); out.write((const char*)&v32, 4);
        int64_t z = 0; out.write((const char*)&z, 8);
    }
    std::fprintf(stderr, "[eagle_dump] %s %s (skip=%ld, rec=%lld B, count=%lld)\n",
                 do_append ? "APPEND" : "neu", magic, skip_samples, rec, (long long)count);
    std::vector<float> hid((size_t)3 * H * W), logits((size_t)V * W);
    std::vector<uint16_t> recbf((size_t)ntaps * H);   // fp16-Pfad: 1 (EGL2) oder 3 (EGL3) Taps
    std::vector<int8_t>   reci8((size_t)H);           // int8-Pfad: je Tap H Werte
    std::vector<int> buf; buf.reserve((size_t)W * 2);
    const long maxtok = []() { const char* e = std::getenv("NOVA_EAGLE_MAXTOK"); return e ? std::atol(e) : -1L; }();
    long total_tok = 0; int windows = 0, samples = 0;
    const auto t0 = std::chrono::steady_clock::now();
    auto flush = [&](int nn) {
        tgt_gpu_.reset();
        tgt_gpu_.forward_batch_hidden(buf.data(), nn, 0, logits.data(), hid.data());
        // hid-Layout [3][H][W]: Tap k, Hidden hh, Position p = hid[(k*H+hh)*W + p]. EGL2 nur Tap 2 (top),
        // EGL3 alle 3 (low=Tap0, mid=Tap1, high=Tap2) hintereinander je Record → Multi-Layer-Fusion-Input.
        const int t0tap = egl3 ? 0 : 2;                // EGL2/EGQ2 = Tap 2 (top), EGL3/EGQ3 = Taps 0..2
        for (int p = 0; p < nn; ++p) {
            const int32_t lbl = argmax_ptr(&logits[(size_t)p * V], V);     // 24B-Greedy für p+1
            const int32_t in_tok = buf[(size_t)p];                        // Input-Token t_p (für embed)
            for (int k = 0; k < ntaps; ++k) {
                const float* tap = &hid[(size_t)(t0tap + k) * H * W];
                if (quant) {                                              // int8: [bf16 scale][H int8] je Tap
                    float amax = 0.f;
                    for (int hh = 0; hh < H; ++hh) { const float a = std::fabs(tap[(size_t)hh * W + p]); if (a > amax) amax = a; }
                    const float scale = amax > 0.f ? amax / 127.f : 1.f;
                    uint32_t su; std::memcpy(&su, &scale, 4); const uint16_t sbf = (uint16_t)(su >> 16);
                    const float inv = 1.f / scale;
                    for (int hh = 0; hh < H; ++hh) {
                        int q = (int)std::lround(tap[(size_t)hh * W + p] * inv);
                        reci8[hh] = (int8_t)(q < -127 ? -127 : (q > 127 ? 127 : q));
                    }
                    out.write((const char*)&sbf, 2);
                    out.write((const char*)reci8.data(), (std::streamsize)H);
                } else {                                                  // bf16 (obere 16 Bit)
                    for (int hh = 0; hh < H; ++hh) {
                        uint32_t u; const float f = tap[(size_t)hh * W + p]; std::memcpy(&u, &f, 4);
                        recbf[(size_t)k * H + hh] = (uint16_t)(u >> 16);
                    }
                }
            }
            if (!quant) out.write((const char*)recbf.data(), (std::streamsize)((size_t)ntaps * H * 2));
            out.write((const char*)&in_tok, 4); out.write((const char*)&lbl, 4); ++count;
        }
        ++windows; total_tok += nn;
        if (windows % 20 == 0) {
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "[eagle_dump] samples=%d windows=%d recs=%lld %.0f tok/s\n",
                         samples, windows, (long long)count, s > 0 ? total_tok / s : 0.0);
        }
    };
    uint32_t len;
    long seen = 0;
    while (in.read((char*)&len, 4)) {
        if (len == 0 || len > 40000u) { in.seekg(len, std::ios::cur); continue; }
        if (seen++ < skip_samples) { in.seekg(len, std::ios::cur); continue; }   // NOVA_EAGLE_SKIP
        std::string text(len, '\0'); in.read(&text[0], len);
        std::vector<int> ids = tok_.encode(text, /*add_bos=*/true);
        ids.push_back(tok_.eos());
        for (int id : ids) buf.push_back(id);
        ++samples;
        while ((int)buf.size() >= W) { flush(W); buf.erase(buf.begin(), buf.begin() + W); }
        if (maxtok > 0 && total_tok >= maxtok) break;
    }
    if ((int)buf.size() >= 2) flush((int)buf.size());
    out.seekp(12, std::ios::beg); out.write((const char*)&count, 8);   // F2: Header-count IMMER aktualisieren (auch Append)
    out.close();
    // F3: letzten konsumierten Sample-Index (seen) als next-skip ausgeben → Treiber setzt lückenlos fort.
    std::fprintf(stderr, "[eagle_dump] FERTIG (%s %s): %lld Records total, %d windows, %ld Tokens, next-skip=%ld -> %s\n",
                 do_append ? "append" : "neu", magic, (long long)count, windows, total_tok, seen, out_path.c_str());
    return 0;
#else
    (void)corpus_path; (void)out_path; return 9;
#endif
}

// Ein Spec-Schritt (greedy): Draft schlägt K Tokens vor, Target verifiziert sie in EINEM
// batched Forward, greedy-Akzeptanz + KV-Rollback. Ergebnis ist bit-identisch zu greedy
// Target-Dekodierung — nur schneller (mehrere Tokens pro 24B-Weight-Read).
void RealInference::spec_step() {
    const int V = w_.cfg.vocab, P = pos_;
    // Perf Phase 6: adaptives K aus der Accept-EMA (Ziel: knapp über der erwarteten Accept-Länge).
    int K = spec_k_;
    if (ss_adaptive_k()) {
        if (acc_ema_ < 0.0) acc_ema_ = spec_k_ * 0.5;   // Init
        int ak = int(acc_ema_ + 2.0 + 0.5);             // round(EMA + 2)
        K = ak < 2 ? 2 : (ak > spec_k_ ? spec_k_ : ak);
    }
#ifdef NOVA_HAVE_CUDA
    // ---- Phase 6A Baum-Verify: m=2 Draft-Ketten (greedy + 1 Verzweigung), ein 24B-Stream, greedy-exakt ----
    if (ss_spec_tree() && ss_single_stream() && draft_gpu_active_ && P >= 1) {
        const int js = 1;                                     // Verzweigungsposition
        std::vector<int> a(K), b(K, -1);
        a[0] = draft_next_;                                   // greedy @P
        for (int j = 0; j < K; ++j) {                         // chain0 greedy; Top-2 nur bei j=js-1 (liefert b[js])
            if (j == js - 1) { int t2[2]; draft_gpu_.step_top2(a[j], P + j, t2); if (j + 1 < K) { a[j + 1] = t2[0]; b[js] = t2[1]; } }
            else            { const int nx = draft_gpu_.step_argmax(a[j], P + j); if (j + 1 < K) a[j + 1] = nx; }
        }
        std::vector<int> c1(K);                               // chain1: a[0..js-1] + b[js] + greedy
        for (int j = 0; j < js; ++j) c1[j] = a[j];
        draft_gpu_.trim(P + js);
        int tok = (b[js] >= 0 ? b[js] : a[js]);
        c1[js] = tok;
        for (int j = js; j < K - 1; ++j) { tok = draft_gpu_.step_argmax(tok, P + j); c1[j + 1] = tok; }
        std::vector<int>& c0 = a;
        // Flacher Baum: [pend@P-1, c0@P.., c1@P..], start_pos=P-1, KV-Slots pend=P-1, c0[j]=P+j, c1[j]=P+K+j.
        const int NB = 1 + 2 * K;
        std::vector<int> seq(NB), tpos(NB); std::vector<unsigned char> tmask((size_t)NB * NB, 0);
        seq[0] = ss_pend_tok_; tpos[0] = P - 1; tmask[0] = 1;
        for (int j = 0; j < K; ++j) {
            const int n0 = 1 + j, n1 = 1 + K + j;
            seq[n0] = c0[j]; tpos[n0] = P + j; tmask[(size_t)n0 * NB + 0] = 1;
            for (int p = 0; p <= j; ++p) tmask[(size_t)n0 * NB + (1 + p)] = 1;
            seq[n1] = c1[j]; tpos[n1] = P + j; tmask[(size_t)n1 * NB + 0] = 1;
            for (int p = 0; p <= j; ++p) tmask[(size_t)n1 * NB + (1 + K + p)] = 1;
        }
        std::vector<float> batch((size_t)NB * V);
        tgt_gpu_.forward_batch_gpu(seq.data(), NB, P - 1, batch.data(), tpos.data(), tmask.data());
        // Accept: greedy-Pfad folgen (c0; bei erster Divergenz an js -> c1 falls Match).
        std::vector<int> emit; int accepted = 0; bool onc1 = false; int pred = 0;
        for (int depth = 0; depth < K; ++depth) {
            const int t = argmax_ptr(&batch[(size_t)pred * V], V);
            if (!onc1 && c0[depth] == t)                       { emit.push_back(t); ++accepted; pred = 1 + depth; }
            else if (!onc1 && depth == js && c1[depth] == t)   { onc1 = true; emit.push_back(t); ++accepted; pred = 1 + K + depth; }
            else if (onc1 && c1[depth] == t)                   { emit.push_back(t); ++accepted; pred = 1 + K + depth; }
            else { emit.push_back(t); break; }                 // Mismatch: Target-Token, stop
        }
        if (accepted == K) emit.push_back(argmax_ptr(&batch[(size_t)pred * V], V));   // Bonus
        const int last = emit.back();
        if (onc1 && accepted > 0) tgt_gpu_.gather_kv_block(P + K, P, accepted);       // c1-KV -> [P..]
        tgt_kv_trim(P + accepted);
        ss_pend_tok_ = last;
        const std::vector<int>& win = onc1 ? c1 : c0;                                 // Draft-KV für Gewinnpfad neu bauen
        draft_gpu_.trim(P);
        for (int i = 0; i < accepted; ++i) draft_gpu_.step_argmax(win[i], P + i);
        draft_next_ = draft_gpu_.step_argmax(last, P + accepted);
        pos_ = P + accepted + 1;
        for (int t : emit) emit_queue_.push_back(t);
        ++spec_steps_; spec_accepted_ += accepted;
        if (ss_adaptive_k()) acc_ema_ = 0.7 * acc_ema_ + 0.3 * double(accepted);
        return;
    }
#endif
    // 1. Draft K Tokens greedy (3B). On-GPU (resident, kein Roundtrip) oder CPU-Fallback.
    std::vector<int> draft(K);
    const bool rankm = ss_spec_rank();                   // Phase 0: 3B-Logits je Position erfassen
    std::vector<std::vector<float>> draftL;              // draftL[j] = 3B-Dist für Position P+j+1
    if (rankm) draftL.resize(K);
#ifdef NOVA_HAVE_CUDA
    if (draft_gpu_active_) {
        draft[0] = draft_next_;                          // greedy-Token @P (aus Prefill/Vorschritt)
        for (int j = 0; j < K; ++j) {
            int nxt;
            if (rankm) { draftL[j].resize(V); nxt = draft_gpu_.step_argmax(draft[j], P + j, draftL[j].data()); }
            else       { nxt = draft_gpu_.step_argmax(draft[j], P + j); }   // KV -> P+j+1, argmax @P+j+1
            if (j + 1 < K) draft[j + 1] = nxt;
        }
    } else
#endif
    {
        std::vector<float> dl = draft_pending_;
        for (int j = 0; j < K; ++j) {
            const int dj = argmax_ptr(dl.data(), V);
            draft[j] = dj;
            dl = forward_step(draft_w_, dj, P + j, draft_kv_);   // draft_kv_ -> P+K
        }
    }
    int accepted = 0; std::vector<int> emit; int last;
    if (ss_single_stream()) {
        // 2s. Single-Stream: den aufgeschobenen Token (ss_pend_tok_ @P-1, KV noch nicht
        //     geschrieben) dem Batch voranstellen. EIN forward_batch schreibt dessen KV UND
        //     liefert die Verify-Logits — kein separater forward_step (kein 2. 24B-Stream).
        std::vector<int> seq; seq.reserve(size_t(K) + 1);
        seq.push_back(ss_pend_tok_);
        for (int j = 0; j < K; ++j) seq.push_back(draft[j]);
        std::vector<float> batch = tgt_forward_batch(seq.data(), K + 1, P - 1);  // -> P+K-1
        // batch[j] = Logit an Position P+j  (batch[0] ersetzt das frühere pending_).
        for (int i = 0; i < K; ++i) {
            const int ai = argmax_ptr(&batch[size_t(i) * V], V);   // Target-Greedy @P+i
            if (rankm && i >= 1) {   // Rang von ai im 3B-Dist für P+i (= draftL[i-1]); Prefix hier korrekt
                const float* dl = draftL[size_t(i) - 1].data(); const float tv = dl[ai];
                int rank = 0; for (int v = 0; v < V; ++v) if (dl[v] > tv) ++rank;
                rank_record(rank, draft[i] != ai);       // divergence = wo linear scheitert
            }
            if (draft[i] == ai) { emit.push_back(draft[i]); ++accepted; }
            else { emit.push_back(ai); break; }
        }
        if (accepted == K) emit.push_back(argmax_ptr(&batch[size_t(K) * V], V));  // Bonus L_{P+K}
        last = emit.back();
        // KV [0..P+accepted-1] behalten (pend_tok_ @P-1 + accepted Drafts); last @P+accepted
        // wird aufgeschoben und ist Kopf des nächsten Batch.
        tgt_kv_trim(P + accepted);
        ss_pend_tok_ = last;
    } else {
        // 2. Target verifiziert alle K in einem Pass.
        std::vector<float> batch = tgt_forward_batch(draft.data(), K, P);   // -> P+K
        // 3. Greedy-Akzeptanz.
        for (int i = 0; i < K; ++i) {
            const float* Lrow = (i == 0) ? pending_.data() : &batch[size_t(i - 1) * V];
            const int ai = argmax_ptr(Lrow, V);                  // Target-Greedy an Position P+i
            if (draft[i] == ai) { emit.push_back(draft[i]); ++accepted; }
            else { emit.push_back(ai); break; }                  // Mismatch: Target-Token, stop
        }
        if (accepted == K) emit.push_back(argmax_ptr(&batch[size_t(K - 1) * V], V));  // Bonus L_{P+K}
        // 4. KV auf akzeptierte Länge zurückschneiden, letzten (neuen) Token einspeisen.
        tgt_kv_trim(P + accepted);
        last = emit.back();
        pending_ = tgt_forward_step(last, P + accepted);            // -> P+accepted+1
    }
#ifdef NOVA_HAVE_CUDA
    if (draft_gpu_active_) {
        draft_gpu_.trim(P + accepted);
        draft_next_ = draft_gpu_.step_argmax(last, P + accepted);    // draft-KV -> P+accepted+1, argmax @P+accepted+1
    } else
#endif
    {
        kv_trim(draft_kv_, P + accepted, draft_cfg_);
        draft_pending_ = forward_step(draft_w_, last, P + accepted, draft_kv_);
    }
    pos_ = P + accepted + 1;
    for (int t : emit) emit_queue_.push_back(t);
    ++spec_steps_; spec_accepted_ += accepted;
    if (ss_adaptive_k()) acc_ema_ = 0.7 * acc_ema_ + 0.3 * double(accepted);   // Accept-EMA für nächstes K
}

int RealInference::sample(const std::vector<float>& logits) const {
    int best = 0; float bv = logits.empty() ? 0.0f : logits[0];
    for (int i = 1; i < int(logits.size()); ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

void RealInference::begin(const GenRequest& req) {
#ifdef NOVA_HAVE_CUDA
    // Lazy-Init des on-GPU-Targets (deckt auch set_weights-Testpfad ab; no-op wenn load() es schon tat).
    if (g_use_gpu && ss_gpu_fwd24() && !tgt_gpu_active_ && loaded_) {
        std::string te; if (tgt_gpu_.init(w_, ss_tgt_kv(), &te, /*streamed=*/true, ss_kv_bits())) tgt_gpu_active_ = true;
    }
#endif
    pos_ = 0; produced_ = 0; done_ = false; emit_queue_.clear();
    spec_steps_ = 0; spec_accepted_ = 0;
    std::string prompt = req.prefix;
    if (!req.dynamic.empty()) { if (!prompt.empty()) prompt += "\n"; prompt += req.dynamic; }
    // Mistral-Instruct-Template: der tekken-Tokenizer matcht [INST]/[/INST] als Control-Tokens,
    // add_bos setzt <s> -> <s>[INST] … [/INST]. Ohne das echot das Instruct-Modell nur den Kontext.
    if (req.instruct) prompt = "[INST] " + prompt + " [/INST]";
    std::vector<int> ids = tok_.ready() ? tok_.encode(prompt, /*add_bos=*/true)
                                        : std::vector<int>{w_.cfg.bos_id};
    if (ids.empty()) ids.push_back(w_.cfg.bos_id);
    std::vector<float> logits, dlogits;
    // Single-Stream (nur Spec): den letzten Prompt-Token NICHT ins Target-KV schreiben —
    // er wird als ss_pend_tok_ aufgeschoben und ist Kopf des ersten Verify-Batch.
    const bool ss = ss_single_stream() && has_draft_;
    const int n = int(ids.size());
    const int tgt_n = ss ? n - 1 : n;                 // Single-Stream: letzten Token aufschieben

    // Perf Phase 2: Longest-Common-Prefix gegen den letzten begin() -> residenten KV wiederverwenden
    // statt neu zu prefillen. reuse=0 => voller Prefill (alter Pfad). Greedy-exakt (identische
    // Prefix-Tokens => identisches KV). Nur so weit wie zuletzt KV geschrieben (pc_tgt_n_), und es
    // bleibt >=1 Target-Token neu zu prefillen (pending_/ss brauchen die letzten Logits).
    int reuse = 0;
    const bool use_pc = pc_override_ < 0 ? ss_prefix_cache() : (pc_override_ == 1);
    if (use_pc && pc_valid_) {
        int lc = 0; const int m = std::min<int>(int(ids.size()), int(pc_ids_.size()));
        while (lc < m && ids[lc] == pc_ids_[lc]) ++lc;
        reuse = std::min(lc, pc_tgt_n_);
        if (reuse > tgt_n - 1) reuse = tgt_n - 1;
        if (reuse < 1) reuse = 0;
    }
    last_reuse_ = reuse;
    if (reuse > 0) {                                   // Prefix behalten: auf `reuse` trimmen (nicht reset)
        kv_trim(kv_, reuse, w_.cfg);
        if (has_draft_) kv_trim(draft_kv_, reuse, draft_cfg_);
#ifdef NOVA_HAVE_CUDA
        if (tgt_gpu_active_) tgt_gpu_.trim(reuse);
        if (has_draft_ && draft_gpu_active_) draft_gpu_.trim(reuse);
#endif
    } else {                                           // voller Reset (alter Pfad)
        kv_.reset(w_.cfg.n_layers);
        if (has_draft_) draft_kv_.reset(draft_cfg_.n_layers);
#ifdef NOVA_HAVE_CUDA
        if (has_draft_ && draft_gpu_active_) draft_gpu_.reset();
        if (tgt_gpu_active_) tgt_gpu_.reset();
#endif
    }
    // TARGET-Prefill GEBATCHT ab `reuse`: Chunks von MB -> das 24B streamt EINMAL je Chunk.
    const int V = w_.cfg.vocab, MB = ss_prefill_chunk();   // Phase 5: default 256 (war 32)
    for (int c0 = reuse; c0 < tgt_n; c0 += MB) {
        const int c1 = std::min(tgt_n, c0 + MB);
        std::vector<float> batch = tgt_forward_batch(ids.data() + c0, c1 - c0, c0);   // schreibt KV[c0..c1-1]
        if (!ss && c1 == tgt_n) {                     // non-Single-Stream: pending_ = Logits des letzten Tokens
            const size_t last = size_t(c1 - c0 - 1);
            logits.assign(batch.begin() + last * V, batch.begin() + (last + 1) * V);
        }
    }
    // DRAFT-Prefill ab `reuse`: resident (kein 24B-Stream), per Token günstig -> draft-KV[reuse..n-1].
    if (has_draft_) {
        for (int i = reuse; i < n; ++i) {
#ifdef NOVA_HAVE_CUDA
            if (draft_gpu_active_) { draft_next_ = draft_gpu_.step_argmax(ids[i], i); continue; }
#endif
            dlogits = forward_step(draft_w_, ids[i], i, draft_kv_);
        }
    }
    pos_ = int(ids.size());
    if (ss) ss_pend_tok_ = ids.back();   // KV[n-1] aufgeschoben, kv_ hat [0..n-2]
    pending_ = std::move(logits);
    if (has_draft_ &&
#ifdef NOVA_HAVE_CUDA
        !draft_gpu_active_ &&
#endif
        true) draft_pending_ = std::move(dlogits);
    have_pending_ = true;
    // Prefix-Cache-State für den nächsten begin(): Tokens + geschriebene Target-KV-Länge.
    pc_ids_ = ids; pc_tgt_n_ = tgt_n; pc_valid_ = true;
}

int RealInference::next_token_id() {
    if (done_ || !loaded_) return -1;
    if (has_draft_) {
        if (emit_queue_.empty()) {
            if (produced_ >= opt_.max_new) { done_ = true; return -1; }
            spec_step();
        }
        if (emit_queue_.empty()) { done_ = true; return -1; }
        const int tid = emit_queue_.front(); emit_queue_.pop_front();
        if (tid == w_.cfg.eos_id) { done_ = true; return -1; }
        ++produced_;
        return tid;
    }
    // Plain greedy (kein Draft).
    if (!have_pending_) return -1;
    const int tid = sample(pending_);
    if (tid == w_.cfg.eos_id || produced_ >= opt_.max_new) { done_ = true; return -1; }
    ++produced_;
    pending_ = tgt_forward_step(tid, pos_);
    ++pos_;
    return tid;
}

std::string RealInference::next_token() {
    const int tid = next_token_id();
    if (tid < 0) return "";
    return tok_.ready() ? tok_.decode(std::vector<int>{tid}) : std::string();
}

}  // namespace nova::infer

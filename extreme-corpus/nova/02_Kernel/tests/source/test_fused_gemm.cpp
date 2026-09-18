// test_fused_gemm.cpp — Fused INT3-Dequant-GEMM Numerik (Kernel-Upgrade ②).
//
// Beweist: die INLINE-Dequantisierung im Matmul (kein FP16-Zwischenbuffer) ist
// numerisch äquivalent zur vollen FP32-GEMV — Cosine-Sim > 0,99 (wie TB3-Kriterium).
#include "InferEngine/fused_gemm.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

using namespace nova::infer;

namespace {
double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
}
// Deterministischer PRNG.
struct Rng { uint64_t s; float next() { s = s * 6364136223846793005ULL + 1ULL; return float(int32_t(s >> 32)) / 2.1e9f; } };
}  // namespace

int main() {
    std::cout << "=== Fused INT3-Dequant-GEMM (②) ===\n";
    const int M = 96, K = 256;
    Rng rng{0xC0FFEE};

    std::vector<float> W(size_t(M) * K), x(K);
    for (auto& w : W) w = rng.next();          // ~[-1,1]
    for (auto& v : x) v = rng.next();

    const Int3Matrix q = pack_int3(W.data(), M, K, /*group=*/32);
    std::printf("  W: %dx%d FP32 = %zu B | INT3-packed = %zu B + %zu Skalen (%.2f×)\n",
                M, K, W.size() * sizeof(float), q.packed.size(), q.scales.size(),
                double(W.size() * sizeof(float)) / double(q.packed.size() + q.scales.size() * 4));

    std::vector<float> y_ref(M), y_fused(M);
    reference_gemv(W.data(), M, K, x.data(), y_ref.data());
    fused_int3_gemv(q, x.data(), y_fused.data());

    const double cs = cosine(y_ref, y_fused);
    std::printf("  Cosine-Sim (fused vs FP32) = %.5f (Ziel > 0,99)\n", cs);

    bool pass = (cs > 0.99);

#ifdef NOVA_HAVE_CUDA
    std::vector<float> y_cuda(M);
    fused_int3_gemv_cuda(q, x.data(), y_cuda.data());
    const double cs_cuda = cosine(y_ref, y_cuda);
    std::printf("  Cosine-Sim (CUDA fused vs FP32) = %.5f (Ziel > 0,99)\n", cs_cuda);
    pass &= (cs_cuda > 0.99);
#endif

    // --- Phase 2 FMT_BLOCK3: zweistufige Skalen. Gate = cosine gegen den ALTEN Pfad (INT3/FP32),
    //     der die reine Skalen-Requantisierung isoliert. Zusätzlich vs FP32-Referenz zur Info.
    //     Größere Matrix (mehrere Superblöcke/Zeile + Skalen-Streuung) als schärferer Test.
    {
        std::cout << "\n=== FMT_BLOCK3 (zweistufige Skalen) ===\n";
        const int Mb = 256, Kb = 640;   // Kb/32 = 20 Gruppen -> 3 Superblöcke/Zeile (Rand-Superblock)
        Rng rb{0xB10C3};
        std::vector<float> Wb(size_t(Mb) * Kb), xb(Kb);
        for (auto& w : Wb) w = rb.next() * (0.2f + 0.8f * std::fabs(rb.next()));  // Skalen-Streuung
        for (auto& v : xb) v = rb.next();
        const Int3Matrix qi = pack_int3  (Wb.data(), Mb, Kb, 32);   // alter Pfad
        const Int3Matrix qb = pack_block3(Wb.data(), Mb, Kb, 32);   // BLOCK3
        // Gewichte müssen bit-identisch sein (nur Skalen unterscheiden sich).
        const bool weights_equal = (qi.packed == qb.packed);
        std::printf("  packed-Bytes identisch (INT3 == BLOCK3): %s\n", weights_equal ? "ja" : "NEIN");
        const size_t sc_fp32 = qi.scales.size() * 4;
        const size_t sc_b3   = qb.superscales.size() * 2 + qb.subscales.size();
        std::printf("  Skalen-Bytes: FP32 %zu -> BLOCK3 %zu (%.1f%% der FP32, effektiv %.3f bit/Gewicht)\n",
                    sc_fp32, sc_b3, 100.0 * double(sc_b3) / double(sc_fp32),
                    8.0 * double(sc_b3) / (double(Mb) * Kb));
        std::vector<float> y_ib(Mb), y_bb(Mb), y_rb(Mb);
        reference_gemv(Wb.data(), Mb, Kb, xb.data(), y_rb.data());
        fused_int3_gemv(qi, xb.data(), y_ib.data());               // alter Pfad
        Int3Matrix qb_fp = qb; qb_fp.fmt = SCALE_FP32;             // host: BLOCK3 -> FP32-Effektivskalen
        block3_expand_scales(qb, qb_fp.scales);
        fused_int3_gemv(qb_fp, xb.data(), y_bb.data());
        const double cs_b_vs_i  = cosine(y_ib, y_bb);             // GATE: vs altem Pfad
        const double cs_b_vs_fp = cosine(y_rb, y_bb);             // Info: vs FP32
        std::printf("  Cosine (BLOCK3 host vs INT3-Pfad) = %.5f  [GATE > 0,99]\n", cs_b_vs_i);
        std::printf("  Cosine (BLOCK3 host vs FP32-ref)  = %.5f  (Info)\n", cs_b_vs_fp);
        pass &= weights_equal && (cs_b_vs_i > 0.99);
#ifdef NOVA_HAVE_CUDA
        std::vector<float> y_bcu(Mb);
        fused_int3_gemv_cuda(qb, xb.data(), y_bcu.data());        // testet expand_block3_scales_kernel
        const double cs_b_cuda = cosine(y_ib, y_bcu);
        std::printf("  Cosine (BLOCK3 CUDA vs INT3-Pfad) = %.5f  [GATE > 0,99]\n", cs_b_cuda);
        pass &= (cs_b_cuda > 0.99);
#endif
    }

    // --- GEMM N>1 (Phase-4-Gate): der bisherige Test deckt nur GEMV/N=1. Der Verify-Pfad
    //     (Spec-Decoding N=K, Prefill-Chunks, Tree-Breite) nutzt den GEMM-Kernel mit N Spalten.
    //     Ein zukünftiger schneller GEMM (dp4a/Tensor-Cores, INT8-Aktivierungen) MUSS dieses Gate
    //     bestehen: GEMM(N) == N unabhängige GEMV (spaltenweise). Deckt N>32 (grid.y-Kachelung) und
    //     das BLOCK3-Skalenformat (das reale gestreamte 24B-Format) ab.
    {
        std::cout << "\n=== Fused INT3-GEMM N>1 (Verify/Prefill-Pfad, Phase-4-Gate) ===\n";
        const int Mg = 128, Kg = 512;                 // Kg/32 = 16 Gruppen -> 2 Superblöcke/Zeile
        Rng rg{0x6E33A};
        std::vector<float> Wg(size_t(Mg) * Kg);
        for (auto& w : Wg) w = rg.next() * (0.2f + 0.8f * std::fabs(rg.next()));   // Skalen-Streuung
        const Int3Matrix q_fp = pack_int3  (Wg.data(), Mg, Kg, 32);   // SCALE_FP32
        Int3Matrix q_b3       = pack_block3(Wg.data(), Mg, Kg, 32);   // BLOCK3 (24B-Format)
        Int3Matrix q_b3_host  = q_b3; q_b3_host.fmt = SCALE_FP32;     // Host-Referenz: BLOCK3 -> FP32-Skalen
        block3_expand_scales(q_b3, q_b3_host.scales);
        // N=8 (Verify), N=40 (>32 -> mehr als eine grid.y-Kachel).
        for (int N : {8, 40}) {
            std::vector<float> Xcol(size_t(Kg) * N);
            for (auto& v : Xcol) v = rg.next();
            // Referenz: N unabhängige GEMV (Spalte n = X[:,n]) über den FP32-Skalen-Pfad.
            std::vector<float> Yref(size_t(Mg) * N), xn(Kg), yn(Mg);
            for (int n = 0; n < N; ++n) {
                for (int k = 0; k < Kg; ++k) xn[k] = Xcol[size_t(k) * N + n];
                fused_int3_gemv(q_fp, xn.data(), yn.data());
                for (int m = 0; m < Mg; ++m) Yref[size_t(m) * N + n] = yn[m];
            }
            std::vector<float> Yg(size_t(Mg) * N);
            fused_int3_gemm(q_fp, Xcol.data(), N, Yg.data());          // Host-GEMM (identischer Pfad)
            const double cs_host = cosine(Yref, Yg);
            std::printf("  N=%2d Cosine (Host-GEMM vs N×GEMV)     = %.5f  [GATE > 0,999]\n", N, cs_host);
            pass &= (cs_host > 0.999);
#ifdef NOVA_HAVE_CUDA
            std::vector<float> Ycu(size_t(Mg) * N), Yb3(size_t(Mg) * N);
            fused_int3_gemm_cuda(q_fp, Xcol.data(), N, Ycu.data());    // CUDA-GEMM (FP32-Skalen)
            const double cs_cuda = cosine(Yref, Ycu);
            std::printf("  N=%2d Cosine (CUDA-GEMM vs N×GEMV)     = %.5f  [GATE > 0,99]\n", N, cs_cuda);
            pass &= (cs_cuda > 0.99);
            fused_int3_gemm_cuda(q_b3, Xcol.data(), N, Yb3.data());    // CUDA-GEMM (BLOCK3 = 24B-Format)
            const double cs_b3 = cosine(Yref, Yb3);
            std::printf("  N=%2d Cosine (CUDA-GEMM BLOCK3 vs GEMV)= %.5f  [GATE > 0,99]\n", N, cs_b3);
            pass &= (cs_b3 > 0.99);
#endif
        }
    }

    std::cout << "\n=== Fused INT3-GEMM: " << (pass ? "BESTANDEN" : "FEHLGESCHLAGEN") << " ===\n";
    return pass ? 0 : 1;
}

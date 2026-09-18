// testbed7_spec_decode.cpp — Speculative Decoding (Design §16, TB 7).
//
// Kriterien: "Accept-Rate > 60%, > 6 tok/s."
//
// Teil 1 (Korrektheit): Beweist die mathematische Äquivalenz aus §8 — die
// emittierten Token sind exakt aus der Target-Verteilung p gesampelt, egal wie
// stark das Draft-q abweicht. Gemessen über die Total-Variation-Distanz zwischen
// empirischer Verteilung des ersten emittierten Tokens und p (muss ~0 sein).
//
// Teil 2 (Durchsatz): Realistisches Draft q = 0.9*p + 0.1*uniform. Gemessen wird
// die Accept-Rate (akzeptiert/angeboten). Da kein reales Modell vorliegt, werden
// die tok/s aus der gemessenen Tokens/Schritt und den Design-Latenzen (§8:
// Verify ~0,5 s, Draft ~8 ms/Token) projiziert — analog zu Block A's Surrogaten.
#include "InferEngine/spec_decoder.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace nova;
using nova::infer::ProbDist;
using nova::infer::SpecConfig;
using nova::infer::SpeculativeDecoder;

namespace {

ProbDist random_peaked(int vocab, std::mt19937& rng, float temp) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    ProbDist p(vocab);
    float mx = -1e30f;
    for (auto& x : p) { x = g(rng) / temp; mx = std::max(mx, x); }
    double sum = 0.0;
    for (auto& x : p) { x = std::exp(x - mx); sum += x; }
    for (auto& x : p) x = float(x / sum);
    return p;
}

ProbDist mix_uniform(const ProbDist& p, float alpha) {
    ProbDist q(p.size());
    const float u = 1.0f / p.size();
    for (size_t i = 0; i < p.size(); ++i) q[i] = alpha * p[i] + (1.0f - alpha) * u;
    return q;
}

int sample(const ProbDist& d, double u) {
    double acc = 0.0;
    for (size_t i = 0; i < d.size(); ++i) { acc += d[i]; if (u < acc) return int(i); }
    return int(d.size()) - 1;
}

double total_variation(const ProbDist& a, const ProbDist& b) {
    double tv = 0.0;
    for (size_t i = 0; i < a.size(); ++i) tv += std::abs(double(a[i]) - b[i]);
    return 0.5 * tv;
}

}  // namespace

int main() {
    std::cout << "=== Testbed 7: Speculative Decoding ===\n";
    const int vocab = 32;
    std::mt19937 rng(7);

    // Feste Verteilungen p (Target) und q (Draft).
    const ProbDist p = random_peaked(vocab, rng, 1.0f);
    const ProbDist q = mix_uniform(p, 0.9f);
    std::printf("  TV(p,q) = %.4f  (theoret. Accept-Rate ~ 1-TV = %.3f)\n",
                total_variation(p, q), 1.0 - total_variation(p, q));

    // --- Teil 1: Äquivalenz (K=1) ------------------------------------------
    // Draft/Target ignorieren ctx und liefern stets q bzw. p.
    std::mt19937 draft_rng(100), acc_rng(200);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    auto draftFn = [&](const std::vector<int>&, int K,
                       std::vector<int>& toks, std::vector<ProbDist>& qd) {
        toks.resize(K); qd.resize(K);
        for (int i = 0; i < K; ++i) { toks[i] = sample(q, U(draft_rng)); qd[i] = q; }
    };
    auto targetFn = [&](const std::vector<int>&, const std::vector<int>& draft,
                        std::vector<ProbDist>& pd) {
        pd.assign(draft.size() + 1, p);
    };
    auto uniFn = [&]() { return U(acc_rng); };

    {
        SpeculativeDecoder dec(SpecConfig{1}, draftFn, targetFn, uniFn);
        std::vector<long> hist(vocab, 0);
        const int iters = 400000;
        for (int it = 0; it < iters; ++it) {
            std::vector<int> ctx = {0};
            auto st = dec.step(ctx);
            if (!st.tokens.empty()) hist[st.tokens[0]]++;
        }
        ProbDist emp(vocab);
        for (int v = 0; v < vocab; ++v) emp[v] = float(double(hist[v]) / iters);
        const double tv = total_variation(emp, p);
        std::printf("  Aequivalenz: TV(empirisch, p) = %.4f  (Ziel < 0.01)\n", tv);
        if (tv >= 0.01) { std::cout << "  FAIL: Sampling nicht aequivalent zu p\n";
                          std::cout << "\n=== Testbed 7: FEHLGESCHLAGEN ===\n"; return 1; }
    }

    // --- Teil 2: Accept-Rate + Durchsatz (K=5) -----------------------------
    SpeculativeDecoder dec(SpecConfig{5}, draftFn, targetFn, uniFn);
    SpeculativeDecoder::RunStats agg;
    const int sequences = 4000, seq_len = 64;
    for (int s = 0; s < sequences; ++s) {
        std::vector<int> ctx = {0};
        auto st = dec.run(ctx, seq_len);
        agg.steps += st.steps; agg.proposed += st.proposed;
        agg.accepted += st.accepted; agg.emitted += st.emitted;
    }
    const double accept_rate = agg.accept_rate();
    const double tokens_per_step = agg.tokens_per_step();

    // Latenz-Modell (§8): Verify-Pass ~0,5 s, Draft ~8 ms/Token.
    const double verify_s = 0.500, draft_s_per_tok = 0.008;
    const double step_s = verify_s + draft_s_per_tok * 5;
    const double tok_s = tokens_per_step / step_s;

    std::printf("  Accept-Rate     = %.1f%%   (Ziel > 60%%)\n", accept_rate * 100.0);
    std::printf("  Tokens/Schritt  = %.2f\n", tokens_per_step);
    std::printf("  Projiziert tok/s= %.1f   (Verify 0,5s + 5x8ms Draft) (Ziel > 6)\n", tok_s);

    bool pass = (accept_rate > 0.60) && (tok_s > 6.0);
    std::cout << "\n=== Testbed 7: " << (pass ? "BESTANDEN" : "FEHLGESCHLAGEN")
              << " ===\n";
    return pass ? 0 : 1;
}

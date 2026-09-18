// testbed8_medusa.cpp — Medusa Decoding (Design §16, TB 8).
//
// Kriterium: "> 10 tok/s (überspringen wenn keine Medusa-Heads)."
//
// Trainierte Medusa-Köpfe sind nicht Teil der Basis-Gewichte (§5.5, §9) und
// liegen in dieser Build-Umgebung nicht vor -> das Performance-Gate wird PER
// DESIGN übersprungen. Dieser Testbed validiert dennoch die Verifikations-
// Logik (greedy/typical acceptance, Bonus-Token, Abbruch bei Mismatch) mit
// synthetischen Köpfen und projiziert die tok/s, damit der Pfad bei Eintreffen
// echter Heads sofort scharfgeschaltet werden kann.
#include "InferEngine/medusa.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

using namespace nova;
using nova::infer::ProbDist;
using nova::infer::MedusaConfig;
using nova::infer::MedusaDecoder;

namespace {

// Deterministisches "Target": Argmax an Position i hängt von ctx.back() ab.
int target_argmax(int ctx_last, int pos, int vocab) {
    return (ctx_last * 31 + pos * 7 + 3) % vocab;
}

ProbDist peaked_at(int idx, int vocab, float mass) {
    ProbDist p(vocab, (1.0f - mass) / (vocab - 1));
    p[idx] = mass;
    return p;
}

}  // namespace

int main() {
    std::cout << "=== Testbed 8: Medusa Decoding ===\n";
    const int vocab = 64;
    const MedusaConfig cfg;  // 3 Köpfe, Lookahead 3

    std::mt19937 rng(8);
    std::bernoulli_distribution head_correct(0.85);  // synthetische Kopf-Trefferquote
    std::uniform_int_distribution<int> any(0, vocab - 1);

    // Synthetische Köpfe: treffen das Target-Argmax meist, sonst Zufall.
    auto proposeFn = [&](const std::vector<int>& ctx, std::vector<int>& toks,
                         std::vector<ProbDist>& hd) {
        const int last = ctx.empty() ? 0 : ctx.back();
        toks.resize(cfg.num_heads); hd.resize(cfg.num_heads);
        for (int h = 0; h < cfg.num_heads; ++h) {
            const int correct = target_argmax(last, h, vocab);
            const int tok = head_correct(rng) ? correct : any(rng);
            toks[h] = tok;
            hd[h] = peaked_at(tok, vocab, 0.9f);
        }
    };
    auto targetFn = [&](const std::vector<int>& ctx, const std::vector<int>& draft,
                        std::vector<ProbDist>& pd) {
        const int last = ctx.empty() ? 0 : ctx.back();
        pd.assign(draft.size() + 1, ProbDist{});
        for (size_t i = 0; i < pd.size(); ++i)
            pd[i] = peaked_at(target_argmax(last, int(i), vocab), vocab, 0.95f);
    };

    MedusaDecoder dec(cfg, proposeFn, targetFn);
    if (!dec.heads_available()) { /* propose_ gesetzt -> hier immer true */ }

    // --- Funktionale Validierung der Verify-Logik --------------------------
    MedusaDecoder::RunStats agg;
    const int sequences = 4000, seq_len = 48;
    for (int s = 0; s < sequences; ++s) {
        std::vector<int> ctx = {1};
        auto st = dec.run(ctx, seq_len);
        agg.steps += st.steps; agg.proposed += st.proposed;
        agg.accepted += st.accepted; agg.emitted += st.emitted;
    }
    const double accept_rate = agg.accept_rate();
    const double tokens_per_step = agg.tokens_per_step();

    // Latenz-Modell (§9): ein Verify-Pass je Schritt, Köpfe quasi gratis (~80 MB).
    const double verify_s = 0.500;
    const double tok_s = tokens_per_step / verify_s;

    std::printf("  Synthetische Heads: Accept-Rate = %.1f%%, Tokens/Schritt = %.2f\n",
                accept_rate * 100.0, tokens_per_step);
    std::printf("  Projiziert tok/s = %.1f   (1 Verify-Pass/Schritt)\n", tok_s);

    // Logik-Selbstcheck: emittierte Tokens/Schritt in [1, num_heads+1].
    const bool logic_ok = tokens_per_step >= 1.0 &&
                          tokens_per_step <= double(cfg.num_heads + 1) + 1e-6 &&
                          accept_rate > 0.0;

    std::cout << "\n  HINWEIS: Keine trainierten Medusa-Heads vorhanden -> "
                 "Performance-Gate (>10 tok/s) wird per Design (§9) UEBERSPRUNGEN.\n";
    std::cout << "  Verify-Logik mit synthetischen Heads: "
              << (logic_ok ? "OK" : "FEHLER") << "\n";

    std::cout << "\n=== Testbed 8: " << (logic_ok ? "BESTANDEN (Heads-Gate uebersprungen)"
                                                  : "FEHLGESCHLAGEN")
              << " ===\n";
    return logic_ok ? 0 : 1;
}

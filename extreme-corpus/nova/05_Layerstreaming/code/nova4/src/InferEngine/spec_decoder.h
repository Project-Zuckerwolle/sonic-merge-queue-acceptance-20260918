// spec_decoder.h — Speculative Decoding / Akzeptanz-Sampling (Design §8, TB 7).
//
// Gemma 3 1B (Draft) schlägt K Token-Kandidaten vor; Gemma 4 E27B (Target)
// verifiziert sie in EINEM Pass. Das hier implementierte Akzeptanz-Sampling
// (Leviathan et al. 2023 / Chen et al. 2023) ist mathematisch exakt äquivalent
// zu direktem Sampling aus der Target-Verteilung — kein Qualitätsverlust (§8).
//
// Voraussetzung: identischer Tokenizer Draft/Target (§4.2) — Hauptgrund der
// Modellwahl. Hier generisch über Funktoren: reale Modelle (chunk-gestreamt)
// werden später eingesteckt, der Algorithmus bleibt identisch.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace nova::infer {

using ProbDist = std::vector<float>;  // über das Vokabular, normalisiert

struct SpecConfig {
    int draft_len = 5;  // K Kandidaten pro Schritt (§8)
};

// Draft generiert autoregressiv K Token + je deren q-Verteilung, gegeben ctx.
//   tokens.size() == K, qdists.size() == K
using DraftFn = std::function<void(const std::vector<int>& ctx, int K,
                                   std::vector<int>& tokens,
                                   std::vector<ProbDist>& qdists)>;

// Target verifiziert in einem Pass: liefert K+1 p-Verteilungen (Position 0 =
// nach ctx, Position i = nach ctx + draft[0..i-1]).
//   pdists.size() == K + 1
using TargetFn = std::function<void(const std::vector<int>& ctx,
                                    const std::vector<int>& draft,
                                    std::vector<ProbDist>& pdists)>;

// Uniform-Sampler U[0,1) — injizierbar für deterministische Tests.
using UniformFn = std::function<double()>;

struct SpecStep {
    std::vector<int> tokens;     // 1..K+1 emittierte Token (exakt aus p gesampelt)
    int proposed = 0;            // K Kandidaten angeboten
    int accepted = 0;            // wie viele Draft-Kandidaten akzeptiert wurden
    bool bonus = false;          // Bonus-Token (alle akzeptiert) erzeugt
};

class SpeculativeDecoder {
public:
    SpeculativeDecoder(SpecConfig cfg, DraftFn draft, TargetFn target, UniformFn uni)
        : cfg_(cfg), draft_(std::move(draft)), target_(std::move(target)),
          uni_(std::move(uni)) {}

    // Ein Spec-Schritt auf ctx; hängt die emittierten Token an ctx an.
    SpecStep step(std::vector<int>& ctx) const;

    // Generiert bis ctx.size() >= target_len oder ein Token == eos erscheint.
    // Liefert die kumulierte Statistik über alle Schritte.
    struct RunStats {
        int steps = 0;
        int proposed = 0;
        int accepted = 0;
        int emitted = 0;
        double accept_rate() const { return proposed ? double(accepted) / proposed : 0.0; }
        double tokens_per_step() const { return steps ? double(emitted) / steps : 0.0; }
    };
    RunStats run(std::vector<int>& ctx, int target_len, int eos = -1) const;

private:
    // Sampelt einen Index aus dist mittels eines Uniform-Werts u in [0,1).
    static int sample_from(const ProbDist& dist, double u);

    SpecConfig cfg_;
    DraftFn    draft_;
    TargetFn   target_;
    UniformFn  uni_;
};

}  // namespace nova::infer

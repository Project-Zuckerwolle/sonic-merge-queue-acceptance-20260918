// medusa.h — Medusa Decoding (Design §9, Testbed 8).
//
// N parallele Ausgabe-Köpfe auf Gemma 3 1B schlagen Token N+1, N+2, N+3
// gleichzeitig vor; Gemma 4 verifiziert alle in einem Pass. Konfiguration:
// 3 Köpfe, Lookahead-Tiefe 3, ~80 MB zusätzlich VRAM (§9).
//
// WICHTIG (§5.5, §9): Medusa-Köpfe müssen TRAINIERT sein — sie sind nicht Teil
// der Basis-Gewichte. Liegen keine Heads vor, läuft Spec Decoding allein und
// TB 8 wird übersprungen. `heads_available` signalisiert das zur Laufzeit.
//
// Akzeptanz: greedy/typical — Kopf-Token i wird akzeptiert solange er dem
// Target-Argmax an Position i entspricht (bzw. p[i][token] >= typical_threshold).
// Bricht beim ersten Mismatch ab; danach ein Bonus-Token aus der Target-Vert.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "InferEngine/spec_decoder.h"  // ProbDist

namespace nova::infer {

struct MedusaConfig {
    int   num_heads        = 3;     // §9
    int   lookahead        = 3;     // Lookahead-Tiefe
    float typical_threshold = 0.0f; // > 0: typical acceptance statt reinem Argmax
};

// Köpfe schlagen num_heads Token (lineare Kette) + je deren Verteilung vor.
//   tokens.size() == num_heads, head_dists.size() == num_heads
using MedusaProposeFn = std::function<void(const std::vector<int>& ctx,
                                           std::vector<int>& tokens,
                                           std::vector<ProbDist>& head_dists)>;

struct MedusaStep {
    std::vector<int> tokens;   // 1..num_heads+1 emittierte Token
    int proposed = 0;
    int accepted = 0;
};

class MedusaDecoder {
public:
    MedusaDecoder(MedusaConfig cfg, MedusaProposeFn propose, TargetFn target)
        : cfg_(cfg), propose_(std::move(propose)), target_(std::move(target)) {}

    bool heads_available() const { return bool(propose_); }

    // Ein Medusa-Schritt; hängt emittierte Token an ctx an.
    MedusaStep step(std::vector<int>& ctx) const;

    struct RunStats {
        int steps = 0, proposed = 0, accepted = 0, emitted = 0;
        double accept_rate() const { return proposed ? double(accepted) / proposed : 0.0; }
        double tokens_per_step() const { return steps ? double(emitted) / steps : 0.0; }
    };
    RunStats run(std::vector<int>& ctx, int target_len, int eos = -1) const;

private:
    static int argmax(const ProbDist& d);

    MedusaConfig    cfg_;
    MedusaProposeFn propose_;
    TargetFn        target_;
};

#ifdef NOVA_HAVE_CUDA
// GPU-Pfad (medusa.cu): Argmax über die Logits jedes Kopfes (num_heads x vocab).
// Der dominante GPU-relevante Schritt der Kopf-Auswertung. out_tokens: num_heads.
bool medusa_argmax_heads_cuda(const float* logits, int num_heads, int vocab,
                              int* out_tokens, std::string* err = nullptr);
#endif

}  // namespace nova::infer

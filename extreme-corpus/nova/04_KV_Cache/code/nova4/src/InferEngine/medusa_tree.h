// medusa_tree.h — Full Medusa Tree-Verify (Kernel-Upgrade ③).
//
// Statt EINER linearen Draft-Kette schlagen die Medusa-Köpfe einen BAUM aus
// Kandidaten-Fortsetzungen vor; der Target verifiziert den ganzen Baum in EINEM
// Pass (Tree-Attention-Maske) und akzeptiert den besten Pfad. Auf einem
// streaming-Target (dense 24B, ~0,5 s/Pass) ist "mehr akzeptierte Token pro
// teurem Pass" der direkte Speed-Hebel — daher hier wichtiger als bei MoE.
//
// Host-Referenz beweist die Akzeptanz-Logik (längster übereinstimmender Pfad +
// Bonus); der CUDA-Pfad (medusa.cu, Server) spiegelt sie mit echter Tree-Maske.
#pragma once

#include <functional>
#include <vector>

namespace nova::infer {

// Kandidaten-Baum: je ein Pfad vorgeschlagener Token-IDs (Länge <= Lookahead).
struct MedusaTree {
    std::vector<std::vector<int>> paths;
};

// Target-Greedy-Vorhersage für ein gegebenes Präfix (base + bisher akzeptiert).
// Modelliert den einen Verifikations-Pass. Liefert das "korrekte" nächste Token.
using TargetPredict = std::function<int(const std::vector<int>& prefix)>;

struct TreeVerifyResult {
    std::vector<int> accepted;   // akzeptierte Token (>=1: mind. ein Bonus-Token)
    int best_path = -1;          // Index des besten Pfads (-1 = keiner passte)
    int matched = 0;             // wie viele Draft-Token akzeptiert (ohne Bonus)
};

// Verifiziert den Baum in EINEM (modellierten) Target-Pass: akzeptiert die
// längste Kette, deren Token je der Target-Greedy-Vorhersage entsprechen, plus
// ein Bonus-Token (immer >=1 emittiert — wie exaktes Spec-Decoding).
TreeVerifyResult medusa_tree_verify(const std::vector<int>& base,
                                    const MedusaTree& tree,
                                    const TargetPredict& target);

}  // namespace nova::infer

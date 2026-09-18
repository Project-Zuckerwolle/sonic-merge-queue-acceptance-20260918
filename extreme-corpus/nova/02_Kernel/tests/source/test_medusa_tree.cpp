// test_medusa_tree.cpp — Medusa Tree-Verify Akzeptanz-Logik (Kernel-Upgrade ③).
//
// Beweist: der Baum akzeptiert die längste passende Kette in EINEM Pass und
// liefert damit MEHR Token pro (teurem) Target-Pass als eine lineare Draft-Kette
// — der direkte Speed-Hebel bei streaming-Target. Bonus-Token stets emittiert.
#include "InferEngine/medusa_tree.h"

#include <cstdio>
#include <iostream>
#include <vector>

using namespace nova::infer;

int main() {
    std::cout << "=== Medusa Tree-Verify (③) ===\n";

    // "Korrekte" Fortsetzung laut Target: C[len(prefix)].
    const std::vector<int> C = {7, 3, 9, 2};
    TargetPredict target = [&](const std::vector<int>& prefix) -> int {
        return prefix.size() < C.size() ? C[prefix.size()] : -1;
    };
    const std::vector<int> base;  // leerer Kontext

    // Baum: ein Pfad trifft die Fortsetzung voll, andere divergieren früh.
    MedusaTree tree;
    tree.paths = {
        {7, 3, 9},   // voll korrekt (Tiefe 3)
        {7, 5, 1},   // divergiert bei Position 1
        {1, 2, 3},   // divergiert sofort
    };
    const TreeVerifyResult r = medusa_tree_verify(base, tree, target);
    std::printf("  best_path=%d matched=%d accepted=[", r.best_path, r.matched);
    for (int t : r.accepted) std::printf("%d ", t);
    std::printf("] (%zu Token)\n", r.accepted.size());

    // Lineare Spec (nur der divergierende Pfad B) als Vergleich.
    MedusaTree linear; linear.paths = {{7, 5, 1}};
    const TreeVerifyResult rl = medusa_tree_verify(base, linear, target);
    std::printf("  linear (Pfad B): matched=%d -> %zu Token\n", rl.matched, rl.accepted.size());

    // Kein passender Pfad -> nur Bonus-Token (>=1, wie exaktes Spec-Decoding).
    MedusaTree miss; miss.paths = {{1, 1, 1}};
    const TreeVerifyResult rm = medusa_tree_verify(base, miss, target);
    std::printf("  kein Treffer: matched=%d -> %zu Token (Bonus)\n", rm.matched, rm.accepted.size());

    bool pass = true;
    pass &= (r.best_path == 0 && r.matched == 3);
    pass &= (r.accepted == std::vector<int>({7, 3, 9, 2}));   // 3 akzeptiert + Bonus C[3]=2
    pass &= (r.accepted.size() > rl.accepted.size());          // Baum > lineare Kette
    pass &= (rm.matched == 0 && rm.accepted.size() == 1);      // stets >= 1 Token

    std::cout << "\n=== Medusa Tree-Verify: " << (pass ? "BESTANDEN" : "FEHLGESCHLAGEN") << " ===\n";
    return pass ? 0 : 1;
}

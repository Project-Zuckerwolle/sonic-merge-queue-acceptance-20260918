// medusa_tree_host.cpp — Host-Referenz für Medusa Tree-Verify (③).
#include "InferEngine/medusa_tree.h"

namespace nova::infer {

TreeVerifyResult medusa_tree_verify(const std::vector<int>& base,
                                    const MedusaTree& tree,
                                    const TargetPredict& target) {
    TreeVerifyResult res;
    std::vector<int> best_matched;   // akzeptierte Draft-Token des besten Pfads

    for (size_t pi = 0; pi < tree.paths.size(); ++pi) {
        const std::vector<int>& path = tree.paths[pi];
        std::vector<int> prefix = base;
        std::vector<int> matched;
        for (int tok : path) {
            if (target(prefix) == tok) { prefix.push_back(tok); matched.push_back(tok); }
            else break;
        }
        if (int(matched.size()) > res.matched) {
            res.matched = int(matched.size());
            res.best_path = int(pi);
            best_matched = matched;
        }
    }

    // Akzeptierte Draft-Token + ein Bonus-Token (immer >= 1 emittiert).
    res.accepted = best_matched;
    std::vector<int> prefix = base;
    for (int t : best_matched) prefix.push_back(t);
    res.accepted.push_back(target(prefix));
    return res;
}

}  // namespace nova::infer

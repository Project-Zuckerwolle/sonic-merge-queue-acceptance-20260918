// apex_batch_host.cpp — Host-Referenz für apex_batch.h (Design §13).
// Spiegelt die 2-Sequenzen-Batch-Struktur des CUDA-Kernels (apex_batch.cu).
#include "Apex/apex_batch.h"

namespace nova::apex {

std::vector<std::string> apex_batch_infer_host(const std::vector<uint8_t>& weights,
                                               const std::vector<std::string>& prompts,
                                               const SubagentFn& fn) {
    std::vector<std::string> out;
    if (weights.empty() || !fn) return out;  // Modell nicht geladen
    out.reserve(prompts.size());

    // Verarbeitung in 2er-Batches (ein GEMM-Pass je Paar) — Reihenfolge bleibt
    // erhalten, beide Sequenzen eines Paares laufen "gleichzeitig".
    for (size_t i = 0; i < prompts.size(); i += 2) {
        const std::string a = fn(prompts[i]);
        std::string b;
        if (i + 1 < prompts.size()) b = fn(prompts[i + 1]);
        out.push_back(a);
        if (i + 1 < prompts.size()) out.push_back(b);
    }
    return out;
}

}  // namespace nova::apex

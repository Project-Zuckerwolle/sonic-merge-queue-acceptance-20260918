// apex_result.cpp — Implementierung von apex_result.h (Design §13.8).
#include "Apex/apex_result.h"

#include <algorithm>
#include <fstream>

namespace nova::apex {

int estimate_tokens(const std::string& s) {
    return s.empty() ? 0 : std::max(1, int((s.size() + 3) / 4));
}

ApexOutput finalize_apex_output(const std::string& full_output, const LlmFn& llm14,
                                int token_limit, const std::string& audit_path) {
    ApexOutput out;
    out.original_tokens = estimate_tokens(full_output);

    // Audit-Trail: vollständigen Output ablegen (BrainCompiler verarbeitet ihn).
    if (!audit_path.empty()) {
        std::ofstream f(audit_path, std::ios::binary | std::ios::trunc);
        if (f) f << full_output;
    }

    if (out.original_tokens <= token_limit || !llm14) {
        out.text = full_output;
        return out;
    }

    // Zusammenfassen via 14B.
    out.text = llm14("Fasse das folgende Apex-Ergebnis prägnant zusammen "
                     "(max " + std::to_string(token_limit) + " Token):\n\n" + full_output);
    out.truncated = true;
    return out;
}

}  // namespace nova::apex

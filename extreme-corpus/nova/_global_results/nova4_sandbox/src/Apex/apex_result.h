// apex_result.h — Apex-Output-Handling (Design §13.8).
//
// Apex-Results > 5.000 Token werden vor Einbettung in den Chat-Context durch
// Ministral 14B zusammengefasst. Der vollständige Output geht (optional) nach
// brain/raw/apex_TIMESTAMP.md (Audit-Trail; BrainCompiler verarbeitet ihn).
#pragma once

#include <string>

#include "Apex/apex_orchestrator.h"  // LlmFn

namespace nova::apex {

struct ApexOutput {
    std::string text;       // ggf. zusammengefasst -> in den Chat-Context
    bool        truncated = false;
    int         original_tokens = 0;
};

// token_limit default 5000 (§13.8). Schreibt full_output nach audit_path (falls
// gesetzt). Bei Überschreitung wird per llm14 zusammengefasst.
ApexOutput finalize_apex_output(const std::string& full_output, const LlmFn& llm14,
                                int token_limit = 5000,
                                const std::string& audit_path = "");

int estimate_tokens(const std::string& s);  // ~4 Zeichen/Token

}  // namespace nova::apex

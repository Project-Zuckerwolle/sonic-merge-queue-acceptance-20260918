// apex_subagent_pool.h — Ministral 3B Subagent-Pool (Design §4.4, §13).
//
// Bei Apex-Start: ministral-3b.bin SSD->RAM (~3s), RAM->VRAM für Batch-Passes.
// 2 Sequenzen gleichzeitig (apex_batch). Nach Apex: 3B aus RAM + VRAM freigeben.
// Der 3B-Generator ist injizierter Callback (SubagentFn) — testbar ohne Modell.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <utility>

#include "Apex/apex_batch.h"

namespace nova::apex {

// --- 3B-Subagent-Kontrakt (Aufgabe 5.2): kontextfrei -----------------------
// Das schwache 3B halluziniert, sobald es Chat-/ReAct-Kontext sieht. Der einzige
// erlaubte Prompt ist {tool_name, params} (+ optional Ergebnis-Schema) — KEIN
// Chat-Verlauf, KEINE ReAct-History, KEINE Persona. Der Orchestrator baut die
// params vollständig; das 3B führt nur aus.
using ToolParams = std::vector<std::pair<std::string, std::string>>;

std::string build_subagent_prompt(const std::string& tool_name,
                                  const ToolParams& params,
                                  const std::string& result_schema = "");

// Prüft, dass ein Prompt kontextfrei ist (kein Chat/ReAct/Persona-Leak).
bool is_context_free_prompt(const std::string& prompt);

class SubagentPool {
public:
    // SSD -> RAM: lädt das Flat-INT4-Binary (ministral-3b.bin).
    bool load(const std::string& bin_path, std::string* err = nullptr);
    bool loaded() const { return !weights_.empty(); }
    size_t bytes() const { return weights_.size(); }
    void unload() { weights_.clear(); weights_.shrink_to_fit(); }

    // 2-Modell-Setup (Aufgabe 2): das 3B ist bereits als Draft resident. Statt
    // erneut zu laden übernimmt der Pool die resident gehaltenen Gewichte —
    // kein On-demand-SSD->RAM->VRAM-Swap mehr.
    void bind_resident_weights(std::vector<uint8_t> resident) { weights_ = std::move(resident); }

    // Batched Inferenz über den 3B (2 Sequenzen je GEMM-Pass).
    std::vector<std::string> batch_infer(const std::vector<std::string>& prompts,
                                         const SubagentFn& fn) const;

private:
    std::vector<uint8_t> weights_;  // RAM-Buffer (Streaming-Quelle)
};

}  // namespace nova::apex

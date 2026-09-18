// brain_linker.h — Cross-Entry Linking (Design §11.2 #6).
//
// Einmal täglich 03:00 (Ministral 14B, LOW-Priorität): Batches von 10 neuen
// Wiki-Seiten -> LLM findet Verbindungen zwischen Seiten, gibt Link-Stärken
// (0.0–1.0) und Begründungen zurück. Ergebnis als <!-- Links: ... --> in den
// Seiten. Der LLM ist injizierter Callback (LlmFn) -> testbar ohne Modell.
//
// Erwartetes LLM-Antwortformat (eine Zeile je Link):
//   QUELLE -> ZIEL | STÄRKE | BEGRÜNDUNG
#pragma once

#include <string>
#include <vector>

#include "Brain/brain_store.h"
#include "Brain/brain_types.h"

namespace nova::brain {

class BrainLinker {
public:
    explicit BrainLinker(LlmFn llm, int batch = 10) : llm_(std::move(llm)), batch_(batch) {}

    // Verlinkt die angegebenen Seiten (Namen). Liefert die Anzahl gesetzter Links.
    int link(BrainStore& store, const std::vector<WikiRef>& pages) const;

    // Parst das LLM-Antwortformat zu (quelle, ziel, stärke, grund).
    struct ParsedLink { std::string source, target, reason; double strength = 0.0; };
    static std::vector<ParsedLink> parse_response(const std::string& resp);

private:
    LlmFn llm_;
    int   batch_;
};

}  // namespace nova::brain

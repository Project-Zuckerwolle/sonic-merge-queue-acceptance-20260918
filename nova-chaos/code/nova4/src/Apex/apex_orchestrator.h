// apex_orchestrator.h — Template + Step/Reflexion-Parser (Design §13.6, §13.9).
//
// Ministral 14B orchestriert den ReAct-Loop. Der Orchestrator stellt bereit:
//   - render_template: {{placeholder}} ersetzen (apex/coding_prompt.md etc.)
//   - parse_step:      LLM-Antwort -> THOUGHT + Aktion (tool_call / task_complete
//                      / task_blocked / task_replan)
//   - parse_reflexion: Reflexions-JSON (§13.6) auswerten
#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "Apex/apex_task.h"

namespace nova::apex {

// Ministral 14B (Orchestrator/Reflexion/Summary): Prompt -> Antwort.
using LlmFn = std::function<std::string(const std::string& prompt)>;

// {{key}} im Template durch vars[key] ersetzen.
std::string render_template(const std::string& tmpl,
                            const std::map<std::string, std::string>& vars);

struct ParsedStep {
    enum Kind { None, ToolCall, Complete, Blocked, Replan } kind = None;
    std::string thought;
    // ToolCall
    std::string tool;
    std::vector<std::pair<std::string, std::string>> params;
    // Complete/Blocked/Replan
    std::string summary;      // task_complete
    std::string reason;       // task_blocked
    std::string next_steps;   // task_blocked
    std::string new_plan;     // task_replan
};

// Extrahiert THOUGHT-Zeile + erstes Steuer-/Tool-Tag aus der LLM-Antwort.
ParsedStep parse_step(const std::string& llm_out);

// Plan-first (Aufgabe 5.3): parst den vollständigen Plan aus <plan>…</plan> mit
// <step id="…" tool="…" deps="a,b" …/>-Einträgen. plan.ok bei >=1 Schritt.
Plan parse_plan(const std::string& llm_out);

struct Reflexion {
    int  progress = 0;             // fortschritt_prozent
    bool strategie_anpassen = false;
    std::string naechster_fokus;
    std::vector<std::string> extrahierte_fakten;
    bool ok = false;
};
Reflexion parse_reflexion(const std::string& json);

// Tag-Attribute key="value" extrahieren (öffentlich für Tests).
std::vector<std::pair<std::string, std::string>> parse_tag_attrs(const std::string& tag);

}  // namespace nova::apex

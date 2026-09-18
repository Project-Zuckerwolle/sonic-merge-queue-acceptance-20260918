// nano_parser.h — Regex-basierter Pre-Router (Design §14, §22 Nova Predator v5).
//
// Läuft pro User-Message VOR dem LLM (<1 ms, kein LLM). Erkennt schnell:
//   - Plan-Trigger    -> sofort Apex-Call vorbereiten (kein BM25 nötig)
//   - Tool-Keyword    -> Tool-Hint für den Context-Builder
//   - Komplexitäts-Signal -> erweiterter Reasoning-Hinweis im Prompt
//   - Todo/Präferenz  -> in Working Memory markieren
//
// Keyword-Listen sind aus den Skill-YAMLs (trigger_keywords) speisbar; Defaults
// decken die Standard-Skills aus §13.3/§13.4 ab.
#pragma once

#include <string>
#include <vector>

namespace nova::web {

struct NanoResult {
    bool        apex_trigger = false;   // Plan-Trigger erkannt
    std::string apex_skill;             // welcher Apex-Skill (coding/news_research/...)
    bool        tool_hint = false;      // Normal-Tool-Keyword erkannt
    std::string tool_name;              // erkanntes Tool
    bool        complex = false;        // Komplexitäts-Signal
    bool        todo_pref = false;      // Todo-/Präferenz-Signal
};

class NanoParser {
public:
    NanoParser();  // lädt Default-Keywords

    NanoResult parse(const std::string& msg) const;

    // Registriert Trigger-Keywords (aus Skill-YAML). is_apex unterscheidet
    // Apex-Skill (Plan-Trigger) von Normal-Tool.
    void add_skill(const std::string& name, const std::vector<std::string>& keywords, bool is_apex);

private:
    struct Entry { std::string name; std::vector<std::string> keywords; bool is_apex; };
    std::vector<Entry> entries_;
    std::vector<std::string> complex_kw_;
    std::vector<std::string> todo_kw_;
};

}  // namespace nova::web

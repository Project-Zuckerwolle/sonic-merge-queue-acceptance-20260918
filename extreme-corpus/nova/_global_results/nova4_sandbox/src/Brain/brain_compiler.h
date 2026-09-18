// brain_compiler.h — Orchestriert den Brain-Zyklus (Design §11.2, §14).
//
// Läuft alle ~30 Min bei Gemma 4 idle (LOW-Priorität). Prüft VOR jedem Lauf das
// apex_running-Flag (§11.2): ist Apex aktiv, wird der Lauf übersprungen.
//
// Pro Lauf (kein LLM für 1+2):
//   1. Konfidenz-Konsolidierung   (brain_consolidator)
//   2. Duplikat-Prüfung           (brain_consolidator)
//   3. raw/ -> wiki/              (Ministral 14B)
//   4. hot.md refresh + log.md
//   5. Idea-Breakthrough-Detection (brain_idea_detector)
// Einmal täglich (03:00):
//   6. Cross-Entry Linking        (brain_linker)
//   7. Evergreen-Flags            (brain_evergreen)
//   8. Daily Briefing
#pragma once

#include <atomic>
#include <string>
#include <vector>

#include "Brain/brain_consolidator.h"
#include "Brain/brain_store.h"
#include "Brain/brain_types.h"

namespace nova::brain {

struct BrainCycleInput {
    std::string today;                    // "YYYY-MM-DD"
    std::string clock = "12:00";          // "HH:MM" — steuert die 03:00-Tagesaufgaben
    std::string session_id;
    std::vector<std::string> mentioned;   // diese Session erwähnte Seiten
    bool        run_daily = false;        // Tagesaufgaben (Linking/Evergreen/Briefing)
    std::string briefing_source;          // Quelle für Daily Briefing (letzte Episode/Todos)
};

struct BrainCycleResult {
    bool skipped_apex = false;
    ConsolidationResult consolidation;
    int  raw_processed = 0;
    int  breakthroughs = 0;   // Seiten mit Score > 0.75
    int  notifications = 0;   // Score > 0.90
    int  links = 0;
    int  evergreen = 0;
    bool daily_run = false;
};

class BrainCompiler {
public:
    BrainCompiler(BrainStore& store, LlmFn llm, std::atomic<bool>& apex_running)
        : store_(store), llm_(std::move(llm)), apex_running_(apex_running) {}

    void set_briefing_path(const std::string& p) { briefing_path_ = p; }

    BrainCycleResult run_cycle(const BrainCycleInput& in);

    // raw/ -> wiki/ (LLM). Erwartet je Datei: "CATEGORY: ..\nNAME: ..\n<inhalt>".
    int process_raw(const std::string& today);

private:
    BrainStore&        store_;
    LlmFn              llm_;
    std::atomic<bool>& apex_running_;
    std::string        briefing_path_;
};

}  // namespace nova::brain

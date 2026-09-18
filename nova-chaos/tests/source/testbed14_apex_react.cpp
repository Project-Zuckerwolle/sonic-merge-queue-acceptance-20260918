// testbed14_apex_react.cpp — Apex ReAct-Loop (Design §16, TB 14).
//
// Kriterium: "Mindestens 3 Iterationen, Tool-Call -> OBS -> THOUGHT korrekt
// verkettet; ReAct-History in apex_tasks/ korrekt geloggt."
//
// Stub-14B treibt eine Coding-Aufgabe über 4 Iterationen: codebase_scan ->
// datei_schreiben -> shell_exec (Tier 3, Bestätigung) -> task_complete. Geprüft
// werden Verkettung, Persistenz (apex_tasks/*.json + Reload) und der Permission-
// Pfad (shell_exec verlangt Bestätigung).
#include "Apex/apex_react_loop.h"
#include "Apex/apex_task.h"
#include "Skills/skill_registry.h"
#include "Skills/tool_exec.h"

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace nova;

int main() {
    std::cout << "=== Testbed 14: Apex ReAct-Loop ===\n";
    bool pass = true;

    const std::string ws = (fs::temp_directory_path() / "nova4_tb14_ws").string();
    const std::string tasks = (fs::temp_directory_path() / "nova4_tb14_tasks").string();
    std::error_code ec; fs::remove_all(ws, ec); fs::remove_all(tasks, ec);
    fs::create_directories(ws, ec);

    skills::WorkspaceGuard guard(ws);
    skills::ToolBox box = skills::default_toolbox();
    skills::ToolContext ctx; ctx.guard = &guard;
    skills::SkillRegistry reg = skills::SkillRegistry::with_defaults();

    apex::ApexTaskManager mgr(tasks);
    pass &= mgr.init(nullptr);

    // Stub-14B: scriptet die Iterationen anhand eines Call-Zählers.
    int call = 0;
    apex::LlmFn llm = [&](const std::string& prompt) -> std::string {
        if (prompt.find("REFLEXION (JSON)") != std::string::npos)
            return "{\"fortschritt_prozent\":50,\"strategie_anpassen\":false,"
                   "\"extrahierte_fakten\":[\"Projekt nutzt CMake\"]}";
        switch (++call) {
            case 1: return "THOUGHT: Zuerst Projektstruktur verstehen.\n"
                           "ACTION: <tool_call name=\"codebase_scan\" path=\".\"/>";
            case 2: return "THOUGHT: Implementierung schreiben.\n"
                           "<tool_call name=\"datei_schreiben\" path=\"out.txt\" content=\"hallo welt\"/>";
            case 3: return "THOUGHT: Build verifizieren.\n"
                           "<tool_call name=\"shell_exec\" cmd=\"echo build-ok\"/>";
            default: return "THOUGHT: Fertig.\n"
                            "<task_complete summary=\"out.txt geschrieben, Build ok\"/>";
        }
    };

    int confirm_calls = 0;
    apex::ReactLoop loop(llm, box, ctx);
    loop.set_registry(&reg, /*tier2_confirm=*/false);   // Tier 2 ohne Bestätigung
    loop.set_confirm_fn([&](const std::string& tool, const std::string&) {
        ++confirm_calls; (void)tool; return true;        // Bestätigung erteilt
    });
    loop.set_task_manager(&mgr);

    apex::ApexTask task; task.task_id = "task_tb14"; task.skill = "coding";
    task.description = "Coding-Demo"; task.max_iterations = 20;
    const std::string status = loop.run(task, "System: ReAct-Coding-Agent.");

    std::printf("  Status                 = %s\n", status.c_str());
    std::printf("  Iterationen            = %d\n", task.iteration);
    std::printf("  History-Einträge       = %zu\n", task.react_history.size());
    std::printf("  shell_exec-Bestätigung = %d (erwartet >=1)\n", confirm_calls);

    pass &= (status == "complete");
    pass &= (task.iteration >= 3);
    pass &= (confirm_calls >= 1);   // shell_exec (Tier 3) immer bestätigen

    // Verkettung prüfen: irgendwo THOUGHT -> ACTION -> OBS in Folge.
    bool chained = false;
    for (size_t i = 0; i + 2 < task.react_history.size(); ++i)
        if (task.react_history[i].type == "THOUGHT" &&
            task.react_history[i + 1].type == "ACTION" &&
            task.react_history[i + 2].type == "OBS") chained = true;
    std::printf("  THOUGHT->ACTION->OBS    = %s\n", chained ? "verkettet" : "NEIN");
    pass &= chained;

    // OBS des codebase_scan muss echtes Tool-Ergebnis sein.
    bool scan_obs = false;
    for (size_t i = 0; i + 1 < task.react_history.size(); ++i)
        if (task.react_history[i].type == "ACTION" && task.react_history[i].tool == "codebase_scan" &&
            task.react_history[i + 1].type == "OBS" &&
            task.react_history[i + 1].content.find("Dateien:") != std::string::npos) scan_obs = true;
    pass &= scan_obs;

    // Datei wurde tatsächlich geschrieben.
    pass &= fs::exists(fs::path(ws) / "out.txt");

    // Persistenz: apex_tasks/*.json geschrieben und reloadbar mit History.
    apex::ApexTask reloaded;
    const bool ld = mgr.load("task_tb14", reloaded, nullptr);
    std::printf("  Task-JSON reload       = %s (%zu Einträge)\n",
                ld ? "ok" : "FAIL", reloaded.react_history.size());
    pass &= ld;
    pass &= (reloaded.status == "complete");
    pass &= (reloaded.react_history.size() == task.react_history.size());
    pass &= (reloaded.iteration == task.iteration);

    fs::remove_all(ws, ec); fs::remove_all(tasks, ec);
    std::cout << "\n=== Testbed 14: " << (pass ? "BESTANDEN" : "FEHLGESCHLAGEN") << " ===\n";
    return pass ? 0 : 1;
}

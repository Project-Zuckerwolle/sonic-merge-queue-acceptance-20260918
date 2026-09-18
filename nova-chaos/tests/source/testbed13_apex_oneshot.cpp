// testbed13_apex_oneshot.cpp — Apex One-Shot + 3B-Batch (Design §16, TB 13).
//
// Kriterium: "Apex-Block erscheint, 3B-Batch läuft korrekt."
//
// (a) Apex-Orchestrator rendert das Template und startet den ReAct-Loop; der
//     Stub-14B liefert sofort <task_complete> -> es entsteht ein Apex-Block
//     (THOUGHT/OBS-Transcript, status=complete).
// (b) Der Subagent-Pool lädt ein (synthetisches) ministral-3b.bin und führt
//     2 Sequenzen in EINEM Batch aus -> 2 korrekte Ausgaben.
#include "Apex/apex_orchestrator.h"
#include "Apex/apex_react_loop.h"
#include "Apex/apex_subagent_pool.h"
#include "Apex/apex_task.h"
#include "Skills/tool_exec.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace nova;

int main() {
    std::cout << "=== Testbed 13: Apex One-Shot + 3B-Batch ===\n";
    bool pass = true;

    const std::string ws = (fs::temp_directory_path() / "nova4_tb13").string();
    std::error_code ec; fs::remove_all(ws, ec); fs::create_directories(ws, ec);

    // --- (a) Apex One-Shot --------------------------------------------------
    std::map<std::string, std::string> vars = {{"task", "Implementiere TurboQuant KV-Manager"}};
    const std::string tmpl = "# Apex: coding\nAufgabe: {{task}}\nFormat: THOUGHT/ACTION/OBS.";
    const std::string rendered = apex::render_template(tmpl, vars);
    pass &= (rendered.find("Implementiere TurboQuant KV-Manager") != std::string::npos);
    pass &= (rendered.find("{{") == std::string::npos);
    std::printf("  Template gerendert     = %s\n",
                rendered.find("TurboQuant") != std::string::npos ? "ja" : "NEIN");

    skills::WorkspaceGuard guard(ws);
    skills::ToolBox box = skills::default_toolbox();
    skills::ToolContext ctx; ctx.guard = &guard;

    // Stub-14B: One-Shot -> sofort task_complete.
    apex::LlmFn llm = [](const std::string&) {
        return "THOUGHT: Aufgabe ist klar und einfach.\n"
               "<task_complete summary=\"KV-Manager-Plan erstellt\"/>";
    };
    apex::ReactLoop loop(llm, box, ctx);
    apex::ApexTask task; task.task_id = "task_tb13"; task.skill = "coding";
    task.description = vars["task"]; task.max_iterations = 20;
    const std::string status = loop.run(task, rendered);

    std::printf("  Apex-Status            = %s\n", status.c_str());
    std::printf("  ReAct-History-Einträge = %zu\n", task.react_history.size());
    pass &= (status == "complete");
    pass &= (!task.react_history.empty());  // Apex-Block existiert
    bool has_thought = false;
    for (auto& s : task.react_history) if (s.type == "THOUGHT") has_thought = true;
    pass &= has_thought;

    // --- (b) 3B-Batch -------------------------------------------------------
    const std::string bin = (fs::path(ws) / "ministral-3b.bin").string();
    { std::ofstream f(bin, std::ios::binary); std::string data(4096, '\x42'); f.write(data.data(), data.size()); }

    apex::SubagentPool pool;
    pass &= pool.load(bin, nullptr);
    std::printf("  3B geladen (RAM)       = %s (%zu Bytes)\n", pool.loaded() ? "ja" : "NEIN", pool.bytes());

    // 2 Sequenzen gleichzeitig (ein GEMM-Batch).
    apex::SubagentFn subfn = [](const std::string& p) { return "antwort:" + p; };
    auto outs = pool.batch_infer({"seqA", "seqB"}, subfn);
    std::printf("  3B-Batch-Ausgaben      = %zu\n", outs.size());
    pass &= (outs.size() == 2);
    pass &= (outs.size() == 2 && outs[0] == "antwort:seqA" && outs[1] == "antwort:seqB");

    // Ohne geladene Gewichte: keine Ausgabe (Batch inaktiv).
    apex::SubagentPool empty;
    pass &= empty.batch_infer({"x"}, subfn).empty();

    fs::remove_all(ws, ec);
    std::cout << "\n=== Testbed 13: " << (pass ? "BESTANDEN" : "FEHLGESCHLAGEN") << " ===\n";
    return pass ? 0 : 1;
}

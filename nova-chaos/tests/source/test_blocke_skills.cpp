// test_blocke_skills.cpp — Block E Logik (Skills + Apex, ohne Netzwerk).
//
// Deckt ab: nano_parser, workspace_guard (Traversal-Schutz), skill_loader (YAML),
// skill_registry (Permission-Tiers, Profile, datei_loeschen-Pflicht), native
// tools (rechner/datei_*/todo), apex_orchestrator (Template/Step/Reflexion),
// apex_task (JSON round-trip), apex_result (Truncation), REPLAN/Reflexion-Logik.
#include "Apex/apex_orchestrator.h"
#include "Apex/apex_react_loop.h"
#include "Apex/apex_result.h"
#include "Apex/apex_task.h"
#include "Skills/native_tools.h"
#include "Skills/skill_loader.h"
#include "Skills/skill_registry.h"
#include "Skills/tool_exec.h"
#include "Web/nano_parser.h"

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace nova;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& name) {
    std::printf("  [%s] %s\n", ok ? "OK" : "XX", name.c_str());
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::cout << "=== Block E: Skills + Apex Logik ===\n";

    // --- NanoParser ---------------------------------------------------------
    {
        web::NanoParser np;
        auto r1 = np.parse("Bitte implementiere einen neuen KV-Manager");
        check(r1.apex_trigger && r1.apex_skill == "coding", "NanoParser: Apex-Trigger coding");
        auto r2 = np.parse("Wie ist das Wetter in Hamburg?");
        check(r2.tool_hint && r2.tool_name == "wetter", "NanoParser: Tool-Hint wetter");
        auto r3 = np.parse("Erkläre die Architektur und vergleiche die Trade-offs");
        check(r3.complex, "NanoParser: Komplexitäts-Signal");
        auto r4 = np.parse("Merk dir bitte: ich bevorzuge kurze Antworten");
        check(r4.todo_pref, "NanoParser: Todo/Präferenz-Signal");
    }

    // --- WorkspaceGuard: Traversal-Schutz ----------------------------------
    {
        const std::string ws = (fs::temp_directory_path() / "nova4_tbE_ws").string();
        std::error_code ec; fs::create_directories(ws, ec);
        skills::WorkspaceGuard g(ws);
        std::string abs, err;
        check(g.resolve("unterordner/datei.txt", abs, &err), "Guard: relativer Pfad erlaubt");
        check(!g.resolve("../../etc/passwd", abs, &err), "Guard: ../../etc/passwd blockiert");
        check(!g.resolve("..\\..\\windows\\system32", abs, &err), "Guard: Windows-Traversal blockiert");
        check(!g.resolve("C:/Windows/System32/cmd.exe", abs, &err), "Guard: absoluter Außen-Pfad blockiert");
        fs::remove_all(ws, ec);
    }

    // --- SkillLoader: YAML --------------------------------------------------
    {
        const std::string yaml =
            "name: websearch\ntype: tool\ntier: 1\n"
            "description: \"DuckDuckGo-Suche\"\n"
            "trigger_keywords: [suche, was ist, finde]\n"
            "executor: native\n"
            "parameters:\n  - name: query\n    type: string\n    required: true\n";
        skills::Skill s; std::string err;
        check(skills::parse_skill_yaml(yaml, s, &err), "SkillLoader: YAML geparst");
        check(s.name == "websearch" && s.type == skills::SkillType::Tool && s.tier == 1,
              "SkillLoader: Felder korrekt");
        check(s.trigger_keywords.size() == 3 && s.trigger_keywords[0] == "suche",
              "SkillLoader: Inline-Liste");
        check(s.parameters.size() == 1 && s.parameters[0].name == "query" && s.parameters[0].required,
              "SkillLoader: Parameter-Block");

        const std::string ay =
            "name: coding\ntype: apex\ntier: 2\nreact_loop: true\n"
            "max_iterations: 20\npersistent: true\nsubagent_executor: 3B\n";
        skills::Skill a;
        check(skills::parse_skill_yaml(ay, a, &err) && a.type == skills::SkillType::Apex &&
              a.react_loop && a.max_iterations == 20 && a.persistent,
              "SkillLoader: Apex-Skill geparst");
    }

    // --- SkillRegistry: Permission + Profile -------------------------------
    {
        auto reg = skills::SkillRegistry::with_defaults();
        check(reg.size() > 20, "Registry: Standard-Skills geladen");
        check(!reg.needs_confirmation("rechner", true), "Permission: Tier 1 nie bestätigen");
        check(reg.needs_confirmation("datei_schreiben", true), "Permission: Tier 2 bei Flag");
        check(!reg.needs_confirmation("datei_schreiben", false), "Permission: Tier 2 ohne Flag");
        check(reg.needs_confirmation("shell_exec", false), "Permission: Tier 3 immer");
        check(reg.needs_confirmation("datei_loeschen", false), "Permission: datei_loeschen Pflicht");

        check(reg.set_active_profile("profile_research"), "Profil: research aktiviert");
        check(reg.is_active("websearch") && !reg.is_active("shell_exec"),
              "Profil: filtert aktive Skills");
    }

    // --- Native Tools: rechner + Datei + Todo ------------------------------
    {
        double v;
        check(skills::tools::eval_expression("2+3*4", v) && v == 14, "rechner: 2+3*4 = 14");
        check(skills::tools::eval_expression("(2+3)*4", v) && v == 20, "rechner: (2+3)*4 = 20");
        check(!skills::tools::eval_expression("2+", v), "rechner: ungültig erkannt");

        const std::string ws = (fs::temp_directory_path() / "nova4_tbE_tools").string();
        std::error_code ec; fs::remove_all(ws, ec); fs::create_directories(ws, ec);
        skills::WorkspaceGuard g(ws);
        skills::ToolContext ctx; ctx.guard = &g;
        ctx.todo_path = (fs::path(ws) / "todos.txt").string();
        auto box = skills::default_toolbox();

        auto w = box.execute("datei_schreiben", {{"path", "a.txt"}, {"content", "hallo"}}, ctx);
        check(w.ok, "datei_schreiben ok");
        auto rd = box.execute("datei_lesen", {{"path", "a.txt"}}, ctx);
        check(rd.ok && rd.output == "hallo", "datei_lesen liefert Inhalt");

        // Traversal über Tool blockiert.
        auto bad = box.execute("datei_schreiben", {{"path", "../escape.txt"}, {"content", "x"}}, ctx);
        check(!bad.ok && bad.output.find("WorkspaceBoundaryError") != std::string::npos,
              "Tool: Traversal blockiert (WorkspaceBoundaryError)");

        box.execute("todo_schreiben", {{"text", "Block E fertigstellen"}}, ctx);
        auto tl = box.execute("todo_lesen", {}, ctx);
        check(tl.output.find("Block E fertigstellen") != std::string::npos, "todo schreiben/lesen");

        // Netzwerk-Tool mit injiziertem http_get.
        ctx.http_get = [](const std::string& url) { return "STUB(" + url + ")"; };
        auto we = box.execute("wetter", {{"location", "Hamburg"}}, ctx);
        check(we.ok && we.output.find("Hamburg") != std::string::npos, "wetter via http_get-Stub");
        fs::remove_all(ws, ec);
    }

    // --- Apex Orchestrator --------------------------------------------------
    {
        auto out = apex::render_template("Aufgabe: {{task}} ({{x}})", {{"task", "Bauen"}, {"x", "1"}});
        check(out == "Aufgabe: Bauen (1)", "render_template");

        auto s1 = apex::parse_step("THOUGHT: Lese Datei.\n<tool_call name=\"datei_lesen\" path=\"a.h\"/>");
        check(s1.kind == apex::ParsedStep::ToolCall && s1.tool == "datei_lesen" &&
              s1.thought == "Lese Datei.", "parse_step: tool_call");
        auto s2 = apex::parse_step("THOUGHT: fertig\n<task_complete summary=\"ok\"/>");
        check(s2.kind == apex::ParsedStep::Complete && s2.summary == "ok", "parse_step: task_complete");
        auto s3 = apex::parse_step("<task_replan new_plan=\"Forward-Declaration\"/>");
        check(s3.kind == apex::ParsedStep::Replan && s3.new_plan == "Forward-Declaration", "parse_step: replan");

        auto rf = apex::parse_reflexion(
            "{\"fortschritt_prozent\":60,\"strategie_anpassen\":true,"
            "\"extrahierte_fakten\":[\"CMake 3.28\",\"C++20\"]}");
        check(rf.ok && rf.progress == 60 && rf.strategie_anpassen && rf.extrahierte_fakten.size() == 2,
              "parse_reflexion");
    }

    // --- Apex Task: JSON round-trip ----------------------------------------
    {
        apex::ApexTask t; t.task_id = "task_x"; t.skill = "coding"; t.description = "Demo";
        t.status = "running"; t.iteration = 7; t.replan_count = 1;
        t.reflexion_facts = {"CMake 3.28"};
        t.log({"THOUGHT", "denke", "", "", -1});
        t.log({"ACTION", "", "datei_lesen", "path=\"a.h\"", -1});
        t.log({"OBS", "Inhalt", "", "", -1});
        const std::string js = apex::ApexTaskManager::serialize(t);
        apex::ApexTask r; std::string err;
        check(apex::ApexTaskManager::deserialize(js, r, &err), "Task: JSON deserialisiert");
        check(r.iteration == 7 && r.replan_count == 1 && r.react_history.size() == 3 &&
              r.react_history[1].tool == "datei_lesen" && r.reflexion_facts.size() == 1,
              "Task: round-trip korrekt");
    }

    // --- Apex Result: Truncation -------------------------------------------
    {
        bool called = false;
        apex::LlmFn sum = [&](const std::string&) { called = true; return std::string("KURZ"); };
        auto small = apex::finalize_apex_output("kleiner output", sum, 5000, "");
        check(!small.truncated && small.text == "kleiner output", "Result: klein -> unverändert");
        std::string big(40000, 'x');  // ~10k Token
        auto large = apex::finalize_apex_output(big, sum, 5000, "");
        check(large.truncated && large.text == "KURZ" && called, "Result: groß -> 14B-Zusammenfassung");
    }

    // --- REPLAN nach 3 Fehlschlägen ----------------------------------------
    {
        const std::string ws = (fs::temp_directory_path() / "nova4_tbE_replan").string();
        std::error_code ec; fs::remove_all(ws, ec); fs::create_directories(ws, ec);
        skills::WorkspaceGuard g(ws);
        skills::ToolBox box = skills::default_toolbox();
        skills::ToolContext ctx; ctx.guard = &g;

        // LLM produziert immer einen fehlschlagenden Tool-Call (Datei außerhalb),
        // bis ein REPLAN kommt; danach (2. REPLAN) -> blocked.
        apex::LlmFn llm = [&](const std::string& p) -> std::string {
            if (p.find("REFLEXION (JSON)") != std::string::npos)
                return "{\"fortschritt_prozent\":10,\"strategie_anpassen\":false}";
            return "THOUGHT: Versuch.\n<tool_call name=\"datei_lesen\" path=\"../nope.txt\"/>";
        };
        apex::ReactLoop loop(llm, box, ctx);
        apex::ApexTask t; t.task_id = "task_replan"; t.max_iterations = 30;
        const std::string st = loop.run(t, "System.");

        int replans = 0;
        for (auto& s : t.react_history) if (s.type == "REPLAN") ++replans;
        check(replans >= 1, "REPLAN: nach 3 Fehlschlägen ausgelöst");
        check(t.replan_count <= 2 + 1, "REPLAN: max 2 begrenzt");
        check(st == "blocked", "REPLAN: nach max_replans blockiert");
        fs::remove_all(ws, ec);
    }

    std::cout << "\n=== Block E Logik: " << (g_fail == 0 ? "BESTANDEN" : "FEHLGESCHLAGEN")
              << " (" << g_fail << " Fehler) ===\n";
    return g_fail == 0 ? 0 : 1;
}

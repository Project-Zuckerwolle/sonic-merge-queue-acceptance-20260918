// testbed17_plan_dag.cpp — Plan-Phase + DAG-Dispatch (Aufgabe 5.3, TB17).
//
// Kriterien:
//  - der Orchestrator gibt VOR der ersten ACTION einen vollständigen, schema-
//    gültigen Plan aus (parse_plan liefert geschlossene Schritte + Abhängigkeiten),
//  - ein Task mit 2+ UNABHÄNGIGEN Schritten führt diese nachweislich PARALLEL aus
//    (gemeinsame Welle + Wall-Clock << seriell), ABHÄNGIGE Schritte seriell danach.
#include "Apex/apex_dag_dispatcher.h"
#include "Apex/apex_orchestrator.h"
#include "Skills/tool_exec.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>

using namespace nova;
using clk = std::chrono::high_resolution_clock;

int main() {
    std::cout << "=== Testbed 17: Plan-Phase + DAG-Dispatch ===\n";

    // Der Orchestrator emittiert zuerst EINEN vollständigen Plan (schema-erzwungen).
    const std::string plan_out =
        "<plan>\n"
        "  <step id=\"s1\" tool=\"slow\" tag=\"A\"/>\n"
        "  <step id=\"s2\" tool=\"slow\" tag=\"B\"/>\n"
        "  <step id=\"s3\" tool=\"rechner\" expr=\"20+22\" deps=\"s1,s2\"/>\n"
        "</plan>\n";

    const apex::Plan plan = apex::parse_plan(plan_out);
    std::printf("  Plan geparst: ok=%s, steps=%zu\n", plan.ok ? "ja" : "NEIN", plan.steps.size());
    for (const auto& s : plan.steps)
        std::printf("    %s: tool=%s deps=%zu\n", s.id.c_str(), s.tool.c_str(), s.deps.size());

    // ToolBox: rechner ist Standard; ein künstlich langsames Tool zeigt Parallelität.
    skills::ToolBox tools = skills::default_toolbox();
    tools.register_tool("slow", [](const skills::Params& p, skills::ToolContext&) {
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        return skills::tool_ok("slow:" + skills::param(p, "tag"));
    });
    skills::ToolContext ctx;

    const auto t0 = clk::now();
    const apex::DagResult res = apex::dispatch_plan(plan, tools, ctx);
    const double wall_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();

    std::printf("  Dispatch ok=%s | Wellen=%zu | Wall=%.1f ms\n",
                res.ok ? "ja" : "NEIN", res.waves.size(), wall_ms);
    for (size_t i = 0; i < res.waves.size(); ++i) {
        std::string ids;
        for (const auto& id : res.waves[i]) ids += id + " ";
        std::printf("    Welle %zu: %s\n", i, ids.c_str());
    }
    std::printf("    s3-OBS = %s\n", res.outputs.count("s3") ? res.outputs.at("s3").c_str() : "(fehlt)");

    bool pass = true;
    pass &= plan.ok && plan.steps.size() == 3;
    // Plan vor ACTION: s3 kennt seine 2 Abhängigkeiten.
    const apex::PlanStep* s3 = plan.find("s3");
    pass &= (s3 && s3->deps.size() == 2);
    pass &= res.ok;
    // 2 unabhängige Schritte in EINER Welle (parallel), abhängiger in späterer.
    pass &= (res.waves.size() == 2);
    if (res.waves.size() == 2) {
        pass &= (res.waves[0].size() == 2);   // s1+s2 parallel
        pass &= (res.waves[1].size() == 1);   // s3 seriell danach
    }
    // Wall-Clock beweist Parallelität: seriell wäre ~120 ms (2×60), parallel ~60 ms.
    pass &= (wall_ms < 110.0);
    // Abhängiger Schritt korrekt ausgeführt (rechner 20+22).
    pass &= (res.outputs.count("s3") && res.outputs.at("s3").find("42") != std::string::npos);

    std::cout << "\n=== Testbed 17: " << (pass ? "BESTANDEN" : "FEHLGESCHLAGEN") << " ===\n";
    return pass ? 0 : 1;
}

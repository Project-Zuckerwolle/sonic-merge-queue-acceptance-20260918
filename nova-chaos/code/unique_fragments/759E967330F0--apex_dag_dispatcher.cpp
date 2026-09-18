// apex_dag_dispatcher.cpp — Implementierung von apex_dag_dispatcher.h (Aufgabe 5.3).
#include "Apex/apex_dag_dispatcher.h"

#include <future>
#include <set>
#include <utility>

namespace nova::apex {

DagResult dispatch_plan(const Plan& plan, const skills::ToolBox& tools,
                        skills::ToolContext& ctx) {
    DagResult res;

    // Abhängigkeiten müssen auf existierende Schritte zeigen.
    for (const auto& s : plan.steps)
        for (const auto& d : s.deps)
            if (!plan.find(d)) { res.error = "Unbekannte Abhängigkeit: " + d; return res; }

    std::set<std::string> done;
    std::vector<const PlanStep*> pending;
    for (const auto& s : plan.steps) pending.push_back(&s);

    while (!pending.empty()) {
        std::vector<const PlanStep*> wave, rest;
        for (const PlanStep* s : pending) {
            bool ready = true;
            for (const auto& d : s->deps) if (!done.count(d)) { ready = false; break; }
            (ready ? wave : rest).push_back(s);
        }
        if (wave.empty()) { res.error = "Zyklus oder unerfüllbare Abhängigkeit"; return res; }

        // Welle parallel ausführen (unabhängige Schritte gleichzeitig).
        std::vector<std::future<std::pair<std::string, std::string>>> futs;
        futs.reserve(wave.size());
        for (const PlanStep* s : wave) {
            futs.push_back(std::async(std::launch::async, [s, &tools, &ctx]() {
                skills::Params p;
                for (const auto& kv : s->params) p.push_back(kv);
                const skills::ToolResult r = tools.execute(s->tool, p, ctx);
                return std::make_pair(s->id, r.output);
            }));
        }

        std::vector<std::string> wave_ids;
        for (size_t i = 0; i < wave.size(); ++i) {
            const auto pr = futs[i].get();
            res.outputs[pr.first] = pr.second;
            done.insert(pr.first);
            wave_ids.push_back(pr.first);
        }
        res.waves.push_back(std::move(wave_ids));
        pending = std::move(rest);
    }

    res.ok = true;
    return res;
}

}  // namespace nova::apex

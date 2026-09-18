// apex_react_loop.cpp — Implementierung von apex_react_loop.h (Design §13.6).
#include "Apex/apex_react_loop.h"

#include "Apex/anti_laziness_scorer.h"
#include "Apex/marcos_mdp.h"

#include <sstream>

namespace nova::apex {

std::string ReactLoop::build_prompt(const std::string& system_prompt, const ApexTask& task) const {
    std::ostringstream os;
    os << system_prompt << "\n\n--- VERLAUF ---\n";
    for (const auto& s : task.react_history) {
        if (s.type == "THOUGHT") os << "THOUGHT: " << s.content << "\n";
        else if (s.type == "ACTION") os << "ACTION: <tool_call name=\"" << s.tool << "\" "
                                        << s.params << "/>\n";
        else if (s.type == "OBS") os << "OBS: " << s.content << "\n";
        else if (s.type == "REPLAN") os << "REPLAN: " << s.content << "\n";
        else if (s.type == "REFLEXION") os << "REFLEXION: Fortschritt " << s.progress << "%\n";
    }
    if (!task.reflexion_facts.empty()) {
        os << "\nBekannte Fakten:\n";
        for (const auto& f : task.reflexion_facts) os << "- " << f << "\n";
    }
    os << "\nNächster Schritt? (THOUGHT + <tool_call.../> oder <task_complete.../>)\n";
    return os.str();
}

void ReactLoop::emit(ApexTask& task, const ReactStep& s) {
    task.log(s);
    if (events_) events_(s);
}

void ReactLoop::do_replan(ApexTask& task, const std::string& plan) {
    ReactStep r; r.type = "REPLAN"; r.content = plan;
    emit(task, r);
    task.consecutive_failures = 0;   // §13.6: Counter-Reset bei REPLAN
    ++task.replan_count;
}

std::string ReactLoop::run(ApexTask& task, const std::string& system_prompt) {
    task.status = "running";

    while (task.iteration < task.max_iterations) {
        ++task.iteration;

        // --- LLM um nächsten Schritt bitten -------------------------------
        const std::string llm_out = llm_(build_prompt(system_prompt, task));
        const ParsedStep step = parse_step(llm_out);

        if (!step.thought.empty()) {
            ReactStep t; t.type = "THOUGHT"; t.content = step.thought; emit(task, t);
        }

        if (step.kind == ParsedStep::Complete) {
            task.status = "complete";
            if (marcos_) marcos_->on_terminal(task, true);
            ReactStep c; c.type = "OBS"; c.content = "task_complete: " + step.summary; emit(task, c);
            break;
        }
        if (step.kind == ParsedStep::Blocked) {
            task.status = "blocked";
            if (marcos_) marcos_->on_terminal(task, false);
            ReactStep c; c.type = "OBS"; c.content = "task_blocked: " + step.reason; emit(task, c);
            break;
        }
        if (step.kind == ParsedStep::Replan) {
            do_replan(task, step.new_plan);
            if (task.replan_count > cfg_.max_replans) {
                task.status = "blocked";
                ReactStep c; c.type = "OBS"; c.content = "task_blocked: max_replans"; emit(task, c);
                break;
            }
            if (mgr_) mgr_->save(task);
            continue;
        }

        if (step.kind == ParsedStep::ToolCall) {
            // ACTION loggen.
            ReactStep a; a.type = "ACTION"; a.tool = step.tool;
            std::ostringstream ps;
            for (size_t i = 0; i < step.params.size(); ++i) {
                if (i) ps << " ";
                ps << step.params[i].first << "=\"" << step.params[i].second << "\"";
            }
            a.params = ps.str();
            emit(task, a);

            // Permission/Bestätigung.
            bool allowed = true;
            if (registry_ && registry_->needs_confirmation(step.tool, tier2_confirm_)) {
                allowed = confirm_ ? confirm_(step.tool, a.params) : true;
            }

            std::string obs;
            bool ok = false;
            if (!allowed) {
                obs = "Abgelehnt durch Nutzer (Bestätigung verweigert).";
            } else {
                const skills::ToolResult r = tools_.execute(step.tool, step.params, ctx_);
                obs = r.output; ok = r.ok;
            }

            ReactStep o; o.type = "OBS"; o.content = obs; emit(task, o);

            if (ok) task.consecutive_failures = 0;
            else    ++task.consecutive_failures;

            if (marcos_) {
                // Anti-Laziness-PRM (§7): THOUGHT-Qualität [0,1] skaliert den MARCOS-Reward (§7.4).
                const float lz = score_thought(step.thought).score;
                // MARCOS steuert Reward/Confidence + Backtracking (ersetzt die lineare Fehler-Heuristik, §6.4).
                const MarcosDecision d = marcos_->on_action(task, step.tool, ok, obs, lz);
                if (d.backtrack) {
                    do_replan(task, "MARCOS-Backtrack: " + d.reason);
                    if (task.replan_count > cfg_.max_replans) {
                        task.status = "blocked";
                        ReactStep c; c.type = "OBS"; c.content = "task_blocked: max_replans"; emit(task, c);
                        break;
                    }
                }
            } else {
                // §13.6: nach 3 Fehlschlägen Strategie wechseln (klassischer Loop).
                if (task.consecutive_failures >= cfg_.failure_threshold) {
                    do_replan(task, "Automatischer REPLAN nach " +
                              std::to_string(cfg_.failure_threshold) + " Fehlschlägen.");
                    if (task.replan_count > cfg_.max_replans) {
                        task.status = "blocked";
                        ReactStep c; c.type = "OBS"; c.content = "task_blocked: max_replans"; emit(task, c);
                        break;
                    }
                }
            }
        } else if (step.kind == ParsedStep::None) {
            // Kein verwertbarer Schritt -> als Fehlschlag werten.
            ++task.consecutive_failures;
            ReactStep o; o.type = "OBS"; o.content = "Keine gültige Aktion erkannt."; emit(task, o);
            if (marcos_) {
                const MarcosDecision d = marcos_->on_no_action(task);
                if (d.backtrack) {
                    do_replan(task, "MARCOS-Backtrack: " + d.reason);
                    if (task.replan_count > cfg_.max_replans) {
                        task.status = "blocked";
                        ReactStep c; c.type = "OBS"; c.content = "task_blocked: max_replans"; emit(task, c);
                        break;
                    }
                }
            }
        }

        // --- Reflexion alle N Iterationen ---------------------------------
        if (task.iteration % cfg_.reflexion_interval == 0) {
            const std::string rj = llm_(
                "REFLEXION (JSON): bewerte den Verlauf. Felder: fortschritt_prozent, "
                "was_funktioniert, was_fehlt, strategie_anpassen(bool), naechster_fokus, "
                "extrahierte_fakten(array).\n\nVerlauf:\n" + build_prompt(system_prompt, task));
            const Reflexion rf = parse_reflexion(rj);
            if (rf.ok) {
                ReactStep r; r.type = "REFLEXION"; r.progress = rf.progress; emit(task, r);
                for (const auto& f : rf.extrahierte_fakten) task.reflexion_facts.push_back(f);
                if (rf.strategie_anpassen)
                    do_replan(task, "Reflexion: " + rf.naechster_fokus);
            }
        }

        if (mgr_) mgr_->save(task);
    }

    if (task.status == "running") {
        task.status = "blocked";
        ReactStep c; c.type = "OBS"; c.content = "task_blocked: max_iterations"; emit(task, c);
    }
    if (mgr_) mgr_->save(task);
    return task.status;
}

}  // namespace nova::apex

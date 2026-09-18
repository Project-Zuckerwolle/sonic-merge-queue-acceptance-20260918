// apex_react_loop.h — Haupt-ReAct-Loop (Design §13.6, §14).
//
// THOUGHT -> ACTION (tool_call) -> OBS, iteriert bis <task_complete> /
// <task_blocked> / max_iterations. Zusätzlich:
//   - REPLAN: nach 3 aufeinanderfolgenden fehlgeschlagenen Actions Strategie-
//     wechsel; max 2 REPLANs pro Task, danach task_blocked.
//   - Reflexion: alle 5 Iterationen ein separater 14B-Pass; extrahierte_fakten
//     -> Task-State; strategie_anpassen:true triggert REPLAN.
// Der State wird je Iteration in apex_tasks/*.json persistiert (Resume bei Wake).
#pragma once

#include <functional>
#include <string>

#include "Apex/apex_orchestrator.h"
#include "Apex/apex_task.h"
#include "Skills/skill_registry.h"
#include "Skills/tool_exec.h"

namespace nova::apex {

struct ReactConfig {
    int reflexion_interval = 5;
    int max_replans = 2;
    int failure_threshold = 3;
};

class ReactLoop {
public:
    using ConfirmFn = std::function<bool(const std::string& tool, const std::string& detail)>;
    using EventFn   = std::function<void(const ReactStep&)>;  // Live-Frontend-Updates

    ReactLoop(LlmFn orchestrator, const skills::ToolBox& tools, skills::ToolContext& ctx,
              ReactConfig cfg = {})
        : llm_(std::move(orchestrator)), tools_(tools), ctx_(ctx), cfg_(cfg) {}

    void set_registry(const skills::SkillRegistry* r, bool tier2_confirm) {
        registry_ = r; tier2_confirm_ = tier2_confirm;
    }
    void set_confirm_fn(ConfirmFn f) { confirm_ = std::move(f); }
    void set_event_fn(EventFn f) { events_ = std::move(f); }
    void set_task_manager(ApexTaskManager* m) { mgr_ = m; }

    // Treibt den Loop auf task. system_prompt = gerendertes Template + Aufgabe.
    // Liefert den Endstatus ("complete"/"blocked").
    std::string run(ApexTask& task, const std::string& system_prompt);

private:
    std::string build_prompt(const std::string& system_prompt, const ApexTask& task) const;
    void emit(ApexTask& task, const ReactStep& s);
    void do_replan(ApexTask& task, const std::string& plan);

    LlmFn                       llm_;
    const skills::ToolBox&      tools_;
    skills::ToolContext&        ctx_;
    ReactConfig                 cfg_;
    const skills::SkillRegistry* registry_ = nullptr;
    bool                        tier2_confirm_ = false;
    ConfirmFn                   confirm_;
    EventFn                     events_;
    ApexTaskManager*            mgr_ = nullptr;
};

}  // namespace nova::apex

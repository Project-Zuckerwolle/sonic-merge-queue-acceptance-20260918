// apex_dag_dispatcher.h — DAG-Dispatch für Plan-Schritte (Aufgabe 5.3, TB17).
//
// Führt voneinander unabhängige Plan-Schritte PARALLEL als Tool-Calls aus
// (Abhängigkeiten aus dem Plan-DAG), sodass der Orchestrator nicht pro Schritt
// über Parallelität reasonieren muss. Ein Schritt = ein Tool-Call. Unabhängige
// Schritte einer Welle laufen gleichzeitig; abhängige in späteren Wellen seriell.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "Apex/apex_task.h"
#include "Skills/tool_exec.h"

namespace nova::apex {

struct DagResult {
    std::map<std::string, std::string> outputs;    // step id -> OBS
    std::vector<std::vector<std::string>> waves;   // parallele Wellen (Beweis Parallelität)
    bool ok = false;
    std::string error;
};

// Topologische Wellen-Planung: jede Welle = alle Schritte deren Abhängigkeiten
// erfüllt sind -> gleichzeitig ausgeführt (std::async). Zyklus/unerfüllbare
// Abhängigkeit -> ok=false + error.
DagResult dispatch_plan(const Plan& plan, const skills::ToolBox& tools,
                        skills::ToolContext& ctx);

}  // namespace nova::apex

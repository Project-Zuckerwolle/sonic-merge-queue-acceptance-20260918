// apex_task.h — Apex Task-State + Persistenz (Design §13.6, §13.7).
//
// Persistenter ReAct-State in apex_tasks/task_TIMESTAMP.json. Bei Hibernate
// geflushst, bei Wake geprüft (Resume-Logik). Enthält die vollständige
// THOUGHT/ACTION/OBS/REPLAN/REFLEXION-History.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace nova::apex {

struct ReactStep {
    std::string type;       // THOUGHT / ACTION / OBS / REPLAN / REFLEXION
    std::string content;    // freier Text (THOUGHT/OBS/REPLAN-Plan)
    std::string tool;       // bei ACTION: Tool-Name
    std::string params;     // bei ACTION: serialisierte Parameter
    int         progress = -1;  // bei REFLEXION: fortschritt_prozent
};

// --- Plan-first (Aufgabe 5.3) ----------------------------------------------
// Der Orchestrator erzeugt zu Task-Beginn EINEN vollständigen, geschlossenen
// Plan (Schritte + Abhängigkeiten), schema-erzwungen. Der ReAct-Loop arbeitet
// ihn ab statt frei zu improvisieren; REPLAN revidiert ihn. Der DAG-Dispatcher
// führt voneinander unabhängige Schritte parallel aus.
struct PlanStep {
    std::string id;
    std::string tool;
    std::vector<std::pair<std::string, std::string>> params;
    std::vector<std::string> deps;   // ids, von denen dieser Schritt abhängt
};

struct Plan {
    std::vector<PlanStep> steps;
    bool ok = false;
    const PlanStep* find(const std::string& id) const {
        for (const auto& s : steps) if (s.id == id) return &s;
        return nullptr;
    }
};

struct ApexTask {
    std::string task_id;
    std::string skill;
    std::string description;
    std::string status = "running";   // running / complete / blocked
    int         iteration = 0;
    int         max_iterations = 20;
    int         consecutive_failures = 0;
    int         replan_count = 0;
    std::vector<std::string> reflexion_facts;
    std::vector<ReactStep>   react_history;
    std::string checkpoint;           // ISO-Zeit (vom Aufrufer gesetzt)

    void log(const ReactStep& s) { react_history.push_back(s); }
};

class ApexTaskManager {
public:
    explicit ApexTaskManager(const std::string& tasks_dir) : dir_(tasks_dir) {}

    bool init(std::string* err = nullptr);

    std::string path_for(const std::string& task_id) const;

    bool save(const ApexTask& t, std::string* err = nullptr) const;
    bool load(const std::string& task_id, ApexTask& out, std::string* err = nullptr) const;

    // Unfertige Tasks (status == "running") für die Resume-Abfrage beim Start.
    std::vector<ApexTask> unfinished() const;

    static std::string serialize(const ApexTask& t);
    static bool         deserialize(const std::string& json, ApexTask& out, std::string* err = nullptr);

private:
    std::string dir_;
};

}  // namespace nova::apex

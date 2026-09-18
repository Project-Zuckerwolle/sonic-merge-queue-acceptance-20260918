// apex_task.cpp — Implementierung von apex_task.h (Design §13.7).
#include "Apex/apex_task.h"

#include "Apex/json.h"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace nova::apex {

bool ApexTaskManager::init(std::string* err) {
    std::error_code ec;
    fs::create_directories(dir_, ec);
    if (ec) { if (err) *err = "apex_tasks init: " + ec.message(); return false; }
    return true;
}

std::string ApexTaskManager::path_for(const std::string& task_id) const {
    return (fs::path(dir_) / (task_id + ".json")).string();
}

std::string ApexTaskManager::serialize(const ApexTask& t) {
    JsonValue o = JsonValue::make_object();
    o.obj["task_id"]   = JsonValue::of(t.task_id);
    o.obj["skill"]     = JsonValue::of(t.skill);
    o.obj["description"] = JsonValue::of(t.description);
    o.obj["status"]    = JsonValue::of(t.status);
    o.obj["iteration"] = JsonValue::of(t.iteration);
    o.obj["max_iterations"] = JsonValue::of(t.max_iterations);
    o.obj["consecutive_failures"] = JsonValue::of(t.consecutive_failures);
    o.obj["replan_count"] = JsonValue::of(t.replan_count);
    { JsonValue v; v.type = JsonValue::Number; v.num = t.confidence;        o.obj["confidence"] = v; }
    { JsonValue v; v.type = JsonValue::Number; v.num = t.cumulative_reward; o.obj["cumulative_reward"] = v; }
    o.obj["checkpoint"] = JsonValue::of(t.checkpoint);

    JsonValue facts = JsonValue::make_array();
    for (const auto& f : t.reflexion_facts) facts.arr.push_back(JsonValue::of(f));
    o.obj["reflexion_facts"] = facts;

    JsonValue hist = JsonValue::make_array();
    for (const auto& s : t.react_history) {
        JsonValue e = JsonValue::make_object();
        e.obj["type"] = JsonValue::of(s.type);
        if (!s.content.empty()) e.obj["content"] = JsonValue::of(s.content);
        if (!s.tool.empty())    e.obj["tool"] = JsonValue::of(s.tool);
        if (!s.params.empty())  e.obj["params"] = JsonValue::of(s.params);
        if (s.progress >= 0)    e.obj["progress"] = JsonValue::of(s.progress);
        hist.arr.push_back(e);
    }
    o.obj["react_history"] = hist;
    return o.serialize();
}

bool ApexTaskManager::deserialize(const std::string& json, ApexTask& out, std::string* err) {
    JsonValue o;
    if (!json_parse(json, o, err)) return false;
    if (!o.is_object()) { if (err) *err = "kein JSON-Objekt"; return false; }

    auto str = [&](const char* k) { auto* v = o.find(k); return v ? v->as_str() : std::string(); };
    auto in  = [&](const char* k, int d) { auto* v = o.find(k); return v ? v->as_int(d) : d; };

    out.task_id = str("task_id");
    out.skill = str("skill");
    out.description = str("description");
    out.status = str("status");
    out.iteration = in("iteration", 0);
    out.max_iterations = in("max_iterations", 20);
    out.consecutive_failures = in("consecutive_failures", 0);
    out.replan_count = in("replan_count", 0);
    if (auto* v = o.find("confidence"))        out.confidence = float(v->num);
    if (auto* v = o.find("cumulative_reward")) out.cumulative_reward = float(v->num);
    out.checkpoint = str("checkpoint");

    if (auto* f = o.find("reflexion_facts"); f && f->is_array())
        for (const auto& e : f->arr) out.reflexion_facts.push_back(e.as_str());

    if (auto* h = o.find("react_history"); h && h->is_array()) {
        for (const auto& e : h->arr) {
            ReactStep s;
            if (auto* v = e.find("type")) s.type = v->as_str();
            if (auto* v = e.find("content")) s.content = v->as_str();
            if (auto* v = e.find("tool")) s.tool = v->as_str();
            if (auto* v = e.find("params")) s.params = v->as_str();
            if (auto* v = e.find("progress")) s.progress = v->as_int(-1);
            out.react_history.push_back(s);
        }
    }
    return true;
}

bool ApexTaskManager::save(const ApexTask& t, std::string* err) const {
    std::ofstream f(path_for(t.task_id), std::ios::binary | std::ios::trunc);
    if (!f) { if (err) *err = "Task speichern fehlgeschlagen"; return false; }
    const std::string s = serialize(t);
    f.write(s.data(), std::streamsize(s.size()));
    return bool(f);
}

bool ApexTaskManager::load(const std::string& task_id, ApexTask& out, std::string* err) const {
    std::ifstream f(path_for(task_id), std::ios::binary);
    if (!f) { if (err) *err = "Task nicht gefunden: " + task_id; return false; }
    std::stringstream ss; ss << f.rdbuf();
    return deserialize(ss.str(), out, err);
}

std::vector<ApexTask> ApexTaskManager::unfinished() const {
    std::vector<ApexTask> out;
    std::error_code ec;
    if (!fs::exists(dir_, ec)) return out;
    for (const auto& e : fs::directory_iterator(dir_, ec)) {
        if (!e.is_regular_file() || e.path().extension() != ".json") continue;
        std::ifstream f(e.path(), std::ios::binary);
        std::stringstream ss; ss << f.rdbuf();
        ApexTask t;
        if (deserialize(ss.str(), t, nullptr) && t.status == "running") out.push_back(std::move(t));
    }
    return out;
}

}  // namespace nova::apex

// apex_orchestrator.cpp — Implementierung von apex_orchestrator.h (Design §13.6).
#include "Apex/apex_orchestrator.h"

#include "Apex/json.h"

#include <cctype>

namespace nova::apex {

std::string render_template(const std::string& tmpl,
                            const std::map<std::string, std::string>& vars) {
    std::string out;
    size_t i = 0;
    while (i < tmpl.size()) {
        const size_t open = tmpl.find("{{", i);
        if (open == std::string::npos) { out += tmpl.substr(i); break; }
        out += tmpl.substr(i, open - i);
        const size_t close = tmpl.find("}}", open);
        if (close == std::string::npos) { out += tmpl.substr(open); break; }
        std::string key = tmpl.substr(open + 2, close - open - 2);
        // Trim key.
        size_t a = key.find_first_not_of(" \t"); size_t b = key.find_last_not_of(" \t");
        key = (a == std::string::npos) ? "" : key.substr(a, b - a + 1);
        auto it = vars.find(key);
        out += (it != vars.end()) ? it->second : std::string();
        i = close + 2;
    }
    return out;
}

std::vector<std::pair<std::string, std::string>> parse_tag_attrs(const std::string& tag) {
    std::vector<std::pair<std::string, std::string>> out;
    size_t i = 0;
    while (i < tag.size()) {
        while (i < tag.size() && !(std::isalpha((unsigned char)tag[i]) || tag[i] == '_')) ++i;
        if (i >= tag.size()) break;
        size_t ks = i;
        while (i < tag.size() && (std::isalnum((unsigned char)tag[i]) || tag[i] == '_')) ++i;
        std::string key = tag.substr(ks, i - ks);
        while (i < tag.size() && (tag[i] == ' ' || tag[i] == '\t')) ++i;
        if (i >= tag.size() || tag[i] != '=') continue;
        ++i;
        while (i < tag.size() && (tag[i] == ' ' || tag[i] == '\t')) ++i;
        if (i >= tag.size() || tag[i] != '"') continue;
        ++i;
        size_t vs = i;
        while (i < tag.size() && tag[i] != '"') ++i;
        std::string val = tag.substr(vs, i - vs);
        if (i < tag.size()) ++i;
        out.push_back({key, val});
    }
    return out;
}

namespace {
// Findet das erste Tag <name ...> ab pos; liefert name + voller Tag-Text.
bool find_tag(const std::string& s, const std::string& name, std::string& tag) {
    const std::string open = "<" + name;
    size_t p = s.find(open);
    if (p == std::string::npos) return false;
    size_t gt = s.find('>', p);
    if (gt == std::string::npos) return false;
    tag = s.substr(p, gt - p + 1);
    return true;
}
std::string attr(const std::vector<std::pair<std::string, std::string>>& a, const std::string& k) {
    for (auto& [kk, v] : a) if (kk == k) return v;
    return {};
}
}  // namespace

ParsedStep parse_step(const std::string& llm_out) {
    ParsedStep st;

    // THOUGHT-Zeile.
    const size_t tp = llm_out.find("THOUGHT:");
    if (tp != std::string::npos) {
        const size_t nl = llm_out.find('\n', tp);
        st.thought = llm_out.substr(tp + 8, (nl == std::string::npos ? llm_out.size() : nl) - tp - 8);
        size_t a = st.thought.find_first_not_of(" \t");
        if (a != std::string::npos) st.thought = st.thought.substr(a);
    }

    std::string tag;
    if (find_tag(llm_out, "task_complete", tag)) {
        st.kind = ParsedStep::Complete;
        st.summary = attr(parse_tag_attrs(tag), "summary");
    } else if (find_tag(llm_out, "task_blocked", tag)) {
        st.kind = ParsedStep::Blocked;
        auto a = parse_tag_attrs(tag);
        st.reason = attr(a, "reason"); st.next_steps = attr(a, "next_steps");
    } else if (find_tag(llm_out, "task_replan", tag)) {
        st.kind = ParsedStep::Replan;
        st.new_plan = attr(parse_tag_attrs(tag), "new_plan");
    } else if (find_tag(llm_out, "tool_call", tag)) {
        st.kind = ParsedStep::ToolCall;
        for (auto& [k, v] : parse_tag_attrs(tag)) {
            if (k == "name") st.tool = v;
            else st.params.push_back({k, v});
        }
    }
    return st;
}

Plan parse_plan(const std::string& s) {
    Plan plan;
    if (s.find("<plan") == std::string::npos) return plan;
    size_t pos = 0;
    for (;;) {
        const size_t p = s.find("<step", pos);
        if (p == std::string::npos) break;
        const size_t gt = s.find('>', p);
        if (gt == std::string::npos) break;
        const std::string tag = s.substr(p, gt - p + 1);
        pos = gt + 1;

        PlanStep st;
        for (auto& [k, v] : parse_tag_attrs(tag)) {
            if (k == "id") st.id = v;
            else if (k == "tool") st.tool = v;
            else if (k == "deps") {
                size_t a = 0;
                while (a <= v.size()) {
                    const size_t c = v.find(',', a);
                    const std::string raw = v.substr(a, c == std::string::npos ? std::string::npos : c - a);
                    const size_t x = raw.find_first_not_of(" \t");
                    const size_t y = raw.find_last_not_of(" \t");
                    if (x != std::string::npos) st.deps.push_back(raw.substr(x, y - x + 1));
                    if (c == std::string::npos) break;
                    a = c + 1;
                }
            } else {
                st.params.push_back({k, v});
            }
        }
        if (!st.id.empty() && !st.tool.empty()) plan.steps.push_back(st);
    }
    plan.ok = !plan.steps.empty();
    return plan;
}

Reflexion parse_reflexion(const std::string& json) {
    Reflexion r;
    JsonValue o;
    if (!json_parse(json, o, nullptr) || !o.is_object()) return r;
    if (auto* v = o.find("fortschritt_prozent")) r.progress = v->as_int(0);
    if (auto* v = o.find("strategie_anpassen")) r.strategie_anpassen = v->as_bool(false);
    if (auto* v = o.find("naechster_fokus")) r.naechster_fokus = v->as_str();
    if (auto* v = o.find("extrahierte_fakten"); v && v->is_array())
        for (const auto& e : v->arr) r.extrahierte_fakten.push_back(e.as_str());
    r.ok = true;
    return r;
}

}  // namespace nova::apex

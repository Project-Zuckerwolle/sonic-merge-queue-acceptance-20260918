// skill_loader.cpp — Implementierung von skill_loader.h (Design §13.1).
#include "Skills/skill_loader.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace nova::skills {

namespace {
std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}
std::string unquote(std::string s) {
    s = trim(s);
    if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') ||
                          (s.front() == '\'' && s.back() == '\'')))
        return s.substr(1, s.size() - 2);
    return s;
}
int indent_of(const std::string& line) {
    int n = 0; for (char c : line) { if (c == ' ') ++n; else break; } return n;
}
std::vector<std::string> parse_inline_list(const std::string& v) {
    std::vector<std::string> out;
    std::string s = trim(v);
    if (s.size() >= 2 && s.front() == '[' && s.back() == ']') s = s.substr(1, s.size() - 2);
    std::stringstream ss(s); std::string item;
    while (std::getline(ss, item, ',')) { item = unquote(item); if (!item.empty()) out.push_back(item); }
    return out;
}
bool truthy(const std::string& v) {
    const std::string t = trim(v);
    return t == "true" || t == "yes" || t == "1";
}
}  // namespace

bool parse_skill_yaml(const std::string& yaml, Skill& out, std::string* err) {
    out = Skill{};
    std::istringstream in(yaml);
    std::string line;
    bool in_params = false;
    SkillParam cur; bool cur_open = false;
    auto flush_param = [&] { if (cur_open) { out.parameters.push_back(cur); cur = SkillParam{}; cur_open = false; } };

    while (std::getline(in, line)) {
        std::string raw = line;
        if (!raw.empty() && raw.back() == '\r') raw.pop_back();
        const std::string t = trim(raw);
        if (t.empty() || t[0] == '#') continue;

        const int ind = indent_of(raw);

        // parameters:-Block (eingerückte "- name:"-Items).
        if (in_params && ind >= 2) {
            std::string item = t;
            if (item.rfind("- ", 0) == 0) { flush_param(); cur_open = true; item = trim(item.substr(2)); }
            const size_t colon = item.find(':');
            if (colon != std::string::npos) {
                const std::string k = trim(item.substr(0, colon));
                const std::string v = trim(item.substr(colon + 1));
                if (k == "name") cur.name = unquote(v);
                else if (k == "type") cur.type = unquote(v);
                else if (k == "required") cur.required = truthy(v);
            }
            continue;
        } else if (in_params && ind < 2) {
            flush_param(); in_params = false;
        }

        const size_t colon = t.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = trim(t.substr(0, colon));
        const std::string val = trim(t.substr(colon + 1));

        if (key == "name") out.name = unquote(val);
        else if (key == "type") out.type = (unquote(val) == "apex") ? SkillType::Apex : SkillType::Tool;
        else if (key == "tier") out.tier = std::atoi(val.c_str());
        else if (key == "description") out.description = unquote(val);
        else if (key == "trigger_keywords") out.trigger_keywords = parse_inline_list(val);
        else if (key == "executor") out.executor = unquote(val);
        else if (key == "react_loop") out.react_loop = truthy(val);
        else if (key == "max_iterations") out.max_iterations = std::atoi(val.c_str());
        else if (key == "persistent") out.persistent = truthy(val);
        else if (key == "template") out.template_path = unquote(val);
        else if (key == "subagent_executor") out.subagent_executor = unquote(val);
        else if (key == "parameters") in_params = true;
    }
    flush_param();

    if (out.name.empty()) { if (err) *err = "Skill ohne name"; return false; }
    return true;
}

std::vector<Skill> load_skill_dir(const std::string& dir) {
    std::vector<Skill> out;
    std::error_code ec;
    if (!fs::exists(dir, ec)) return out;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_regular_file()) continue;
        const auto ext = e.path().extension().string();
        if (ext != ".yaml" && ext != ".yml") continue;
        std::ifstream f(e.path(), std::ios::binary);
        std::stringstream ss; ss << f.rdbuf();
        Skill s;
        if (parse_skill_yaml(ss.str(), s, nullptr)) out.push_back(std::move(s));
    }
    return out;
}

}  // namespace nova::skills

// apex_subagent_pool.cpp — Implementierung von apex_subagent_pool.h (Design §4.4).
#include "Apex/apex_subagent_pool.h"

#include <cctype>
#include <fstream>

namespace nova::apex {

namespace {
std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else o += c;
    }
    return o;
}
std::string to_lower(const std::string& s) {
    std::string o = s;
    for (char& c : o) c = char(std::tolower((unsigned char)c));
    return o;
}
}  // namespace

std::string build_subagent_prompt(const std::string& tool_name,
                                  const ToolParams& params,
                                  const std::string& result_schema) {
    // Striktes, kontextfreies Kontrakt-JSON — ausschließlich {tool_name, params}.
    std::string p = "{\"tool\":\"" + json_escape(tool_name) + "\",\"params\":{";
    for (size_t i = 0; i < params.size(); ++i) {
        if (i) p += ",";
        p += "\"" + json_escape(params[i].first) + "\":\"" + json_escape(params[i].second) + "\"";
    }
    p += "}";
    if (!result_schema.empty()) p += ",\"result_schema\":\"" + json_escape(result_schema) + "\"";
    p += "}";
    return p;
}

bool is_context_free_prompt(const std::string& prompt) {
    static const char* kForbidden[] = {
        "thought", "action:", "obs:", "reflexion", "persona", "identity",
        "verlauf", "system:", "assistant:", "user:", "# dreaming", "working memory",
    };
    const std::string low = to_lower(prompt);
    for (const char* f : kForbidden)
        if (low.find(f) != std::string::npos) return false;
    return true;
}

bool SubagentPool::load(const std::string& bin_path, std::string* err) {
    std::ifstream f(bin_path, std::ios::binary | std::ios::ate);
    if (!f) { if (err) *err = "3B-Binary nicht gefunden: " + bin_path; return false; }
    const std::streamoff sz = f.tellg();
    f.seekg(0);
    weights_.resize(size_t(sz < 0 ? 0 : sz));
    if (sz > 0) {
        f.read(reinterpret_cast<char*>(weights_.data()), sz);
        if (f.gcount() != sz) { if (err) *err = "3B-Binary Lesefehler"; weights_.clear(); return false; }
    }
    return true;
}

std::vector<std::string> SubagentPool::batch_infer(const std::vector<std::string>& prompts,
                                                   const SubagentFn& fn) const {
    return apex_batch_infer_host(weights_, prompts, fn);
}

}  // namespace nova::apex

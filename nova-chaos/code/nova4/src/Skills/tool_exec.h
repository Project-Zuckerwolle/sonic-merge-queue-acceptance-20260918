// tool_exec.h — Tool-Ausführungs-Framework (Design §13.3, §13.5).
//
// ToolBox bildet Tool-Name -> Executor ab. Jeder native Tool-Executor bekommt
// die Parameter (aus dem <tool_call>-Tag) und einen ToolContext (WorkspaceGuard,
// Bestätigungs-Callback, HTTP-Fetch für Netzwerk-Tools, Pfade). Permission/
// Bestätigung wird VOR der Ausführung über die SkillRegistry geprüft (separat).
#pragma once

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "Skills/workspace_guard.h"

namespace nova::skills {

using Params = std::vector<std::pair<std::string, std::string>>;
std::string param(const Params& p, const std::string& key, const std::string& def = "");
bool        has_param(const Params& p, const std::string& key);

struct ToolResult {
    bool        ok = true;
    std::string output;     // OBS-Text fürs Chat/ReAct
};
inline ToolResult tool_ok(const std::string& s)  { return {true, s}; }
inline ToolResult tool_err(const std::string& s) { return {false, "FEHLER: " + s}; }

struct ToolContext {
    WorkspaceGuard* guard = nullptr;
    // Netzwerk-Tools (wetter/websearch/...): URL -> Antworttext. Injizierbar.
    std::function<std::string(const std::string& url)> http_get;
    std::string todo_path;   // Datei für Todos (im Workspace)
};

using ToolFn = std::function<ToolResult(const Params&, ToolContext&)>;

class ToolBox {
public:
    void register_tool(const std::string& name, ToolFn fn) { tools_[name] = std::move(fn); }
    bool has(const std::string& name) const { return tools_.count(name) > 0; }
    ToolResult execute(const std::string& name, const Params& p, ToolContext& ctx) const;

private:
    std::map<std::string, ToolFn> tools_;
};

// Registriert alle nativen Standard-Tools (§13.3).
ToolBox default_toolbox();

}  // namespace nova::skills

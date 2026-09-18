// tool_exec.cpp — Implementierung von tool_exec.h (Design §13.3).
#include "Skills/tool_exec.h"

#include "Skills/native_tools.h"

namespace nova::skills {

ToolResult ToolBox::execute(const std::string& name, const Params& p, ToolContext& ctx) const {
    auto it = tools_.find(name);
    if (it == tools_.end()) return tool_err("unbekanntes Tool: " + name);
    return it->second(p, ctx);
}

ToolBox default_toolbox() {
    using namespace tools;
    ToolBox b;
    b.register_tool("rechner", rechner);
    b.register_tool("uhrzeit", uhrzeit);
    b.register_tool("datei_lesen", datei_lesen);
    b.register_tool("datei_suchen", datei_suchen);
    b.register_tool("codebase_scan", codebase_scan);
    b.register_tool("todo_lesen", todo_lesen);
    b.register_tool("wetter", wetter);
    b.register_tool("websearch", websearch);
    b.register_tool("news_rss", news_rss);
    b.register_tool("finanzen", finanzen);
    b.register_tool("datei_schreiben", datei_schreiben);
    b.register_tool("datei_verschieben", datei_verschieben);
    b.register_tool("datei_sortieren", datei_sortieren);
    b.register_tool("ordner_erstellen", ordner_erstellen);
    b.register_tool("datei_loeschen", datei_loeschen);
    b.register_tool("todo_schreiben", todo_schreiben);
    b.register_tool("git_ops", git_ops);
    b.register_tool("shell_exec", shell_exec);
    b.register_tool("power_sleep", power_sleep);
    return b;
}

}  // namespace nova::skills

// native_tools.h — Implementierungen der nativen Standard-Tools (Design §13.3).
// Einzeln deklariert, in tool_exec.cpp registriert. Jede Funktion ist ein ToolFn.
#pragma once

#include "Skills/tool_exec.h"

namespace nova::skills::tools {

// Tier 1 (READ).
ToolResult rechner(const Params&, ToolContext&);        // Ausdruck auswerten
ToolResult uhrzeit(const Params&, ToolContext&);
ToolResult datei_lesen(const Params&, ToolContext&);
ToolResult datei_suchen(const Params&, ToolContext&);
ToolResult codebase_scan(const Params&, ToolContext&);
ToolResult todo_lesen(const Params&, ToolContext&);
ToolResult wetter(const Params&, ToolContext&);          // http_get
ToolResult websearch(const Params&, ToolContext&);       // http_get
ToolResult news_rss(const Params&, ToolContext&);        // http_get
ToolResult finanzen(const Params&, ToolContext&);        // http_get

// Tier 2 (WRITE_LOCAL) — alle WorkspaceFS-geschützt.
ToolResult datei_schreiben(const Params&, ToolContext&);
ToolResult datei_verschieben(const Params&, ToolContext&);
ToolResult datei_sortieren(const Params&, ToolContext&);
ToolResult ordner_erstellen(const Params&, ToolContext&);
ToolResult datei_loeschen(const Params&, ToolContext&);
ToolResult todo_schreiben(const Params&, ToolContext&);
ToolResult git_ops(const Params&, ToolContext&);

// Tier 3 (SYSTEM).
ToolResult shell_exec(const Params&, ToolContext&);      // CreateProcess, 60s, Sandbox
ToolResult power_sleep(const Params&, ToolContext&);     // SetSuspendState(S4)

// Ausdrucks-Parser (öffentlich für Tests).
bool eval_expression(const std::string& expr, double& out);

}  // namespace nova::skills::tools

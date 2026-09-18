// testbed12_tool.cpp — Normal-Tool im Chat (Design §16, TB 12).
//
// Kriterium: "<tool_call> intercepted, Ergebnis korrekt eingebettet."
//
// Verbindet den nova_ws-Interceptor (Block C) mit der ToolBox (Block E): der
// Modell-Stream enthält ein <tool_call name="rechner".../>; der Interceptor fängt
// es ab, die ToolBox führt das native Tool aus, und die OBS wird als Text in den
// Chat-Stream eingebettet.
#include "Skills/skill_registry.h"
#include "Skills/tool_exec.h"
#include "Web/nova_ws.h"

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace nova;

int main() {
    std::cout << "=== Testbed 12: Normal-Tool im Chat ===\n";

    const std::string ws = (fs::temp_directory_path() / "nova4_tb12_ws").string();
    std::error_code ec; fs::create_directories(ws, ec);
    skills::WorkspaceGuard guard(ws);
    skills::ToolBox box = skills::default_toolbox();
    skills::ToolContext ctx; ctx.guard = &guard;
    skills::SkillRegistry reg = skills::SkillRegistry::with_defaults();

    // Bridge web::ToolCall -> skills::ToolBox.
    int confirmations = 0;
    auto tool_exec = [&](const web::ToolCall& tc) -> std::string {
        skills::Params p;
        for (const auto& kv : tc.params) p.push_back(kv);
        // Permission-Check (rechner = Tier 1 -> keine Bestätigung).
        if (reg.needs_confirmation(tc.name, /*tier2_confirm=*/true)) ++confirmations;
        return box.execute(tc.name, p, ctx).output;
    };

    // Synthetischer Modell-Stream mit eingebettetem tool_call.
    std::vector<std::string> toks = {
        "Das ", "Ergebnis ", "ist ", "<tool_call name=\"rechner\" ",
        "expr=\"2+3*4\"/>", " — fertig."
    };
    size_t i = 0;
    auto model = [&]() -> std::string { return i < toks.size() ? toks[i++] : std::string(); };

    std::string chat;
    bool tool_seen = false;
    web::ChatHandler ch;
    ch.stream(model,
        [&](const web::ChatEvent& e) {
            if (e.type == web::ChatEvent::Text) chat += e.text;
            else if (e.type == web::ChatEvent::Tool) tool_seen = true;
        },
        tool_exec);

    std::printf("  Chat-Ausgabe: %s\n", chat.c_str());
    std::printf("  Tool-Event gesehen     = %s\n", tool_seen ? "ja" : "NEIN");
    std::printf("  Bestätigungen (Tier 1) = %d (erwartet 0)\n", confirmations);

    bool pass = true;
    pass &= tool_seen;
    pass &= (chat.find("= 14") != std::string::npos);   // Rechner-OBS eingebettet
    pass &= (chat.find("Das Ergebnis ist") != std::string::npos);
    pass &= (chat.find("fertig") != std::string::npos);
    pass &= (confirmations == 0);                        // Tier 1 -> nie bestätigen

    fs::remove_all(ws, ec);
    std::cout << "\n=== Testbed 12: " << (pass ? "BESTANDEN" : "FEHLGESCHLAGEN") << " ===\n";
    return pass ? 0 : 1;
}

// testbed16_subagent_ctx.cpp — 3B-Subagent kontextfrei (Aufgabe 5.2, TB16).
//
// Kriterien:
//  - der an das 3B übergebene Prompt enthält nachweislich nur {tool_name, params}
//    (is_context_free_prompt == true; Chat/ReAct/Persona werden erkannt & abgelehnt),
//  - ein 3B-Batch-Task liefert korrekte Ergebnisse ohne Kontext-Leakage,
//  - die resident gehaltenen 3B-Gewichte werden ohne erneuten Load genutzt.
#include "Apex/apex_subagent_pool.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace nova;

namespace {
// Extrahiert den "tool"-Wert aus dem Kontrakt-JSON (winziger Parser für den Test).
std::string tool_of(const std::string& prompt) {
    const std::string key = "\"tool\":\"";
    const size_t p = prompt.find(key);
    if (p == std::string::npos) return {};
    const size_t s = p + key.size();
    const size_t e = prompt.find('"', s);
    return e == std::string::npos ? std::string() : prompt.substr(s, e - s);
}
}  // namespace

int main() {
    std::cout << "=== Testbed 16: 3B-Subagent kontextfrei ===\n";
    bool pass = true;

    // (a) Kontrakt-Prompt ist kontextfrei und enthält nur Name+Params.
    apex::ToolParams params = {{"location", "Hamburg"}, {"units", "metric"}};
    const std::string p1 = apex::build_subagent_prompt("wetter", params);
    std::printf("  Kontrakt-Prompt: %s\n", p1.c_str());
    pass &= apex::is_context_free_prompt(p1);
    pass &= (p1.find("wetter") != std::string::npos);
    pass &= (p1.find("Hamburg") != std::string::npos);
    pass &= (p1.find("params") != std::string::npos);

    // (b) Ein durchgesickerter ReAct/Chat-Prompt wird als NICHT kontextfrei erkannt.
    const std::string leaky =
        "THOUGHT: zuerst Datei lesen\nOBS: ...\nPersona: Du bist Nova\n" + p1;
    pass &= (apex::is_context_free_prompt(leaky) == false);
    std::printf("  Leaky-Prompt abgelehnt   = %s\n", apex::is_context_free_prompt(leaky) ? "NEIN" : "ja");

    // (c) Resident-Gewichte (2-Modell): kein Load von Disk, loaded()==true.
    apex::SubagentPool pool;
    pool.bind_resident_weights(std::vector<uint8_t>(1024, 0x42));
    pass &= pool.loaded();
    std::printf("  Resident-Gewichte        = %zu Bytes (loaded=%s)\n",
                pool.bytes(), pool.loaded() ? "ja" : "NEIN");

    // (d) Batch über 2 Sequenzen: jede FN-Eingabe MUSS kontextfrei sein.
    const std::vector<std::string> prompts = {
        apex::build_subagent_prompt("rechner", {{"expr", "2+2"}}),
        apex::build_subagent_prompt("uhrzeit", {}),
    };
    bool all_clean = true;
    auto mock_3b = [&](const std::string& prompt) -> std::string {
        if (!apex::is_context_free_prompt(prompt)) all_clean = false;  // Leakage-Wächter
        return "ergebnis:" + tool_of(prompt);
    };
    const std::vector<std::string> outs = pool.batch_infer(prompts, mock_3b);

    pass &= all_clean;
    pass &= (outs.size() == 2);
    if (outs.size() == 2) {
        pass &= (outs[0] == "ergebnis:rechner");
        pass &= (outs[1] == "ergebnis:uhrzeit");
        std::printf("  Batch-Ergebnisse         = [%s, %s]\n", outs[0].c_str(), outs[1].c_str());
    }
    std::printf("  Kein Kontext-Leakage     = %s\n", all_clean ? "ja" : "NEIN");

    std::cout << "\n=== Testbed 16: " << (pass ? "BESTANDEN" : "FEHLGESCHLAGEN") << " ===\n";
    return pass ? 0 : 1;
}

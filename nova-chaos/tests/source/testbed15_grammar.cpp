// testbed15_grammar.cpp — Schema-Zwang beim Decodieren (Aufgabe 5.1, TB15).
//
// Kriterien:
//  (a) über 50 erzwungene Ausgaben sind 100% syntaktisch gültig (parse_step != None),
//  (b) ein Prompt/Logit-Strom der gezielt zum Formatbruch verleitet erzeugt trotzdem
//      gültige Struktur,
//  (c) ein erfundener Tool-Name ist unmöglich — jeder erzeugte Tool-Name liegt im
//      Skill-Registry.
// Zusätzlich: die Reflexions-JSON-Grammatik liefert stets parse_reflexion-gültiges JSON.
#include "Apex/apex_orchestrator.h"
#include "InferEngine/grammar.h"
#include "Skills/skill_registry.h"

#include <cstdio>
#include <iostream>
#include <set>
#include <string>
#include <vector>

using namespace nova;

namespace {
// Deterministischer LCG — variiert die (adversarialen) Logits pro Lauf.
struct Lcg {
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed) {}
    uint32_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return uint32_t(s >> 33); }
    float unit() { return float(next() % 1000) / 1000.0f; }
};
}  // namespace

int main() {
    std::cout << "=== Testbed 15: Schema-Zwang (constrained decoding) ===\n";

    skills::SkillRegistry reg = skills::SkillRegistry::with_defaults();
    std::vector<std::string> tool_names;
    std::set<std::string> valid_tools;
    for (const skills::Skill* s : reg.by_type(skills::SkillType::Tool)) {
        tool_names.push_back(s->name);
        valid_tools.insert(s->name);
    }
    std::printf("  Registry-Tools: %zu\n", tool_names.size());

    int valid = 0, tool_calls = 0, invented = 0;
    const int N = 50;
    for (int i = 0; i < N; ++i) {
        auto g = infer::make_action_grammar(tool_names);
        Lcg rng(0xA5A5 + uint64_t(i) * 2654435761ULL);

        // Adversariale Logits: legen die Masse bewusst auf (im jeweiligen Zustand)
        // OFT VERBOTENE Indizes 0..3 und den Close-Index, um Formatbruch zu erzwingen.
        auto logits = [&](int /*step*/) {
            infer::Logits d(g->vocab().size(), 0.01f);
            // Struktur-Literale + ein zufälliger Index bekommen hohe Masse.
            for (int k = 0; k < 4 && k < int(d.size()); ++k) d[size_t(k)] = 5.0f;
            const size_t r = rng.next() % d.size();
            d[r] += 9.0f;
            return d;
        };

        const std::string out = infer::constrained_generate(*g, logits);
        const apex::ParsedStep ps = apex::parse_step(out);
        if (ps.kind != apex::ParsedStep::None) ++valid;
        if (ps.kind == apex::ParsedStep::ToolCall) {
            ++tool_calls;
            if (valid_tools.find(ps.tool) == valid_tools.end()) ++invented;
        }
        if (i < 4) std::printf("  [%d] %s -> kind=%d tool=%s\n", i, out.c_str(),
                               int(ps.kind), ps.tool.c_str());
    }
    std::printf("  Gültige Ausgaben         = %d/%d\n", valid, N);
    std::printf("  davon Tool-Calls         = %d\n", tool_calls);
    std::printf("  erfundene Tool-Namen     = %d (erwartet 0)\n", invented);

    // (b) Extrem-Fall: ALLE Masse auf einen im ToolName-Zustand verbotenen Index.
    auto g2 = infer::make_action_grammar(tool_names);
    auto evil = [&](int) {
        infer::Logits d(g2->vocab().size(), 0.0f);
        d[0] = 1000.0f;  // "<tool_call name=\"" — nur im Start erlaubt, danach verboten
        return d;
    };
    const std::string evil_out = infer::constrained_generate(*g2, evil);
    const apex::ParsedStep evil_ps = apex::parse_step(evil_out);
    std::printf("  Adversarial-Ausgabe      = %s (kind=%d, tool=%s)\n",
                evil_out.c_str(), int(evil_ps.kind), evil_ps.tool.c_str());
    const bool evil_valid = evil_ps.kind != apex::ParsedStep::None &&
                            (evil_ps.kind != apex::ParsedStep::ToolCall ||
                             valid_tools.count(evil_ps.tool));

    // Reflexions-JSON-Grammatik.
    int refl_valid = 0;
    for (int i = 0; i < 5; ++i) {
        auto rg = infer::make_reflexion_grammar();
        Lcg rng(0x1234 + uint64_t(i));
        auto logits = [&](int) {
            infer::Logits d(rg->vocab().size(), 0.0f);
            const size_t r = rng.next() % d.size();
            d[r] = 100.0f;  // beliebiger Index — Maske erzwingt Gültigkeit
            return d;
        };
        const std::string js = infer::constrained_generate(*rg, logits);
        if (apex::parse_reflexion(js).ok) ++refl_valid;
        if (i == 0) std::printf("  Reflexion-JSON: %s\n", js.c_str());
    }
    std::printf("  Reflexion gültig         = %d/5\n", refl_valid);

    bool pass = true;
    pass &= (valid == N);          // 100% gültig
    pass &= (invented == 0);       // kein erfundener Tool-Name
    pass &= (tool_calls > 0);      // Tool-Calls wurden tatsächlich erzeugt
    pass &= evil_valid;            // Formatbruch-Versuch bleibt gültig
    pass &= (refl_valid == 5);     // Reflexions-JSON stets gültig

    std::cout << "\n=== Testbed 15: " << (pass ? "BESTANDEN" : "FEHLGESCHLAGEN") << " ===\n";
    return pass ? 0 : 1;
}

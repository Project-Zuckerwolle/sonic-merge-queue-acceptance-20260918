// test_blockc_memory.cpp — Block C Logik (Memory + Filter + Interceptor + Settings).
//
// Deckt die nicht-netzwerkbasierten Block-C-Komponenten ab:
//   identity_store (LOCKED/Dreaming), working_memory (Overflow), context_builder,
//   phrase_filter, nova_ws StreamInterceptor (über Token-Grenzen), settings.
#include "Memory/context_builder.h"
#include "Memory/identity_store.h"
#include "Memory/working_memory.h"
#include "Web/nova_ws.h"
#include "Web/phrase_filter.h"
#include "Web/settings.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace nova;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& name) {
    std::printf("  [%s] %s\n", ok ? "OK" : "XX", name.c_str());
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::cout << "=== Block C: Memory / Filter / Interceptor / Settings ===\n";

    // --- IdentityStore: LOCKED + Dreaming ----------------------------------
    {
        memory::IdentityStore id;
        id.parse("# LOCKED\nIch entwickle Nova 4 in C++.\nSprache: Deutsch.\n\n"
                 "# Dreaming\nAlter Fokus.\n");
        check(id.find("LOCKED") != nullptr, "LOCKED-Block geparst");
        check(id.write_dreaming("Neuer Fokus: Block C"), "write_dreaming akzeptiert");
        const std::string* dr = id.find("Dreaming");
        check(dr && *dr == "Neuer Fokus: Block C", "Dreaming-Block aktualisiert");
        const std::string* lk = id.find("LOCKED");
        check(lk && lk->find("Nova 4 in C++") != std::string::npos, "LOCKED unverändert");
        check(!id.set_block("LOCKED", "HACK"), "set_block auf LOCKED verweigert");
        check(id.find("LOCKED")->find("HACK") == std::string::npos, "LOCKED nicht überschrieben");
        // Round-trip
        memory::IdentityStore id2; id2.parse(id.serialize());
        check(id2.find("LOCKED") && id2.find("Dreaming"), "Round-trip bewahrt Blöcke");
    }

    // --- WorkingMemory: Overflow + Kompression ------------------------------
    {
        memory::WorkingMemory wm(50);  // kleines Budget zum Auslösen
        int compress_calls = 0;
        wm.set_compressor([&](const std::vector<memory::Turn>& old) {
            ++compress_calls;
            return std::string("[Summary von ") + std::to_string(old.size()) + " Turns]";
        });
        for (int i = 0; i < 30; ++i)
            wm.add_turn("user", "Dies ist Turn Nummer " + std::to_string(i) + " mit etwas Text.");
        check(compress_calls > 0, "Overflow löste Kompression aus");
        check(wm.total_tokens() <= 50, "Token unter Budget gehalten");
        check(wm.turns().front().role == "summary", "Ältestes ist Summary");
        check(wm.turns().back().text.find("29") != std::string::npos, "Jüngster Turn erhalten");
    }

    // --- ContextBuilder -----------------------------------------------------
    {
        memory::ContextBuilder cb;
        memory::ContextStack s;
        s.persona = "Du bist Nova.";
        s.identity = "# LOCKED\nC++ Projekt.";
        s.hot = "Aktueller Fokus: Block C.";
        s.daily_briefing = "Gestern: Block B fertig.";
        s.episodes = {"Episode 1", "Episode 2"};
        s.working_memory = "user: Hallo\nassistant: Hi";
        s.wiki = {"turboquant.md", "kv_manager.md"};
        const auto r = cb.assemble(s);
        check(r.text.find("PERSONA") != std::string::npos, "Context enthält PERSONA");
        check(r.text.find("WIKI") != std::string::npos, "Context enthält WIKI");
        check(r.text.find("WORKING MEMORY") != std::string::npos, "Context enthält Working Memory");
        check(r.total_tokens > 0, "Token-Schätzung > 0");
    }

    // --- PhraseFilter -------------------------------------------------------
    {
        web::PhraseFilter pf;
        const std::string in = "Natürlich! Das geht so. Super Frage! Hier die Antwort.";
        const std::string out = pf.filter(in);
        check(out.find("Natürlich!") == std::string::npos, "Floskel 'Natürlich!' entfernt");
        check(out.find("Super Frage!") == std::string::npos, "Floskel 'Super Frage!' entfernt");
        check(out.find("Hier die Antwort") != std::string::npos, "Inhalt bewahrt");
    }

    // --- StreamInterceptor: Tags über Token-Grenzen -------------------------
    {
        web::StreamInterceptor it;
        std::string text;
        std::vector<web::ToolCall> tools;
        std::vector<web::ApexCall> apex;
        it.on_text([&](const std::string& s) { text += s; });
        it.on_tool([&](const web::ToolCall& t) { tools.push_back(t); });
        it.on_apex([&](const web::ApexCall& a) { apex.push_back(a); });

        // Tokenstrom mit zerstückelten Tags (Tag-Start über Chunk-Grenze).
        std::vector<std::string> stream = {
            "Wetter: ", "<tool_", "call name=\"wetter\" location=\"Ham", "burg\"/>",
            " und ", "<apex_call skill=\"coding\" task=\"Implementiere X\"/>", " fertig."
        };
        for (auto& tok : stream) it.feed(tok);
        it.finish();

        check(text == "Wetter:  und  fertig.", "Text korrekt (Tags entfernt)");
        check(tools.size() == 1 && tools[0].name == "wetter", "tool_call abgefangen");
        check(tools.size() == 1 && tools[0].param("location") == "Hamburg", "tool_call Parameter geparst");
        check(apex.size() == 1 && apex[0].skill == "coding" && apex[0].task == "Implementiere X",
              "apex_call abgefangen");
    }

    // --- StreamInterceptor: literales '<' bleibt Text -----------------------
    {
        web::StreamInterceptor it;
        std::string text;
        it.on_text([&](const std::string& s) { text += s; });
        it.feed("a < b und c < d");
        it.finish();
        check(text == "a < b und c < d", "Literales '<' nicht als Tag interpretiert");
    }

    // --- Settings: Round-trip ----------------------------------------------
    {
        web::Settings st = web::Settings::defaults();
        st.set("log_level", "DEBUG");
        const std::string j = st.serialize();
        web::Settings st2; st2.parse(j);
        check(st2.get("log_level") == "DEBUG", "Settings String round-trip");
        check(st2.get_int("port_http", 0) == 8000, "Settings Zahl round-trip");
        check(st2.get("bind_address") == "127.0.0.1", "Settings Default vorhanden");
    }

    std::cout << "\n=== Block C Logik: " << (g_fail == 0 ? "BESTANDEN" : "FEHLGESCHLAGEN")
              << " (" << g_fail << " Fehler) ===\n";
    return g_fail == 0 ? 0 : 1;
}

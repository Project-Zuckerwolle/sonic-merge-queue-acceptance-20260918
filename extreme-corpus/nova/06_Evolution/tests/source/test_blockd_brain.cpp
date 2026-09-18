// test_blockd_brain.cpp — Block D Logik (Brain-Pipeline + Server-Features).
//
// Deckt ab: brain_store, brain_consolidator (Konfidenz + Duplikat), evergreen,
// idea_detector, brain_linker (LLM-Stub), brain_compiler (apex_running-Lock,
// raw->wiki, daily), dreaming (LOCKED-Schutz), idle_monitor, ram_model_store,
// startup_loader, service-Command-Parser.
#include "Brain/brain_compiler.h"
#include "Brain/brain_consolidator.h"
#include "Brain/brain_evergreen.h"
#include "Brain/brain_idea_detector.h"
#include "Brain/brain_linker.h"
#include "Brain/brain_store.h"
#include "Brain/wiki_meta.h"
#include "Memory/dreaming.h"
#include "Memory/identity_store.h"
#include "System/idle_monitor.h"
#include "System/ram_model_store.h"
#include "System/startup_loader.h"
#include "System/service.h"

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace nova;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& name) {
    std::printf("  [%s] %s\n", ok ? "OK" : "XX", name.c_str());
    if (!ok) ++g_fail;
}
std::string mkpage(const std::string& body, const brain::WikiMeta& m) {
    return brain::apply_meta(body, m);
}
}  // namespace

int main() {
    std::cout << "=== Block D: Brain-Pipeline + Server-Features ===\n";

    const std::string root = (fs::temp_directory_path() / "nova4_tb_blockd").string();
    std::error_code ec; fs::remove_all(root, ec);
    brain::BrainStore store(root);
    std::string err;
    check(store.init(&err), "brain_store init");

    // --- Wiki R/W -----------------------------------------------------------
    store.write_wiki(brain::WikiCategory::Concepts, "turboquant",
                     "# TurboQuant\nKV-Cache Kompression PolarQuant QJL.");
    store.write_wiki(brain::WikiCategory::Concepts, "spec_decoding",
                     "# Speculative Decoding\nDraft Modell verifiziert Token.");
    store.write_wiki(brain::WikiCategory::Entities, "ministral14b",
                     "# Ministral 14B\nReasoning Orchestrator Brain Apex.");
    {
        std::string out;
        check(store.read_wiki(brain::WikiCategory::Concepts, "turboquant", out) &&
              out.find("PolarQuant") != std::string::npos, "Wiki read/write");
        check(store.list_wiki().size() == 3, "list_wiki zählt 3 Seiten");
    }

    // --- Consolidator: Konfidenz -------------------------------------------
    {
        brain::BrainConsolidator con;
        auto r = con.consolidate(store, "2026-06-24", "sess-1", {"turboquant"});
        check(r.updated >= 1, "Konsolidierung: erwähnte Seite aktualisiert");
        std::string page; store.read_wiki(brain::WikiCategory::Concepts, "turboquant", page);
        auto m = brain::parse_meta(page);
        check(m.last_mentioned == "2026-06-24", "last_mentioned gesetzt");
        check(m.sessions.size() == 1 && m.sessions[0] == "sess-1", "Session-ID erfasst");
    }

    // --- Consolidator: Inaktiv nach >30 Tagen ------------------------------
    {
        brain::WikiMeta old; old.last_mentioned = "2026-01-01";  // ~175 Tage alt
        store.write_wiki(brain::WikiCategory::Synthesis, "altseite",
                         mkpage("# Alt\nLange nicht erwähnt.", old));
        brain::BrainConsolidator con;
        auto r = con.consolidate(store, "2026-06-24", "sess-2", {});
        check(r.deactivated >= 1, "Alte Seite als [inaktiv] markiert");
        std::string page; store.read_wiki(brain::WikiCategory::Synthesis, "altseite", page);
        check(brain::parse_meta(page).inactive, "Inaktiv-Flag gesetzt");
    }

    // --- Evergreen: >=3 Sessions -------------------------------------------
    {
        brain::WikiMeta m; m.sessions = {"s1", "s2", "s3"}; m.last_mentioned = "2026-06-24";
        store.write_wiki(brain::WikiCategory::Concepts, "evergreen_kandidat",
                         mkpage("# Wichtig\nOft erwähnt.", m));
        const int marked = brain::mark_evergreen(store, 3);
        check(marked >= 1, "Evergreen markiert (>=3 Sessions)");
        std::string page; store.read_wiki(brain::WikiCategory::Concepts, "evergreen_kandidat", page);
        check(brain::parse_meta(page).evergreen, "Evergreen-Flag gesetzt");

        // Evergreen-Seite darf nicht inaktiv werden, auch wenn alt.
        brain::WikiMeta em = brain::parse_meta(page); em.last_mentioned = "2026-01-01";
        store.write_wiki(brain::WikiCategory::Concepts, "evergreen_kandidat",
                         mkpage("# Wichtig\nOft erwähnt.", em));
        brain::BrainConsolidator con;
        con.consolidate(store, "2026-06-24", "sess-x", {});
        std::string p2; store.read_wiki(brain::WikiCategory::Concepts, "evergreen_kandidat", p2);
        check(!brain::parse_meta(p2).inactive, "Evergreen bleibt aktiv trotz Alter");
    }

    // --- Duplikat-Prüfung ---------------------------------------------------
    {
        store.write_wiki(brain::WikiCategory::Entities, "dup_a",
                         "# Mistral\nMinistral 14B Reasoning Orchestrator Brain Apex Modell.");
        store.write_wiki(brain::WikiCategory::Entities, "dup_b",
                         "# Mistral\nMinistral 14B Reasoning Orchestrator Brain Apex Modell.");
        brain::BrainConsolidator con;
        auto r = con.consolidate(store, "2026-06-24", "sess-3", {});
        check(r.duplicates >= 1, "Duplikat-Paar erkannt");
        check(store.wiki_exists(brain::WikiCategory::Synthesis, "duplikat_vorschlaege"),
              "duplikat_vorschlaege.md geschrieben");
    }

    // --- Brain Linker (LLM-Stub) -------------------------------------------
    {
        brain::LlmFn llm = [](const std::string&) {
            return "turboquant -> kv_manager | 0.9 | direkte Abhängigkeit\n"
                   "turboquant -> spec_decoding | 0.4 | beide Decoding\n";
        };
        store.write_wiki(brain::WikiCategory::Concepts, "kv_manager", "# KV Manager\nH2O Trim.");
        brain::BrainLinker linker(llm);
        std::vector<brain::WikiRef> pages = {{brain::WikiCategory::Concepts, "turboquant"}};
        const int n = linker.link(store, pages);
        check(n == 2, "Linker schrieb 2 Links");
        std::string page; store.read_wiki(brain::WikiCategory::Concepts, "turboquant", page);
        auto m = brain::parse_meta(page);
        check(m.links.size() == 2 && m.links[0].target == "kv_manager" && m.links[0].strength > 0.85,
              "Links korrekt geparst (Stärke)");
    }

    // --- Idea Detector ------------------------------------------------------
    {
        brain::IdeaDetector det;
        auto scores = det.detect(store);
        check(!scores.empty(), "Idea-Detector liefert Scores");
        bool found_tq = false;
        for (auto& s : scores) if (s.name == "turboquant") { found_tq = true;
            check(s.score >= 0.0 && s.score <= 1.0, "Score in [0,1]"); }
        check(found_tq, "turboquant (concepts) bewertet");
    }

    // --- Brain Compiler: apex_running-Lock ---------------------------------
    {
        std::atomic<bool> apex{true};
        brain::LlmFn llm = [](const std::string&) { return "CATEGORY: synthesis\nNAME: x\n# X\ninhalt"; };
        brain::BrainCompiler bc(store, llm, apex);
        brain::BrainCycleInput in; in.today = "2026-06-24"; in.session_id = "s";
        auto r = bc.run_cycle(in);
        check(r.skipped_apex, "Brain-Cycle übersprungen bei apex_running");

        apex.store(false);
        store.append_raw("note1.md", "Notiz: Nova nutzt CUDA auf RTX 3080.");
        auto r2 = bc.run_cycle(in);
        check(!r2.skipped_apex, "Brain-Cycle läuft bei apex_running=false");
        check(r2.raw_processed >= 1, "raw -> wiki verarbeitet");
        check(store.wiki_exists(brain::WikiCategory::Synthesis, "x"), "Wiki-Seite aus raw erzeugt");
    }

    // --- Brain Compiler: Daily (Linking/Evergreen/Briefing) ----------------
    {
        std::atomic<bool> apex{false};
        const std::string brief_path = (fs::path(root) / "daily_briefing.md").string();
        brain::LlmFn llm = [](const std::string& p) {
            if (p.find("Daily Briefing") != std::string::npos) return std::string("# Briefing\nHeute: Block D.");
            return std::string("turboquant -> kv_manager | 0.8 | Abhängigkeit");
        };
        brain::BrainCompiler bc(store, llm, apex);
        bc.set_briefing_path(brief_path);
        brain::BrainCycleInput in; in.today = "2026-06-24"; in.session_id = "s2"; in.run_daily = true;
        in.briefing_source = "Letzte Episode: Block D gebaut.";
        auto r = bc.run_cycle(in);
        check(r.daily_run, "Daily-Lauf ausgeführt");
        check(fs::exists(brief_path), "daily_briefing.md geschrieben");
    }

    // --- Dreaming: LOCKED-Schutz -------------------------------------------
    {
        memory::IdentityStore id;
        id.parse("# LOCKED\nNova 4 in C++.\n\n# Dreaming\nAlt.\n");
        brain::LlmFn llm = [](const std::string&) { return "Neuer Stand: Block D fertig."; };
        check(memory::run_dreaming(llm, "user: Block D done", id, &err), "Dreaming ausgeführt");
        check(id.find("Dreaming") && *id.find("Dreaming") == "Neuer Stand: Block D fertig.",
              "Dreaming-Block aktualisiert");
        check(id.find("LOCKED") && id.find("LOCKED")->find("Nova 4 in C++") != std::string::npos,
              "LOCKED unberührt");
    }

    // --- Idle Monitor: Zustandsmaschine ------------------------------------
    {
        system::IdleConfig cfg; cfg.idle_timeout_ms = 1000; cfg.countdown_ms = 500;
        system::IdleState st; st.last_activity_ms = 0;
        // t=500: noch aktiv genug -> None
        check(system::evaluate_idle(500, false, cfg, st) == system::IdleAction::None, "Idle: noch aktiv");
        // t=1000: Schwelle -> CountdownStart
        check(system::evaluate_idle(1000, false, cfg, st) == system::IdleAction::CountdownStart, "Idle: Countdown start");
        // t=1200: Countdown läuft -> None
        check(system::evaluate_idle(1200, false, cfg, st) == system::IdleAction::None, "Idle: Countdown läuft");
        // t=1500: Countdown abgelaufen -> Hibernate
        check(system::evaluate_idle(1500, false, cfg, st) == system::IdleAction::Hibernate, "Idle: Hibernate");
        // Apex blockiert Hibernate
        system::IdleState st2; st2.last_activity_ms = 0;
        system::evaluate_idle(2000, false, cfg, st2);  // -> Countdown
        check(system::evaluate_idle(2100, true, cfg, st2) == system::IdleAction::AbortedByActivity,
              "Idle: Apex bricht Countdown ab");
    }

    // --- RAM Model Store + Startup Loader -----------------------------------
    {
        const std::string mp = (fs::path(root) / "fakemodel.bin").string();
        { std::ofstream f(mp, std::ios::binary); std::string data(1 << 20, 'X'); f.write(data.data(), data.size()); }
        system::RamModelStore rms;
        std::vector<system::PreloadEntry> models = {{"gemma3-1b", mp}};
        auto res = system::preload_models(rms, models);
        check(res.size() == 1 && res[0].ok && res[0].bytes == (1u << 20), "Startup-Preload lädt Modell");
        check(rms.has("gemma3-1b") && rms.total_bytes() == (1u << 20), "RAM-Store hält Modell");
        rms.free("gemma3-1b");
        check(!rms.has("gemma3-1b") && rms.total_bytes() == 0, "RAM-Store free()");
    }

    // --- Service Command Parser --------------------------------------------
    {
        const char* a1[] = {"nova4.exe", "--install-service"};
        const char* a2[] = {"nova4.exe", "--uninstall-service"};
        const char* a3[] = {"nova4.exe", "--service"};
        const char* a4[] = {"nova4.exe"};
        check(system::parse_service_command(2, const_cast<char**>(a1)) == system::ServiceCommand::Install, "Service: --install");
        check(system::parse_service_command(2, const_cast<char**>(a2)) == system::ServiceCommand::Uninstall, "Service: --uninstall");
        check(system::parse_service_command(2, const_cast<char**>(a3)) == system::ServiceCommand::Run, "Service: --service");
        check(system::parse_service_command(1, const_cast<char**>(a4)) == system::ServiceCommand::None, "Service: keine");
    }

    fs::remove_all(root, ec);
    std::cout << "\n=== Block D Logik: " << (g_fail == 0 ? "BESTANDEN" : "FEHLGESCHLAGEN")
              << " (" << g_fail << " Fehler) ===\n";
    return g_fail == 0 ? 0 : 1;
}

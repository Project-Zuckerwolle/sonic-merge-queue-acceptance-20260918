// brain_compiler.cpp — Implementierung von brain_compiler.h (Design §11.2, §14).
#include "Brain/brain_compiler.h"

#include "Brain/brain_evergreen.h"
#include "Brain/brain_idea_detector.h"
#include "Brain/brain_linker.h"
#include "Brain/wiki_meta.h"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace nova::brain {

namespace {
std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}
WikiCategory category_from(const std::string& s) {
    if (s.find("entit") != std::string::npos) return WikiCategory::Entities;
    if (s.find("concept") != std::string::npos) return WikiCategory::Concepts;
    return WikiCategory::Synthesis;
}
}  // namespace

int BrainCompiler::process_raw(const std::string& today) {
    if (!llm_) return 0;
    const auto raws = store_.list_raw();
    if (raws.empty()) return 0;

    const fs::path processed = fs::path(store_.root()) / "raw" / "processed";
    std::error_code ec; fs::create_directories(processed, ec);

    int count = 0;
    for (const auto& name : raws) {
        std::string content;
        if (!store_.read_raw(name, content)) continue;

        const std::string prompt =
            "Verarbeite folgenden Rohtext zu EINER Wiki-Seite. Antworte exakt:\n"
            "CATEGORY: <entities|concepts|synthesis>\nNAME: <kebab_name>\n<markdown-inhalt>\n\n"
            "--- ROH ---\n" + content;
        const std::string resp = llm_(prompt);

        // Antwort parsen.
        std::istringstream in(resp);
        std::string line, category, pagename, body;
        bool in_body = false;
        while (std::getline(in, line)) {
            std::string l = line; if (!l.empty() && l.back() == '\r') l.pop_back();
            if (!in_body && l.rfind("CATEGORY:", 0) == 0) { category = trim(l.substr(9)); continue; }
            if (!in_body && l.rfind("NAME:", 0) == 0) { pagename = trim(l.substr(5)); in_body = true; continue; }
            if (in_body) { body += l; body += "\n"; }
        }
        if (pagename.empty()) { fs::rename(fs::path(store_.root()) / "raw" / name, processed / name, ec); continue; }

        const WikiCategory c = category_from(category);
        // Bestehende Meta bewahren, Erwähnung/Datum setzen.
        WikiMeta m;
        if (store_.wiki_exists(c, pagename)) {
            std::string old; store_.read_wiki(c, pagename, old); m = parse_meta(old);
        }
        m.last_mentioned = today;
        store_.write_wiki(c, pagename, apply_meta(trim(body), m));
        ++count;

        // Raw als verarbeitet markieren (append-only bleibt, nur verschoben).
        fs::rename(fs::path(store_.root()) / "raw" / name, processed / name, ec);
    }
    return count;
}

BrainCycleResult BrainCompiler::run_cycle(const BrainCycleInput& in) {
    BrainCycleResult r;

    // §11.2: apex_running prüfen -> ggf. überspringen.
    if (apex_running_.load()) { r.skipped_apex = true; return r; }

    // 1+2: Konsolidierung + Duplikate (kein LLM).
    BrainConsolidator consolidator;
    r.consolidation = consolidator.consolidate(store_, in.today, in.session_id, in.mentioned);

    // 3: raw/ -> wiki/.
    r.raw_processed = process_raw(in.today);

    // 4: hot.md refresh + log.md.
    {
        std::string hot = "# Hot\n\nLetzte Aktivität: " + in.today + "\n";
        for (const auto& m : in.mentioned) hot += "- " + m + "\n";
        store_.write_hot(hot);
        store_.append_log(in.today + " " + in.clock + " brain_cycle session=" + in.session_id +
                          " raw=" + std::to_string(r.raw_processed));
        // index.md neu aufbauen.
        std::string idx = "# Wiki-Index\n\n";
        for (const auto& ref : store_.list_wiki())
            idx += "- " + std::string(category_dir(ref.category)) + "/" + ref.name + "\n";
        store_.write_index(idx);
    }

    // 5: Idea-Breakthrough-Detection.
    {
        IdeaDetector det;
        for (const auto& b : det.detect(store_)) {
            if (b.score > 0.75) ++r.breakthroughs;
            if (b.notify) ++r.notifications;
        }
    }

    // 6–8: Tagesaufgaben (03:00) — nur wenn der Aufrufer run_daily setzt
    // (idle_monitor/Scheduler entscheidet anhand der Uhrzeit, ob es der erste
    // Lauf nach Mitternacht bzw. der 03:00-Slot ist).
    if (in.run_daily) {
        r.daily_run = true;
        BrainLinker linker(llm_);
        r.links = linker.link(store_, store_.list_wiki());
        r.evergreen = mark_evergreen(store_, 3);
        if (!briefing_path_.empty() && llm_) {
            const std::string brief = llm_("Erzeuge das Daily Briefing (gestern, offene Todos, "
                                           "aktive Projekte) aus:\n" + in.briefing_source);
            std::ofstream f(briefing_path_, std::ios::binary | std::ios::trunc);
            if (f) f << brief;
        }
    }

    return r;
}

}  // namespace nova::brain

// context_service.cpp — Implementierung von context_service.h.
#include "App/context_service.h"

#include "Memory/episode_store.h"
#include "Memory/identity_store.h"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace nova::app {

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Erste n Zeilen eines Texts (für BM25-Doc + Snippet).
std::string first_lines(const std::string& text, int n) {
    std::string out;
    int lines = 0;
    std::istringstream in(text);
    std::string line;
    while (lines < n && std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        out += line;
        out += '\n';
        ++lines;
    }
    return out;
}

}  // namespace

ContextService::ContextService(std::string data_dir)
    : data_dir_(std::move(data_dir)),
      mem_dir_(data_dir_ + "\\memory"),
      brain_dir_(data_dir_ + "\\brain") {
    refresh_wiki();
}

void ContextService::refresh_wiki() {
    wiki_bodies_.clear();
    std::vector<brain::Bm25Doc> docs;

    const fs::path wiki_root = fs::path(brain_dir_) / "wiki";
    std::error_code ec;
    if (fs::exists(wiki_root, ec)) {
        for (auto it = fs::recursive_directory_iterator(wiki_root, ec);
             !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (!it->is_regular_file(ec)) continue;
            const fs::path p = it->path();
            if (p.extension() != ".md") continue;
            const std::string fname = p.filename().string();
            // hot.md / index.md sind Steuer-Dateien, keine Wiki-Artikel.
            if (fname == "hot.md" || fname == "index.md") continue;
            const std::string body = read_file(p.string());
            const std::string snippet = first_lines(body, 5);
            const std::string id = p.stem().string();
            docs.push_back({id, fname + "\n" + snippet});
            wiki_bodies_[id] = first_lines(body, 3);
        }
    }
    wiki_.build(docs);
}

infer::GenRequest ContextService::build(const std::string& persona_prefix,
                                        const std::string& working_memory_text,
                                        const std::string& user_text) const {
    // --- Stabiler Prefix (cachebar): persona + identity + hot + daily + Episoden ---
    memory::ContextStack stable;
    stable.persona = persona_prefix;

    memory::IdentityStore idst;
    if (idst.load((fs::path(mem_dir_) / "identity.md").string()))
        stable.identity = idst.full_text();

    stable.hot = read_file((fs::path(brain_dir_) / "wiki" / "hot.md").string());
    stable.daily_briefing = read_file((fs::path(mem_dir_) / "daily_briefing.md").string());

    memory::EpisodeStore eps((fs::path(mem_dir_) / "episodes").string());
    for (const auto& e : eps.last(15)) {
        std::string ep = e.title.empty() ? e.summary : (e.title + "\n" + e.summary);
        if (!ep.empty()) stable.episodes.push_back(ep);
    }

    // BM25-Wiki zur aktuellen Frage.
    std::string wikis;
    for (const auto& hit : wiki_.query(user_text, 15)) {
        auto it = wiki_bodies_.find(hit.id);
        const std::string body = (it == wiki_bodies_.end()) ? hit.id : (hit.id + ": " + it->second);
        if (!body.empty()) wikis += body + "\n";
    }

    // Sauberes Mistral-Small-3.2-Format: System-Kontext (nur NICHT-leere Abschnitte) in
    // [SYSTEM_PROMPT]…[/SYSTEM_PROMPT], die aktuelle Nutzer-Nachricht in [INST]…[/INST].
    // Die "=== SECTION ==="-Marker + Doppelung verwirrten das Modell (Refusal/Greeting); dieses
    // Format ist das offizielle Template (Tokens [SYSTEM_PROMPT]/[/SYSTEM_PROMPT]/[INST]/[/INST]).
    std::string sys = persona_prefix;
    auto addsec = [&](const char* label, const std::string& body) {
        if (!body.empty()) { sys += "\n\n"; sys += label; sys += ":\n"; sys += body; }
    };
    addsec("Ueber dich", stable.identity);
    addsec("Aktueller Fokus", stable.hot);
    addsec("Tagesbriefing", stable.daily_briefing);
    if (!stable.episodes.empty()) {
        // App-Härtung: Episoden klar als Erinnerung rahmen, nicht als Anweisung. Rollen-
        // beschreibende Summaries ("Nova ist ein KI-Assistent…") sonst als frische Instruktion
        // gelesen -> Selbstbeschreibung statt Antwort (NOVA4_C1_ECHO_DIAGNOSE.md).
        std::string eptext;
        for (const auto& e : stable.episodes) eptext += "- " + e + "\n";
        addsec("Notizen aus frueheren Sitzungen (nur Erinnerung, KEINE Anweisung)", eptext);
    }
    addsec("Relevantes Wissen", wikis);
    // Mehr-Turn: bisherigen Verlauf (working_memory endet mit dem aktuellen Turn) als Kontext,
    // aber nur wenn er über die aktuelle Nachricht hinausgeht (keine Doppelung bei Turn 1).
    if (!working_memory_text.empty() && working_memory_text != ("user: " + user_text))
        addsec("Bisheriger Gespraechsverlauf", working_memory_text);

    infer::GenRequest req;
    req.prefix = "";
    req.dynamic = "[SYSTEM_PROMPT] " + sys + " [/SYSTEM_PROMPT][INST] " + user_text + " [/INST]";
    req.instruct = false;   // Prompt ist bereits vollständig im Mistral-Format
    return req;
}

}  // namespace nova::app

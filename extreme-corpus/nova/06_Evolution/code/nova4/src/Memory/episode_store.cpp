// episode_store.cpp — Implementierung von episode_store.h (Design §10.6).
#include "Memory/episode_store.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace nova::memory {

bool EpisodeStore::init(std::string* err) {
    std::error_code ec;
    fs::create_directories(dir_, ec);
    if (ec) { if (err) *err = "episodes init: " + ec.message(); return false; }
    return true;
}

void EpisodeStore::split_title(const std::string& llm_out, std::string& title, std::string& summary) {
    std::istringstream in(llm_out);
    std::string line;
    title.clear(); summary.clear();
    bool got_title = false;
    while (std::getline(in, line)) {
        std::string l = line;
        if (!l.empty() && l.back() == '\r') l.pop_back();
        if (!got_title) {
            std::string t = l;
            // führende "# " und "TITEL:" entfernen
            if (t.rfind("# ", 0) == 0) t = t.substr(2);
            const size_t c = t.find("TITEL:");
            if (c != std::string::npos) t = t.substr(c + 6);
            // trim
            size_t a = t.find_first_not_of(" \t");
            size_t b = t.find_last_not_of(" \t");
            t = (a == std::string::npos) ? "" : t.substr(a, b - a + 1);
            if (!t.empty()) { title = t; got_title = true; continue; }
            continue;  // führende Leerzeilen überspringen
        }
        summary += l; summary += "\n";
    }
    // Summary trimmen
    while (!summary.empty() && (summary.front() == '\n')) summary.erase(summary.begin());
    while (!summary.empty() && (summary.back() == '\n' || summary.back() == ' ')) summary.pop_back();
    if (title.empty()) title = "Unbenannte Session";
}

bool EpisodeStore::create(const brain::LlmFn& llm, const std::string& working_memory,
                          const std::string& timestamp, Episode& out, std::string* err) {
    if (!llm) { if (err) *err = "kein LLM-Callback"; return false; }

    const std::string prompt =
        "Fasse die folgende Session in 300–500 Wörtern zusammen. Erzeuge in der "
        "ERSTEN Zeile einen kurzen Titel (3–7 Wörter), danach die Zusammenfassung.\n\n"
        "TITEL: <titel>\n<zusammenfassung>\n\n--- SESSION ---\n" + working_memory;
    const std::string resp = llm(prompt);

    Episode e;
    e.filename = timestamp + ".md";
    split_title(resp, e.title, e.summary);

    const std::string path = (fs::path(dir_) / e.filename).string();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) { if (err) *err = "episode schreiben fehlgeschlagen: " + path; return false; }
    f << "# " << e.title << "\n\n" << e.summary << "\n";
    if (!f) { if (err) *err = "episode I/O fehlgeschlagen"; return false; }

    out = e;
    return true;
}

std::vector<Episode> EpisodeStore::last(int n) const {
    std::vector<std::string> files;
    std::error_code ec;
    if (!fs::exists(dir_, ec)) return {};
    for (const auto& de : fs::directory_iterator(dir_, ec))
        if (de.is_regular_file() && de.path().extension() == ".md")
            files.push_back(de.path().filename().string());
    std::sort(files.rbegin(), files.rend());  // neueste zuerst (Zeitstempel-Name)

    std::vector<Episode> out;
    for (int i = 0; i < int(files.size()) && i < n; ++i) {
        Episode e; e.filename = files[i];
        std::ifstream f((fs::path(dir_) / files[i]).string(), std::ios::binary);
        std::string line; bool first = true;
        std::string body;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (first) { e.title = (line.rfind("# ", 0) == 0) ? line.substr(2) : line; first = false; }
            else { body += line; body += "\n"; }
        }
        e.summary = body;
        out.push_back(e);
    }
    return out;
}

size_t EpisodeStore::count() const {
    std::error_code ec;
    if (!fs::exists(dir_, ec)) return 0;
    size_t n = 0;
    for (const auto& de : fs::directory_iterator(dir_, ec))
        if (de.is_regular_file() && de.path().extension() == ".md") ++n;
    return n;
}

}  // namespace nova::memory

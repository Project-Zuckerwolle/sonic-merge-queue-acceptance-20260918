// brain_linker.cpp — Implementierung von brain_linker.h (Design §11.2 #6).
#include "Brain/brain_linker.h"

#include "Brain/wiki_meta.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <sstream>

namespace nova::brain {

namespace {
std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}
}  // namespace

std::vector<BrainLinker::ParsedLink> BrainLinker::parse_response(const std::string& resp) {
    std::vector<ParsedLink> out;
    std::stringstream ss(resp);
    std::string line;
    while (std::getline(ss, line)) {
        // Format: QUELLE -> ZIEL | STÄRKE | BEGRÜNDUNG
        const size_t arrow = line.find("->");
        if (arrow == std::string::npos) continue;
        const size_t p1 = line.find('|', arrow);
        if (p1 == std::string::npos) continue;
        const size_t p2 = line.find('|', p1 + 1);

        ParsedLink l;
        l.source = trim(line.substr(0, arrow));
        l.target = trim(line.substr(arrow + 2, p1 - arrow - 2));
        const std::string sv = trim(line.substr(p1 + 1, (p2 == std::string::npos ? line.size() : p2) - p1 - 1));
        l.strength = std::atof(sv.c_str());
        if (p2 != std::string::npos) l.reason = trim(line.substr(p2 + 1));
        if (!l.source.empty() && !l.target.empty()) out.push_back(l);
    }
    return out;
}

int BrainLinker::link(BrainStore& store, const std::vector<WikiRef>& pages) const {
    if (!llm_ || pages.empty()) return 0;

    // Quelle-Name -> Ref (zum Zurückschreiben).
    std::map<std::string, WikiRef> ref_of;
    for (const auto& r : pages) ref_of[r.name] = r;

    int total = 0;
    for (size_t i = 0; i < pages.size(); i += size_t(batch_)) {
        const size_t end = std::min(pages.size(), i + size_t(batch_));
        // Prompt mit Seitennamen + ersten Zeilen aufbauen.
        std::string prompt = "Finde Verbindungen zwischen diesen Wiki-Seiten. "
                             "Antworte je Zeile: QUELLE -> ZIEL | STÄRKE(0..1) | BEGRÜNDUNG\n\n";
        for (size_t j = i; j < end; ++j) {
            prompt += "## " + pages[j].name + "\n";
            prompt += store.first_lines(pages[j].category, pages[j].name, 5);
            prompt += "\n";
        }

        const std::string resp = llm_(prompt);
        // Links je Quelle sammeln.
        std::map<std::string, std::vector<WikiLink>> by_source;
        for (const auto& pl : parse_response(resp)) {
            if (!ref_of.count(pl.source)) continue;
            by_source[pl.source].push_back({pl.target, pl.strength, pl.reason});
        }
        // In die Quell-Seiten schreiben (bestehende Links ersetzen).
        for (auto& [src, links] : by_source) {
            const WikiRef& r = ref_of[src];
            std::string page;
            if (!store.read_wiki(r.category, r.name, page)) continue;
            WikiMeta m = parse_meta(page);
            m.links = links;
            store.write_wiki(r.category, r.name, apply_meta(strip_meta(page), m));
            total += int(links.size());
        }
    }
    return total;
}

}  // namespace nova::brain

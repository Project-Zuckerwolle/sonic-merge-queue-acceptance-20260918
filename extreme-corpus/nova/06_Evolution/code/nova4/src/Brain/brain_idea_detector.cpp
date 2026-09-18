// brain_idea_detector.cpp — Implementierung von brain_idea_detector.h (Design §11.2 #5).
#include "Brain/brain_idea_detector.h"

#include "Brain/brain_consolidator.h"  // similarity
#include "Brain/wiki_meta.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <set>

namespace nova::brain {

double IdeaDetector::clamp01(double x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

std::vector<BreakthroughScore> IdeaDetector::detect(BrainStore& store) const {
    const auto refs = store.list_wiki();

    // Inbound-Links + Kategorie je Seite einsammeln.
    std::map<std::string, WikiCategory> category_of;
    std::map<std::string, int> inbound;
    std::map<std::string, std::set<int>> inbound_categories;  // Kategorien der Quellen
    std::map<std::string, std::string> first5;
    for (const auto& r : refs) {
        category_of[r.name] = r.category;
        first5[r.name] = store.first_lines(r.category, r.name, 5);
    }
    for (const auto& r : refs) {
        std::string page;
        if (!store.read_wiki(r.category, r.name, page)) continue;
        for (const auto& l : parse_meta(page).links) {
            inbound[l.target]++;
            inbound_categories[l.target].insert(int(r.category));
        }
    }

    std::vector<BreakthroughScore> out;
    std::string hot_add;
    for (const auto& r : refs) {
        if (r.category != WikiCategory::Concepts) continue;

        BreakthroughScore b; b.name = r.name;

        // Vernetzungsgrad: bis 5 Inbound-Links -> voll.
        b.connectivity = 0.30 * clamp01(inbound[r.name] / 5.0);

        // Themen-Diversität: distinkte Quell-Kategorien (max 3).
        b.diversity = 0.25 * clamp01(inbound_categories[r.name].size() / 3.0);

        // Neuheit: 1 - max Ähnlichkeit zu anderen Concepts-Seiten.
        double max_sim = 0.0;
        for (const auto& o : refs) {
            if (o.category != WikiCategory::Concepts || o.name == r.name) continue;
            max_sim = std::max(max_sim, BrainConsolidator::similarity(
                r.name + " " + first5[r.name], o.name + " " + first5[o.name]));
        }
        b.novelty = 0.25 * clamp01(1.0 - max_sim);

        // Reife: distinkte Sessions (max 3).
        std::string page;
        store.read_wiki(r.category, r.name, page);
        const auto m = parse_meta(page);
        b.maturity = 0.20 * clamp01(m.sessions.size() / 3.0);

        b.score = b.connectivity + b.diversity + b.novelty + b.maturity;
        b.notify = b.score > 0.90;
        if (b.score > 0.75) {
            char buf[16]; std::snprintf(buf, sizeof(buf), "%.2f", b.score);
            hot_add += "[BREAKTHROUGH] " + r.name + " (Score " + buf + ")\n";
        }
        out.push_back(b);
    }

    if (!hot_add.empty()) {
        std::string hot; store.read_hot(hot);
        store.write_hot(hot + "\n" + hot_add);
    }
    std::sort(out.begin(), out.end(),
              [](const BreakthroughScore& a, const BreakthroughScore& b) { return a.score > b.score; });
    return out;
}

}  // namespace nova::brain

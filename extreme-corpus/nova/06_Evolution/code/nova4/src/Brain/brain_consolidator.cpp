// brain_consolidator.cpp — Implementierung von brain_consolidator.h (Design §11.2).
#include "Brain/brain_consolidator.h"

#include "Brain/brain_types.h"
#include "Brain/keyword_index.h"  // tokenize
#include "Brain/wiki_meta.h"

#include <algorithm>
#include <set>

namespace nova::brain {

double BrainConsolidator::similarity(const std::string& a, const std::string& b) {
    const auto ta = KeywordIndex::tokenize(a);
    const auto tb = KeywordIndex::tokenize(b);
    if (ta.empty() && tb.empty()) return 1.0;
    std::set<std::string> sa(ta.begin(), ta.end()), sb(tb.begin(), tb.end());
    size_t inter = 0;
    for (const auto& t : sa) if (sb.count(t)) ++inter;
    const size_t uni = sa.size() + sb.size() - inter;
    return uni ? double(inter) / uni : 0.0;
}

ConsolidationResult BrainConsolidator::consolidate(BrainStore& store, const std::string& today,
                                                   const std::string& session_id,
                                                   const std::vector<std::string>& mentioned) const {
    ConsolidationResult r;
    std::set<std::string> ment(mentioned.begin(), mentioned.end());
    const auto refs = store.list_wiki();

    for (const auto& ref : refs) {
        std::string page;
        if (!store.read_wiki(ref.category, ref.name, page)) continue;
        WikiMeta m = parse_meta(page);
        const std::string body = strip_meta(page);
        bool changed = false;

        if (ment.count(ref.name)) {
            m.last_mentioned = today;
            const size_t before = m.sessions.size();
            m.add_session(session_id);
            if (m.sessions.size() != before) changed = true;
            if (m.inactive) { m.inactive = false; ++r.reactivated; changed = true; }
            ++r.updated; changed = true;
        }

        // Inaktiv-Regel: > 30 Tage, nicht evergreen, nicht gerade erwähnt.
        if (!m.evergreen && !ment.count(ref.name) && !m.last_mentioned.empty()) {
            const int age = days_between(m.last_mentioned, today);
            if (age > 30 && !m.inactive) { m.inactive = true; ++r.deactivated; changed = true; }
        }

        if (changed) store.write_wiki(ref.category, ref.name, apply_meta(body, m));
    }

    // --- Duplikat-Prüfung (FuzzyMatch über Name + erste 5 Zeilen) ----------
    std::string dup_report = "# Duplikat-Vorschläge\n\n";
    for (size_t i = 0; i < refs.size(); ++i) {
        const std::string ki = refs[i].name + " " +
            store.first_lines(refs[i].category, refs[i].name, 5);
        for (size_t j = i + 1; j < refs.size(); ++j) {
            const std::string kj = refs[j].name + " " +
                store.first_lines(refs[j].category, refs[j].name, 5);
            if (similarity(ki, kj) >= dup_threshold_) {
                ++r.duplicates;
                dup_report += "- " + refs[i].name + " <-> " + refs[j].name + "\n";
            }
        }
    }
    if (r.duplicates > 0)
        store.write_wiki(WikiCategory::Synthesis, "duplikat_vorschlaege", dup_report);

    return r;
}

}  // namespace nova::brain

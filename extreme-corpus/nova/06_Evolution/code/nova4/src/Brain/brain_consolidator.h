// brain_consolidator.h — Konfidenz-Konsolidierung + Duplikat-Prüfung (Design §11.2 #1,#2).
//
// Kein LLM. Läuft in jedem Brain-Zyklus:
//   1. Konfidenz: erwähnte Seiten -> last_mentioned=heute + Session-ID merken.
//      Seiten > 30 Tage nicht erwähnt und NICHT evergreen -> "# [inaktiv]".
//      Wieder erwähnte Seiten verlieren die Inaktiv-Markierung.
//   2. Duplikate: FuzzyMatch (Token-Jaccard) über Seitenname + erste 5 Zeilen.
//      Kandidaten -> wiki/synthesis/duplikat_vorschlaege.md (manueller Review).
#pragma once

#include <string>
#include <vector>

#include "Brain/brain_store.h"

namespace nova::brain {

struct ConsolidationResult {
    int updated     = 0;  // Seiten mit aktualisiertem last_mentioned
    int deactivated = 0;  // neu als [inaktiv] markiert
    int reactivated = 0;  // Inaktiv-Markierung entfernt
    int duplicates  = 0;  // gefundene Duplikat-Paare
};

class BrainConsolidator {
public:
    explicit BrainConsolidator(double dup_threshold = 0.6) : dup_threshold_(dup_threshold) {}

    // mentioned: in dieser Session erwähnte Seitennamen. today: "YYYY-MM-DD".
    ConsolidationResult consolidate(BrainStore& store, const std::string& today,
                                    const std::string& session_id,
                                    const std::vector<std::string>& mentioned) const;

    // Token-Jaccard-Ähnlichkeit zweier Texte (0..1).
    static double similarity(const std::string& a, const std::string& b);

private:
    double dup_threshold_;
};

}  // namespace nova::brain

// brain_idea_detector.h — Idea-Breakthrough-Score (Design §11.2 #5).
//
// Für jede Seite in wiki/concepts/ wird ein Durchbruch-Score berechnet:
//   Vernetzungsgrad   (max 0.30) — wie viele andere Seiten verlinken hierher
//   Themen-Diversität (max 0.25) — Kategorien der verlinkten Seiten
//   Neuheit           (max 0.25) — Verschiedenheit zu ähnlichen Seiten
//   Reife             (max 0.20) — Erwähnungen in den letzten 30 Sessions
//   > 0.75 -> hot.md als [BREAKTHROUGH]
//   > 0.90 -> Frontend-Benachrichtigung
#pragma once

#include <string>
#include <vector>

#include "Brain/brain_store.h"

namespace nova::brain {

struct BreakthroughScore {
    std::string name;
    double score = 0.0;
    double connectivity = 0.0, diversity = 0.0, novelty = 0.0, maturity = 0.0;
    bool   notify = false;  // > 0.90
};

class IdeaDetector {
public:
    // Berechnet Scores; markiert > 0.75 in hot.md (anhängend) und liefert alle
    // Scores. notify==true für > 0.90 (Frontend-Benachrichtigung).
    std::vector<BreakthroughScore> detect(BrainStore& store) const;

private:
    static double clamp01(double x);
};

}  // namespace nova::brain

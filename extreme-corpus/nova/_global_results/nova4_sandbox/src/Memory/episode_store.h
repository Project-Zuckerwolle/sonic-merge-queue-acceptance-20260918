// episode_store.h — Session -> Episode Summary (Design §10.6).
//
// Ministral 14B fasst jede Session zu 300–500 Wörtern zusammen und erzeugt einen
// kurzen Titel (3–7 Wörter). Datei: memory/episodes/DATUM_UHRZEIT.md mit Titel in
// der ersten Zeile (# TITEL). Die letzten 15 Summaries landen im Context-Stack;
// das Frontend zeigt die Gesprächsliste (Titel + Datum).
//
// Der 14B-Aufruf ist injizierter Callback (brain::LlmFn) -> testbar ohne Modell.
#pragma once

#include <string>
#include <vector>

#include "Brain/brain_types.h"

namespace nova::memory {

struct Episode {
    std::string filename;  // DATUM_UHRZEIT.md
    std::string title;
    std::string summary;
};

class EpisodeStore {
public:
    explicit EpisodeStore(const std::string& episodes_dir) : dir_(episodes_dir) {}

    bool init(std::string* err = nullptr);

    // Erzeugt eine Episode aus dem Working-Memory-Text. timestamp z.B.
    // "20260624_143022". Schreibt die Datei und füllt out.
    bool create(const brain::LlmFn& llm, const std::string& working_memory,
                const std::string& timestamp, Episode& out, std::string* err = nullptr);

    std::vector<Episode> last(int n = 15) const;  // neueste zuerst
    size_t count() const;

    // Trennt LLM-Antwort in Titel (erste Zeile, "TITEL:"-Präfix optional) + Summary.
    static void split_title(const std::string& llm_out, std::string& title, std::string& summary);

private:
    std::string dir_;
};

}  // namespace nova::memory

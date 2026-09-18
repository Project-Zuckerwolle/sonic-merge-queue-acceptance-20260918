// phrase_filter.h — Forbidden-Phrase Post-Processing-Filter (Design §12.3).
//
// String-Match (Mikrosekunden), läuft nach der Generierung. Entfernt
// konfigurierte Floskeln aus dem Ausgabetext und räumt zurückbleibende
// Doppel-Leerzeichen / führende Satzzeichen auf. Liste konfigurierbar
// (Settings-Menü); Standard-Liste aus §12.3.
#pragma once

#include <string>
#include <vector>

namespace nova::web {

class PhraseFilter {
public:
    PhraseFilter();  // lädt Standard-Liste (§12.3)
    explicit PhraseFilter(std::vector<std::string> phrases)
        : phrases_(std::move(phrases)) {}

    void set_phrases(std::vector<std::string> phrases) { phrases_ = std::move(phrases); }
    const std::vector<std::string>& phrases() const { return phrases_; }

    // Entfernt alle Vorkommen (case-insensitiv) und normalisiert Whitespace.
    std::string filter(const std::string& text) const;

    static std::vector<std::string> default_phrases();

private:
    std::vector<std::string> phrases_;
};

}  // namespace nova::web

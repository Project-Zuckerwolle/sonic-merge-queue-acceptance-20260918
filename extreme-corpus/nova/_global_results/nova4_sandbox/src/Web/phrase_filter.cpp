// phrase_filter.cpp — Implementierung von phrase_filter.h (Design §12.3).
#include "Web/phrase_filter.h"

#include <algorithm>
#include <cctype>

namespace nova::web {

std::vector<std::string> PhraseFilter::default_phrases() {
    // §12.3 Standard-Liste.
    return {
        "Natürlich!", "Gerne!", "Absolut!", "Super Frage!", "Großartig!",
        "Sehr gerne!", "Selbstverständlich", "Das ist eine tolle Idee",
    };
}

PhraseFilter::PhraseFilter() : phrases_(default_phrases()) {}

namespace {
// ASCII-case-insensitive Suche (UTF-8-Bytes >= 0x80 bleiben unverändert
// vergleichbar, da Floskeln byte-genau übereinstimmen müssen).
char lower(char c) {
    return (c >= 'A' && c <= 'Z') ? char(c - 'A' + 'a') : c;
}
size_t find_ci(const std::string& hay, const std::string& needle, size_t from) {
    if (needle.empty() || needle.size() > hay.size()) return std::string::npos;
    for (size_t i = from; i + needle.size() <= hay.size(); ++i) {
        size_t j = 0;
        for (; j < needle.size(); ++j)
            if (lower(hay[i + j]) != lower(needle[j])) break;
        if (j == needle.size()) return i;
    }
    return std::string::npos;
}
}  // namespace

std::string PhraseFilter::filter(const std::string& text) const {
    std::string out = text;
    for (const auto& p : phrases_) {
        if (p.empty()) continue;
        size_t pos;
        while ((pos = find_ci(out, p, 0)) != std::string::npos)
            out.erase(pos, p.size());
    }

    // Aufräumen: Mehrfach-Whitespace -> einfaches Leerzeichen, führende
    // Satzzeichen/Spaces je Zeile entfernen, Trim.
    std::string cleaned;
    cleaned.reserve(out.size());
    bool prev_space = false;
    for (char c : out) {
        if (c == '\n') { while (!cleaned.empty() && cleaned.back() == ' ') cleaned.pop_back();
                         cleaned.push_back('\n'); prev_space = false; continue; }
        if (c == ' ' || c == '\t') { if (!prev_space) cleaned.push_back(' '); prev_space = true; }
        else { cleaned.push_back(c); prev_space = false; }
    }
    // Führende Spaces/Satzzeichen am Anfang jeder Zeile trimmen.
    std::string result;
    result.reserve(cleaned.size());
    bool line_start = true;
    for (char c : cleaned) {
        if (line_start) {
            if (c == ' ' || c == ',' || c == '.' || c == '!' || c == ';' || c == ':') continue;
            line_start = false;
        }
        if (c == '\n') line_start = true;
        result.push_back(c);
    }
    // Gesamt-Trim.
    while (!result.empty() && (result.front() == '\n' || result.front() == ' ')) result.erase(result.begin());
    while (!result.empty() && (result.back() == '\n' || result.back() == ' ')) result.pop_back();
    return result;
}

}  // namespace nova::web

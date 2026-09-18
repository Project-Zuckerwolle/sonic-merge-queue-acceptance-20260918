// nano_parser.cpp — Implementierung von nano_parser.h (Design §14).
#include "Web/nano_parser.h"

#include <algorithm>
#include <cctype>

namespace nova::web {

namespace {
std::string lower(const std::string& s) {
    std::string o = s;
    for (auto& c : o) c = char(::tolower((unsigned char)c));
    return o;
}
bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}
}  // namespace

NanoParser::NanoParser() {
    // Apex-Skills (§13.4) — Plan-Trigger.
    add_skill("coding", {"code schreiben", "implementiere", "programmiere", "fix", "refactor", "baue"}, true);
    add_skill("news_research", {"news", "nachrichten", "berichte", "recherchiere"}, true);
    add_skill("deep_research", {"recherchiere ausführlich", "erkläre tiefgehend"}, true);
    add_skill("datei_projekt", {"dateien organisieren", "projekt aufräumen"}, true);

    // Normal-Tools (§13.3) — Tool-Hint.
    add_skill("wetter", {"wetter", "temperatur"}, false);
    add_skill("websearch", {"suche", "was ist", "finde", "google"}, false);
    add_skill("news_rss", {"schlagzeilen", "feed"}, false);
    add_skill("finanzen", {"kurs", "aktie", "börse"}, false);
    add_skill("rechner", {"rechne", "berechne", "wieviel ist"}, false);
    add_skill("uhrzeit", {"uhrzeit", "wie spät", "datum"}, false);
    add_skill("codebase_scan", {"scanne projekt", "projektstruktur"}, false);

    complex_kw_ = {"warum", "vergleiche", "analysiere", "erkläre", "konzept", "architektur",
                   "trade-off", "abwägung", "strategie"};
    todo_kw_    = {"todo", "merk dir", "erinnere mich", "aufgabe", "nicht vergessen",
                   "ich bevorzuge", "ich mag", "lieber"};
}

void NanoParser::add_skill(const std::string& name, const std::vector<std::string>& keywords, bool is_apex) {
    std::vector<std::string> lk;
    lk.reserve(keywords.size());
    for (const auto& k : keywords) lk.push_back(lower(k));
    entries_.push_back({name, std::move(lk), is_apex});
}

NanoResult NanoParser::parse(const std::string& msg) const {
    const std::string m = lower(msg);
    NanoResult r;

    // Apex-Trigger haben Vorrang (längstes Match gewinnt grob über Reihenfolge).
    for (const auto& e : entries_) {
        if (!e.is_apex) continue;
        for (const auto& k : e.keywords)
            if (contains(m, k)) { r.apex_trigger = true; r.apex_skill = e.name; break; }
        if (r.apex_trigger) break;
    }

    // Normal-Tool-Hint (auch wenn Apex erkannt — kann zusätzlich nützlich sein).
    for (const auto& e : entries_) {
        if (e.is_apex) continue;
        for (const auto& k : e.keywords)
            if (contains(m, k)) { r.tool_hint = true; r.tool_name = e.name; break; }
        if (r.tool_hint) break;
    }

    for (const auto& k : complex_kw_) if (contains(m, k)) { r.complex = true; break; }
    for (const auto& k : todo_kw_)    if (contains(m, k)) { r.todo_pref = true; break; }
    return r;
}

}  // namespace nova::web

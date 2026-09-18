// brain_types.h — gemeinsame Typen des Brain-Systems (Design §11).
//
// Der "LLM" (Ministral 14B: raw->wiki, Linking, Episode, Dreaming, Briefing)
// wird als injizierbarer Callback modelliert — so ist die gesamte Brain-Logik
// ohne Modell testbar (gleicher Surrogat-Ansatz wie in Block A/B).
#pragma once

#include <functional>
#include <string>

namespace nova::brain {

// Ministral-14B-Aufruf: Prompt -> Antworttext.
using LlmFn = std::function<std::string(const std::string& prompt)>;

// Datums-Helfer im Format "YYYY-MM-DD" (deterministisch, kein time() im Kern).
int  days_between(const std::string& from, const std::string& to);  // to - from in Tagen
bool valid_date(const std::string& d);

}  // namespace nova::brain

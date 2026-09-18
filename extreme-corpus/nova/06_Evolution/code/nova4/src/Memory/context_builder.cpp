// context_builder.cpp — Implementierung von context_builder.h (Design §10.9).
#include "Memory/context_builder.h"

#include "Memory/working_memory.h"  // estimate_tokens

#include <algorithm>

namespace nova::memory {

ContextResult ContextBuilder::assemble(const ContextStack& s) const {
    std::string out;
    auto section = [&](const char* title, const std::string& body) {
        if (body.empty()) return;
        out += "=== "; out += title; out += " ===\n";
        out += body;
        if (out.back() != '\n') out += '\n';
        out += "\n";
    };

    section("PERSONA", s.persona);
    section("IDENTITY", s.identity);
    section("HOT", s.hot);
    section("DAILY BRIEFING", s.daily_briefing);

    if (!s.episodes.empty()) {
        std::string ep;
        const int n = std::min(int(s.episodes.size()), max_episodes_);
        for (int i = 0; i < n; ++i) { ep += s.episodes[i]; ep += "\n\n"; }
        section("EPISODE SUMMARIES", ep);
    }

    if (!s.wiki.empty()) {
        std::string w;
        const int n = std::min(int(s.wiki.size()), max_wiki_);
        for (int i = 0; i < n; ++i) { w += "- "; w += s.wiki[i]; w += "\n"; }
        section("WIKI (BM25 Top-15)", w);
    }

    if (!s.tool_hint.empty()) section("TOOL-HINT", s.tool_hint);

    section("WORKING MEMORY", s.working_memory);

    ContextResult r;
    r.text = out;
    r.total_tokens = WorkingMemory::estimate_tokens(out);
    return r;
}

}  // namespace nova::memory

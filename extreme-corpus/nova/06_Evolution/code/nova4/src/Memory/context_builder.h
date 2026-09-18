// context_builder.h — 6-Schichten-Context-Stack (Design §10.1, §10.9).
//
// Lens-First: relevanter Kontext ist immer da (§10). Der Stack:
//   1 persona.md            immer
//   2 identity.md           immer (LOCKED + Dreaming)
//   3 hot.md                immer (~500 Wörter)
//   4 daily_briefing.md     brain-kompiliert
//   5 Episode Summaries     letzte 15
//   6 Working Memory        Rolling Window (Session)
//   + BM25 Top-15 Wiki      dynamisch pro User-Message
//
// Ziel-Gesamt ~103.500 Token (§10.9). assemble() fügt die Schichten mit klaren
// Trennern zu einem einzigen Prompt-Kontext zusammen; estimate_tokens() liefert
// die Schätzung. Ein optionaler Tool-Hint (NanoParser, Block E) wird eingebettet.
#pragma once

#include <string>
#include <vector>

namespace nova::memory {

struct ContextStack {
    std::string persona;
    std::string identity;
    std::string hot;
    std::string daily_briefing;
    std::vector<std::string> episodes;   // bis zu 15
    std::string working_memory;
    std::vector<std::string> wiki;       // BM25 Top-15 (gerenderte Snippets)
    std::string tool_hint;               // optional (NanoParser)
};

struct ContextResult {
    std::string text;
    int         total_tokens = 0;
};

class ContextBuilder {
public:
    explicit ContextBuilder(int max_episodes = 15, int max_wiki = 15)
        : max_episodes_(max_episodes), max_wiki_(max_wiki) {}

    ContextResult assemble(const ContextStack& s) const;

private:
    int max_episodes_;
    int max_wiki_;
};

}  // namespace nova::memory

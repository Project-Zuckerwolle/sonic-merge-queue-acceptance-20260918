// working_memory.h — In-Session Rolling Window (Design §10.7, §14).
//
// 90.000 Token Rolling Window. Bei Overflow komprimiert Gemma 3 1B die ältesten
// Turns on-the-fly (§10.7) — hier als injizierbarer Compressor-Callback, damit
// die Logik ohne Modell testbar ist. Ohne Callback werden älteste Turns hart
// gekürzt (Fallback). Token-Schätzung heuristisch (~4 Zeichen/Token).
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace nova::memory {

struct Turn {
    std::string role;  // "user" / "assistant" / "system" / "summary"
    std::string text;
};

// Komprimiert eine Folge alter Turns zu EINEM Summary-Text (Gemma 3 1B).
using CompressorFn = std::function<std::string(const std::vector<Turn>&)>;

class WorkingMemory {
public:
    explicit WorkingMemory(int max_tokens = 90000) : max_tokens_(max_tokens) {}

    void set_compressor(CompressorFn fn) { compress_ = std::move(fn); }

    // Hängt einen Turn an; löst bei Überschreiten des Budgets Overflow aus.
    void add_turn(const std::string& role, const std::string& text);

    int  total_tokens() const { return total_tokens_; }
    size_t turn_count() const { return turns_.size(); }
    const std::vector<Turn>& turns() const { return turns_; }

    std::string text() const;   // zusammengefügter Verlauf für den Context-Stack
    void clear();

    static int estimate_tokens(const std::string& s);  // ~4 Zeichen/Token

private:
    void enforce_budget();

    int max_tokens_;
    int total_tokens_ = 0;
    std::vector<Turn> turns_;
    CompressorFn compress_;
};

}  // namespace nova::memory

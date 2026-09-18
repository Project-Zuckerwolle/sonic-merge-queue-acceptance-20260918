// working_memory.cpp — Implementierung von working_memory.h (Design §10.7).
#include "Memory/working_memory.h"

#include <algorithm>

namespace nova::memory {

int WorkingMemory::estimate_tokens(const std::string& s) {
    // Grobe Heuristik (~4 Zeichen/Token); mind. 1 Token für nicht-leeren Text.
    if (s.empty()) return 0;
    return std::max(1, int((s.size() + 3) / 4));
}

void WorkingMemory::add_turn(const std::string& role, const std::string& text) {
    turns_.push_back({role, text});
    total_tokens_ += estimate_tokens(text);
    enforce_budget();
}

void WorkingMemory::enforce_budget() {
    auto recount = [&] {
        total_tokens_ = 0;
        for (const auto& t : turns_) total_tokens_ += estimate_tokens(t.text);
    };

    // Iterativ die älteste Hälfte zu EINEM Summary komprimieren, bis unter Budget.
    // Konvergiert (Turn-Zahl sinkt je Runde), der jüngste Turn bleibt unberührt.
    while (total_tokens_ > max_tokens_ && turns_.size() > 1) {
        const size_t cut = (turns_.size() + 1) / 2;  // älteste Hälfte (aufgerundet)
        std::vector<Turn> old(turns_.begin(), turns_.begin() + cut);
        std::string summary = compress_ ? compress_(old)
                                        : std::string("[gekürzt: ") + std::to_string(cut) +
                                          " ältere Turns]";

        std::vector<Turn> rest(turns_.begin() + cut, turns_.end());
        turns_.clear();
        turns_.push_back({"summary", summary});
        for (auto& t : rest) turns_.push_back(std::move(t));
        recount();
    }

    // Nur noch ein Turn und immer noch zu groß -> hart kürzen.
    if (total_tokens_ > max_tokens_ && turns_.size() == 1) {
        Turn& s = turns_.front();
        const int allowed_chars = std::max(0, max_tokens_ * 4);
        if (int(s.text.size()) > allowed_chars) s.text.resize(size_t(allowed_chars));
        recount();
    }
}

std::string WorkingMemory::text() const {
    std::string out;
    for (const auto& t : turns_) {
        out += t.role; out += ": "; out += t.text; out += "\n";
    }
    return out;
}

void WorkingMemory::clear() {
    turns_.clear();
    total_tokens_ = 0;
}

}  // namespace nova::memory

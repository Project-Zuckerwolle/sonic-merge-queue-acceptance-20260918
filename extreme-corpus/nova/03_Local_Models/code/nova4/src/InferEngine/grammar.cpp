// grammar.cpp — Implementierung von grammar.h (Aufgabe 5.1).
#include "InferEngine/grammar.h"

namespace nova::infer {

bool apply_mask(Logits& d, const std::vector<char>& mask) {
    double sum = 0.0;
    for (size_t i = 0; i < d.size(); ++i) {
        if (i >= mask.size() || !mask[i]) d[i] = 0.0f;
        else sum += d[i];
    }
    if (sum <= 0.0) return false;
    for (auto& x : d) x = float(x / sum);
    return true;
}

std::string constrained_generate(Grammar& g,
                                 const std::function<Logits(int step)>& logits,
                                 int max_steps) {
    g.reset();
    for (int step = 0; step < max_steps && !g.complete(); ++step) {
        const std::vector<char> mask = g.allowed_mask();
        Logits d = logits ? logits(step) : Logits{};
        if (d.size() < mask.size()) d.resize(mask.size(), 0.0f);

        int choice = -1;
        Logits masked = d;
        if (apply_mask(masked, mask)) {
            float best = -1.0f;
            for (size_t i = 0; i < masked.size(); ++i)
                if (masked[i] > best) { best = masked[i]; choice = int(i); }
        }
        if (choice < 0) {  // Nullmasse auf allen erlaubten -> erstes erlaubtes Token
            for (size_t i = 0; i < mask.size(); ++i)
                if (mask[i]) { choice = int(i); break; }
        }
        if (choice < 0) break;  // kein erlaubtes Token (darf nicht vorkommen)
        g.advance(choice);
    }
    return g.text();
}

// ---------------------------------------------------------------------------
// Action-Grammatik
// ---------------------------------------------------------------------------
namespace {

class ActionGrammar : public Grammar {
public:
    explicit ActionGrammar(const std::vector<std::string>& tool_names) {
        vocab_.push_back("<tool_call name=\"");        // 0
        vocab_.push_back("<task_complete summary=\"");  // 1
        vocab_.push_back("<task_blocked reason=\"");    // 2
        vocab_.push_back("<task_replan new_plan=\"");   // 3
        tool_base_ = int(vocab_.size());                // 4
        for (const auto& n : tool_names) vocab_.push_back(n);
        close_idx_ = int(vocab_.size());                // "/>-close
        vocab_.push_back("\"/>");
        value_idx_ = int(vocab_.size());                // Platzhalter-Wert
        vocab_.push_back("ok");
        reset();
    }

    const std::vector<std::string>& vocab() const override { return vocab_; }

    std::vector<char> allowed_mask() const override {
        std::vector<char> m(vocab_.size(), 0);
        switch (state_) {
            case Start:
                m[0] = m[1] = m[2] = m[3] = 1;
                break;
            case ToolName:
                for (int i = tool_base_; i < close_idx_; ++i) m[i] = 1;
                break;
            case ToolAfterName:
            case AfterValue:
                m[close_idx_] = 1;
                break;
            case Value:
                m[value_idx_] = 1;
                break;
            case Done:
                break;
        }
        return m;
    }

    void advance(int id) override {
        if (id < 0 || id >= int(vocab_.size())) return;
        produced_ += vocab_[size_t(id)];
        switch (state_) {
            case Start:         state_ = (id == 0) ? ToolName : Value; break;
            case ToolName:      state_ = ToolAfterName; break;
            case ToolAfterName: state_ = Done; break;
            case Value:         state_ = AfterValue; break;
            case AfterValue:    state_ = Done; break;
            case Done:          break;
        }
    }

    bool complete() const override { return state_ == Done; }
    void reset() override { produced_.clear(); state_ = Start; }

private:
    enum State { Start, ToolName, ToolAfterName, Value, AfterValue, Done };
    std::vector<std::string> vocab_;
    int tool_base_ = 0, close_idx_ = 0, value_idx_ = 0;
    State state_ = Start;
};

// ---------------------------------------------------------------------------
// Reflexions-JSON-Grammatik (lineare FSM mit festen Feldern)
// ---------------------------------------------------------------------------
class ReflexionGrammar : public Grammar {
public:
    ReflexionGrammar() {
        // Vokabular: feste JSON-Stücke + Wert-Slots.
        vocab_ = {
            "{\"fortschritt_prozent\": ",         // 0
            "50",                                    // 1  NUM
            ", \"was_funktioniert\": \"",          // 2
            "ok",                                    // 3  V
            "\", \"was_fehlt\": \"",               // 4
            "ok",                                    // 5  V
            "\", \"strategie_anpassen\": ",         // 6
            "true",                                  // 7  BOOL
            "false",                                 // 8  BOOL
            ", \"naechster_fokus\": \"",           // 9
            "ok",                                    // 10 V
            "\", \"extrahierte_fakten\": [\"",     // 11
            "fakt",                                  // 12 V
            "\"]}",                                  // 13
        };
        // Erlaubte Token je Schritt (BOOL-Schritt bietet true/false).
        allowed_by_step_ = {
            {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7, 8},
            {9}, {10}, {11}, {12}, {13},
        };
        reset();
    }

    const std::vector<std::string>& vocab() const override { return vocab_; }

    std::vector<char> allowed_mask() const override {
        std::vector<char> m(vocab_.size(), 0);
        if (pos_ < int(allowed_by_step_.size()))
            for (int id : allowed_by_step_[size_t(pos_)]) m[size_t(id)] = 1;
        return m;
    }

    void advance(int id) override {
        if (id < 0 || id >= int(vocab_.size())) return;
        produced_ += vocab_[size_t(id)];
        ++pos_;
    }

    bool complete() const override { return pos_ >= int(allowed_by_step_.size()); }
    void reset() override { produced_.clear(); pos_ = 0; }

private:
    std::vector<std::string> vocab_;
    std::vector<std::vector<int>> allowed_by_step_;
    int pos_ = 0;
};

}  // namespace

std::unique_ptr<Grammar> make_action_grammar(const std::vector<std::string>& tool_names) {
    return std::make_unique<ActionGrammar>(tool_names);
}

std::unique_ptr<Grammar> make_reflexion_grammar() {
    return std::make_unique<ReflexionGrammar>();
}

}  // namespace nova::infer

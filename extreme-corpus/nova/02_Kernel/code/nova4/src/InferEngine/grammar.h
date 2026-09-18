// grammar.h — Grammatik-gebundenes Decodieren (Aufgabe 5.1, TB15).
//
// Ein schwaches Modell bricht sonst das Schema (ungültige Tags, kaputtes JSON,
// erfundene Tool-Namen). Grammatik-gebundenes Decodieren maskiert im Sampling-
// Schritt ungültige Tokens, sodass NUR gültige Struktur entstehen KANN.
//
// Modell: Token-Level-FSM über ein explizites Vokabular von String-Stücken.
// allowed_mask() liefert je Vokabel-Token 1/0 (erlaubt im aktuellen Zustand),
// advance() konsumiert die Wahl, complete() meldet eine vollständige Struktur.
// apply_mask() ist die eigentliche Decoder-Kopplung (auf p- und q-Verteilung
// anwendbar, um die Akzeptanz-Sampling-Exaktheit von spec_decoder zu wahren).
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nova::infer {

using Logits = std::vector<float>;  // nichtnegative Scores über das Vokabular

class Grammar {
public:
    virtual ~Grammar() = default;
    virtual const std::vector<std::string>& vocab() const = 0;
    virtual std::vector<char> allowed_mask() const = 0;  // Größe == vocab, 1=erlaubt
    virtual void advance(int token_id) = 0;
    virtual bool complete() const = 0;
    virtual void reset() = 0;
    const std::string& text() const { return produced_; }

protected:
    std::string produced_;
};

// Verbotene Tokens -> 0, renormalisiert. false wenn die Maske alle Masse tilgt.
bool apply_mask(Logits& dist, const std::vector<char>& mask);

// Constrained Generation: maskiert je Schritt die (beliebig verzerrten) Logits
// und wählt das höchstbewertete ERLAUBTE Token; fällt bei Nullmasse auf das erste
// erlaubte Token zurück. Ergebnis ist garantiert grammatik-gültig.
std::string constrained_generate(Grammar& g,
                                 const std::function<Logits(int step)>& logits,
                                 int max_steps = 256);

// Action-Grammatik (Design §13.6): <tool_call name="T" .../> | <task_complete/>
// | <task_blocked/> | <task_replan/>. Tool-Namen an `tool_names` gebunden —
// ein erfundener Name existiert nicht im Vokabular und ist damit unmöglich.
std::unique_ptr<Grammar> make_action_grammar(const std::vector<std::string>& tool_names);

// Reflexions-JSON-Grammatik (Design §13.6, feste Felder).
std::unique_ptr<Grammar> make_reflexion_grammar();

}  // namespace nova::infer

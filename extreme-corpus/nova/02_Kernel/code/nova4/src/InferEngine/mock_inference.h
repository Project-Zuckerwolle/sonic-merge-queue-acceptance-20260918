// mock_inference.h — Deterministische CPU-Inferenz (Aufgabe 0).
//
// KEIN Sprachmodell: reagiert auf Schlüsselwörter im Request und emittiert eine
// feste, kohärente deutsche Antwort — optional mit einem <tool_call .../>, damit
// der Tool-Pfad end-to-end beweisbar ist. Tokens sind bewusst zerstückelt (Tags
// erstrecken sich über mehrere Tokens), um den inkrementellen StreamInterceptor
// (nova_ws) real zu treiben. Dient als Default-Engine solange keine GPU/Modelle
// vorhanden sind, und als Fixture für App-/Apex-/Memory-Testbeds.
#pragma once

#include "InferEngine/inference.h"

#include <string>
#include <vector>

namespace nova::infer {

class MockInference : public IInference {
public:
    void begin(const GenRequest& req) override;
    std::string next_token() override;

    // Baut die Token-Liste aus dem Nutzertext (öffentlich für Tests).
    static std::vector<std::string> build_response(const std::string& user_text);
    // Extrahiert einen arithmetischen Ausdruck ("2+3*4") oder "" wenn keiner.
    static std::string extract_expr(const std::string& text);

private:
    std::vector<std::string> tokens_;
    size_t idx_ = 0;
};

}  // namespace nova::infer

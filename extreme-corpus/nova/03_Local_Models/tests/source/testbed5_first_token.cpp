// testbed5_first_token.cpp — Erster Token / Generierungs-Pipeline (Aufgabe 2, TB5).
//
// [DEV-PC] Beweist den Generierungs-/Streaming-Pfad END-TO-END über die
// IInference-Seam mit MockInference: Prompt -> kohärenter Token-Stream -> Stop,
// deterministisch reproduzierbar. [SERVER] tauscht MockInference gegen
// GpuInference (echtes Mistral 24B, chunk-gestreamt) — gleiche Seam.
#include "InferEngine/mock_inference.h"

#include <cstdio>
#include <iostream>
#include <string>

using namespace nova;

int main() {
    std::cout << "=== Testbed 5: Erster Token (Mock-Pipeline) ===\n";

    infer::MockInference eng;
    infer::GenRequest req;
    req.prefix = "Du bist Nova 4. Sprache: Deutsch.";
    req.dynamic = "Hallo, sage ein Wort.";

    eng.begin(req);
    std::string out;
    int toks = 0;
    for (;;) {
        const std::string t = eng.next_token();
        if (t.empty()) break;
        out += t;
        ++toks;
    }
    std::printf("  Stream (%d Token): %s\n", toks, out.c_str());

    // complete() muss deterministisch dasselbe liefern.
    const std::string out2 = eng.complete(req);

    bool pass = true;
    pass &= (toks >= 3);                                  // mehrere Token gestreamt
    pass &= (!out.empty());
    pass &= (out.find("Nova") != std::string::npos);      // kohärente Antwort
    pass &= (out2 == out);                                // deterministisch/reproduzierbar

    std::cout << "\n=== Testbed 5: " << (pass ? "BESTANDEN" : "FEHLGESCHLAGEN") << " ===\n";
    return pass ? 0 : 1;
}

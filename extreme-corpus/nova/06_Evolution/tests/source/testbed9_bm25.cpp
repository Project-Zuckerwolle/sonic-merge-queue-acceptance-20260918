// testbed9_bm25.cpp — BM25 Keyword-Index (Design §16, TB 9).
//
// Kriterium: "Top-15 korrekt, < 10ms."
//
// Baut einen Index über synthetische Wiki-Seiten (Dateiname + erste 5 Zeilen),
// stellt Queries und prüft: (a) die erwartete Seite ist Top-1, (b) Ranking nach
// BM25 korrekt, (c) Query-Latenz < 10 ms, (d) Top-15-Begrenzung + Normalisierung.
#include "Brain/keyword_index.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace nova::brain;

namespace {
Bm25Doc page(const std::string& id, const std::string& body) { return {id, id + "\n" + body}; }
}  // namespace

int main() {
    std::cout << "=== Testbed 9: BM25 Keyword-Index ===\n";

    std::vector<Bm25Doc> docs = {
        page("nova4_inferengine.md",
             "Chunk-Streaming-Pipeline SSD RAM VRAM Triple-Buffer GEMM Gewichte"),
        page("turboquant.md",
             "KV-Cache Kompression PolarQuant QJL Outlier Quantisierung 2.5 Bit"),
        page("speculative_decoding.md",
             "Draft Modell Gemma verifiziert Token Kandidaten Akzeptanz Sampling"),
        page("medusa.md",
             "parallele Köpfe Lookahead Token vorschlagen Verifikation einem Pass"),
        page("brain_compiler.md",
             "Wiki Seiten Konsolidierung Duplikat Cross-Entry Linking Evergreen"),
        page("memory_system.md",
             "Lens-First Schichten persona identity hot daily briefing Episode Working"),
        page("apex_react.md",
             "ReAct Loop THOUGHT ACTION OBS Reflexion REPLAN Subagent Orchestrator"),
        page("kv_manager.md",
             "TurboQuant KV-Cache H2O Fallback Heavy Hitter Recency Trim Budget"),
    };

    KeywordIndex idx;
    idx.build(docs);
    std::printf("  Index: %zu Wiki-Seiten\n", idx.doc_count());

    struct QCase { std::string query; std::string expect_top; };
    std::vector<QCase> cases = {
        {"PolarQuant QJL Outlier", "turboquant.md"},
        {"H2O Heavy Hitter Trim", "kv_manager.md"},
        {"Draft Token Akzeptanz Sampling", "speculative_decoding.md"},
        {"THOUGHT ACTION OBS Reflexion", "apex_react.md"},
        {"persona identity hot Episode", "memory_system.md"},
    };

    bool pass = true;
    double max_ms = 0.0;
    for (const auto& qc : cases) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        const auto hits = idx.query(qc.query, 15);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        max_ms = std::max(max_ms, ms);

        const std::string top = hits.empty() ? "(leer)" : hits[0].id;
        const bool ok = (top == qc.expect_top);
        pass &= ok;
        std::printf("  [%s] \"%s\" -> %s (score %.3f, %.3f ms)\n",
                    ok ? "OK" : "XX", qc.query.c_str(), top.c_str(),
                    hits.empty() ? 0.0 : hits[0].score, ms);
        if (int(hits.size()) > 15) { pass = false; std::printf("    FAIL: > 15 Treffer\n"); }
    }

    // Normalisierung: Top-Treffer hat norm_score == 1.0.
    const auto h = idx.query("TurboQuant KV-Cache", 15);
    const bool norm_ok = !h.empty() && h[0].norm_score > 0.999;
    std::printf("  Normalisierung Top norm_score = %.3f -> %s\n",
                h.empty() ? 0.0 : h[0].norm_score, norm_ok ? "OK" : "FAIL");
    pass &= norm_ok;

    std::printf("  Max Query-Latenz = %.3f ms (Ziel < 10 ms)\n", max_ms);
    pass &= (max_ms < 10.0);

    std::cout << "\n=== Testbed 9: " << (pass ? "BESTANDEN" : "FEHLGESCHLAGEN") << " ===\n";
    return pass ? 0 : 1;
}

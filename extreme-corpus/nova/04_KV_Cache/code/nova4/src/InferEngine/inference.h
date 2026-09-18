// inference.h — Streaming-Inferenz-Seam (Aufgabe 0, Design §6/§8/§14).
//
// Ein einziges Interface, hinter dem entweder das echte GPU-Streaming
// (GpuInference: chunk_streamer -> fused GEMM -> spec_decoder -> kv_manager,
// nur unter NOVA_HAVE_CUDA + Modellen) oder die deterministische CPU-Mock
// (MockInference) steckt. So ist die gesamte App-, Tool- und Apex-Verdrahtung
// ohne GPU beweisbar; auf dem Server wird nur die Impl getauscht.
//
// Der Kontext ist in prefix/dynamic geteilt: ein echtes Engine cached den KV
// des stabilen System-Prefix (persona/identity/hot/daily) über begin()-Aufrufe
// und prefillt nur den wechselnden Suffix neu (KV bounded-per-turn + Prefix-Cache).
#pragma once

#include <functional>
#include <string>

namespace nova::infer {

// Vollständig assemblierter Generierungs-Request eines Turns.
struct GenRequest {
    std::string prefix;   // stabil: persona/identity/hot/daily_briefing (cachebar)
    std::string dynamic;  // wechselnd: Working Memory + BM25-Wiki + aktueller Turn
    bool instruct = false; // App-Chat: Prompt in <s>[INST]…[/INST] hüllen (Mistral-Instruct).
                           // Default false -> Roh-Completion (c1_measure/Benchmark unverändert).
};

// Streaming-Inferenz. next_token() liefert Anzeigetext; "" == Turn-Ende (passt
// exakt auf web::ChatHandler::ModelFn). complete() fährt einen Turn zu Ende.
class IInference {
public:
    virtual ~IInference() = default;

    // Startet einen neuen Turn. Verwirft Per-Turn-State; ein echtes Engine hält
    // den Prefix-KV über begin()-Aufrufe, solange req.prefix unverändert ist.
    virtual void begin(const GenRequest& req) = 0;

    // Nächstes Ausgabe-Token als Text. Leerer String == Turn-Ende.
    virtual std::string next_token() = 0;

    // Nicht-streamende Vollgenerierung (Brain/Apex-LLM-Callbacks): treibt per
    // Default begin()+next_token(); ein echtes Engine darf spezialisieren.
    virtual std::string complete(const GenRequest& req);
};

// Adapter: Engine -> Token-Quelle (web::ChatHandler::ModelFn-kompatibel).
// begin() muss für den aktuellen Turn bereits aufgerufen sein.
std::function<std::string()> as_token_source(IInference& eng);

// Adapter: Engine -> LLM-Callback (prompt -> Volltext) für Brain/Apex, mit
// `prefix` als stabilem System-Header für jeden Aufruf.
std::function<std::string(const std::string&)> as_llm_fn(IInference& eng,
                                                         std::string prefix = {});

}  // namespace nova::infer

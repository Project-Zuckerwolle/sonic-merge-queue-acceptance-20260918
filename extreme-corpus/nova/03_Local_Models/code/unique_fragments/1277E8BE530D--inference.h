// inference.h — die Naht zwischen App und Modell.
//
// Gegenüber Nova 4 (inference.h:29-42) erweitert um genau das, was das neue
// Kontextmodell braucht und was vorher fehlte:
//
//   - append(): der Kontext wächst, statt bei jedem Turn neu gebaut zu werden.
//     Nova 4 baute in ContextService::build den kompletten Prompt jedes Mal neu
//     (context_service.cpp:74-131) und setzte req.prefix = "" (:128); der
//     Prefix-Cache griff dadurch auf ~800 von 90.000 Token. Ein voller
//     90k-Prefill kostet ~2,6e15 FLOP, also ~100 s auf der 7900 XT. Das ist
//     nicht optimierbar, es darf nur nicht pro Turn anfallen.
//   - drop_range(): Kontext-Hygiene. Alte Werkzeug-Ergebnisse fliegen raus,
//     ohne dass alles dahinter neu prefillt werden muss (siehe Kommentar unten).
//   - park()/unpark(): Gewichte VRAM <-> RAM, damit die Karte frei wird.
//   - grammar: erzwungene Struktur beim Decodieren. In Nova 4 vollständig
//     implementiert (grammar.cpp) und NIE aufgerufen — LlmFn hatte nicht mal
//     einen Parameter dafür.
//   - stats(): tok/s, Kontextfüllung, VRAM. Nova 4 hatte cudaMemGetInfo in
//     src/ überhaupt nicht, nur in tests/.
#pragma once

#include "Core/nova_core.h"

#include <memory>
#include <string>
#include <vector>

namespace nova::infer {

// ---------------------------------------------------------------------------
// Kontext-Segmente
// ---------------------------------------------------------------------------
// Der Kontext ist eine Folge von Segmenten mit bekannter Token-Position. Das
// ist die Voraussetzung dafür, dass die Engine weiß, ab wo sie neu prefillen
// muss, wenn etwas in der Mitte entfernt oder ersetzt wird.
enum class SegKind {
    System,     // persona + identity + Kernfakten + Brain-Inhaltsverzeichnis
    User,
    Assistant,
    ToolResult, // darf später verworfen werden (Hygiene)
    Summary     // Ergebnis einer Verdichtung
};

struct Segment {
    SegKind     kind = SegKind::User;
    std::string text;
    int         tokens = 0;       // von der Engine gefüllt
    int64_t     id = 0;           // stabile Kennung für drop_range/replace
    bool        pinned = false;   // nie verwerfen (System, LOCKED)
};

struct EngineStats {
    double  tokens_per_s = 0.0;
    int     context_tokens = 0;
    int     context_max = 0;
    // Bytes, die EINE Kontextposition im KV-Cache wirklich belegt — aus der
    // Geometrie des geladenen Modells, nicht geschätzt. Damit ist der Regler
    // (HipEngineConfig::kv_budget_bytes) in der Oberfläche beschriftbar:
    // "90.000 Token = X GiB". 0, solange kein Modell geladen ist.
    size_t  kv_bytes_pro_token = 0;
    int     prefill_pending = 0;   // wie viele Token beim nächsten Turn neu müssen
    size_t  vram_used = 0;
    size_t  vram_total = 0;
    bool    parked = false;
    double  last_prefill_ms = 0.0;
    double  last_decode_ms = 0.0;
};

// ---------------------------------------------------------------------------
// Grammatik
// ---------------------------------------------------------------------------
// Ein Trie über echte Modell-Token (Tekken, vocab 131072). Nova 4s
// ActionGrammar lief über ~8 String-Stücke statt über das Vokabular — sie
// hätte auch dann nicht funktioniert, wenn sie aufgerufen worden wäre.
class ITokenTrie;

struct GrammarSpec {
    // Erlaubte Fortsetzungen ab dem Moment, in dem `<tool name="` gesehen wurde.
    std::vector<std::string> tool_names;
    // Pro Werkzeug die erlaubten Parameternamen.
    std::vector<std::pair<std::string, std::vector<std::string>>> params_by_tool;
    // Freier Inhalt wird nie maskiert; das Ende bestimmt der Byte-Zähler.
    bool enforce = true;

    // --- DIE ERZWUNGENE ERÖFFNUNG, 31.07.2026 -----------------------------
    // WAS HIER VORHER FEHLTE UND WARUM DAS DER TEURE FEHLER WAR.
    //
    // `enforce = true` hiess bis heute nur: WENN das Modell `<tool name="`
    // tippt, sind danach nur Registry-Namen erlaubt. Der `TokenTrie` startet
    // aber in `Mode::Free` (token_trie.cpp:343), und `scan_triggers` feuert
    // erst, wenn der Ausloeser bereits im Strom steht. OB ueberhaupt ein
    // Aufruf beginnt, entschied damit allein das Modell.
    //
    // GEMESSEN ueber vier Agentenlaeufe mit derselben Aufgabe
    // (C:\nova\lauf-01-abbruch\fortschritt.tsv, C:\nova\lauf-02-nulllinie.tsv,
    // lauf-03-p1messung.tsv, lauf-04-formatfix.tsv): 7 Werkzeugstarts in
    // zusammen 21 Iterationen, dagegen 64x `denken.aufgegeben` und
    // 14x `react.keine_aktion`. Das Modell schrieb Markdown (Median 2231 Token
    // je Ausgabe) und fing nie an. Die Grammatik war ein Formatwaechter ohne
    // Startbefehl.
    //
    // JETZT: ist `eroeffnung` gesetzt, MUSS das allererste Token eine dieser
    // Zeichenketten fortsetzen. Danach laeuft alles wie bisher — die fertige
    // Eroeffnung steht im `tail_`, und `scan_triggers` uebernimmt.
    // Leer = altes Verhalten (Freiphase, Chat, Blickwinkel).
    //
    // Mehrere Alternativen sind Absicht: der Werkzeugschritt muss auch
    // `<task_complete>` erreichen koennen, sonst waere der Agent in die
    // Werkzeugschleife eingemauert.
    std::vector<std::string> eroeffnung;
};

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------
struct SampleParams {
    float temperature = 0.0f;   // 0 = argmax
    float top_p = 1.0f;
    int   top_k = 0;
    // Wiederholungsbremse. Nova 4 samplete reinen Argmax
    // (real_inference.cpp:1190) ohne jede Schleifenerkennung — bei einem 24B
    // im Agentenmodus die häufigste Art, eine Aufgabe zu verlieren.
    float repeat_penalty = 1.05f;
    int   repeat_window = 256;
    int   max_new_tokens = 4096;
};

class IEngine {
public:
    virtual ~IEngine() = default;

    // --- Kontext ---------------------------------------------------------
    // Setzt den System-Kopf. Ändert er sich, wird alles neu prefillt — deshalb
    // gehört dort nur hinein, was sich selten ändert.
    virtual Status set_system(const std::string& text) = 0;

    // Hängt ein Segment an. Kostet nur seinen eigenen Prefill.
    virtual Result<int64_t> append(const Segment& seg) = 0;

    // Entfernt Segmente. Die Engine prefillt ab der frühesten Lücke neu und
    // meldet über stats().prefill_pending, was das kostet — der Aufrufer kann
    // also entscheiden, ob sich die Hygiene lohnt.
    virtual Status drop(const std::vector<int64_t>& segment_ids) = 0;

    // Ersetzt eine Folge von Segmenten durch eines (Verdichtung).
    virtual Result<int64_t> replace(const std::vector<int64_t>& ids, const Segment& with) = 0;

    virtual std::vector<Segment> segments() const = 0;
    virtual Status               clear_context() = 0;

    // --- Generierung -----------------------------------------------------
    virtual Status      begin(const SampleParams& sp, const GrammarSpec* g = nullptr) = 0;
    virtual std::string next_token() = 0;   // "" == Ende
    virtual void        cancel() = 0;       // aus einem anderen Thread erlaubt

    // Vollgenerierung für Hintergrundaufgaben (Brain, Verdichtung).
    // WICHTIG: läuft über dieselbe Engine, also serialisiert. Der Scheduler
    // (System/scheduler.h) entscheidet, wann so etwas laufen darf — nicht der
    // Aufrufer. In Nova 4 nahm der Brain-Thread einfach das Generierungs-Mutex
    // (nova_main.cpp:310) und blockierte damit den Chat für Minuten.
    virtual Result<std::string> complete(const std::string& prompt,
                                         const SampleParams& sp) = 0;

    // --- Zustand ---------------------------------------------------------
    virtual EngineStats stats() const = 0;

    // Gemittelter Hidden-State des letzten Layers über einen Positionsbereich.
    //
    // WOZU: die Frage "ist das noch dasselbe Thema" muss sprachunabhängig
    // beantwortet werden. Über Wortmengen geht das schlecht — dieselbe Sache
    // mit anderen Wörtern sieht wie ein Wechsel aus, und ein Sprachwechsel
    // zerstört den Vergleich vollständig. Über den Modellzustand geht es, und
    // zwar UMSONST: der Hidden-State ist beim Forward-Pass ohnehin entstanden,
    // hier wird nur noch über die Positionen gemittelt und normiert.
    //
    // Das ist ausdrücklich KEIN trainierter Satz-Encoder und soll keiner sein.
    // Für "gleiches Thema innerhalb einer Sitzung" reicht ein gemittelter
    // Decoder-Zustand; für eine Suche über Tausende von Dokumenten reichte er
    // nicht — dafür gibt es BM25 im Brain. Die Trennung ist Absicht.
    //
    // Leerer Vektor => die Engine hält keine Zustände (Mock). Der Aufrufer
    // muss diesen Fall behandeln und darf NICHT stillschweigend auf einen
    // schlechteren Vergleich ausweichen, ohne es zu melden.
    //
    // STAND IM BAUM: NICHT GEBAUT — WEDER IN DER HIP- NOCH IN DER MOCK-ENGINE.
    // Beide liefern den leeren Vektor; die HipEngine sagt es zusätzlich einmal
    // im Log. Der Grund steht bei HipEngine::pooled_state: der Hidden-State
    // liegt in den Aktivierungspuffern eines CHUNKS und wird vom nächsten Chunk
    // überschrieben, es gibt keinen Puffer je Segment und keinen Reduktions-
    // kernel dafür. Alles, was oben über die sprachunabhängige Themenerkennung
    // steht, beschreibt also eine ABSICHT und keinen vorhandenen Pfad. Wer sich
    // darauf stützt (Memory/substanz.h:194-207), bekommt heute den
    // Mengenvergleich — und muss das melden.
    virtual std::vector<float> pooled_state(int64_t segment_id) const = 0;

    // Gewichte VRAM -> pinned RAM (park) und zurück. Blockweise, weil das
    // Windows-Lock-Limit (~5-6 GB) das Pinnen am Stück verweigert — steht so
    // im Nova-4-Perf-Log und ist der Grund, warum park() eine Blockgröße hat.
    virtual Status park() = 0;
    virtual Status unpark() = 0;
    virtual bool   parked() const = 0;

    // KV-Snapshot des System-Kopfs auf Platte. Spart den einmaligen Prefill
    // beim Sitzungsstart, solange sich der Kopf nicht geändert hat.
    virtual Status save_system_kv(const std::string& path) = 0;
    virtual Status load_system_kv(const std::string& path, const std::string& expect_hash) = 0;
};

// ---------------------------------------------------------------------------
// Implementierungen
// ---------------------------------------------------------------------------
// HipEngine  — die echte: INT3 resident, HIP, gfx1100.
// MockEngine — deterministisch, ohne GPU. Trägt die gesamte App-, Werkzeug- und
//              Agenten-Verdrahtung im Test. Das ist der Grund, warum Brain,
//              Memory und Agent hier ohne Karte prüfbar sind.
std::unique_ptr<IEngine> make_mock_engine();

struct HipEngineConfig {
    std::string model_path;      // .nv3w
    std::string tokenizer_path;  // tekken
    int         max_context = 90000;
    int         kv_bits = 3;
    size_t      park_block_bytes = 512ull << 20;   // 512 MB, Windows-Lock-Limit

    // DER KV-REGLER. > 0: `max_context` wird VERWORFEN und aus diesem
    // VRAM-Budget gerechnet.
    //
    // WARUM HIER UND NICHT IN DER APP: die Bytes je Position hängen an der
    // Modellgeometrie (n_layers, n_kv_heads, head_dim) und an kv_bits. Diese
    // Zahlen kennt erst die Engine, nachdem sie den Artefaktkopf gelesen hat.
    // Jede Rechnung ausserhalb wäre eine zweite Quelle — und genau darüber sagt
    // Core/nova_metrics.h, dass die 33,3-KiB-Rechnung, auf der das Budget stand,
    // Padding und den Puffer-Nachlauf nicht kannte.
    size_t kv_budget_bytes = 0;
    // Untergrenze der Ableitung: ein Budget, das nur für 2.000 Token reicht,
    // macht die Anwendung unbenutzbar — dann ist die Einstellung falsch und
    // soll auffallen, nicht stillschweigend gelten.
    int kv_budget_min_context = 8192;
};
Result<std::unique_ptr<IEngine>> make_hip_engine(const HipEngineConfig& cfg);

}  // namespace nova::infer

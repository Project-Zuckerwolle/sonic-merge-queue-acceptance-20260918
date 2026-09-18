// dreaming.cpp — Implementierung von dreaming.h.
#include "Memory/dreaming.h"

#include "Memory/system_head.h"   // hash_of — EINE Hash-Funktion im Segment

#include <atomic>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

namespace nova::memory {

namespace {

std::string trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    const size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Ein Prozess, ein Exit-Handler. std::atexit nimmt nur einen Funktionszeiger
// ohne Argument, also braucht es den Umweg über eine Prozessvariable.
std::atomic<Dreaming*>& exit_target() {
    static std::atomic<Dreaming*> t{nullptr};
    return t;
}

void run_exit_flush() {
    Dreaming* d = exit_target().load();
    if (!d) return;
    // Fehler beim Exit sind nicht mehr behandelbar, aber sie dürfen den Prozess
    // nicht mit einer Ausnahme aus einem atexit-Handler beenden.
    Status s = d->flush(DreamTrigger::ProcessExit);
    s.discard("Prozess-Exit: Ergebnis nicht mehr auswertbar, gesicherte Sitzung bleibt liegen");
}

// Schwanz behalten: das Ende einer Sitzung trägt Entscheidungen und offene
// Punkte, der Anfang ist bereits verdichtet (substanz.h, Hochwasser-Zug).
std::string tail_to_budget(const std::string& text, int budget_tokens) {
    if (count_tokens(text) <= budget_tokens) return text;
    // Näherung über Bytes: der Tokenizer ist hier nicht zeichenweise abfragbar,
    // und eine grobe Untergrenze ist besser als ein Prompt, der die Engine
    // minutenlang blockiert.
    const int tok = count_tokens(text);
    if (tok <= 0) return text;
    const double keep_ratio = double(budget_tokens) / double(tok);
    size_t keep = size_t(double(text.size()) * keep_ratio);
    if (keep >= text.size()) return text;
    size_t start = text.size() - keep;
    const size_t nl = text.find('\n', start);
    if (nl != std::string::npos) start = nl + 1;   // nie mitten in einer Zeile beginnen
    return "[Anfang der Sitzung gekürzt]\n" + text.substr(start);
}

}  // namespace

const char* dream_trigger_name(DreamTrigger t) {
    switch (t) {
        case DreamTrigger::SessionEnd:  return "Sitzungsende";
        case DreamTrigger::Idle:        return "Idle";
        case DreamTrigger::ProcessExit: return "Prozess-Exit";
        case DreamTrigger::Recovered:   return "Nachholung nach unsauberem Ende";
    }
    return "?";
}

Dreaming::Dreaming(IdentityStore& identity, LlmFn llm, DreamingConfig cfg)
    : identity_(identity), llm_(std::move(llm)), cfg_(cfg),
      pending_path_(Paths::join(Paths::get().memory(), "pending_session.md")) {}

Dreaming::~Dreaming() {
    Dreaming* self = this;
    exit_target().compare_exchange_strong(self, nullptr);
}

void Dreaming::set_event_sink(EventSink sink) {
    st_.with([&](State& s) { s.sink = std::move(sink); });
}

void Dreaming::emit(EventKind k, std::string a, std::string b) const {
    EventSink s = st_.with([](const State& st) { return st.sink; });
    if (s) s(Event{k, std::move(a), std::move(b), 0});
}

Status Dreaming::stage(const std::string& session_text) {
    if (trim(session_text).empty()) return ok();
    // Atomar: eine halb geschriebene Sicherung wäre schlimmer als keine, weil
    // der nächste Start sie für eine vollständige Sitzung hielte.
    return write_atomic(pending_path_, session_text);
}

Status Dreaming::run(const std::string& session_text, DreamTrigger trigger) {
    const std::string text = trim(session_text);
    if (text.empty()) return ok();
    if (!llm_) {
        emit(EventKind::Error, "memory.dreaming.no_llm",
             "Kein Hintergrund-LLM — Sitzung bleibt gesichert und wird nachgeholt.");
        return Status(err("memory.dreaming.no_llm", "Kein Hintergrund-LLM.", NOVA_HERE));
    }

    const std::string hash = SystemHead::hash_of(text);

    // Betreten: nur einer gleichzeitig, und dieselbe Sitzung nicht zweimal.
    // In Nova 4 gab es beides nicht — der WS-Close-Pfad konnte parallel zu einem
    // zweiten Client laufen und schrieb ungeschützt in dieselbe Instanz
    // (chat_service.cpp:331).
    const bool enter = st_.with([&](State& s) {
        if (s.busy) return false;
        if (s.last_hash == hash) return false;
        s.busy = true;
        return true;
    });
    if (!enter) return ok();

    const std::optional<std::string> prev = identity_.find(IdentityStore::kDreaming);
    const std::string prompt =
        "Aktualisiere den Dreaming-Block. Extrahiere dauerhafte Fakten, Präferenzen und "
        "Projektstände. Übernimm den bisherigen Block und ergänze ihn — lösche nichts, "
        "was weiterhin gilt. Wenn etwas dem bisherigen Stand widerspricht, schreibe beides "
        "mit Datum hin, statt zu überschreiben. Antworte NUR mit dem Blockinhalt, ohne "
        "Code-Fences.\n\n"
        "--- BISHER ---\n" + (prev ? *prev : std::string()) +
        "\n\n--- SESSION ---\n" + tail_to_budget(text, cfg_.max_session_tokens);

    Result<std::string> resp = llm_(prompt);

    if (!resp) {
        st_.with([](State& s) { s.busy = false; });
        emit(EventKind::Error, "memory.dreaming.llm", resp.error().message);
        return Status(resp.error());
    }

    Status w = identity_.write_dreaming(resp.value());
    st_.with([&](State& s) {
        s.busy = false;
        if (w) s.last_hash = hash;
    });
    if (!w) return w;   // write_dreaming hat bereits ein Event gesendet

    // Erst NACH erfolgreichem Schreiben die Sicherung wegräumen. Andersherum
    // wäre ein Fehler dazwischen der Verlust der Sitzung.
    std::error_code ec;
    fs::remove(pending_path_, ec);

    emit(EventKind::MemoryWrite, "dreaming",
         std::string("Dreaming aktualisiert (") + dream_trigger_name(trigger) + ").");
    return ok();
}

Status Dreaming::run_session_end(const std::string& session_text) {
    if (Status s = stage(session_text); !s) return s;
    return run(session_text, DreamTrigger::SessionEnd);
}

Status Dreaming::maybe_run_idle(const std::string& session_text, int64_t now_ms) {
    const bool due = st_.with([&](State& s) {
        if (s.busy) return false;
        if (s.last_run_ms != 0 && now_ms - s.last_run_ms < cfg_.idle_interval_ms) return false;
        s.last_run_ms = now_ms;
        return true;
    });
    if (!due) return ok();
    if (Status s = stage(session_text); !s) return s;
    return run(session_text, DreamTrigger::Idle);
}

Status Dreaming::flush(DreamTrigger trigger) {
    if (!file_exists(pending_path_)) return ok();
    auto r = read_file(pending_path_);
    if (!r) {
        emit(EventKind::Error, "memory.dreaming.pending", r.error().message);
        return Status(r.error());
    }
    return run(r.value(), trigger);
}

Status Dreaming::recover_pending() {
    if (!file_exists(pending_path_)) return ok();
    emit(EventKind::MemoryWrite, "dreaming.nachholung",
         "Eine Sitzung wurde beim letzten Lauf nicht mehr verarbeitet und wird nachgeholt.");
    return flush(DreamTrigger::Recovered);
}

Status Dreaming::install_exit_flush(Dreaming* d) {
    Dreaming* expected = nullptr;
    if (!exit_target().compare_exchange_strong(expected, d))
        return Status(err("memory.dreaming.exit_handler",
                          "Es ist bereits ein Dreaming-Exit-Handler registriert.", NOVA_HERE));
    if (std::atexit(&run_exit_flush) != 0) {
        exit_target().store(nullptr);
        return Status(err("memory.dreaming.atexit",
                          "atexit-Handler konnte nicht registriert werden.", NOVA_HERE));
    }
    return ok();
}

}  // namespace nova::memory

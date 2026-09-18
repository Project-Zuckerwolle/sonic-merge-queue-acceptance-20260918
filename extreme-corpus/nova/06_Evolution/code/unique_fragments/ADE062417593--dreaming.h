// dreaming.h — Nova verarbeitet die Sitzung und schreibt in `# Dreaming`.
//
// Das Verfahren aus Nova 4 bleibt: inkrementell, der bisherige Block geht als
// `--- BISHER ---` mit in den Prompt, die Sitzung als `--- SESSION ---`
// (dreaming.cpp:10-18), und geschrieben wird ausschließlich über
// IdentityStore::write_dreaming, damit `# LOCKED…` unberührt bleibt.
//
// WAS HIER ANDERS IST — DER AUSLÖSER.
// Nova 4 hat Dreaming und Episode am WebSocket-Close gestartet
// (chat_service.cpp:316-336): der Code läuft NACH der `while (c.recv_text(msg))`-
// Schleife, also nur, wenn die Verbindung sauber endet. Konkret bedeutet das:
//
//   - Absturz des Prozesses      -> Sitzung komplett verloren
//   - Browser-Tab hart geschlossen, WLAN weg, Tunnel-Abbruch -> verloren
//   - Hibernate nach Idle (§5)   -> verloren
//   - Nutzer lässt den Tab tagelang offen -> passiert nie
//
// Es gibt keine Rettung: der Verlauf lebte nur im RAM (WorkingMemory), und die
// einzige Stelle, die ihn verewigt, war genau die, die nicht lief.
//
// Hier drei Auslöser plus eine Sicherung:
//   (a) Sitzungsende
//   (b) periodisch bei Idle
//   (c) Prozess-Exit über einen Flush-Handler (std::atexit)
//   +   stage(): der Sitzungstext wird nach jedem Turn ATOMAR auf Platte
//       gesichert. Selbst wenn alle drei Auslöser ausfallen — Stromausfall,
//       SIGKILL —, findet der nächste Start die Sitzung und arbeitet sie nach.
//       Ohne diese Sicherung ist jeder Auslöser nur eine Wette darauf, dass der
//       Prozess ordentlich endet.
#pragma once

#include "Core/nova_core.h"
#include "Memory/episode_store.h"    // LlmFn
#include "Memory/identity_store.h"

#include <cstdint>
#include <string>

namespace nova::memory {

enum class DreamTrigger { SessionEnd, Idle, ProcessExit, Recovered };

const char* dream_trigger_name(DreamTrigger t);

struct DreamingConfig {
    // Wie oft im Idle. 15 Minuten: oft genug, dass ein Absturz wenig kostet,
    // selten genug, dass die gestreamte Engine nicht ständig belegt ist.
    int64_t idle_interval_ms = 15 * 60 * 1000;
    // Obergrenze für die Eingabe. Ein 90k-Verlauf durch ein Modell mit ~35 tok/s
    // zu schieben dauert Minuten; der Anfang der Sitzung ist ohnehin schon
    // verdichtet (substanz.h, Hochwasser-Zug).
    int max_session_tokens = 20000;
};

class Dreaming {
public:
    Dreaming(IdentityStore& identity, LlmFn llm, DreamingConfig cfg = DreamingConfig());

    // Meldet einen registrierten Exit-Handler wieder ab. Ohne das ruft der
    // atexit-Handler in ein zerstörtes Objekt — ein Absturz beim Beenden, also
    // genau dort, wo niemand mehr hinschaut.
    ~Dreaming();

    Dreaming(const Dreaming&) = delete;
    Dreaming& operator=(const Dreaming&) = delete;

    void set_event_sink(EventSink sink);

    // Nach jedem Turn aufrufen. Billig (eine atomare Datei), verhindert den
    // Totalverlust bei unsauberem Ende.
    Status stage(const std::string& session_text);

    // (a) Sitzungsende.
    Status run_session_end(const std::string& session_text);

    // (b) Idle. Tut nichts, wenn das Intervall nicht um oder nichts neu ist.
    Status maybe_run_idle(const std::string& session_text, int64_t now_ms);

    // (c) Prozess-Exit bzw. Start nach Absturz: verarbeitet die gesicherte
    // Sitzung. Ohne gesicherte Sitzung ein No-Op.
    Status flush(DreamTrigger trigger = DreamTrigger::ProcessExit);

    // Registriert flush() als atexit-Handler. Nur EINE Instanz je Prozess.
    static Status install_exit_flush(Dreaming* d);

    // Beim Start aufrufen: arbeitet eine Sitzung nach, die der letzte Lauf nicht
    // mehr verarbeitet hat.
    Status recover_pending();

    const std::string& pending_path() const { return pending_path_; }

private:
    struct State {
        EventSink   sink;
        int64_t     last_run_ms = 0;
        std::string last_hash;      // gleiche Sitzung nicht zweimal verarbeiten
        bool        busy = false;   // Idle-Thread und Sitzungsende gleichzeitig
    };

    Status run(const std::string& session_text, DreamTrigger trigger);
    void   emit(EventKind k, std::string a, std::string b) const;

    IdentityStore&         identity_;
    LlmFn                  llm_;
    DreamingConfig         cfg_;
    std::string            pending_path_;
    mutable Guarded<State> st_;
};

}  // namespace nova::memory

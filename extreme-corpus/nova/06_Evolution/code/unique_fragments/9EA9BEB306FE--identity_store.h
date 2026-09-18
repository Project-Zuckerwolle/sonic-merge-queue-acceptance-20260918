// identity_store.h — identity.md: `# HEADING`-Blöcke, `# LOCKED…` unantastbar.
//
// Übernommen aus Nova 4 (Memory/identity_store.h) ist die Idee: eine
// Markdown-Datei aus Überschriften-Blöcken, in die das Modell (Dreaming) nur
// EINEN Block schreiben darf, während vom Nutzer gesetzte `# LOCKED…`-Blöcke
// nie überschrieben werden.
//
// WAS HIER ANDERS IST — drei Löcher der Nova-4-Fassung:
//
//   1. Der LOCKED-Schutz galt nur gegen set_block (identity_store.cpp:90).
//      Geschrieben wurde die Datei aber mit einem nackten ofstream
//      (identity_store.cpp:70-76), also truncate-dann-schreiben. Ein Absturz,
//      ein voller Datenträger oder ein Hibernate (§5: der Rechner geht nach
//      Idle in S4) mitten im write() hinterlässt eine halbe oder leere
//      identity.md — inklusive halber LOCKED-Blöcke. Der Schutz war damit
//      gegen genau den Fall wirkungslos, der ihn braucht. Hier: write_atomic()
//      (temp + fsync + rename, nova_core.h §4) plus Rotation der letzten drei
//      Fassungen, damit auch ein inhaltlich kaputtes Dreaming rückholbar ist.
//
//   2. Die Instanz war geteilt und ungeschützt: chat_service.cpp:331 ruft
//      run_dreaming(...deps.identity) beim WS-Close auf, während ein zweiter
//      Client über denselben ChatDeps-Zeiger liest. std::vector<IdentityBlock>
//      wird dabei gleichzeitig gelesen und umgebaut. Hier liegt der Zustand
//      hinter Guarded<> (nova_core.h §5).
//
//   3. write_dreaming schrieb die LLM-Antwort ungeprüft durch. In den echten
//      Testdaten (testdata/memory/identity.md) steht deshalb als kompletter
//      Dreaming-Block: "```berlin```" — ein Code-Fence um ein Wort. Eine leere
//      Antwort hätte den Block ersatzlos geleert. Hier werden Fences entfernt
//      und eine leere Antwort abgelehnt, statt den Bestand zu vernichten.
#pragma once

#include "Core/nova_core.h"

#include <optional>
#include <string>
#include <vector>

namespace nova::memory {

struct IdentityBlock {
    std::string heading;  // ohne führendes "# "; "" = Text vor dem ersten Heading
    std::string body;     // Zeilen bis zum nächsten Heading, ohne Leerzeilen am Ende
};

class IdentityStore {
public:
    explicit IdentityStore(std::string path = {});

    // Muss vor dem ersten schreibenden Zugriff gesetzt werden. Jede Änderung an
    // der Identität ist für den Nutzer sichtbar (EventKind::MemoryWrite) — Nova 4
    // schrieb den Dreaming-Block wortlos.
    void set_event_sink(EventSink sink);

    Status load();               // fehlende Datei ist KEIN Fehler (frische Installation)
    Status save() const;         // atomar + Backup der letzten 3 Fassungen

    // Generisches Setzen. Verweigert jeden `LOCKED…`-Kopf.
    Status set_block(const std::string& heading, const std::string& body);

    // Dreaming schreibt ausschließlich hierhin. Räumt Code-Fences ab und lehnt
    // leeren Inhalt ab, statt einen gefüllten Block zu leeren.
    Status write_dreaming(const std::string& new_body);

    std::optional<std::string> find(const std::string& heading) const;
    std::vector<IdentityBlock>  blocks() const;   // Kopie — der Zustand bleibt im Lock
    std::string                 serialize() const;

    // Nur der Nicht-LOCKED-Teil, für den System-Kopf. LOCKED wird separat und
    // ungekürzt vorangestellt (system_head.cpp), damit ein Token-Budget niemals
    // eine Nutzer-Regel abschneidet.
    std::string locked_text() const;
    std::string unlocked_text() const;

    void parse(const std::string& md);            // ersetzt den Zustand
    static bool is_locked(const std::string& heading);

    const std::string& path() const { return path_; }

    static constexpr const char* kDreaming = "Dreaming";

private:
    struct State {
        std::vector<IdentityBlock> blocks;
        EventSink                  sink;
    };

    // Schreibt OHNE den Zustands-Lock zu nehmen und OHNE Events zu senden — es
    // wird aus einem with()-Block heraus aufgerufen, und std::mutex ist nicht
    // rekursiv. Eine Backup-Warnung kommt über den Ausgabeparameter zurück und
    // wird erst nach dem Verlassen des Locks als Event gemeldet.
    Status write_file_unlocked(const std::string& text, std::string* backup_warn) const;
    void   emit(EventKind k, std::string a, std::string b) const;

    std::string            path_;
    mutable Guarded<State> st_;
};

}  // namespace nova::memory

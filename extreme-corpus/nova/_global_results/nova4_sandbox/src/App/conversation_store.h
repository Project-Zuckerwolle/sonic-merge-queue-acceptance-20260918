// conversation_store.h — Mehrere Chat-Konversationen (Design-Erweiterung).
//
// Nova war ursprünglich Single-Session (§10.3). Für die Claude/ChatGPT-artige UI
// (Sidebar mit mehreren Chats) hält dieser Store je Konversation eine eigene
// WorkingMemory + einen Titel und persistiert den Verlauf nach <data>\chats\.
// Die Single-Stream-Engine erzwingt serielle Generierung (gen_mutex in ChatDeps);
// der Wechsel zwischen Chats ist reines State-Umschalten.
#pragma once

#include "Memory/working_memory.h"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace nova::app {

struct Conversation {
    std::string           id;
    std::string           title;
    memory::WorkingMemory wm;
};

class ConversationStore {
public:
    explicit ConversationStore(std::string dir);  // <data>\chats

    void load_all();  // vorhandene Transcripts beim Start laden

    // Legt eine neue Konversation an und liefert ihre id.
    std::string create(const std::string& title = "Neuer Chat");

    Conversation* get(const std::string& id);  // nullptr wenn unbekannt
    bool  remove(const std::string& id);
    bool  rename(const std::string& id, const std::string& title);

    // (id, title), neueste zuerst.
    std::vector<std::pair<std::string, std::string>> list() const;

    void   persist(const std::string& id) const;  // Transcript -> Datei
    size_t size() const { return convs_.size(); }

private:
    std::string path_for(const std::string& id) const;

    std::string                        dir_;
    std::map<std::string, Conversation> convs_;
    std::vector<std::string>            order_;   // Erstellungsreihenfolge (neueste hinten)
    uint64_t                            counter_ = 0;
};

}  // namespace nova::app

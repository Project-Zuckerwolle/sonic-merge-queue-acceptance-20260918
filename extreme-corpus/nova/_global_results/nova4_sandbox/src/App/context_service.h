// context_service.h — Baut den echten Chat-Kontext (Design §10) für den Live-Chat.
//
// Löst das TODO(gpu) in chat_service: statt Roh-User-Text wird der 6-Schichten-
// Context-Stack assembliert. prefix = stabiler, cachebarer Kopf
// (persona/identity/hot/daily/15 Episoden), dynamic = BM25-Wiki + Working Memory
// + aktueller Turn. Fehlende Dateien -> leere Schicht (frische Installation OK).
#pragma once

#include "Brain/keyword_index.h"
#include "InferEngine/inference.h"
#include "Memory/context_builder.h"

#include <map>
#include <string>
#include <vector>

namespace nova::app {

class ContextService {
public:
    // data_dir = %ProgramData%\Nova4 (bzw. %APPDATA%\Nova4).
    explicit ContextService(std::string data_dir);

    // (Neu-)Aufbau des BM25-Wiki-Index aus brain/wiki/. Beim Start + nach Brain-Zyklus.
    void refresh_wiki();

    // Baut den Turn-Request. working_memory_text = bisheriger Session-Verlauf
    // (inkl. gerade angehängtem User-Turn) oder leer; user_text = aktuelle Nachricht
    // (für die BM25-Abfrage und als Fallback).
    infer::GenRequest build(const std::string& persona_prefix,
                            const std::string& working_memory_text,
                            const std::string& user_text) const;

    size_t wiki_docs() const { return wiki_bodies_.size(); }

private:
    std::string data_dir_;
    std::string mem_dir_;    // <data>/memory
    std::string brain_dir_;  // <data>/brain

    memory::ContextBuilder builder_;
    brain::KeywordIndex     wiki_;
    std::map<std::string, std::string> wiki_bodies_;  // id -> gerendertes Snippet
};

}  // namespace nova::app

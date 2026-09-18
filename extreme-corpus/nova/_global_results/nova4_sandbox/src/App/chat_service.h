// chat_service.h — Wiederverwendbarer WS-Chat-Handler (Aufgabe 0).
//
// Verdrahtet die Inferenz-Seam (IInference) mit dem Tool/Apex-Interceptor
// (nova_ws) und der ToolBox. Sowohl nova4.exe als auch das App-Pipeline-Testbed
// nutzen exakt diesen Pfad — der Test beweist damit denselben Code den die App
// fährt (nur die Engine-Impl wird getauscht: Mock vs. GPU).
#pragma once

#include "Apex/apex_task.h"
#include "Brain/brain_types.h"
#include "InferEngine/inference.h"
#include "Skills/skill_registry.h"
#include "Skills/tool_exec.h"
#include "Web/server.h"

#include <atomic>
#include <mutex>
#include <string>

namespace nova::memory { class WorkingMemory; class IdentityStore; class EpisodeStore; }

namespace nova::app {

class ContextService;
class ConversationStore;

struct ChatDeps {
    infer::IInference*     engine   = nullptr;
    skills::ToolBox*       tools    = nullptr;
    skills::ToolContext*   tool_ctx = nullptr;
    skills::SkillRegistry* registry = nullptr;
    std::string            prefix;               // stabiler System-Header (Prefix-Cache)
    bool                   tier2_confirm = true;

    // Optional (Increment B/C). Ohne diese verhält sich der Handler wie zuvor
    // (Roh-User-Text als dynamic), damit test_app_pipeline unverändert läuft.
    ContextService*        context   = nullptr;  // baut echten 6-Schichten-Kontext (§10)
    memory::WorkingMemory* wm        = nullptr;  // Fallback-Verlauf ohne Multi-Chat-Store
    std::mutex*            gen_mutex = nullptr;  // serialisiert die Single-Stream-Engine
    ConversationStore*     chats     = nullptr;  // Multi-Chat: mehrere Konversationen

    // C2 (Hintergrund/Agentic). Bei Session-Ende (WS-Close) laufen Dreaming +
    // Episode-Erstellung über den gen_mutex-gekapselten Hintergrund-LLM (§10.6, §12.4).
    // Ohne diese Felder verhält sich der Handler wie zuvor (Tests unverändert).
    memory::IdentityStore* identity      = nullptr;  // identity.md (# Dreaming-Block)
    memory::EpisodeStore*  episodes      = nullptr;  // memory/episodes/*.md
    std::string            identity_path;            // Pfad zu identity.md (zum Speichern)
    brain::LlmFn           llm;                       // Hintergrund-LLM (gen_mutex-gekapselt)

    // Apex-ReAct-Loop (§13.6): <apex_call> im Chat startet echte Agent-Tasks.
    apex::ApexTaskManager* apex_tasks   = nullptr;   // apex_tasks/*.json Persistenz
    std::atomic<bool>*     apex_running = nullptr;   // sperrt BrainCompiler während Apex
    // Fertigstellung: Apex-Prompt-Templates (apex/<skill>_prompt.md) + Audit-Trail-Ziel (brain/raw/).
    std::string            apex_dir;                 // §13.9 Template-Verzeichnis
    std::string            brain_raw_dir;            // §13.8 Voll-Output-Audit-Trail

    // Idle-Monitor (§14): steady_clock-ms der letzten WS-Aktivität. Der Idle-Thread
    // liest dies, um Hibernate zu verzögern solange Aktivität kommt. Optional.
    std::atomic<int64_t>*  activity_ms  = nullptr;
};

// Baut einen WS-Handler: empfängt {"type":"message","text":...}, generiert die
// Antwort über engine, fängt <tool_call> ab, führt Normal-Tools aus und streamt
// JSON-Events {text|tool|apex|done} zurück. Loopt bis die Verbindung schließt.
// `deps` muss den Server überleben.
web::HttpServer::WsHandler make_chat_handler(ChatDeps& deps);

}  // namespace nova::app

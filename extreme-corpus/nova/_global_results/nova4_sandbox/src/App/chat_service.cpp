// chat_service.cpp — Implementierung des WS-Chat-Handlers (Aufgabe 0).
#include "App/chat_service.h"

#include "Apex/apex_react_loop.h"
#include "Apex/apex_orchestrator.h"
#include "Apex/apex_result.h"
#include "App/context_service.h"
#include "App/conversation_store.h"
#include "Memory/dreaming.h"
#include "Memory/episode_store.h"
#include "Memory/identity_store.h"
#include "Memory/working_memory.h"
#include "Web/nova_ws.h"

#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <sstream>
#include <vector>

namespace nova::app {

namespace {

// Echter Zeitstempel "YYYYMMDD_HHMMSS" (nur in der App-Verdrahtung, nicht im Kern).
std::string now_stamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%d_%H%M%S", &tm);
    return buf;
}

// Monotone ms für den Idle-Monitor (§14) — steady_clock, unabhängig von Wall-Clock.
int64_t steady_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:   o += c;
        }
    }
    return o;
}

std::string event_json(const web::ChatEvent& e) {
    switch (e.type) {
        case web::ChatEvent::Text: return "{\"type\":\"text\",\"text\":\"" + esc(e.text) + "\"}";
        case web::ChatEvent::Tool: return "{\"type\":\"tool\",\"name\":\"" + esc(e.tool.name) + "\"}";
        case web::ChatEvent::Apex: return "{\"type\":\"apex\",\"skill\":\"" + esc(e.apex.skill) + "\"}";
        case web::ChatEvent::Done: return "{\"type\":\"done\"}";
    }
    return "{\"type\":\"done\"}";
}

// Minimaler Extraktor für ein String-Feld "key":"..." aus der JSON-Nachricht.
std::string json_str_field(const std::string& msg, const std::string& key) {
    const std::string pat = "\"" + key + "\"";
    const size_t k = msg.find(pat);
    if (k == std::string::npos) return {};
    const size_t colon = msg.find(':', k + pat.size());
    if (colon == std::string::npos) return {};
    const size_t q = msg.find('"', colon + 1);
    if (q == std::string::npos) return {};
    std::string out;
    for (size_t i = q + 1; i < msg.size(); ++i) {
        const char ch = msg[i];
        if (ch == '\\' && i + 1 < msg.size()) {
            const char n = msg[++i];
            switch (n) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                default: out += n; break;
            }
        } else if (ch == '"') {
            break;
        } else {
            out += ch;
        }
    }
    return out;
}

// Chat-Liste als JSON-Event für die Sidebar.
std::string chats_json(ConversationStore& cs) {
    std::string o = "{\"type\":\"chats\",\"chats\":[";
    bool first = true;
    for (const auto& [id, title] : cs.list()) {
        if (!first) o += ",";
        o += "{\"id\":\"" + esc(id) + "\",\"title\":\"" + esc(title) + "\"}";
        first = false;
    }
    o += "]}";
    return o;
}

// Startet den echten Apex-ReAct-Loop für einen <apex_call> und streamt die
// THOUGHT/ACTION/OBS-Schritte live ans Frontend (§13.6). Owner-Auto-Freigabe auf
// dem dedizierten Server (Nutzer-Entscheid „Voll + Auto-Freigabe"). Persistiert
// den Task-State (apex_tasks/*.json) für Resume nach Hibernate.
void run_apex(ChatDeps& deps, web::WsConn& c, const web::ApexCall& ac) {
    if (!deps.llm || !deps.tools || !deps.tool_ctx) return;
    if (deps.apex_running) deps.apex_running->store(true);

    apex::ApexTask task;
    task.task_id = "task_" + now_stamp();
    task.skill = ac.skill;
    task.description = ac.task;

    apex::ReactLoop loop(deps.llm, *deps.tools, *deps.tool_ctx);
    if (deps.registry) loop.set_registry(deps.registry, deps.tier2_confirm);
    loop.set_confirm_fn([](const std::string&, const std::string&) { return true; });
    loop.set_event_fn([&c](const apex::ReactStep& s) {
        c.send_text("{\"type\":\"apex_step\",\"step\":\"" + esc(s.type) +
                    "\",\"content\":\"" + esc(s.content) +
                    "\",\"tool\":\"" + esc(s.tool) + "\"}");
    });
    if (deps.apex_tasks) loop.set_task_manager(deps.apex_tasks);

    // System-Prompt: gerendertes Template (apex/<skill>_prompt.md, §13.9) statt hartcodiert.
    // Fallback auf den Inline-Prompt, wenn kein Template vorliegt.
    std::string sys;
    if (!deps.apex_dir.empty()) {
        std::ifstream tf(deps.apex_dir + "\\" + ac.skill + "_prompt.md", std::ios::binary);
        if (tf) { std::stringstream ss; ss << tf.rdbuf();
                  sys = apex::render_template(ss.str(), {{"task", ac.task}, {"skill", ac.skill}}); }
    }
    if (sys.empty()) {
        sys = "Du bist ein Apex-Agent (Skill: " + ac.skill + ") im ReAct-Modus.\n"
              "Aufgabe: " + ac.task + "\n"
              "Denke (THOUGHT), handle via <tool_call name=\"..\" .../>, beobachte OBS. "
              "Schliesse mit <task_complete summary=\"..\"/> oder <task_blocked reason=\"..\"/>.";
    }

    std::string status;
    try { status = loop.run(task, sys); }
    catch (...) { status = "error"; }

    // Result-Handling (§13.8): Voll-Output aus der ReAct-History als Audit-Trail nach brain/raw/;
    // >5k Token werden per Hintergrund-LLM zusammengefasst (finalize_apex_output).
    if (!deps.brain_raw_dir.empty() && deps.llm) {
        std::string full;
        for (const auto& s : task.react_history) {
            full += s.type; if (!s.tool.empty()) full += " [" + s.tool + "]";
            full += ": " + s.content + "\n";
        }
        const std::string audit = deps.brain_raw_dir + "\\apex_" + task.task_id + ".md";
        try { apex::finalize_apex_output(full, deps.llm, 5000, audit); } catch (...) {}
    }
    c.send_text("{\"type\":\"apex_done\",\"status\":\"" + esc(status) + "\"}");

    if (deps.apex_running) deps.apex_running->store(false);
}

}  // namespace

web::HttpServer::WsHandler make_chat_handler(ChatDeps& deps) {
    return [&deps](web::WsConn& c) {
        std::string active;  // aktiver Chat dieser Verbindung
        std::string msg;
        while (c.recv_text(msg)) {
            if (deps.activity_ms) deps.activity_ms->store(steady_now_ms());  // Idle-Monitor (§14)
            const std::string type = json_str_field(msg, "type");

            // Ein globaler Lock (Single-Stream-Engine): serialisiert Generierung UND
            // strukturelle Chat-Operationen, damit Conversation-Pointer gültig bleiben.
            auto lock = [&] {
                std::unique_lock<std::mutex> lk;
                if (deps.gen_mutex) lk = std::unique_lock<std::mutex>(*deps.gen_mutex);
                return lk;
            };

            // ---------- Multi-Chat-Verwaltung ----------
            if (deps.chats && type == "new_chat") {
                auto lk = lock();
                active = deps.chats->create("Neuer Chat");
                c.send_text("{\"type\":\"chat_created\",\"chat\":\"" + esc(active) + "\",\"title\":\"Neuer Chat\"}");
                c.send_text(chats_json(*deps.chats));
                continue;
            }
            if (deps.chats && type == "list_chats") {
                auto lk = lock();
                c.send_text(chats_json(*deps.chats));
                continue;
            }
            if (deps.chats && type == "rename_chat") {
                auto lk = lock();
                deps.chats->rename(json_str_field(msg, "chat"), json_str_field(msg, "title"));
                c.send_text(chats_json(*deps.chats));
                continue;
            }
            if (deps.chats && type == "delete_chat") {
                auto lk = lock();
                const std::string id = json_str_field(msg, "chat");
                deps.chats->remove(id);
                if (active == id) active.clear();
                c.send_text(chats_json(*deps.chats));
                continue;
            }
            if (deps.chats && type == "switch_chat") {
                auto lk = lock();
                active = json_str_field(msg, "chat");
                if (Conversation* conv = deps.chats->get(active)) {
                    for (const auto& t : conv->wm.turns())
                        c.send_text("{\"type\":\"history_turn\",\"role\":\"" + esc(t.role) +
                                    "\",\"text\":\"" + esc(t.text) + "\"}");
                }
                c.send_text("{\"type\":\"history_done\",\"chat\":\"" + esc(active) + "\"}");
                continue;
            }

            // ---------- Nachricht ----------
            const std::string user = json_str_field(msg, "text");
            const std::string chat_arg = json_str_field(msg, "chat");

            // App-Härtung: leere/nur-Whitespace Nachrichten NIE an das Modell geben. Ein leeres
            // [INST][/INST] lässt das schwache 24B den System-Kontext fortsetzen (Selbstbeschreibung/
            // "Echo") statt zu antworten — der einzige reproduzierte Echo-Pfad (NOVA4_C1_ECHO_DIAGNOSE.md).
            // Canned Rückfrage + done, ohne Turn/Kontext/Generierung.
            {
                auto is_blank = [](const std::string& s) {
                    for (char ch : s) if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') return false;
                    return true;
                };
                if (is_blank(user)) {
                    c.send_text("{\"type\":\"text\",\"text\":\"" + esc("Bitte stelle eine Frage.") + "\"}");
                    c.send_text("{\"type\":\"done\"}");
                    continue;
                }
            }

            auto lk = lock();  // Generierung + Chat-Struktur unter einem Lock

            // Ziel-WorkingMemory: aus dem Multi-Chat-Store, sonst Fallback (deps.wm).
            memory::WorkingMemory* wm = deps.wm;
            if (deps.chats) {
                if (!chat_arg.empty()) active = chat_arg;
                Conversation* conv = active.empty() ? nullptr : deps.chats->get(active);
                if (!conv) {
                    active = deps.chats->create("Neuer Chat");
                    conv = deps.chats->get(active);
                    c.send_text("{\"type\":\"chat_created\",\"chat\":\"" + esc(active) +
                                "\",\"title\":\"Neuer Chat\"}");
                }
                // Auto-Titel aus der ersten User-Nachricht.
                const bool first_msg = conv->wm.turn_count() == 0;
                wm = &conv->wm;
                if (first_msg && !user.empty()) {
                    std::string title = user.substr(0, 40);
                    deps.chats->rename(active, title);
                    c.send_text(chats_json(*deps.chats));
                }
            }

            if (wm) wm->add_turn("user", user);

            infer::GenRequest req;
            if (deps.context) {
                req = deps.context->build(deps.prefix, wm ? wm->text() : std::string(), user);
            } else {
                req.prefix = deps.prefix;
                req.dynamic = user;
                req.instruct = true;   // auch der Fallback-Pfad braucht die Instruct-Hülle
            }

            auto tool_exec = [&deps](const web::ToolCall& tc) -> std::string {
                skills::Params p;
                for (const auto& kv : tc.params) p.push_back(kv);
                return deps.tools->execute(tc.name, p, *deps.tool_ctx).output;
            };

            std::string assistant;
            std::vector<web::ApexCall> apex_calls;
            auto sink = [&c, &assistant, &apex_calls](const web::ChatEvent& e) {
                if (e.type == web::ChatEvent::Text) assistant += e.text;
                if (e.type == web::ChatEvent::Apex) apex_calls.push_back(e.apex);
                c.send_text(event_json(e));
            };

            deps.engine->begin(req);
            auto model = infer::as_token_source(*deps.engine);
            web::ChatHandler ch;
            ch.stream(model, sink, tool_exec);

            if (wm) wm->add_turn("assistant", assistant);

            // Generierungs-Lock JETZT freigeben: der Apex-Loop nutzt deps.llm
            // (locked_llm), das gen_mutex selbst nimmt — sonst rekursiver Deadlock.
            if (lk.owns_lock()) lk.unlock();

            // Apex-Calls im Output -> echten ReAct-Loop starten (Block E).
            for (const auto& ac : apex_calls) run_apex(deps, c, ac);

            if (deps.chats && !active.empty()) deps.chats->persist(active);
        }

        // ---------- Session-Ende (WS-Close): Dreaming + Episode (§10.6, §12.4) ----------
        // Läuft über den gen_mutex-gekapselten Hintergrund-LLM (deps.llm), damit es
        // nie mit einer Chat-Generierung kollidiert. Nur wenn C2 verdrahtet ist.
        // deps.llm (locked_llm) serialisiert die Engine bereits selbst — hier KEIN
        // gen_mutex nehmen, sonst Doppel-Lock (std::mutex nicht rekursiv). Der
        // wm.text()-Lesezugriff ist unkritisch (Single-User, andere Konversation).
        if (deps.llm && deps.chats && !active.empty()) {
            if (Conversation* conv = deps.chats->get(active)) {
                if (conv->wm.turn_count() > 0) {
                    const std::string wmtext = conv->wm.text();
                    if (deps.episodes) {
                        memory::Episode ep;
                        deps.episodes->create(deps.llm, wmtext, now_stamp(), ep);
                    }
                    if (deps.identity) {
                        memory::run_dreaming(deps.llm, wmtext, *deps.identity);
                        if (!deps.identity_path.empty()) deps.identity->save(deps.identity_path);
                    }
                }
            }
        }
    };
}

}  // namespace nova::app

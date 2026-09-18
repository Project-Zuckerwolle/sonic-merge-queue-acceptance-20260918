// nova_main.cpp — Applikations-Entry-Point nova4.exe (Aufgabe 0, Design §14, §18).
//
// Verdrahtet die Block-A–E-Bibliotheken zu einem lauffähigen Server:
//   config/Pfade -> SkillRegistry + ToolBox -> IInference -> HttpServer(WS+GET).
// Default-Engine ist MockInference (deterministisch, kein GPU/Modell nötig), so
// dass die gesamte Chat-/Tool-Pipeline hier beweisbar ist; auf dem Server wird
// GpuInference (NOVA_HAVE_CUDA + Modelle) eingesteckt — gleiche Verdrahtung.
#include "Apex/apex_task.h"
#include "App/chat_service.h"
#include "App/context_service.h"
#include "App/conversation_store.h"
#include "Brain/brain_compiler.h"
#include "Brain/brain_store.h"
#include "Brain/brain_types.h"
#include "InferEngine/inference.h"
#include "InferEngine/mock_inference.h"
#include "InferEngine/real_inference.h"
#include "Memory/episode_store.h"
#include "Memory/identity_store.h"
#include "Memory/working_memory.h"
#include "Skills/skill_loader.h"
#include "Skills/skill_registry.h"
#include "Skills/tool_exec.h"
#include "Skills/workspace_guard.h"
#include "System/idle_monitor.h"
#include "System/service.h"
#include "Web/frontend_assets.h"
#include "Web/server.h"
#include "Web/settings.h"
#include "Web/websocket.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace fs = std::filesystem;
using namespace nova;

namespace {

// Laufzeit-Datenpfad: %ProgramData%\Nova4 (Service/SYSTEM) bzw. %APPDATA%\Nova4.
std::string data_dir() {
    if (const char* pd = std::getenv("ProgramData")) return std::string(pd) + "\\Nova4";
    if (const char* ad = std::getenv("APPDATA")) return std::string(ad) + "\\Nova4";
    return "Nova4";
}

const char* arg_value(int argc, char** argv, const char* key, const char* def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == key) return argv[i + 1];
    return def;
}
bool has_flag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == key) return true;
    return false;
}

// Default-System-Prefix wenn keine persona.md existiert (Prefix-Cache-Kandidat).
std::string load_prefix(const std::string& mem_dir) {
    const fs::path persona = fs::path(mem_dir) / "persona.md";
    std::error_code ec;
    if (fs::exists(persona, ec)) {
        FILE* f = std::fopen(persona.string().c_str(), "rb");
        if (f) {
            std::string s;
            char buf[4096]; size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
            std::fclose(f);
            if (!s.empty()) return s;
        }
    }
    return "Du bist Nova 4, ein lokaler KI-Assistent. Sprache: Deutsch. "
           "Keine Floskeln. Kurze Antworten fuer kurze Fragen.";
}

// --- HTTP-Helfer fuer die UI (Upload/Download/Settings/Skill-Install) ---
std::string url_decode(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size()) {
            auto hex = [](char c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; };
            o += char(hex(s[i + 1]) * 16 + hex(s[i + 2])); i += 2;
        } else if (s[i] == '+') o += ' ';
        else o += s[i];
    }
    return o;
}
// Query-Param aus "/pfad?key=value&..." (URL-dekodiert).
std::string query_param(const std::string& path, const std::string& key) {
    const size_t q = path.find('?');
    if (q == std::string::npos) return {};
    std::istringstream in(path.substr(q + 1));
    std::string pair;
    while (std::getline(in, pair, '&')) {
        const size_t eq = pair.find('=');
        if (eq != std::string::npos && pair.substr(0, eq) == key)
            return url_decode(pair.substr(eq + 1));
    }
    return {};
}
std::string path_only(const std::string& path) {
    const size_t q = path.find('?');
    return q == std::string::npos ? path : path.substr(0, q);
}
std::string read_file_bin(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}
bool write_file_bin(const std::string& p, const std::string& data) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(data.data(), std::streamsize(data.size())); return bool(f);
}

}  // namespace

int main(int argc, char** argv) {
    if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        std::printf(
            "nova4.exe — lokaler KI-Assistent\n"
            "  --mock                 deterministische CPU-Engine (Default ohne GPU)\n"
            "  --bind <addr>          Bind-Adresse (Default 127.0.0.1)\n"
            "  --port <n>             Port (Default 8000)\n"
            "  --install-service      als Windows-Dienst registrieren\n"
            "  --uninstall-service    Dienst entfernen\n");
        return 0;
    }

#ifdef _WIN32
    if (has_flag(argc, argv, "--install-service")) {
        std::string err;
        const bool ok = system::install_service(&err);
        std::printf("install-service: %s\n", ok ? "OK" : err.c_str());
        return ok ? 0 : 1;
    }
    if (has_flag(argc, argv, "--uninstall-service")) {
        std::string err;
        const bool ok = system::uninstall_service(&err);
        std::printf("uninstall-service: %s\n", ok ? "OK" : err.c_str());
        return ok ? 0 : 1;
    }
#endif

    const std::string dd_default = data_dir();
    const std::string root = arg_value(argc, argv, "--data-dir", dd_default.c_str());  // Test-Instanz isolierbar
    std::error_code ec;
    fs::create_directories(root, ec);

    // config.json laden bzw. beim ersten Start anlegen. Auf diesem dedizierten
    // Nova-Server ist Vollzugriff (unrestricted) der gewuenschte Default.
    const std::string cfg_path = root + "\\config.json";
    web::Settings cfg = web::Settings::defaults();
    if (fs::exists(cfg_path, ec)) {
        cfg.load(cfg_path);
    } else {
        cfg.set("unrestricted", "true");   // dedizierter Server: Nova steuert den ganzen PC
        cfg.set("active_profile", "");      // leer = alle Tools aktiv
        cfg.save(cfg_path);
        std::printf("config.json angelegt: %s\n", cfg_path.c_str());
    }

    // Bind/Port: CLI hat Vorrang, sonst config.json.
    const std::string bind = arg_value(argc, argv, "--bind", cfg.get("bind_address", "127.0.0.1").c_str());
    const int port = std::atoi(arg_value(argc, argv, "--port", cfg.get("port_http", "8000").c_str()));

    // Workspace + Vollzugriff aus config.
    const std::string ws = cfg.get("workspace", "workspace");
    const std::string workspace = fs::path(ws).is_absolute() ? ws : (root + "\\" + ws);
    const bool unrestricted = cfg.get_bool("unrestricted", false);
    const std::string mem_dir = root + "\\memory";
    fs::create_directories(workspace, ec);

    // Skills, Tools, Kontext.
    skills::SkillRegistry registry = skills::SkillRegistry::with_defaults();
    // Nutzer-eigene Skills aus %ProgramData%\Nova4\skills\ nachladen (§13.1).
    const std::string skills_dir = root + "\\skills";
    fs::create_directories(skills_dir, ec);
    int user_skills = 0;
    for (const auto& sk : skills::load_skill_dir(skills_dir)) { registry.add(sk); ++user_skills; }
    if (!cfg.get("active_profile").empty()) registry.set_active_profile(cfg.get("active_profile"));

    skills::ToolBox tools = skills::default_toolbox();
    skills::WorkspaceGuard guard(workspace, unrestricted);
    skills::ToolContext tool_ctx;
    tool_ctx.guard = &guard;
    tool_ctx.todo_path = workspace + "\\todos.md";

    // Engine-Auswahl (config "engine": "mock" | "real"). "real" = INT3-RAM-Engine (C1):
    // lädt die Textgewichte aus einer GGUF (gguf_path) beim Start nach RAM, KV-gecacht.
    // Fällt bei Fehler/fehlendem Pfad auf Mock zurück, damit der Server immer startet.
    infer::MockInference mock_engine;
    std::unique_ptr<infer::RealInference> real_engine;
    infer::IInference* engine_ptr = &mock_engine;
    const std::string engine_kind = cfg.get("engine", "mock");
    const char* engine_name = "Mock";
    if (engine_kind == "real") {
        const std::string gguf = cfg.get("gguf_path", "");
        const std::string tekken = cfg.get("tekken_dir", root + "\\models\\tekken");
        if (gguf.empty()) {
            std::printf("engine=real, aber gguf_path leer -> Mock.\n");
        } else {
            std::printf("Lade Real-Engine (INT3 RAM) aus %s ... (dauert)\n", gguf.c_str());
            // Fertigstellung Phase 1: schnellen C1-Pfad als Default konfigurieren (config.json übersteuerbar,
            // Env hätte sonst Vorrang). MUSS vor load()/load_draft() gesetzt werden (ss_*-Helfer lesen einmalig).
            infer::real_inference_set_flag("NOVA_GPU_FWD24",          cfg.get_bool("gpu_forward", true)        ? "1" : "0");
            infer::real_inference_set_flag("NOVA_FMT_BLOCK3",         cfg.get_bool("fmt_block3", true)         ? "1" : "0");
            infer::real_inference_set_flag("NOVA_SPEC_SINGLE_STREAM", cfg.get_bool("spec_single_stream", true) ? "1" : "0");
            infer::real_inference_set_flag("NOVA_PINNED_STREAM",      cfg.get_bool("pinned_stream", true)      ? "1" : "0");
            infer::real_inference_set_flag("NOVA_KV_BITS",            cfg.get("kv_bits", "3"));
            infer::real_inference_set_flag("NOVA_TGT_BITS",           cfg.get("tgt_bits", "3"));
            infer::real_inference_set_flag("NOVA_TGT_KV",             cfg.get("target_kv", "80000"));
            infer::real_inference_set_flag("NOVA_DRAFT_KV",           cfg.get("draft_kv", "2048"));
            // Perf: verifizierte Software-Gewinne als Default AN (greedy-exakt, config-übersteuerbar).
            infer::real_inference_set_flag("NOVA_ADAPTIVE_K",         cfg.get_bool("adaptive_k", true)  ? "1" : "0");  // +18-20% Decode
            infer::real_inference_set_flag("NOVA_PREFIX_CACHE",       cfg.get_bool("prefix_cache", true) ? "1" : "0"); // TTFT Turn 2+
            real_engine = std::make_unique<infer::RealInference>();
            std::string ee;
            infer::RealInference::Options ropt; ropt.max_new = cfg.get_int("max_new", 512);
            if (real_engine->load(gguf, infer::Mistral3Config::m24b(), tekken, ropt, &ee)) {
                engine_ptr = real_engine.get(); engine_name = "Real-INT3";
                // Draft (3B) laden -> Spec-Decoding aktiv (sonst plain greedy). Optional via draft_gguf_path.
                const std::string draft_gguf = cfg.get("draft_gguf_path", "");
                if (!draft_gguf.empty()) {
                    std::string de;
                    if (real_engine->load_draft(draft_gguf, infer::Mistral3Config::m3b(), &de)) {
                        engine_name = "Real-INT3+Spec";
                        std::printf("Draft (3B) geladen -> Spec-Decoding aktiv.\n");
                    } else {
                        std::printf("Draft-Load fehlgeschlagen (%s) -> greedy ohne Spec.\n", de.c_str());
                    }
                }
            } else {
                std::printf("Real-Engine fehlgeschlagen (%s) -> Mock.\n", ee.c_str());
                real_engine.reset();
            }
        }
    }

    app::ChatDeps deps;
    deps.engine = engine_ptr;
    deps.tools = &tools;
    deps.tool_ctx = &tool_ctx;
    deps.registry = &registry;
    deps.prefix = load_prefix(mem_dir);
    deps.tier2_confirm = cfg.get_bool("tier2_confirm", false);

    // Echter Kontext (§10) + Session-Verlauf + serialisierte Generierung.
    // (Increment C ersetzt die eine WorkingMemory durch einen Multi-Chat-Store.)
    app::ContextService context(root);
    memory::WorkingMemory wm;
    std::mutex gen_mutex;
    deps.context = &context;
    deps.wm = &wm;
    deps.gen_mutex = &gen_mutex;

    // Multi-Chat: mehrere Konversationen (Sidebar) mit Verlaufs-Persistenz.
    app::ConversationStore chats(root + "\\chats");
    chats.load_all();
    deps.chats = &chats;

    // ---------- C2: Hintergrund/Agentic (§10.6, §11.2, §12.4) ----------
    // Ein gen_mutex-gekapselter Hintergrund-LLM für Brain/Dreaming/Episode: alle
    // teilen sich die Single-Stream-Engine mit dem Chat, daher IMMER unter demselben
    // Lock — nie parallele Nutzung derselben Engine.
    std::string err;
    fs::create_directories(mem_dir, ec);
    std::atomic<bool> apex_running{false};
    auto raw_llm = infer::as_llm_fn(*engine_ptr, deps.prefix);
    brain::LlmFn locked_llm = [raw_llm, &gen_mutex](const std::string& prompt) {
        std::lock_guard<std::mutex> lk(gen_mutex);
        return raw_llm(prompt);
    };

    // Identity (# Dreaming-Block) + Episoden — für Session-Ende im Chat-Handler.
    memory::IdentityStore identity;
    const std::string identity_path = mem_dir + "\\identity.md";
    identity.load(identity_path);            // fehlend = leer, ok
    memory::EpisodeStore episodes(mem_dir + "\\episodes");
    episodes.init();
    deps.identity = &identity;
    deps.episodes = &episodes;
    deps.identity_path = identity_path;
    deps.llm = locked_llm;

    // Apex-Task-Persistenz (apex_tasks/*.json) + apex_running-Lock für Brain.
    apex::ApexTaskManager apex_mgr(root + "\\apex_tasks");
    apex_mgr.init(&err);
    deps.apex_tasks = &apex_mgr;
    deps.apex_running = &apex_running;
    deps.apex_dir = root + "\\apex";            // Prompt-Templates (§13.9)
    deps.brain_raw_dir = root + "\\brain\\raw"; // Apex-Audit-Trail (§13.8)
    // Resume (§13.7): unfertige Apex-Tasks nach Wake melden.
    for (const auto& t : apex_mgr.unfinished())
        std::printf("[apex] unfertiger Task '%s' (%s) bei Iteration %d/%d — Resume moeglich.\n",
                    t.description.c_str(), t.task_id.c_str(), t.iteration, t.max_iterations);

    // BrainCompiler-Zyklus (~30 Min, prüft apex_running, LOW-Prio). Läuft im
    // Hintergrund und verarbeitet brain/raw -> wiki; 03:00 die Tagesaufgaben.
    brain::BrainStore brain_store(root + "\\brain");
    brain_store.init(&err);
    brain::BrainCompiler compiler(brain_store, locked_llm, apex_running);
    compiler.set_briefing_path(mem_dir + "\\daily_briefing.md");
    const int brain_min = std::atoi(cfg.get("brain_interval_min", "30").c_str());
    std::thread([&compiler, &episodes, brain_min]() {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::minutes(brain_min > 0 ? brain_min : 30));
            std::time_t t = std::time(nullptr);
            std::tm tm{};
#ifdef _WIN32
            localtime_s(&tm, &t);
#else
            localtime_r(&t, &tm);
#endif
            char day[16]; std::strftime(day, sizeof day, "%Y-%m-%d", &tm);
            char hm[8];   std::strftime(hm, sizeof hm, "%H:%M", &tm);
            brain::BrainCycleInput in;
            in.today = day; in.clock = hm; in.run_daily = (tm.tm_hour == 3);
            in.session_id = std::string(day) + "_" + hm;
            // Echte Session-Daten (G24): Daily-Briefing-Quelle + erwähnte Seite aus der letzten Episode,
            // statt leerem Source (sonst Konsolidierung nur altersbasiert, Briefing aus Leer-String).
            auto last = episodes.last(1);
            if (!last.empty()) {
                in.briefing_source = last.front().title + "\n" + last.front().summary;
                if (!last.front().title.empty()) in.mentioned.push_back(last.front().title);
            }
            compiler.run_cycle(in);          // überspringt intern bei apex_running
        }
    }).detach();

    // Idle-Monitor/Hibernate (§14, Server-PC). Standardmäßig AUS — würde den Server
    // suspendieren; per config idle_hibernate=true opt-in. Apex blockiert Hibernate
    // (apex_running). WS-Aktivität setzt activity_ms zurück -> Countdown-Abbruch.
    static std::atomic<int64_t> activity_ms{0};
    deps.activity_ms = &activity_ms;
    if (cfg.get_bool("idle_hibernate", false)) {
        system::IdleConfig icfg;
        const int idle_min = std::atoi(cfg.get("idle_timeout_min", "30").c_str());
        icfg.idle_timeout_ms = int64_t(idle_min > 0 ? idle_min : 30) * 60 * 1000;
        std::thread([&apex_running, icfg]() {
            auto now = [] {
                return int64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
            };
            system::IdleMonitor mon(icfg, apex_running);
            mon.touch(now());
            int64_t seen = activity_ms.load();
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(30));
                const int64_t a = activity_ms.load();
                if (a != seen) { seen = a; mon.touch(now()); }  // neue WS-Aktivität
                mon.tick(now());                                 // löst Hibernate intern aus
            }
        }).detach();
        std::printf("Idle-Hibernate aktiv: Timeout %d Min (config idle_timeout_min)\n", idle_min);
    }

    if (!web::net_init(&err)) { std::printf("net_init: %s\n", err.c_str()); return 1; }

    web::HttpServer server;
    server.set_ws_handler("/ws", app::make_chat_handler(deps));
    server.set_get_handler([&](const std::string& path) {
        web::HttpResponse r;
        const std::string p = path_only(path);
        if (p == "/api/config") {
            r.content_type = "application/json; charset=utf-8";
            r.body = read_file_bin(cfg_path);
        } else if (p == "/api/persona") {
            r.content_type = "text/plain; charset=utf-8";
            r.body = read_file_bin(mem_dir + "\\persona.md");
        } else if (p == "/download") {
            std::string abs, e;
            if (guard.resolve(query_param(path, "path"), abs, &e)) {
                r.content_type = "application/octet-stream";
                r.body = read_file_bin(abs);
            } else { r.status = 403; r.body = e; }
        } else if (p == "/m" || p == "/mobile") {
            r.body = web::mobile_html();
        } else {
            r.body = web::index_html();
        }
        return r;
    });

    server.set_post_handler([&](const std::string& path, const std::string& body) {
        web::HttpResponse r;
        r.content_type = "application/json; charset=utf-8";
        const std::string p = path_only(path);
        if (p == "/upload") {
            std::string abs, e;
            if (!guard.resolve(query_param(path, "path"), abs, &e)) { r.status = 403; r.body = "{\"ok\":false}"; return r; }
            std::error_code ec2; fs::create_directories(fs::path(abs).parent_path(), ec2);
            const bool ok = write_file_bin(abs, body);
            r.body = ok ? "{\"ok\":true}" : "{\"ok\":false}";
        } else if (p == "/api/config") {
            r.body = write_file_bin(cfg_path, body) ? "{\"ok\":true,\"note\":\"Neustart uebernimmt Aenderungen\"}"
                                                    : "{\"ok\":false}";
        } else if (p == "/skill") {
            const std::string name = query_param(path, "name");
            if (name.empty()) { r.status = 400; r.body = "{\"ok\":false}"; return r; }
            const bool ok = write_file_bin(skills_dir + "\\" + name + ".yaml", body);
            r.body = ok ? "{\"ok\":true,\"note\":\"Neustart laedt den Skill\"}" : "{\"ok\":false}";
        } else { r.status = 404; r.body = "{\"ok\":false}"; }
        return r;
    });

    if (!server.start(bind, port, &err)) { std::printf("start: %s\n", err.c_str()); return 1; }
    std::printf("Nova 4 laeuft auf http://%s:%d  (Engine: %s, unrestricted=%s, %d User-Skills, Strg+C)\n",
                bind.c_str(), server.port(), engine_name, unrestricted ? "AN" : "aus", user_skills);

#ifdef _WIN32
    // Als Windows-Dienst gestartet (SCM ruft "nova4.exe --service"): die Server-
    // Setup oben ist mit Mock instant (<1s), also weit unter dem SCM-Start-Timeout.
    // Der Dispatcher meldet RUNNING und blockiert bis Stop; g_body wartet darauf.
    // (Bei C1/GPU-Preload später ggf. START_PENDING-Checkpoints ergänzen.)
    if (system::parse_service_command(argc, argv) == system::ServiceCommand::Run) {
        system::run_as_service([] {
            while (!system::service_stop_requested())
                std::this_thread::sleep_for(std::chrono::seconds(1));
        }, &err);
        return 0;
    }
#endif

    for (;;) std::this_thread::sleep_for(std::chrono::seconds(1));
    return 0;  // via Strg+C beendet
}

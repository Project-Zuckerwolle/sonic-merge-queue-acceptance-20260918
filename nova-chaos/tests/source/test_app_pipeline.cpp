// test_app_pipeline.cpp — nova4.exe End-to-End-Pipeline (Aufgabe 0).
//
// Beweist die komplette App-Verdrahtung OHNE GPU/Modell: echter HttpServer +
// WebSocket-Loopback-Client -> app::make_chat_handler -> MockInference ->
// StreamInterceptor -> ToolBox. Kriterien: Token-für-Token-Streaming, ein
// <tool_call> wird abgefangen, die Tool-OBS wird eingebettet, {done} kommt,
// und ein zweiter Turn beweist den Handler-Loop (Multi-Turn).
#include "App/chat_service.h"
#include "InferEngine/mock_inference.h"
#include "Skills/skill_registry.h"
#include "Skills/tool_exec.h"
#include "Skills/workspace_guard.h"
#include "Web/server.h"
#include "Web/websocket.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace nova;
using clk = std::chrono::high_resolution_clock;

namespace {
double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// Sendet eine Nachricht, liest bis {done}; sammelt Text + Tool-Events.
struct TurnResult {
    std::string text;
    int text_frames = 0;
    int tool_frames = 0;
    double first_ms = -1.0;
    bool done = false;
};
TurnResult run_turn(web::WsConn& cli, const std::string& user) {
    TurnResult r;
    const auto t0 = clk::now();
    cli.send_text("{\"type\":\"message\",\"text\":\"" + user + "\"}");
    std::string frame;
    while (cli.recv_text(frame)) {
        if (frame.find("\"type\":\"text\"") != std::string::npos) {
            if (r.first_ms < 0) r.first_ms = ms_since(t0);
            ++r.text_frames;
            r.text += frame;
        } else if (frame.find("\"type\":\"tool\"") != std::string::npos) {
            ++r.tool_frames;
        } else if (frame.find("\"type\":\"done\"") != std::string::npos) {
            r.done = true;
            break;
        }
    }
    return r;
}
}  // namespace

int main() {
    std::cout << "=== App-Pipeline: nova4.exe End-to-End (Mock) ===\n";
    std::string err;
    if (!web::net_init(&err)) { std::cout << "  net_init: " << err << "\n"; return 2; }

    const std::string ws = (fs::temp_directory_path() / "nova4_app_ws").string();
    std::error_code ec; fs::create_directories(ws, ec);

    skills::SkillRegistry registry = skills::SkillRegistry::with_defaults();
    skills::ToolBox tools = skills::default_toolbox();
    skills::WorkspaceGuard guard(ws);
    skills::ToolContext tool_ctx; tool_ctx.guard = &guard;
    infer::MockInference engine;

    app::ChatDeps deps;
    deps.engine = &engine; deps.tools = &tools;
    deps.tool_ctx = &tool_ctx; deps.registry = &registry;
    deps.prefix = "Testkontext.";

    web::HttpServer server;
    server.set_ws_handler("/ws", app::make_chat_handler(deps));
    if (!server.start("127.0.0.1", 0, &err)) { std::cout << "  start: " << err << "\n"; return 2; }
    std::printf("  Server: 127.0.0.1:%d\n", server.port());

    web::WsConn cli;
    if (!web::ws_client_connect("127.0.0.1", server.port(), "/ws", cli, &err)) {
        std::cout << "  client: " << err << "\n"; server.stop(); return 2;
    }

    // Turn 1: Arithmetik -> rechner-Tool.
    const TurnResult t1 = run_turn(cli, "Bitte rechne 2+3*4 aus.");
    std::printf("  T1 erstes Token %.1f ms | text_frames=%d | tool=%d | done=%s\n",
                t1.first_ms, t1.text_frames, t1.tool_frames, t1.done ? "ja" : "nein");
    std::printf("  T1 Chat: %s\n", t1.text.c_str());

    // Turn 2: Begrüßung -> kein Tool (beweist Handler-Loop).
    const TurnResult t2 = run_turn(cli, "Hallo, wer bist du?");
    std::printf("  T2 text_frames=%d | tool=%d | done=%s\n",
                t2.text_frames, t2.tool_frames, t2.done ? "ja" : "nein");

    cli.close();
    server.stop();
    fs::remove_all(ws, ec);

    bool pass = true;
    pass &= (t1.first_ms >= 0 && t1.first_ms < 100.0);   // schnelles erstes Token
    pass &= (t1.text_frames >= 5);                       // gestreamt, nicht ein Batch
    pass &= (t1.tool_frames == 1);                        // genau ein tool_call
    pass &= (t1.text.find("= 14") != std::string::npos); // rechner-OBS eingebettet
    pass &= (t1.text.find("fertig") != std::string::npos);
    pass &= t1.done;
    pass &= (t2.done && t2.tool_frames == 0);             // 2. Turn, kein Tool
    pass &= (t2.text.find("Nova") != std::string::npos);

    std::cout << "\n=== App-Pipeline: " << (pass ? "BESTANDEN" : "FEHLGESCHLAGEN") << " ===\n";
    return pass ? 0 : 1;
}

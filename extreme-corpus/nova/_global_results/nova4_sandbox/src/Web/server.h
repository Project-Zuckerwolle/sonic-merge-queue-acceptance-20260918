// server.h — Minimaler HTTP/WebSocket-Server auf Winsock (Design §14, TB 10).
//
// Accept-Loop in eigenem Thread, ein Thread je Verbindung. Routen:
//   - WebSocket-Upgrade auf ws_path -> WsHandler(WsConn&) (Chat-Streaming)
//   - sonst GET -> GetHandler(path) liefert (content_type, body)
//
// Ersetzt Crow für den hermetischen Testbed; gleiche Rolle (WebSocket + REST),
// gleiche Handler-Signaturen wie sie ein Crow-Adapter später anbieten würde.
#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "Web/websocket.h"

namespace nova::web {

struct HttpResponse {
    int         status = 200;
    std::string content_type = "text/html; charset=utf-8";
    std::string body;
};

class HttpServer {
public:
    using WsHandler   = std::function<void(WsConn&)>;
    using GetHandler  = std::function<HttpResponse(const std::string& path)>;
    using PostHandler = std::function<HttpResponse(const std::string& path, const std::string& body)>;

    ~HttpServer() { stop(); }

    void set_ws_handler(const std::string& path, WsHandler h) { ws_path_ = path; ws_ = std::move(h); }
    void set_get_handler(GetHandler h) { get_ = std::move(h); }
    void set_post_handler(PostHandler h) { post_ = std::move(h); }

    // bind_addr z.B. "127.0.0.1". port==0 -> OS wählt freien Port (port() abfragen).
    bool start(const std::string& bind_addr, int port, std::string* err = nullptr);
    void stop();
    int  port() const { return port_; }
    bool running() const { return running_.load(); }

private:
    void accept_loop();
    void handle_conn(Socket s);

    std::string ws_path_ = "/ws";
    WsHandler   ws_;
    GetHandler  get_;
    PostHandler post_;

    Socket listen_sock_ = invalid_socket();
    int    port_ = 0;
    std::atomic<bool> running_{false};
    std::thread accept_thread_;
    std::vector<std::thread> conn_threads_;
};

}  // namespace nova::web

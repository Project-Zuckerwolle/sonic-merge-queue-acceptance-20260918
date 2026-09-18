// server.cpp — Implementierung von server.h (Winsock HTTP/WS).
#include "Web/server.h"

#include <algorithm>
#include <sstream>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

namespace nova::web {

namespace {
// Liest HTTP-Request bis "\r\n\r\n". Liefert Roh-Header.
bool read_headers(Socket s, std::string& out) {
    char ch;
    while (out.find("\r\n\r\n") == std::string::npos) {
        const int n = ::recv(int(s), &ch, 1, 0);
        if (n <= 0) return false;
        out.push_back(ch);
        if (out.size() > 65536) return false;
    }
    return true;
}

std::string header_value(const std::string& req, const std::string& key) {
    // case-insensitiver Zeilen-Scan "Key: value".
    std::istringstream in(req);
    std::string line;
    auto lower = [](std::string x) { for (auto& c : x) c = char(::tolower((unsigned char)c)); return x; };
    const std::string k = lower(key);
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        if (lower(line.substr(0, colon)) == k) {
            size_t v = colon + 1;
            while (v < line.size() && (line[v] == ' ' || line[v] == '\t')) ++v;
            return line.substr(v);
        }
    }
    return {};
}

std::string request_path(const std::string& req) {
    // "GET /path HTTP/1.1"
    const size_t sp1 = req.find(' ');
    if (sp1 == std::string::npos) return "/";
    const size_t sp2 = req.find(' ', sp1 + 1);
    if (sp2 == std::string::npos) return "/";
    return req.substr(sp1 + 1, sp2 - sp1 - 1);
}

std::string request_method(const std::string& req) {
    const size_t sp = req.find(' ');
    return sp == std::string::npos ? "GET" : req.substr(0, sp);
}

// Liest exakt n Bytes (POST-Body). false bei Verbindungsabbruch.
bool recv_exact(Socket s, std::string& out, size_t n) {
    out.clear(); out.reserve(n);
    char buf[65536];
    while (out.size() < n) {
        const int want = int(std::min(sizeof(buf), n - out.size()));
        const int got = ::recv(int(s), buf, want, 0);
        if (got <= 0) return false;
        out.append(buf, size_t(got));
    }
    return true;
}

const char* reason_phrase(int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 500: return "Internal Server Error";
        default:  return "OK";
    }
}

bool send_all_raw(Socket s, const std::string& data) {
    size_t sent = 0;
    while (sent < data.size()) {
        const int n = ::send(int(s), data.data() + sent, int(data.size() - sent), 0);
        if (n <= 0) return false;
        sent += size_t(n);
    }
    return true;
}
}  // namespace

bool HttpServer::start(const std::string& bind_addr, int port, std::string* err) {
    if (!net_init(err)) return false;
#ifdef _WIN32
    listen_sock_ = Socket(::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
    if (!is_valid(listen_sock_)) { if (err) *err = "socket() fehlgeschlagen"; return false; }

    BOOL yes = TRUE;
    ::setsockopt(int(listen_sock_), SOL_SOCKET, SO_REUSEADDR,
                 reinterpret_cast<const char*>(&yes), sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(uint16_t(port));
    if (::inet_pton(AF_INET, bind_addr.c_str(), &addr.sin_addr) != 1) {
        if (err) *err = "inet_pton fehlgeschlagen: " + bind_addr;
        ::closesocket(int(listen_sock_)); listen_sock_ = invalid_socket(); return false;
    }
    if (::bind(int(listen_sock_), reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        if (err) *err = "bind fehlgeschlagen"; ::closesocket(int(listen_sock_));
        listen_sock_ = invalid_socket(); return false;
    }
    if (::listen(int(listen_sock_), SOMAXCONN) != 0) {
        if (err) *err = "listen fehlgeschlagen"; ::closesocket(int(listen_sock_));
        listen_sock_ = invalid_socket(); return false;
    }
    // Tatsächlich gebundenen Port ermitteln (bei port==0).
    sockaddr_in bound{}; int blen = sizeof(bound);
    if (::getsockname(int(listen_sock_), reinterpret_cast<sockaddr*>(&bound), &blen) == 0)
        port_ = ntohs(bound.sin_port);
    else
        port_ = port;

    running_.store(true);
    accept_thread_ = std::thread([this] { accept_loop(); });
    return true;
#else
    (void)bind_addr; (void)port; if (err) *err = "nur Windows"; return false;
#endif
}

void HttpServer::accept_loop() {
#ifdef _WIN32
    while (running_.load()) {
        sockaddr_in caddr{}; int clen = sizeof(caddr);
        const Socket cs = Socket(::accept(int(listen_sock_),
                                          reinterpret_cast<sockaddr*>(&caddr), &clen));
        if (!is_valid(cs)) { if (!running_.load()) break; continue; }
        conn_threads_.emplace_back([this, cs] { handle_conn(cs); });
    }
#endif
}

void HttpServer::handle_conn(Socket s) {
#ifdef _WIN32
    std::string req;
    if (!read_headers(s, req)) { ::closesocket(int(s)); return; }

    const std::string path = request_path(req);
    const std::string upgrade = header_value(req, "Upgrade");
    const std::string wskey = header_value(req, "Sec-WebSocket-Key");

    bool is_ws = false;
    for (char c : upgrade) if (::tolower((unsigned char)c) == 'w') { is_ws = true; break; }
    is_ws = is_ws && !wskey.empty();

    if (is_ws && path == ws_path_ && ws_) {
        const std::string accept = ws_accept_key(wskey);
        const std::string resp =
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            "Sec-WebSocket-Accept: " + accept + "\r\n\r\n";
        if (send_all_raw(s, resp)) {
            WsConn conn(s, /*is_server=*/true);
            ws_(conn);
            conn.close();
            return;  // close() schließt den Socket
        }
        ::closesocket(int(s));
        return;
    }

    // GET oder POST.
    const std::string method = request_method(req);
    HttpResponse r;
    if (method == "POST") {
        std::string body;
        const long clen = std::atol(header_value(req, "Content-Length").c_str());
        if (clen > 0) recv_exact(s, body, size_t(clen));
        if (post_) r = post_(path, body);
        else { r.status = 404; r.body = "Not Found"; }
    } else {
        if (get_) r = get_(path);
        else { r.status = 404; r.body = "Not Found"; }
    }
    std::ostringstream os;
    os << "HTTP/1.1 " << r.status << " " << reason_phrase(r.status) << "\r\n"
       << "Content-Type: " << r.content_type << "\r\n"
       << "Content-Length: " << r.body.size() << "\r\n"
       << "Connection: close\r\n\r\n";
    // Header + Body getrennt senden (Body kann binär/gross sein).
    if (send_all_raw(s, os.str())) send_all_raw(s, r.body);
    ::closesocket(int(s));
#else
    (void)s;
#endif
}

void HttpServer::stop() {
    if (!running_.exchange(false)) return;
#ifdef _WIN32
    if (is_valid(listen_sock_)) {
        ::closesocket(int(listen_sock_));  // bricht accept() ab
        listen_sock_ = invalid_socket();
    }
#endif
    if (accept_thread_.joinable()) accept_thread_.join();
    for (auto& t : conn_threads_) if (t.joinable()) t.join();
    conn_threads_.clear();
}

}  // namespace nova::web

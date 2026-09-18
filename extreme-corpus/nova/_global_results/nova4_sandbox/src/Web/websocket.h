// websocket.h — Minimale WebSocket-Schicht auf Winsock (Design §13.5, TB 10).
//
// Statt Crow (Asio/Boost) eine selbst-enthaltene, hermetisch testbare native
// Implementierung — passt zu "ein Binary, Windows-native, keine schweren Deps".
// Enthält RFC-6455-Handshake (SHA1 + Base64), Frame-Encode/Decode (Text + Close
// + Ping/Pong) und Client-Connect für die Testbeds. Die Chat-Logik (nova_ws) ist
// transport-agnostisch und kann später hinter Crow gehängt werden.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nova::web {

#ifdef _WIN32
using Socket = uintptr_t;  // == SOCKET (UINT_PTR)
#else
using Socket = int;
#endif
Socket invalid_socket();
bool   is_valid(Socket s);

bool net_init(std::string* err = nullptr);   // WSAStartup (idempotent)
void net_shutdown();

// Krypto-Helfer (für Handshake; auch separat getestet).
std::vector<uint8_t> sha1(const uint8_t* data, size_t len);
std::string          base64_encode(const uint8_t* data, size_t len);
std::string          ws_accept_key(const std::string& client_key);  // §RFC6455

// Eine WebSocket-Verbindung über einen bereits verbundenen Socket.
// is_server: Server sendet UNMASKED, Client sendet MASKED (RFC 6455).
class WsConn {
public:
    WsConn() : sock_(invalid_socket()), server_(true) {}
    WsConn(Socket s, bool is_server) : sock_(s), server_(is_server) {}

    bool send_text(const std::string& payload);
    // Blockiert bis eine vollständige Textnachricht vorliegt. false bei Close/Fehler.
    bool recv_text(std::string& out);
    void close();
    bool valid() const { return is_valid(sock_); }
    Socket sock() const { return sock_; }

private:
    bool send_all(const uint8_t* data, size_t len);
    bool fill(size_t need);  // liest bis buf_ >= need Bytes hat

    Socket sock_;
    bool   server_;
    std::vector<uint8_t> buf_;  // ungeparste Empfangsbytes
};

// Client: TCP-Connect + WS-Handshake gegen host:port/path. Liefert offene
// WsConn (is_server=false). Für Testbeds.
bool ws_client_connect(const std::string& host, int port, const std::string& path,
                       WsConn& out, std::string* err = nullptr);

}  // namespace nova::web

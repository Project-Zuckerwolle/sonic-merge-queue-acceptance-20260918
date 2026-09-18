// websocket.cpp — Implementierung von websocket.h (RFC 6455 auf Winsock).
#include "Web/websocket.h"

#include <cstring>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

namespace nova::web {

Socket invalid_socket() {
#ifdef _WIN32
    return Socket(INVALID_SOCKET);
#else
    return -1;
#endif
}
bool is_valid(Socket s) { return s != invalid_socket(); }

bool net_init(std::string* err) {
#ifdef _WIN32
    static bool done = false;
    if (done) return true;
    WSADATA w;
    if (WSAStartup(MAKEWORD(2, 2), &w) != 0) {
        if (err) *err = "WSAStartup fehlgeschlagen";
        return false;
    }
    done = true;
#endif
    return true;
}
void net_shutdown() {
#ifdef _WIN32
    // Bewusst kein WSACleanup() — Prozess-Lebensdauer, idempotenter Init.
#endif
}

// ---- SHA1 (RFC 3174) ------------------------------------------------------
std::vector<uint8_t> sha1(const uint8_t* data, size_t len) {
    uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE,
             h3 = 0x10325476, h4 = 0xC3D2E1F0;
    // Padding
    std::vector<uint8_t> msg(data, data + len);
    const uint64_t bitlen = uint64_t(len) * 8;
    msg.push_back(0x80);
    while (msg.size() % 64 != 56) msg.push_back(0x00);
    for (int i = 7; i >= 0; --i) msg.push_back(uint8_t((bitlen >> (i * 8)) & 0xFF));

    auto rol = [](uint32_t x, int c) { return (x << c) | (x >> (32 - c)); };
    for (size_t off = 0; off < msg.size(); off += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(msg[off + i * 4]) << 24) | (uint32_t(msg[off + i * 4 + 1]) << 16) |
                   (uint32_t(msg[off + i * 4 + 2]) << 8) | uint32_t(msg[off + i * 4 + 3]);
        for (int i = 16; i < 80; ++i) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
        for (int i = 0; i < 80; ++i) {
            uint32_t f, k;
            if (i < 20)      { f = (b & c) | ((~b) & d);          k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d;                     k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d);   k = 0x8F1BBCDC; }
            else             { f = b ^ c ^ d;                     k = 0xCA62C1D6; }
            const uint32_t t = rol(a, 5) + f + e + k + w[i];
            e = d; d = c; c = rol(b, 30); b = a; a = t;
        }
        h0 += a; h1 += b; h2 += c; h3 += d; h4 += e;
    }
    std::vector<uint8_t> out(20);
    const uint32_t hs[5] = {h0, h1, h2, h3, h4};
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 4; ++j) out[i * 4 + j] = uint8_t((hs[i] >> ((3 - j) * 8)) & 0xFF);
    return out;
}

std::string base64_encode(const uint8_t* data, size_t len) {
    static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((len + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 3 <= len; i += 3) {
        const uint32_t n = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8) | data[i + 2];
        out.push_back(T[(n >> 18) & 63]); out.push_back(T[(n >> 12) & 63]);
        out.push_back(T[(n >> 6) & 63]);  out.push_back(T[n & 63]);
    }
    if (len - i == 1) {
        const uint32_t n = uint32_t(data[i]) << 16;
        out.push_back(T[(n >> 18) & 63]); out.push_back(T[(n >> 12) & 63]);
        out.push_back('='); out.push_back('=');
    } else if (len - i == 2) {
        const uint32_t n = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8);
        out.push_back(T[(n >> 18) & 63]); out.push_back(T[(n >> 12) & 63]);
        out.push_back(T[(n >> 6) & 63]); out.push_back('=');
    }
    return out;
}

std::string ws_accept_key(const std::string& client_key) {
    static const char* GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string s = client_key + GUID;
    auto d = sha1(reinterpret_cast<const uint8_t*>(s.data()), s.size());
    return base64_encode(d.data(), d.size());
}

// ---- Socket-I/O -----------------------------------------------------------
bool WsConn::send_all(const uint8_t* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        const int n = ::send(int(sock_), reinterpret_cast<const char*>(data + sent),
                             int(len - sent), 0);
        if (n <= 0) return false;
        sent += size_t(n);
    }
    return true;
}

bool WsConn::fill(size_t need) {
    uint8_t tmp[4096];
    while (buf_.size() < need) {
        const int n = ::recv(int(sock_), reinterpret_cast<char*>(tmp), int(sizeof(tmp)), 0);
        if (n <= 0) return false;
        buf_.insert(buf_.end(), tmp, tmp + n);
    }
    return true;
}

static std::vector<uint8_t> encode_frame(const std::string& payload, uint8_t opcode, bool mask) {
    std::vector<uint8_t> f;
    f.push_back(uint8_t(0x80 | opcode));  // FIN + opcode
    const size_t len = payload.size();
    uint8_t b1 = mask ? 0x80 : 0x00;
    if (len < 126) { f.push_back(b1 | uint8_t(len)); }
    else if (len <= 0xFFFF) {
        f.push_back(b1 | 126);
        f.push_back(uint8_t((len >> 8) & 0xFF)); f.push_back(uint8_t(len & 0xFF));
    } else {
        f.push_back(b1 | 127);
        for (int i = 7; i >= 0; --i) f.push_back(uint8_t((uint64_t(len) >> (i * 8)) & 0xFF));
    }
    uint8_t mk[4] = {0, 0, 0, 0};
    if (mask) {
        // Deterministischer, aber pro-Frame variierender Maskenschlüssel.
        static uint32_t seed = 0x12345678u;
        seed = seed * 1664525u + 1013904223u;
        for (int i = 0; i < 4; ++i) mk[i] = uint8_t((seed >> (i * 8)) & 0xFF);
        f.insert(f.end(), mk, mk + 4);
    }
    for (size_t i = 0; i < len; ++i)
        f.push_back(mask ? uint8_t(payload[i]) ^ mk[i & 3] : uint8_t(payload[i]));
    return f;
}

bool WsConn::send_text(const std::string& payload) {
    const auto f = encode_frame(payload, 0x1, /*mask=*/!server_);
    return send_all(f.data(), f.size());
}

bool WsConn::recv_text(std::string& out) {
    out.clear();
    for (;;) {
        if (!fill(2)) return false;
        const bool fin = (buf_[0] & 0x80) != 0;
        const uint8_t opcode = buf_[0] & 0x0F;
        const bool masked = (buf_[1] & 0x80) != 0;
        uint64_t len = buf_[1] & 0x7F;
        size_t hdr = 2;
        if (len == 126) { if (!fill(4)) return false; len = (uint64_t(buf_[2]) << 8) | buf_[3]; hdr = 4; }
        else if (len == 127) {
            if (!fill(10)) return false;
            len = 0; for (int i = 0; i < 8; ++i) len = (len << 8) | buf_[2 + i]; hdr = 10;
        }
        uint8_t mk[4] = {0, 0, 0, 0};
        if (masked) { if (!fill(hdr + 4)) return false;
                      for (int i = 0; i < 4; ++i) mk[i] = buf_[hdr + i]; hdr += 4; }
        if (!fill(hdr + size_t(len))) return false;

        std::string payload(size_t(len), '\0');
        for (uint64_t i = 0; i < len; ++i)
            payload[size_t(i)] = char(masked ? buf_[hdr + size_t(i)] ^ mk[i & 3] : buf_[hdr + size_t(i)]);
        buf_.erase(buf_.begin(), buf_.begin() + hdr + size_t(len));

        if (opcode == 0x8) { return false; }              // Close
        if (opcode == 0x9) {                              // Ping -> Pong
            const auto pong = encode_frame(payload, 0xA, !server_);
            send_all(pong.data(), pong.size());
            continue;
        }
        if (opcode == 0xA) continue;                      // Pong ignorieren
        // Text (0x1) oder Continuation (0x0)
        out += payload;
        if (fin) return true;
    }
}

void WsConn::close() {
    if (!is_valid(sock_)) return;
    const auto f = encode_frame("", 0x8, !server_);
    send_all(f.data(), f.size());
#ifdef _WIN32
    ::closesocket(int(sock_));
#endif
    sock_ = invalid_socket();
}

// ---- Client-Connect -------------------------------------------------------
static bool recv_until(Socket s, const std::string& marker, std::string& out) {
    char ch;
    while (out.find(marker) == std::string::npos) {
        const int n = ::recv(int(s), &ch, 1, 0);
        if (n <= 0) return false;
        out.push_back(ch);
        if (out.size() > 65536) return false;
    }
    return true;
}

bool ws_client_connect(const std::string& host, int port, const std::string& path,
                       WsConn& out, std::string* err) {
    if (!net_init(err)) return false;
#ifdef _WIN32
    Socket s = Socket(::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
    if (!is_valid(s)) { if (err) *err = "socket() fehlgeschlagen"; return false; }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(uint16_t(port));
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        if (err) *err = "inet_pton fehlgeschlagen: " + host;
        ::closesocket(int(s)); return false;
    }
    if (::connect(int(s), reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        if (err) *err = "connect fehlgeschlagen";
        ::closesocket(int(s)); return false;
    }

    // Statischer Client-Key (Test) — Server akzeptiert jeden gültigen Key.
    const std::string key = "dGhlIHNhbXBsZSBub25jZQ==";
    std::string req = "GET " + path + " HTTP/1.1\r\n"
                      "Host: " + host + ":" + std::to_string(port) + "\r\n"
                      "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                      "Sec-WebSocket-Key: " + key + "\r\nSec-WebSocket-Version: 13\r\n\r\n";
    if (::send(int(s), req.data(), int(req.size()), 0) <= 0) {
        if (err) *err = "send handshake fehlgeschlagen"; ::closesocket(int(s)); return false;
    }
    std::string resp;
    if (!recv_until(s, "\r\n\r\n", resp)) { if (err) *err = "kein Handshake-Response"; ::closesocket(int(s)); return false; }
    if (resp.find(" 101 ") == std::string::npos) { if (err) *err = "kein 101 Upgrade"; ::closesocket(int(s)); return false; }

    out = WsConn(s, /*is_server=*/false);
    return true;
#else
    (void)host; (void)port; (void)path; (void)out;
    if (err) *err = "nur Windows";
    return false;
#endif
}

}  // namespace nova::web

// bridge.cpp — Implementierung von bridge.h (Aufgabe 6.2).
#include "Tunnel/bridge.h"

#include <cstdlib>
#include <sstream>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

namespace nova::tunnel {

#ifdef _WIN32
BridgeResponse bridge_get(const std::string& host, int port, const std::string& path,
                          std::string* err) {
    BridgeResponse r;
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) { if (err) *err = "WSAStartup"; return r; }

    SOCKET s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(uint16_t(port));
    ::inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

    if (s == INVALID_SOCKET || ::connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        if (err) *err = "connect zu nova4 fehlgeschlagen";
        if (s != INVALID_SOCKET) ::closesocket(s);
        WSACleanup();
        return r;
    }

    std::ostringstream req;
    req << "GET " << path << " HTTP/1.1\r\n"
        << "Host: " << host << ":" << port << "\r\n"
        << "Connection: close\r\n\r\n";
    const std::string rs = req.str();
    ::send(s, rs.data(), int(rs.size()), 0);

    std::string resp;
    char buf[4096];
    for (;;) {
        const int n = ::recv(s, buf, sizeof(buf), 0);
        if (n <= 0) break;
        resp.append(buf, size_t(n));
    }
    ::closesocket(s);
    WSACleanup();

    // Status + Body extrahieren.
    const size_t sp1 = resp.find(' ');
    if (sp1 != std::string::npos) r.status = std::atoi(resp.c_str() + sp1 + 1);
    const size_t hdr_end = resp.find("\r\n\r\n");
    if (hdr_end != std::string::npos) r.body = resp.substr(hdr_end + 4);
    r.ok = (r.status >= 200 && r.status < 400);
    return r;
}
#else
BridgeResponse bridge_get(const std::string&, int, const std::string&, std::string* err) {
    BridgeResponse r; if (err) *err = "nur Windows"; return r;
}
#endif

}  // namespace nova::tunnel

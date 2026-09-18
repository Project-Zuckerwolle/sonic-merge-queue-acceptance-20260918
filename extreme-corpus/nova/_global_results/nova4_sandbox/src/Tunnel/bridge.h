// bridge.h — HTTP-Bridge zu nova4.exe (Aufgabe 6.2, TT5).
//
// nova-tunnel.exe entschlüsselt (TLS + NOISE) und leitet die reine App-Anfrage
// an das lokale nova4.exe (127.0.0.1:8000) weiter — nova4 sieht nur normale
// localhost-HTTP-Requests und weiß nichts von TLS/NOISE (Design §Multi-Session).
#pragma once

#include <string>

namespace nova::tunnel {

struct BridgeResponse {
    int         status = 0;
    std::string body;
    bool        ok = false;
};

// Minimaler localhost-HTTP-GET an nova4.exe. Blockierend, Connection: close.
BridgeResponse bridge_get(const std::string& host, int port, const std::string& path,
                          std::string* err = nullptr);

}  // namespace nova::tunnel

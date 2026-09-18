// tls_server.h — TLS-1.3/1.2 (Schannel) Serve-Loop des NovaTunnel (Aufgabe 6, TT1/2/5/8).
//
// Kette pro Verbindung:
//   accept -> Schannel-TLS-Handshake -> WS-Upgrade -> NOISE-XX (Responder) ->
//   Geräte-Whitelist -> Transport: decrypt -> nova4:/ws (ws_client_connect) ->
//   Token-Stream zurück, jeder als NOISE-verschlüsselter WS-Frame (Streaming).
//
// Selbst-signiertes Zertifikat wird bei Bedarf zur Laufzeit erzeugt (der Client
// pinnt den NOISE-Static-Key, nicht das TLS-Zert — self-signed ist erwartet, TT2).
#pragma once

#include "Tunnel/noise_xx.h"

#include <string>

namespace nova::tunnel {

struct TlsServerConfig {
    std::string bind        = "0.0.0.0";
    int         port        = 443;
    std::string nova4_host  = "127.0.0.1";
    int         nova4_port  = 8000;
    std::string root;                 // %ProgramData%\NovaTunnel
    std::string cert_cn     = "nova-live.duckdns.org";
    std::string enroll_password;      // Passwort-Enrollment (POST /enroll); leer = deaktiviert
    Key32       static_priv{};
    Key32       static_pub{};
};

// Blockiert bis zum Abbruch. Rückgabe = Prozess-Exit-Code (0 = sauber gestoppt).
int run_tls_server(const TlsServerConfig& cfg, std::string* err = nullptr);

}  // namespace nova::tunnel

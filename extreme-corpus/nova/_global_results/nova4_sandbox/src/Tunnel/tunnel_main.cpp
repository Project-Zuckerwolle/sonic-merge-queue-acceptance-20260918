// tunnel_main.cpp — nova-tunnel.exe Entry-Point + CLI (Aufgabe 6.4).
//
// CLI (Design §NovaTunnel):
//   --setup                 Server-Static-Key erzeugen (DPAPI), config.json anlegen
//   --show-fingerprint      SHA-256 des TLS-Zertifikats (für client.js PINNED_SHA256)
//   --add-device "Name"     Geräte-Credential erzeugen, Public Key whitelisten
//   --list-devices          alle Geräte
//   --revoke-device "Name"  Gerät sperren
//   --revoke-all            Panic: alle sperren
//   --install-service       Windows-Dienst (sc-Befehl ausgeben)
//
// Der eigentliche TLS+NOISE-Serve-Loop ist der Deploy-Schritt (Schannel + Browser,
// siehe uebergabe.md); die Management-Kommandos hier sind vollständig funktionsfähig.
#include "Tunnel/bridge.h"
#include "Tunnel/device_store.h"
#include "Tunnel/dpapi.h"
#include "Tunnel/file_transfer.h"
#include "Tunnel/monocypher_backend.h"
#include "Tunnel/tls_server.h"
#include "Web/settings.h"
#include "Web/websocket.h"   // base64_encode

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;
using namespace nova;

namespace {

std::string root_dir() {
    if (const char* pd = std::getenv("ProgramData")) return std::string(pd) + "\\NovaTunnel";
    return "NovaTunnel";
}
std::string b64(const std::vector<uint8_t>& v) {
    return web::base64_encode(v.data(), v.size());
}
std::string arg_after(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc - 1; ++i) if (std::string(argv[i]) == key) return argv[i + 1];
    return {};
}
bool has(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == key) return true;
    return false;
}
std::string fingerprint_colons(const std::string& hex) {
    std::string o;
    for (size_t i = 0; i < hex.size(); i += 2) {
        if (!o.empty()) o += ':';
        o += char(std::toupper(hex[i]));
        if (i + 1 < hex.size()) o += char(std::toupper(hex[i + 1]));
    }
    return o;
}

int cmd_setup(const std::string& root) {
    fs::create_directories(fs::path(root) / "keys");
    fs::create_directories(fs::path(root) / "certs");
    fs::create_directories(fs::path(root) / "logs");

    // Server-Static-Keypair (X25519) erzeugen, Private via DPAPI schützen.
    tunnel::Key32 priv, pub;
    tunnel::monocypher_backend().keypair(priv, pub);
    std::vector<uint8_t> priv_v(priv.begin(), priv.end()), enc;
    std::string err;
    if (!tunnel::dpapi_protect(priv_v, enc, &err)) { std::printf("DPAPI: %s\n", err.c_str()); return 1; }
    { std::ofstream f(fs::path(root) / "keys" / "server_static.key", std::ios::binary | std::ios::trunc);
      f.write(reinterpret_cast<const char*>(enc.data()), std::streamsize(enc.size())); }

    // config.json (Ports + Domain-Platzhalter).
    { std::ofstream f(fs::path(root) / "config.json", std::ios::trunc);
      f << "{\n  \"port_https\": 443,\n  \"nova4_port\": 8000,\n"
           "  \"duckdns_domain\": \"nova-live.duckdns.org\",\n"
           "  \"server_static_pub_b64\": \"" << b64(std::vector<uint8_t>(pub.begin(), pub.end())) << "\"\n}\n"; }

    // Leere Whitelist anlegen falls nicht vorhanden.
    tunnel::DeviceStore ds((fs::path(root) / "devices.json").string());
    ds.load(); ds.save();

    std::printf("Setup OK -> %s\n  server_static (Public): %s\n  config.json + leere devices.json angelegt.\n"
                "  Naechster Schritt: TLS-Zertifikat nach certs/tunnel.crt (siehe uebergabe.md).\n",
                root.c_str(), b64(std::vector<uint8_t>(pub.begin(), pub.end())).c_str());
    return 0;
}

int cmd_show_fingerprint(const std::string& root) {
    const fs::path crt = fs::path(root) / "certs" / "tunnel.crt";
    std::ifstream f(crt, std::ios::binary);
    if (!f) { std::printf("Kein Zertifikat: %s\n  (erst mit openssl erzeugen, siehe uebergabe.md)\n", crt.string().c_str()); return 1; }
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const std::string fp = fingerprint_colons(tunnel::sha256_hex(data.data(), data.size()));
    std::printf("SHA-256 Fingerprint (fuer client.js PINNED_SHA256):\n%s\n", fp.c_str());
    return 0;
}

int cmd_add_device(const std::string& root, const std::string& name) {
    tunnel::DeviceStore ds((fs::path(root) / "devices.json").string());
    ds.load();
    tunnel::Key32 priv, pub;
    tunnel::monocypher_backend().keypair(priv, pub);
    const std::string pub_b64 = b64(std::vector<uint8_t>(pub.begin(), pub.end()));
    const std::string priv_b64 = b64(std::vector<uint8_t>(priv.begin(), priv.end()));
    std::string err;
    if (!ds.add(name, pub_b64, /*now*/ 0, &err) || !ds.save(&err)) { std::printf("Fehler: %s\n", err.c_str()); return 1; }
    // Server-Static + Domain aus config.json mitliefern: die Android-/Browser-App
    // braucht genau diese drei Werte (Domain, Server-Static, Credential) — hier
    // gebuendelt, damit ein Kommando pro Geraet reicht.
    web::Settings s; s.load((fs::path(root) / "config.json").string());
    const std::string domain = s.get("duckdns_domain", "nova-live.duckdns.org");
    // Server-Static robust aus dem DPAPI-Key-File ableiten (wie load_serve_config),
    // nicht auf config.json verlassen — die App braucht ihn zwingend zum Pinnen.
    std::string server_pub = s.get("server_static_pub_b64", "");
    {
        std::ifstream kf(fs::path(root) / "keys" / "server_static.key", std::ios::binary);
        std::vector<uint8_t> enc((std::istreambuf_iterator<char>(kf)), std::istreambuf_iterator<char>()), priv_raw;
        if (tunnel::dpapi_unprotect(enc, priv_raw) && priv_raw.size() >= 32) {
            tunnel::Key32 sp; std::copy_n(priv_raw.begin(), 32, sp.begin());
            const tunnel::Key32 pub_k = tunnel::x25519_public(sp);
            server_pub = b64(std::vector<uint8_t>(pub_k.begin(), pub_k.end()));
        }
    }
    std::printf("Geraet '%s' registriert.\n  Public (Whitelist): %s\n\n"
                "  === In die Nova-App eintragen (Einstellungen) ===\n"
                "  DuckDNS-Domain     : %s\n"
                "  Server-Static (B64): %s\n"
                "  Credential (B64)   : %s   <-- GEHEIM, nur dieses Geraet\n\n"
                "  Als QR: obigen Credential-String in einen QR-Generator geben.\n",
                name.c_str(), pub_b64.c_str(), domain.c_str(),
                server_pub.empty() ? "(fehlt in config.json — 'nova-tunnel --setup' erneut)" : server_pub.c_str(),
                priv_b64.c_str());
    return 0;
}

int cmd_list(const std::string& root) {
    tunnel::DeviceStore ds((fs::path(root) / "devices.json").string());
    ds.load();
    std::printf("%zu Geraet(e):\n", ds.size());
    for (const auto& d : ds.list())
        std::printf("  - %-20s pub=%s\n", d.name.c_str(), d.public_key_b64.c_str());
    return 0;
}

int cmd_revoke(const std::string& root, const std::string& name, bool all) {
    tunnel::DeviceStore ds((fs::path(root) / "devices.json").string());
    ds.load();
    if (all) { ds.revoke_all(); ds.save(); std::printf("Alle Geraete gesperrt (Panic).\n"); return 0; }
    const bool ok = ds.revoke_by_name(name);
    ds.save();
    std::printf(ok ? "Geraet '%s' gesperrt.\n" : "Geraet '%s' nicht gefunden.\n", name.c_str());
    return ok ? 0 : 1;
}

// Baut die TlsServerConfig aus config.json + DPAPI-Static-Key.
tunnel::TlsServerConfig load_serve_config(const std::string& root) {
    tunnel::TlsServerConfig cfg; cfg.root = root;
    web::Settings s; s.load((fs::path(root) / "config.json").string());
    cfg.port = s.get_int("port_https", 443);
    cfg.nova4_port = s.get_int("nova4_port", 8000);
    cfg.cert_cn = s.get("duckdns_domain", "nova-live.duckdns.org");
    cfg.enroll_password = s.get("enroll_password", "");   // leer = Enrollment aus
    std::ifstream f(fs::path(root) / "keys" / "server_static.key", std::ios::binary);
    std::vector<uint8_t> enc((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()), priv;
    if (tunnel::dpapi_unprotect(enc, priv) && priv.size() >= 32) {
        std::copy_n(priv.begin(), 32, cfg.static_priv.begin());
        cfg.static_pub = tunnel::x25519_public(cfg.static_priv);
    }
    return cfg;
}

#ifdef _WIN32
static SERVICE_STATUS_HANDLE g_ssh; static SERVICE_STATUS g_ss{}; static std::string g_root;
void WINAPI svc_ctrl(DWORD c) {
    if (c == SERVICE_CONTROL_STOP || c == SERVICE_CONTROL_SHUTDOWN) {
        g_ss.dwCurrentState = SERVICE_STOPPED; SetServiceStatus(g_ssh, &g_ss);
    }
}
void WINAPI svc_main(DWORD, LPWSTR*) {
    g_ssh = RegisterServiceCtrlHandlerW(L"NovaTunnel", svc_ctrl);
    g_ss.dwServiceType = SERVICE_WIN32_OWN_PROCESS; g_ss.dwCurrentState = SERVICE_RUNNING;
    g_ss.dwControlsAccepted = SERVICE_ACCEPT_STOP | SERVICE_ACCEPT_SHUTDOWN;
    SetServiceStatus(g_ssh, &g_ss);
    std::string err; tunnel::run_tls_server(load_serve_config(g_root), &err);
    g_ss.dwCurrentState = SERVICE_STOPPED; SetServiceStatus(g_ssh, &g_ss);
}
int cmd_install_service() {
    wchar_t path[MAX_PATH]; GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring bin = std::wstring(L"\"") + path + L"\" --service-run";
    SC_HANDLE scm = OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CREATE_SERVICE);
    if (!scm) { std::printf("OpenSCManager fehlgeschlagen (als Admin ausfuehren).\n"); return 1; }
    SC_HANDLE svc = CreateServiceW(scm, L"NovaTunnel", L"Nova 4 Secure Tunnel", SERVICE_ALL_ACCESS,
        SERVICE_WIN32_OWN_PROCESS, SERVICE_AUTO_START, SERVICE_ERROR_NORMAL, bin.c_str(),
        nullptr, nullptr, nullptr, nullptr, nullptr);
    if (!svc) { const DWORD e = GetLastError();
        std::printf("CreateService fehlgeschlagen (%lu)%s\n", e, e == 1073 ? " - existiert bereits" : "");
        CloseServiceHandle(scm); return 1; }
    std::printf("Dienst 'NovaTunnel' installiert (Autostart -> --service-run).\n");
    CloseServiceHandle(svc); CloseServiceHandle(scm); return 0;
}
int cmd_uninstall_service() {
    SC_HANDLE scm = OpenSCManagerW(nullptr, nullptr, SC_MANAGER_ALL_ACCESS);
    if (!scm) return 1;
    if (SC_HANDLE svc = OpenServiceW(scm, L"NovaTunnel", DELETE)) {
        DeleteService(svc); CloseServiceHandle(svc); std::printf("Dienst 'NovaTunnel' entfernt.\n");
    }
    CloseServiceHandle(scm); return 0;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    const std::string root = root_dir();

    if (has(argc, argv, "--setup"))            return cmd_setup(root);
    if (has(argc, argv, "--show-fingerprint")) return cmd_show_fingerprint(root);
    if (has(argc, argv, "--add-device"))       return cmd_add_device(root, arg_after(argc, argv, "--add-device"));
    if (has(argc, argv, "--list-devices"))     return cmd_list(root);
    if (has(argc, argv, "--revoke-device"))    return cmd_revoke(root, arg_after(argc, argv, "--revoke-device"), false);
    if (has(argc, argv, "--revoke-all"))       return cmd_revoke(root, "", true);

#ifdef _WIN32
    if (has(argc, argv, "--install-service"))   return cmd_install_service();
    if (has(argc, argv, "--uninstall-service")) return cmd_uninstall_service();
    if (has(argc, argv, "--service-run")) {
        g_root = root;
        SERVICE_TABLE_ENTRYW t[] = { {const_cast<LPWSTR>(L"NovaTunnel"), svc_main}, {nullptr, nullptr} };
        StartServiceCtrlDispatcherW(t);
        return 0;
    }
#endif
    if (has(argc, argv, "--serve")) {
        std::string e;
        const int rc = tunnel::run_tls_server(load_serve_config(root), &e);
        if (rc != 0) std::printf("serve: %s\n", e.c_str());
        return rc;
    }

    std::printf("nova-tunnel.exe — Nova 4 Secure Tunnel\n"
                "  --setup | --show-fingerprint | --add-device \"Name\" | --list-devices\n"
                "  --revoke-device \"Name\" | --revoke-all | --install-service | --uninstall-service\n");
    return 0;
}

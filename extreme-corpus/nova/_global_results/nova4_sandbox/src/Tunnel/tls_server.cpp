// tls_server.cpp — Implementierung von tls_server.h (Schannel + WS + NOISE + Relay).
#include "Tunnel/tls_server.h"

#include "Tunnel/device_store.h"
#include "Tunnel/monocypher_backend.h"
#include "Tunnel/rate_limiter.h"
#include "Web/websocket.h"   // ws_client_connect, WsConn, base64_encode, sha1/ws_accept_key

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#define SECURITY_WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <wincrypt.h>
#include <ncrypt.h>
#include <schannel.h>
#include <security.h>
#include <sspi.h>
#pragma comment(lib, "secur32.lib")
#pragma comment(lib, "crypt32.lib")
#pragma comment(lib, "ncrypt.lib")
#pragma comment(lib, "ws2_32.lib")
#endif

namespace nova::tunnel {

#ifdef _WIN32
namespace {

std::string b64(const std::vector<uint8_t>& v) { return web::base64_encode(v.data(), v.size()); }

// ---- Zertifikat aus dem Windows-Cert-Store (LocalMachine\My) laden ----------
// Diese Maschine haengt bei Key-ERZEUGUNG/-IMPORT im C++-Prozess (CryptoAPI UND
// CNG, PFX-Import). Loesung: das Cert wird EINMAL via PowerShell erstellt
// (New-SelfSignedCertificate -> LocalMachine\My, KSP funktioniert dort) und hier
// nur GEOEFFNET — C++ erzeugt/importiert keinen Key -> kein Hang. Schannel nutzt
// den persistierten Key des Store-Kontexts direkt.
PCCERT_CONTEXT load_store_cert(const std::wstring& subject_cn) {
    HCERTSTORE store = CertOpenStore(CERT_STORE_PROV_SYSTEM_W, 0, 0,
        CERT_SYSTEM_STORE_LOCAL_MACHINE | CERT_STORE_READONLY_FLAG, L"MY");
    if (!store) { std::fprintf(stderr, "CertOpenStore LocalMachine\\My fail 0x%lx\n", GetLastError()); return nullptr; }
    PCCERT_CONTEXT c = CertFindCertificateInStore(store, X509_ASN_ENCODING | PKCS_7_ASN_ENCODING, 0,
        CERT_FIND_SUBJECT_STR_W, subject_cn.c_str(), nullptr);
    if (!c) std::fprintf(stderr, "Cert '%ls' nicht in LocalMachine\\My — erst via New-SelfSignedCertificate anlegen.\n",
                         subject_cn.c_str());
    return c;  // Store bewusst offen lassen (Key haengt am Kontext)
}

bool acquire_server_creds(PCCERT_CONTEXT cert, CredHandle* out) {
    SCHANNEL_CRED sc{};
    sc.dwVersion = SCHANNEL_CRED_VERSION;
    sc.cCreds = 1;
    sc.paCred = &cert;
    sc.dwFlags = SCH_USE_STRONG_CRYPTO;
    TimeStamp ts;
    const SECURITY_STATUS st = AcquireCredentialsHandleW(
        nullptr, const_cast<LPWSTR>(UNISP_NAME_W), SECPKG_CRED_INBOUND,
        nullptr, &sc, nullptr, nullptr, out, &ts);
    if (st != SEC_E_OK) std::fprintf(stderr, "AcquireCredentialsHandle fail 0x%lx\n", (unsigned long)st);
    return st == SEC_E_OK;
}

// ---- Roh-Socket-Helfer ----------------------------------------------------
int raw_recv(SOCKET s, char* buf, int n) { return ::recv(s, buf, n, 0); }
bool raw_send(SOCKET s, const char* buf, int n) {
    int sent = 0;
    while (sent < n) { int k = ::send(s, buf + sent, n - sent, 0); if (k <= 0) return false; sent += k; }
    return true;
}

// ---- TLS-Verbindung: Handshake + Stream-I/O ------------------------------
struct TlsConn {
    SOCKET      sock = INVALID_SOCKET;
    CtxtHandle  ctx{};
    SecPkgContext_StreamSizes sizes{};
    std::string enc;   // gelesene, noch nicht entschlüsselte Bytes
    std::string dec;   // entschlüsselte, noch nicht konsumierte Bytes
    bool        ok = false;

    bool handshake(CredHandle* cred) {
        DWORD req = ASC_REQ_SEQUENCE_DETECT | ASC_REQ_REPLAY_DETECT | ASC_REQ_CONFIDENTIALITY |
                    ASC_REQ_EXTENDED_ERROR | ASC_REQ_ALLOCATE_MEMORY | ASC_REQ_STREAM;
        bool ctx_new = false;
        for (;;) {
            // Mehr Bytes lesen falls nötig.
            if (enc.empty()) {
                char tmp[8192]; int r = raw_recv(sock, tmp, sizeof(tmp));
                if (r <= 0) return false;
                enc.append(tmp, size_t(r));
            }
            SecBuffer in[2]{};
            in[0].BufferType = SECBUFFER_TOKEN; in[0].pvBuffer = enc.data(); in[0].cbBuffer = ULONG(enc.size());
            in[1].BufferType = SECBUFFER_EMPTY;
            SecBufferDesc ind{SECBUFFER_VERSION, 2, in};
            SecBuffer out[1]{}; out[0].BufferType = SECBUFFER_TOKEN;
            SecBufferDesc outd{SECBUFFER_VERSION, 1, out};
            DWORD attr = 0; TimeStamp ts;
            SECURITY_STATUS st = AcceptSecurityContext(
                cred, ctx_new ? &ctx : nullptr, &ind, req, 0,
                ctx_new ? nullptr : &ctx, &outd, &attr, &ts);
            ctx_new = true;
            if (out[0].pvBuffer && out[0].cbBuffer) {
                raw_send(sock, static_cast<char*>(out[0].pvBuffer), int(out[0].cbBuffer));
                FreeContextBuffer(out[0].pvBuffer);
            }
            if (st == SEC_E_INCOMPLETE_MESSAGE) {
                char tmp[8192]; int r = raw_recv(sock, tmp, sizeof(tmp));
                if (r <= 0) return false; enc.append(tmp, size_t(r)); continue;
            }
            if (st == SEC_I_CONTINUE_NEEDED || st == SEC_E_OK) {
                // Verbrauchte Eingabe entfernen (extra = unverbraucht).
                if (in[1].BufferType == SECBUFFER_EXTRA)
                    enc = enc.substr(enc.size() - in[1].cbBuffer);
                else
                    enc.clear();
                if (st == SEC_E_OK) {
                    QueryContextAttributes(&ctx, SECPKG_ATTR_STREAM_SIZES, &sizes);
                    ok = true; return true;
                }
                continue;
            }
            return false;  // Fehler
        }
    }

    // Liefert bis zu `want` entschlüsselte Bytes (blockierend).
    bool recv_plain(std::string& out, size_t want) {
        out.clear();
        while (out.size() < want) {
            if (!dec.empty()) {
                const size_t take = (dec.size() < want - out.size()) ? dec.size() : (want - out.size());
                out.append(dec, 0, take); dec.erase(0, take); continue;
            }
            if (!decrypt_more()) return false;
        }
        return true;
    }
    bool decrypt_more() {
        for (;;) {
            if (!enc.empty()) {
                std::vector<char> work(enc.begin(), enc.end());
                SecBuffer b[4]{};
                b[0].BufferType = SECBUFFER_DATA; b[0].pvBuffer = work.data(); b[0].cbBuffer = ULONG(work.size());
                b[1].BufferType = SECBUFFER_EMPTY; b[2].BufferType = SECBUFFER_EMPTY; b[3].BufferType = SECBUFFER_EMPTY;
                SecBufferDesc d{SECBUFFER_VERSION, 4, b};
                SECURITY_STATUS st = DecryptMessage(&ctx, &d, 0, nullptr);
                if (st == SEC_E_OK) {
                    size_t extra = 0;
                    for (int i = 0; i < 4; ++i) {
                        if (b[i].BufferType == SECBUFFER_DATA && b[i].cbBuffer)
                            dec.append(static_cast<char*>(b[i].pvBuffer), b[i].cbBuffer);
                        if (b[i].BufferType == SECBUFFER_EXTRA) extra = b[i].cbBuffer;
                    }
                    enc = extra ? enc.substr(enc.size() - extra) : std::string();
                    if (!dec.empty()) return true;
                    continue;
                }
                if (st == SEC_I_CONTEXT_EXPIRED) return false;
                if (st != SEC_E_INCOMPLETE_MESSAGE) return false;
            }
            char tmp[8192]; int r = raw_recv(sock, tmp, sizeof(tmp));
            if (r <= 0) return false; enc.append(tmp, size_t(r));
        }
    }

    bool send_plain(const char* data, size_t len) {
        const size_t chunk = sizes.cbMaximumMessage ? sizes.cbMaximumMessage : 16384;
        std::vector<char> buf(sizes.cbHeader + chunk + sizes.cbTrailer);
        size_t off = 0;
        while (off < len) {
            const size_t n = (len - off < chunk) ? (len - off) : chunk;
            memcpy(buf.data() + sizes.cbHeader, data + off, n);
            SecBuffer b[4]{};
            b[0].BufferType = SECBUFFER_STREAM_HEADER; b[0].pvBuffer = buf.data(); b[0].cbBuffer = sizes.cbHeader;
            b[1].BufferType = SECBUFFER_DATA; b[1].pvBuffer = buf.data() + sizes.cbHeader; b[1].cbBuffer = ULONG(n);
            b[2].BufferType = SECBUFFER_STREAM_TRAILER; b[2].pvBuffer = buf.data() + sizes.cbHeader + n; b[2].cbBuffer = sizes.cbTrailer;
            b[3].BufferType = SECBUFFER_EMPTY;
            SecBufferDesc d{SECBUFFER_VERSION, 4, b};
            if (EncryptMessage(&ctx, 0, &d, 0) != SEC_E_OK) return false;
            const int total = int(b[0].cbBuffer + b[1].cbBuffer + b[2].cbBuffer);
            if (!raw_send(sock, buf.data(), total)) return false;
            off += n;
        }
        return true;
    }
    void shutdown_close() {
        DeleteSecurityContext(&ctx);
        if (sock != INVALID_SOCKET) { closesocket(sock); sock = INVALID_SOCKET; }
    }
};

// ---- WS-Framing über TlsConn ---------------------------------------------
bool ws_read_line_block(TlsConn& t, std::string& req) {
    req.clear(); std::string ch;
    while (req.find("\r\n\r\n") == std::string::npos) {
        if (!t.recv_plain(ch, 1)) return false;
        req += ch; if (req.size() > 65536) return false;
    }
    return true;
}
std::string header_val(const std::string& req, const std::string& key) {
    auto low = [](std::string s){ for(auto&c:s)c=char(::tolower((unsigned char)c)); return s; };
    const std::string k = low(key); size_t p = 0;
    std::string r = req;
    std::string ll = low(r);
    size_t pos = ll.find(k + ":");
    if (pos == std::string::npos) return {};
    size_t s = pos + k.size() + 1; while (s < r.size() && (r[s]==' '||r[s]=='\t')) ++s;
    size_t e = r.find("\r\n", s); return r.substr(s, e - s);
    (void)p;
}
// Ein WS-Frame lesen (Client->Server, maskiert). Liefert Payload (binär) in out.
bool ws_recv_frame(TlsConn& t, std::string& out) {
    std::string h;
    if (!t.recv_plain(h, 2)) return false;
    const uint8_t b0 = uint8_t(h[0]), b1 = uint8_t(h[1]);
    const uint8_t opcode = b0 & 0x0F; const bool masked = (b1 & 0x80) != 0;
    uint64_t len = b1 & 0x7F;
    if (len == 126) { std::string e; if(!t.recv_plain(e,2)) return false; len=(uint64_t(uint8_t(e[0]))<<8)|uint8_t(e[1]); }
    else if (len == 127) { std::string e; if(!t.recv_plain(e,8)) return false; len=0; for(int i=0;i<8;++i) len=(len<<8)|uint8_t(e[i]); }
    uint8_t mk[4]={0,0,0,0};
    if (masked) { std::string m; if(!t.recv_plain(m,4)) return false; for(int i=0;i<4;++i) mk[i]=uint8_t(m[i]); }
    std::string p; if (len && !t.recv_plain(p, size_t(len))) return false;
    if (masked) for (size_t i=0;i<p.size();++i) p[i]=char(uint8_t(p[i])^mk[i&3]);
    if (opcode == 0x8) return false;   // Close
    out.swap(p); return true;
}
// Ein WS-Frame senden (Server->Client, unmaskiert). opcode 0x2 = binär.
bool ws_send_frame(TlsConn& t, const std::string& payload, uint8_t opcode = 0x2) {
    std::string f; f.push_back(char(0x80 | opcode));
    const size_t n = payload.size();
    if (n < 126) f.push_back(char(n));
    else if (n <= 0xFFFF) { f.push_back(char(126)); f.push_back(char((n>>8)&0xFF)); f.push_back(char(n&0xFF)); }
    else { f.push_back(char(127)); for(int i=7;i>=0;--i) f.push_back(char((uint64_t(n)>>(i*8))&0xFF)); }
    f += payload;
    return t.send_plain(f.data(), f.size());
}

std::vector<uint8_t> to_vec(const std::string& s) { return std::vector<uint8_t>(s.begin(), s.end()); }
std::string to_str(const std::vector<uint8_t>& v) { return std::string(v.begin(), v.end()); }

// Minimaler JSON-String-Feld-Extraktor: "key":"value" (nur einfache Werte).
std::string json_field(const std::string& body, const std::string& key) {
    const std::string pat = "\"" + key + "\"";
    size_t k = body.find(pat); if (k == std::string::npos) return {};
    size_t c = body.find(':', k + pat.size()); if (c == std::string::npos) return {};
    size_t q = body.find('"', c + 1); if (q == std::string::npos) return {};
    std::string out;
    for (size_t i = q + 1; i < body.size(); ++i) {
        if (body[i] == '\\' && i + 1 < body.size()) { out += body[++i]; continue; }
        if (body[i] == '"') break;
        out += body[i];
    }
    return out;
}
std::string http_body(const std::string& mime, const std::string& body) {
    return "HTTP/1.1 200 OK\r\nContent-Type: " + mime +
           "\r\nContent-Length: " + std::to_string(body.size()) +
           "\r\nConnection: close\r\n\r\n" + body;
}
std::string read_bin_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

}  // namespace

int run_tls_server(const TlsServerConfig& cfg, std::string* err) {
    if (!web::net_init(err)) return 1;

    // TLS-Init (Cert + Schannel-Creds) in einem DETACHTEN Thread mit Timeout: der
    // Crypto-Key-Service kann in eingeschraenktem Kontext (nicht-Dienst, kein Profil)
    // blockieren. Als SYSTEM-Dienst (--install-service) laeuft es normal.
    // (std::async waere falsch: sein Future-Destruktor wuerde auf den Hang joinen.)
    struct Init { std::atomic<bool> done{false}, ok{false}; PCCERT_CONTEXT cert{nullptr}; CredHandle cred{}; };
    auto in = std::make_shared<Init>();
    const std::wstring wcn(cfg.cert_cn.begin(), cfg.cert_cn.end());
    std::thread([in, wcn]() {
        in->cert = load_store_cert(wcn);
        in->ok = in->cert && acquire_server_creds(in->cert, &in->cred);
        in->done = true;
    }).detach();
    for (int i = 0; i < 200 && !in->done.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (!in->done.load()) {
        std::fprintf(stderr, "NovaTunnel: TLS-Init blockiert (Crypto-Key-Service im aktuellen Kontext).\n"
                     "  Als Dienst starten: nova-tunnel --install-service  (laeuft als SYSTEM, dort OK).\n");
        std::fflush(stderr);
        // Harter Exit: der haengende Crypto-Thread blockiert sonst den normalen Prozess-Exit.
        TerminateProcess(GetCurrentProcess(), 2);
        return 2;
    }
    if (!in->ok.load()) { if (err) *err = "Zertifikat/Schannel-Creds fehlgeschlagen"; return 1; }
    PCCERT_CONTEXT cert = in->cert; CredHandle cred = in->cred;

    SOCKET ls = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    BOOL yes = TRUE; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (char*)&yes, sizeof(yes));
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons(uint16_t(cfg.port));
    inet_pton(AF_INET, cfg.bind.c_str(), &addr.sin_addr);
    if (::bind(ls, (sockaddr*)&addr, sizeof(addr)) != 0) { if(err)*err="bind :"+std::to_string(cfg.port)+" fehlgeschlagen (Admin? Port frei?)"; return 1; }
    if (::listen(ls, SOMAXCONN) != 0) { if(err)*err="listen fehlgeschlagen"; return 1; }

    DeviceStore devices((cfg.root.empty()? std::string("devices.json") : cfg.root + "\\devices.json"));
    devices.load();
    std::mutex dev_mtx;   // serialisiert devices-Zugriff (Enrollment schreibt, NOISE-Pfad liest)
    RateLimiter limiter;
    std::atomic<int64_t> clock{1};   // monotone Pseudo-Zeit (ms) pro Verbindung

    printf("NovaTunnel: TLS-Serve auf %s:%d -> nova4 %s:%d (%zu Geraete)\n",
           cfg.bind.c_str(), cfg.port, cfg.nova4_host.c_str(), cfg.nova4_port, devices.size());

    std::vector<std::thread> workers;
    for (;;) {
        sockaddr_in ca{}; int cl = sizeof(ca);
        SOCKET cs = ::accept(ls, (sockaddr*)&ca, &cl);
        if (cs == INVALID_SOCKET) continue;
        char ipbuf[64]{}; inet_ntop(AF_INET, &ca.sin_addr, ipbuf, sizeof(ipbuf));
        const std::string ip = ipbuf;
        const int64_t now = clock.fetch_add(1000);
        if (limiter.on_connection(ip, now) != RateVerdict::Allow) { closesocket(cs); continue; }

        workers.emplace_back([cs, &cred, &devices, &dev_mtx, &limiter, cfg, ip, now]() {
            TlsConn t; t.sock = cs;
            if (!t.handshake(const_cast<CredHandle*>(&cred))) { closesocket(cs); return; }

            // Request-Header lesen (bis \r\n\r\n).
            std::string req;
            if (!ws_read_line_block(t, req)) { t.shutdown_close(); return; }

            // Methode + Pfad aus der Request-Zeile.
            const size_t sp1 = req.find(' ');
            const size_t sp2 = (sp1 == std::string::npos) ? std::string::npos : req.find(' ', sp1 + 1);
            const std::string method = (sp1 == std::string::npos) ? "" : req.substr(0, sp1);
            const std::string path = (sp2 == std::string::npos) ? "" : req.substr(sp1 + 1, sp2 - sp1 - 1);

            // ---- OTA: unverschlüsselte (TLS) GETs. APK ist kein Geheimnis; die
            // Sicherheit liegt im NOISE-Credential, nicht in der App-Binary. ----
            if (method == "GET" && (path == "/update/version" || path == "/update/app.apk")) {
                const std::string file = (cfg.root.empty() ? std::string() : cfg.root + "\\update\\") +
                                         (path == "/update/version" ? "version.json" : "app.apk");
                const std::string data = read_bin_file(file);
                if (data.empty()) {
                    const std::string r = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                    t.send_plain(r.data(), r.size());
                } else {
                    const std::string resp = http_body(path == "/update/version"
                        ? "application/json" : "application/vnd.android.package-archive", data);
                    t.send_plain(resp.data(), resp.size());
                }
                t.shutdown_close(); return;
            }

            // ---- Passwort-Enrollment: POST /enroll {"password":"..."} ----
            if (method == "POST" && path == "/enroll") {
                size_t clen = 0;
                { const std::string cl = header_val(req, "Content-Length");
                  if (!cl.empty()) clen = size_t(std::strtoul(cl.c_str(), nullptr, 10)); }
                std::string body; if (clen && clen < 8192) t.recv_plain(body, clen);
                const std::string pw = json_field(body, "password");
                if (cfg.enroll_password.empty() || pw.empty() || pw != cfg.enroll_password) {
                    limiter.on_noise_failure(ip, now);   // Fehlversuch -> IP-Sperre nach 5×
                    const std::string r = "HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                    t.send_plain(r.data(), r.size()); t.shutdown_close(); return;
                }
                Key32 priv, pub; monocypher_backend().keypair(priv, pub);
                const std::string pub_b64  = b64(std::vector<uint8_t>(pub.begin(), pub.end()));
                const std::string priv_b64 = b64(std::vector<uint8_t>(priv.begin(), priv.end()));
                { std::lock_guard<std::mutex> lk(dev_mtx);
                  devices.add("app-" + std::to_string(now), pub_b64, 0, nullptr);
                  devices.save(nullptr); }
                const std::string json = "{\"domain\":\"" + cfg.cert_cn +
                    "\",\"server_static\":\"" + b64(std::vector<uint8_t>(cfg.static_pub.begin(), cfg.static_pub.end())) +
                    "\",\"credential\":\"" + priv_b64 + "\"}";
                const std::string resp = http_body("application/json", json);
                t.send_plain(resp.data(), resp.size());
                t.shutdown_close(); return;
            }

            // ---- Sonst: WS-Upgrade (Chat über NOISE). ----
            const std::string wskey = header_val(req, "Sec-WebSocket-Key");
            if (wskey.empty()) { t.shutdown_close(); return; }
            const std::string resp = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                "Connection: Upgrade\r\nSec-WebSocket-Accept: " + web::ws_accept_key(wskey) + "\r\n\r\n";
            if (!t.send_plain(resp.data(), resp.size())) { t.shutdown_close(); return; }

            // NOISE-XX Responder.
            NoiseSession noise(monocypher_backend(), /*initiator=*/false);
            noise.set_static(cfg.static_priv, cfg.static_pub);
            std::string m1; if (!ws_recv_frame(t, m1)) { t.shutdown_close(); return; }
            noise.read_msg1(to_vec(m1));
            if (!ws_send_frame(t, to_str(noise.write_msg2()))) { t.shutdown_close(); return; }
            std::string m3; if (!ws_recv_frame(t, m3)) { t.shutdown_close(); return; }
            noise.read_msg3(to_vec(m3));
            if (!noise.established()) { limiter.on_noise_failure(ip, now); t.shutdown_close(); return; }

            // Geräte-Whitelist: unbekannter Static-Key -> sofort trennen (TT3).
            const std::string peer_b64 = b64(std::vector<uint8_t>(noise.remote_static().begin(), noise.remote_static().end()));
            if (!devices.contains(peer_b64)) { t.shutdown_close(); return; }

            // Bridge zu nova4:/ws (persistente WS-Verbindung).
            web::WsConn nova; std::string e2;
            if (!web::ws_client_connect(cfg.nova4_host, cfg.nova4_port, "/ws", nova, &e2)) {
                t.shutdown_close(); return;
            }

            std::atomic<bool> alive{true};
            // Thread B: nova4 -> (encrypt) -> Client.
            std::thread down([&]() {
                std::string frame;
                while (alive.load() && nova.recv_text(frame)) {
                    if (!ws_send_frame(t, to_str(noise.encrypt(to_vec(frame))))) break;
                }
                alive.store(false);
            });
            // Thread A (hier): Client -> (decrypt) -> nova4.
            std::string cf;
            while (alive.load() && ws_recv_frame(t, cf)) {
                std::vector<uint8_t> pt;
                if (!noise.decrypt(to_vec(cf), pt)) continue;   // Replay/kaputt -> verwerfen
                if (!nova.send_text(to_str(pt))) break;
            }
            alive.store(false);
            nova.close();
            if (down.joinable()) down.join();
            t.shutdown_close();
        });
        // Fertige Worker aufräumen (grobe Obergrenze).
        if (workers.size() > 64) { for (auto& w : workers) if (w.joinable()) w.detach(); workers.clear(); }
    }
}

#else
int run_tls_server(const TlsServerConfig&, std::string* err) { if (err) *err = "nur Windows"; return 1; }
#endif

}  // namespace nova::tunnel

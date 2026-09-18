// rate_limiter.h — NovaTunnel Rate-Limiting (Aufgabe 6.1, TT7).
//
// Reine Zustandsmaschine (Zeit injiziert -> deterministisch testbar, wie
// vram_monitor::evaluate_sample). Grenzen (Design §Aufgabe 6):
//   > 10 Verbindungen/IP/min   -> IP 10 Min blockieren
//   > 5 NOISE-Fehler/IP        -> IP 30 Min blockieren
//   > 30 Nachrichten/Session/s -> Session trennen
#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>

namespace nova::tunnel {

struct RateConfig {
    int max_conns_per_ip_min   = 10;
    int max_noise_fails_per_ip = 5;
    int max_msgs_per_sec       = 30;
    int64_t conn_block_ms      = 10 * 60 * 1000;
    int64_t noise_block_ms     = 30 * 60 * 1000;
};

enum class RateVerdict { Allow, BlockedConns, BlockedNoiseFails, SessionFlood };
const char* to_string(RateVerdict v);

class RateLimiter {
public:
    explicit RateLimiter(RateConfig cfg = {}) : cfg_(cfg) {}

    // now_ms injiziert (Tests deterministisch). Registriert eine neue Verbindung.
    RateVerdict on_connection(const std::string& ip, int64_t now_ms);
    // Registriert einen NOISE-Handshake-Fehler dieser IP.
    RateVerdict on_noise_failure(const std::string& ip, int64_t now_ms);
    // Registriert eine Nachricht innerhalb einer Session.
    RateVerdict on_message(const std::string& session, int64_t now_ms);

    bool is_blocked(const std::string& ip, int64_t now_ms) const;

private:
    struct IpState {
        std::deque<int64_t> conn_times;   // Zeitstempel im 60s-Fenster
        int     noise_fails = 0;
        int64_t block_until = 0;
    };
    struct SessState {
        int64_t sec_bucket = -1;
        int     count = 0;
    };

    RateConfig cfg_;
    std::unordered_map<std::string, IpState>   ips_;
    std::unordered_map<std::string, SessState> sessions_;
};

}  // namespace nova::tunnel

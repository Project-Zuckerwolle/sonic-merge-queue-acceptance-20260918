// rate_limiter.cpp — Implementierung von rate_limiter.h (Aufgabe 6.1).
#include "Tunnel/rate_limiter.h"

namespace nova::tunnel {

const char* to_string(RateVerdict v) {
    switch (v) {
        case RateVerdict::Allow:             return "Allow";
        case RateVerdict::BlockedConns:      return "BlockedConns";
        case RateVerdict::BlockedNoiseFails: return "BlockedNoiseFails";
        case RateVerdict::SessionFlood:      return "SessionFlood";
    }
    return "?";
}

bool RateLimiter::is_blocked(const std::string& ip, int64_t now_ms) const {
    auto it = ips_.find(ip);
    return it != ips_.end() && now_ms < it->second.block_until;
}

RateVerdict RateLimiter::on_connection(const std::string& ip, int64_t now_ms) {
    IpState& st = ips_[ip];
    if (now_ms < st.block_until) return RateVerdict::BlockedConns;

    // 60s-Sliding-Window pflegen.
    st.conn_times.push_back(now_ms);
    while (!st.conn_times.empty() && st.conn_times.front() <= now_ms - 60'000)
        st.conn_times.pop_front();

    if (int(st.conn_times.size()) > cfg_.max_conns_per_ip_min) {
        st.block_until = now_ms + cfg_.conn_block_ms;
        return RateVerdict::BlockedConns;
    }
    return RateVerdict::Allow;
}

RateVerdict RateLimiter::on_noise_failure(const std::string& ip, int64_t now_ms) {
    IpState& st = ips_[ip];
    ++st.noise_fails;
    if (st.noise_fails > cfg_.max_noise_fails_per_ip) {
        st.block_until = now_ms + cfg_.noise_block_ms;
        return RateVerdict::BlockedNoiseFails;
    }
    return RateVerdict::Allow;
}

RateVerdict RateLimiter::on_message(const std::string& session, int64_t now_ms) {
    SessState& st = sessions_[session];
    const int64_t sec = now_ms / 1000;
    if (sec != st.sec_bucket) { st.sec_bucket = sec; st.count = 0; }
    ++st.count;
    if (st.count > cfg_.max_msgs_per_sec) return RateVerdict::SessionFlood;
    return RateVerdict::Allow;
}

}  // namespace nova::tunnel

// idle_monitor.h — Idle-Erkennung + Hibernate S4 (Design §14, §1b).
//
// [Server-PC] Trackt die letzte Aktivität (WebSocket-Turn, laufende Apex-Tasks).
// Wenn idle > N Minuten UND kein Apex läuft: Countdown -> SetSuspendState(S4).
// Die Entscheidung (evaluate) ist eine reine Zustandsmaschine ohne Threads/Sleeps
// und damit voll testbar; IdleMonitor verdrahtet sie mit Timer + Hibernate-Call.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

namespace nova::system {

enum class IdleAction {
    None,
    CountdownStart,   // Schwelle erreicht -> Frontend "Hibernate in 60s" + Abbrechen
    Hibernate,        // Countdown abgelaufen, nicht abgebrochen -> S4
    AbortedByActivity // Aktivität während Countdown -> Countdown abbrechen
};
const char* to_string(IdleAction a);

struct IdleConfig {
    int64_t idle_timeout_ms   = 30 * 60 * 1000;  // §12.2 Default 30 Min
    int64_t countdown_ms      = 60 * 1000;       // §14 "Hibernate in 60s"
};

struct IdleState {
    int64_t last_activity_ms = 0;
    bool    counting_down    = false;
    int64_t countdown_start_ms = 0;
};

// Reine Auswertung. now_ms monoton. apex_running blockiert Hibernate (§14).
IdleAction evaluate_idle(int64_t now_ms, bool apex_running,
                         const IdleConfig& cfg, IdleState& st);

class IdleMonitor {
public:
    using HibernateFn = std::function<void()>;   // default: SetSuspendState(S4)
    using NotifyFn    = std::function<void(IdleAction)>;

    IdleMonitor(IdleConfig cfg, std::atomic<bool>& apex_running)
        : cfg_(cfg), apex_running_(apex_running) {}

    void set_hibernate_fn(HibernateFn f) { hibernate_ = std::move(f); }
    void set_notify_fn(NotifyFn f) { notify_ = std::move(f); }

    void touch(int64_t now_ms);                  // Aktivität registrieren
    IdleAction tick(int64_t now_ms);             // ein Poll-Schritt

    const IdleState& state() const { return st_; }

    // Echte Hibernate-Auslösung (Windows SetSuspendState). No-op außerhalb Windows.
    static void system_hibernate();

private:
    IdleConfig         cfg_;
    IdleState          st_;
    std::atomic<bool>& apex_running_;
    HibernateFn        hibernate_;
    NotifyFn           notify_;
};

}  // namespace nova::system

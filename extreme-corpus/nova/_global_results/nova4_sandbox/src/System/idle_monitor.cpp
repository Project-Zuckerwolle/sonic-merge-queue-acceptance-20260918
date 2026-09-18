// idle_monitor.cpp — Implementierung von idle_monitor.h (Design §14).
#include "System/idle_monitor.h"

#ifdef _WIN32
#include <windows.h>
#include <powrprof.h>
#pragma comment(lib, "PowrProf.lib")
#endif

namespace nova::system {

const char* to_string(IdleAction a) {
    switch (a) {
        case IdleAction::None:              return "None";
        case IdleAction::CountdownStart:    return "CountdownStart";
        case IdleAction::Hibernate:         return "Hibernate";
        case IdleAction::AbortedByActivity: return "AbortedByActivity";
    }
    return "?";
}

IdleAction evaluate_idle(int64_t now_ms, bool apex_running,
                         const IdleConfig& cfg, IdleState& st) {
    // Apex blockiert Hibernate komplett (§14): laufender Countdown wird abgebrochen.
    if (apex_running) {
        if (st.counting_down) { st.counting_down = false; return IdleAction::AbortedByActivity; }
        return IdleAction::None;
    }

    const int64_t idle = now_ms - st.last_activity_ms;

    if (st.counting_down) {
        if (now_ms - st.countdown_start_ms >= cfg.countdown_ms) {
            st.counting_down = false;
            return IdleAction::Hibernate;
        }
        return IdleAction::None;  // Countdown läuft weiter
    }

    if (idle >= cfg.idle_timeout_ms) {
        st.counting_down = true;
        st.countdown_start_ms = now_ms;
        return IdleAction::CountdownStart;
    }
    return IdleAction::None;
}

void IdleMonitor::touch(int64_t now_ms) {
    st_.last_activity_ms = now_ms;
    if (st_.counting_down) {
        st_.counting_down = false;
        if (notify_) notify_(IdleAction::AbortedByActivity);
    }
}

IdleAction IdleMonitor::tick(int64_t now_ms) {
    const IdleAction a = evaluate_idle(now_ms, apex_running_.load(), cfg_, st_);
    if (a != IdleAction::None && notify_) notify_(a);
    if (a == IdleAction::Hibernate) {
        if (hibernate_) hibernate_();
        else system_hibernate();
    }
    return a;
}

void IdleMonitor::system_hibernate() {
#ifdef _WIN32
    // SetSuspendState(Hibernate=TRUE, ForceCritical=FALSE, DisableWakeEvent=FALSE).
    ::SetSuspendState(TRUE, FALSE, FALSE);
#endif
}

}  // namespace nova::system

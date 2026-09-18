// vram_monitor.h — VRAM-/Thermal-Überwachung (Design §7.3, §15.3).
//
// Aufgaben:
//   - Gaming-Schutz (§3, §7.3): freier VRAM unter Schwelle -> H2O-KV-Trim
//   - Thermal-Notfall (§15.3): GPU-Temp 85/90/95°C -> Warn/Entladen/Kritisch,
//     Erholung bei < 75°C -> Modelle dürfen neu laden
//
// Telemetrie kommt über ein GpuTelemetry-Interface:
//   - SimulatedTelemetry (Host/Tests): skriptbare Samples
//   - NvmlTelemetry (Server/Gaming mit NVIDIA): nur unter NOVA_HAVE_NVML
//
// Die Auswertung (evaluate_sample) ist eine reine Zustandsmaschine und ohne
// Threads/Sleeps testbar; VramMonitor verdrahtet sie mit einem Polling-Thread.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace nova::infer {

struct GpuSample {
    uint64_t vram_used_bytes  = 0;
    uint64_t vram_total_bytes = 0;
    double   temperature_c    = 0.0;
    bool     valid            = false;
    uint64_t vram_free_bytes() const {
        return vram_total_bytes > vram_used_bytes ? vram_total_bytes - vram_used_bytes : 0;
    }
};

class GpuTelemetry {
public:
    virtual ~GpuTelemetry() = default;
    virtual GpuSample sample() = 0;
};

// Aktionen (Design §7.3, §15.3).
enum class VramAction {
    ThermalWarn,       // > 85°C   -> WARN/Frontend-Hinweis
    ThermalUnload,     // > 90°C   -> Modelle aus VRAM entladen, Generierung abbrechen
    ThermalCritical,   // > 95°C   -> zusätzlich Hibernate(Server)/Frontend-Warnung
    ThermalRecovered,  // < 75°C   -> Modelle dürfen neu laden
    GamingTrim,        // freier VRAM < Gaming-Hardcap -> H2O KV-Trim (Gaming-PC)
    FloorTrim,         // freier VRAM < weichem Server-Floor -> proaktiver KV-Trim (Aufgabe 5.4)
};

const char* to_string(VramAction a);

struct VramConfig {
    bool     gaming_protection   = true;            // Gaming-PC: ja, Server-PC: nein
    uint64_t gaming_min_free     = 4ull << 30;      // 4 GB immer frei (§3)
    bool     server_floor        = false;           // Server: proaktiver Frei-Puffer (kein Gaming-Cap)
    uint64_t floor_min_free      = 500ull << 20;    // 500 MB Mindest-frei (Aufgabe 5.4, config.json)
    double   temp_warn_c         = 85.0;
    double   temp_unload_c       = 90.0;
    double   temp_critical_c     = 95.0;
    double   temp_recover_c      = 75.0;
    int      vram_poll_ms        = 500;             // §7.3
    int      thermal_poll_ms     = 10000;           // §15.3
};

struct VramState {
    bool models_unloaded = false;  // wegen Thermal entladen (Hysterese bis < recover)
    bool warned          = false;  // Warn bereits ausgelöst
    bool gaming_trimmed  = false;  // Gaming-Trim aktiv (bis freier VRAM wieder ok)
    bool floor_trimmed   = false;  // Server-Floor-Trim aktiv (Aufgabe 5.4)
};

// Verify-before-load (Aufgabe 5.4): nur laden, wenn nach dem Load der Floor noch
// hält — nie der Budget-Tabelle vertrauen, echten freien VRAM prüfen.
inline bool vram_can_load(uint64_t free_bytes, uint64_t need_bytes, uint64_t floor_bytes) {
    return free_bytes >= need_bytes && (free_bytes - need_bytes) >= floor_bytes;
}

// Reine Auswertung: erzeugt Aktionen + aktualisiert state. check_thermal/
// check_vram steuern, welche Prüfungen laufen (verschiedene Poll-Intervalle).
std::vector<VramAction> evaluate_sample(const GpuSample& s, const VramConfig& cfg,
                                        VramState& state,
                                        bool check_thermal = true,
                                        bool check_vram = true);

class VramMonitor {
public:
    using ActionCallback = std::function<void(VramAction, const GpuSample&)>;

    VramMonitor(std::shared_ptr<GpuTelemetry> tele, VramConfig cfg)
        : tele_(std::move(tele)), cfg_(cfg) {}
    ~VramMonitor() { stop(); }

    void set_callback(ActionCallback cb) { cb_ = std::move(cb); }

    void start();   // startet Polling-Thread
    void stop();    // stoppt + joint

    GpuSample last_sample() const;
    VramState state() const { return state_; }

private:
    void loop();

    std::shared_ptr<GpuTelemetry> tele_;
    VramConfig       cfg_;
    VramState        state_;
    ActionCallback   cb_;
    std::thread      thread_;
    std::atomic<bool> running_{false};

    mutable std::mutex sample_mutex_;
    GpuSample last_;
};

// Proaktiver Lade-Wächter (Aufgabe 5.4): prüft VOR jedem RAM->VRAM-Load den
// echten freien VRAM (cudaMemGetInfo/NVML via Telemetry). Hält der Floor nach
// dem Load nicht, wird erst getrimmt (trim_cb -> z.B. KvCache::trim_to_bytes) und
// erneut geprüft; hält er dann noch nicht, wird der Load verweigert (defer) —
// KEIN OOM, kein Session-Abbruch.
class VramLoadGuard {
public:
    VramLoadGuard(std::shared_ptr<GpuTelemetry> tele, uint64_t floor_bytes)
        : tele_(std::move(tele)), floor_(floor_bytes) {}

    // trim_cb(target_free_bytes) soll bis mindestens target frei machen und den
    // (neuen) geschätzten freien VRAM zurückliefern. true == Load darf erfolgen.
    bool acquire(uint64_t need_bytes,
                 const std::function<uint64_t(uint64_t target_free)>& trim_cb,
                 std::string* err = nullptr);

    uint64_t last_free() const { return last_free_; }
    bool     trimmed()   const { return trimmed_; }

private:
    std::shared_ptr<GpuTelemetry> tele_;
    uint64_t floor_;
    uint64_t last_free_ = 0;
    bool     trimmed_   = false;
};

// ---- Simulierte Telemetrie (Host/Tests) -----------------------------------
class SimulatedTelemetry : public GpuTelemetry {
public:
    // Liefert nacheinander die gepushten Samples; danach wiederholt das letzte.
    void push(const GpuSample& s);
    void set_constant(const GpuSample& s);
    GpuSample sample() override;

private:
    std::vector<GpuSample> queue_;
    size_t idx_ = 0;
    std::mutex m_;
};

#ifdef NOVA_HAVE_NVML
// ---- NVML-Telemetrie (NVIDIA, Server/Gaming) ------------------------------
class NvmlTelemetry : public GpuTelemetry {
public:
    bool init(unsigned device_index, std::string* err = nullptr);
    ~NvmlTelemetry() override;
    GpuSample sample() override;
private:
    void* device_ = nullptr;  // nvmlDevice_t
    bool  ready_  = false;
};
#endif

}  // namespace nova::infer

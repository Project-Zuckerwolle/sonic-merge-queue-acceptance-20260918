// vram_monitor.cpp — siehe vram_monitor.h
#include "InferEngine/vram_monitor.h"

#include <chrono>

#ifdef NOVA_HAVE_NVML
#include <nvml.h>
#endif

namespace nova::infer {

const char* to_string(VramAction a) {
    switch (a) {
        case VramAction::ThermalWarn:      return "ThermalWarn";
        case VramAction::ThermalUnload:    return "ThermalUnload";
        case VramAction::ThermalCritical:  return "ThermalCritical";
        case VramAction::ThermalRecovered: return "ThermalRecovered";
        case VramAction::GamingTrim:       return "GamingTrim";
        case VramAction::FloorTrim:        return "FloorTrim";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// Reine Zustandsmaschine
// ---------------------------------------------------------------------------
std::vector<VramAction> evaluate_sample(const GpuSample& s, const VramConfig& cfg,
                                        VramState& st, bool check_thermal, bool check_vram) {
    std::vector<VramAction> out;
    if (!s.valid) return out;

    if (check_thermal) {
        const double t = s.temperature_c;
        if (t >= cfg.temp_critical_c) {
            // > 95°C: kritisch (impliziert Entladen).
            st.models_unloaded = true;
            out.push_back(VramAction::ThermalCritical);
        } else if (t >= cfg.temp_unload_c) {
            // > 90°C: Modelle entladen, laufende Generierung abbrechen.
            if (!st.models_unloaded) out.push_back(VramAction::ThermalUnload);
            st.models_unloaded = true;
        } else if (t >= cfg.temp_warn_c) {
            // > 85°C: einmalige Warnung (solange nicht ohnehin entladen).
            if (!st.warned && !st.models_unloaded) {
                out.push_back(VramAction::ThermalWarn);
                st.warned = true;
            }
        } else if (t < cfg.temp_recover_c) {
            // < 75°C: Erholung -> Modelle dürfen neu laden.
            if (st.models_unloaded) out.push_back(VramAction::ThermalRecovered);
            st.models_unloaded = false;
            st.warned = false;
        }
        // Zwischen recover und warn (75..85): keine Aktion, Hysterese hält Zustand.
    }

    if (check_vram && cfg.gaming_protection) {
        const uint64_t free = s.vram_free_bytes();
        if (free < cfg.gaming_min_free) {
            if (!st.gaming_trimmed) {
                out.push_back(VramAction::GamingTrim);
                st.gaming_trimmed = true;
            }
        } else {
            st.gaming_trimmed = false;  // wieder genug frei -> Trim kann erneut auslösen
        }
    }

    // Proaktiver Server-Floor (Aufgabe 5.4): weicher Mindest-frei-Puffer, der VOR
    // einem OOM greift. Auf dem Server (kein Gaming-Cap) verhindert das reaktives
    // Crash+Neustart; Unterschreitung triggert KV-Trim, bevor geladen/expandiert wird.
    if (check_vram && cfg.server_floor) {
        const uint64_t free = s.vram_free_bytes();
        if (free < cfg.floor_min_free) {
            if (!st.floor_trimmed) {
                out.push_back(VramAction::FloorTrim);
                st.floor_trimmed = true;
            }
        } else {
            st.floor_trimmed = false;
        }
    }
    return out;
}

bool VramLoadGuard::acquire(uint64_t need_bytes,
                            const std::function<uint64_t(uint64_t)>& trim_cb,
                            std::string* err) {
    trimmed_ = false;
    const GpuSample s = tele_ ? tele_->sample() : GpuSample{};
    uint64_t free = s.valid ? s.vram_free_bytes() : 0;
    last_free_ = free;
    if (vram_can_load(free, need_bytes, floor_)) return true;   // genug frei -> laden

    // Nicht genug: erst trimmen (H2O/KV) bis need+floor frei, dann erneut prüfen.
    if (trim_cb) {
        free = trim_cb(need_bytes + floor_);
        trimmed_ = true;
        last_free_ = free;
        if (vram_can_load(free, need_bytes, floor_)) return true;
    }
    if (err) *err = "VRAM-Floor: Load verweigert (defer) statt OOM";
    return false;   // defer — KEIN OOM/Abbruch
}

// ---------------------------------------------------------------------------
// VramMonitor — Polling-Thread
// ---------------------------------------------------------------------------
void VramMonitor::start() {
    if (running_.exchange(true)) return;
    thread_ = std::thread([this] { loop(); });
}

void VramMonitor::stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
}

GpuSample VramMonitor::last_sample() const {
    std::lock_guard<std::mutex> lk(sample_mutex_);
    return last_;
}

void VramMonitor::loop() {
    using clock = std::chrono::steady_clock;
    auto last_thermal = clock::now() - std::chrono::milliseconds(cfg_.thermal_poll_ms);

    while (running_.load()) {
        GpuSample s = tele_ ? tele_->sample() : GpuSample{};
        {
            std::lock_guard<std::mutex> lk(sample_mutex_);
            last_ = s;
        }
        const bool do_thermal =
            std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - last_thermal)
                .count() >= cfg_.thermal_poll_ms;
        if (do_thermal) last_thermal = clock::now();

        for (VramAction a : evaluate_sample(s, cfg_, state_, do_thermal, /*check_vram=*/true))
            if (cb_) cb_(a, s);

        // In kleinen Schritten schlafen, damit stop() zügig greift.
        int slept = 0;
        while (running_.load() && slept < cfg_.vram_poll_ms) {
            const int step = cfg_.vram_poll_ms - slept < 20 ? cfg_.vram_poll_ms - slept : 20;
            std::this_thread::sleep_for(std::chrono::milliseconds(step));
            slept += step;
        }
    }
}

// ---------------------------------------------------------------------------
// SimulatedTelemetry
// ---------------------------------------------------------------------------
void SimulatedTelemetry::push(const GpuSample& s) {
    std::lock_guard<std::mutex> lk(m_);
    queue_.push_back(s);
}
void SimulatedTelemetry::set_constant(const GpuSample& s) {
    std::lock_guard<std::mutex> lk(m_);
    queue_.assign(1, s);
    idx_ = 0;
}
GpuSample SimulatedTelemetry::sample() {
    std::lock_guard<std::mutex> lk(m_);
    if (queue_.empty()) return GpuSample{};
    const GpuSample s = queue_[idx_];
    if (idx_ + 1 < queue_.size()) ++idx_;  // letztes Sample wird wiederholt
    return s;
}

// ---------------------------------------------------------------------------
// NvmlTelemetry (nur unter NOVA_HAVE_NVML)
// ---------------------------------------------------------------------------
#ifdef NOVA_HAVE_NVML
bool NvmlTelemetry::init(unsigned device_index, std::string* err) {
    if (nvmlInit_v2() != NVML_SUCCESS) { if (err) *err = "nvmlInit fehlgeschlagen"; return false; }
    nvmlDevice_t dev{};
    if (nvmlDeviceGetHandleByIndex_v2(device_index, &dev) != NVML_SUCCESS) {
        if (err) *err = "nvmlDeviceGetHandleByIndex fehlgeschlagen";
        nvmlShutdown();
        return false;
    }
    device_ = reinterpret_cast<void*>(dev);
    ready_  = true;
    return true;
}

NvmlTelemetry::~NvmlTelemetry() {
    if (ready_) nvmlShutdown();
}

GpuSample NvmlTelemetry::sample() {
    GpuSample s;
    if (!ready_) return s;
    auto dev = reinterpret_cast<nvmlDevice_t>(device_);
    nvmlMemory_t mem{};
    unsigned temp = 0;
    bool ok = (nvmlDeviceGetMemoryInfo(dev, &mem) == NVML_SUCCESS);
    ok = ok && (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS);
    if (!ok) return s;
    s.vram_used_bytes  = mem.used;
    s.vram_total_bytes = mem.total;
    s.temperature_c    = double(temp);
    s.valid            = true;
    return s;
}
#endif  // NOVA_HAVE_NVML

}  // namespace nova::infer

// test_prefetch_vram.cpp — Host-Test für prefetch_stats + vram_monitor.
//
// Keine nummerierte Design-Testbed-Stufe, aber Teil der Block-A-Validierung:
//   - prefetch_stats: Zugriffs-Tracking, hot-Schwelle (>5/h), Persistenz (§6.4)
//   - vram_monitor:   Thermal-Zustandsmaschine 85/90/95/75°C + Gaming-Trim
//                     (§7.3, §15.3), inkl. kurzem Polling-Thread-Smoke-Test.
#include "InferEngine/prefetch_stats.h"
#include "InferEngine/vram_monitor.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace nova::infer;

namespace {
int g_fails = 0;
void check(bool cond, const std::string& what) {
    std::cout << (cond ? "  [OK] " : "  [FAIL] ") << what << "\n";
    if (!cond) ++g_fails;
}

GpuSample mk(double temp, uint64_t used, uint64_t total) {
    GpuSample s; s.temperature_c = temp; s.vram_used_bytes = used;
    s.vram_total_bytes = total; s.valid = true; return s;
}
bool has(const std::vector<VramAction>& v, VramAction a) {
    for (auto x : v) if (x == a) return true;
    return false;
}
}  // namespace

static void test_prefetch() {
    std::cout << "[prefetch_stats]\n";
    PrefetchStats ps;                      // Schwelle = 5 (> 5/h = hot)
    const int64_t t0 = 1'000'000;          // beliebige Unix-Basiszeit

    const std::string hot = "models/g4/chunks/L0.nv4";
    const std::string cold = "models/g4/chunks/L1.nv4";
    for (int i = 0; i < 7; ++i) ps.record_access(hot, t0 + i);  // 7 Zugriffe in <1h
    ps.record_access(cold, t0);

    check(ps.accesses_last_hour(hot, t0 + 10) == 7, "7 Zugriffe im Fenster gezählt");
    check(ps.is_hot(hot, t0 + 10), "Hot-Chunk erkannt (>5/h)");
    check(!ps.is_hot(cold, t0 + 10), "Cold-Chunk nicht hot");
    check(ps.hot_chunks(t0 + 10).size() == 1, "genau 1 Hot-Chunk");

    // Fenster läuft nach 1h ab -> Zähler zurückgesetzt.
    check(ps.accesses_last_hour(hot, t0 + 3601) == 0, "Fenster nach 1h abgelaufen");
    check(!ps.is_hot(hot, t0 + 3601), "nach Ablauf nicht mehr hot");

    // Persistenz-Round-Trip.
    const std::string p = (fs::temp_directory_path() / "nova4_chunk_access.bin").string();
    check(ps.save(p), "save() chunk_access.bin");
    PrefetchStats ps2;
    check(ps2.load(p), "load() chunk_access.bin");
    check(ps2.size() == ps.size(), "Record-Anzahl nach Load gleich");
    check(ps2.accesses_last_hour(hot, t0 + 10) == 7, "Zählerstand nach Load erhalten");
    std::error_code ec; fs::remove(p, ec);
}

static void test_thermal() {
    std::cout << "[vram_monitor: Thermal-Zustandsmaschine]\n";
    VramConfig cfg;                 // Defaults: 85/90/95/75, gaming_protection an
    cfg.gaming_protection = false;  // hier nur Thermal prüfen
    VramState st;
    const uint64_t T = 20ull << 30, U = 1ull << 30;  // viel frei -> kein Trim

    check(evaluate_sample(mk(60, U, T), cfg, st).empty(), "60°C: keine Aktion");

    auto a85 = evaluate_sample(mk(86, U, T), cfg, st);
    check(has(a85, VramAction::ThermalWarn), "86°C: ThermalWarn");
    check(evaluate_sample(mk(87, U, T), cfg, st).empty(), "87°C: Warn nur einmal");

    auto a90 = evaluate_sample(mk(91, U, T), cfg, st);
    check(has(a90, VramAction::ThermalUnload), "91°C: ThermalUnload");
    check(st.models_unloaded, "Zustand: Modelle entladen");

    auto a95 = evaluate_sample(mk(96, U, T), cfg, st);
    check(has(a95, VramAction::ThermalCritical), "96°C: ThermalCritical");

    // Zwischen 75 und 85 keine Erholung (Hysterese).
    check(evaluate_sample(mk(80, U, T), cfg, st).empty(), "80°C: noch keine Erholung");
    check(st.models_unloaded, "bei 80°C weiterhin entladen");

    auto rec = evaluate_sample(mk(70, U, T), cfg, st);
    check(has(rec, VramAction::ThermalRecovered), "70°C: ThermalRecovered");
    check(!st.models_unloaded, "Zustand: wieder ladbar");
}

static void test_gaming_trim() {
    std::cout << "[vram_monitor: Gaming-Schutz]\n";
    VramConfig cfg;                       // gaming_min_free = 4 GB
    VramState st;
    const uint64_t T = 20ull << 30;
    // Nur VRAM prüfen, Temperatur unkritisch.
    auto trim = evaluate_sample(mk(50, 17ull << 30, T), cfg, st, /*thermal=*/false, /*vram=*/true);
    check(has(trim, VramAction::GamingTrim), "freier VRAM 3 GB < 4 GB -> GamingTrim");
    auto trim2 = evaluate_sample(mk(50, 17ull << 30, T), cfg, st, false, true);
    check(!has(trim2, VramAction::GamingTrim), "Trim nur einmal bis Entspannung");
    auto ok = evaluate_sample(mk(50, 8ull << 30, T), cfg, st, false, true);
    check(ok.empty(), "12 GB frei -> kein Trim, Zustand entspannt");
    auto trim3 = evaluate_sample(mk(50, 17ull << 30, T), cfg, st, false, true);
    check(has(trim3, VramAction::GamingTrim), "erneuter Engpass -> Trim wieder");
}

static void test_monitor_thread() {
    std::cout << "[vram_monitor: Polling-Thread Smoke-Test]\n";
    auto tele = std::make_shared<SimulatedTelemetry>();
    tele->set_constant(mk(92, 1ull << 30, 20ull << 30));  // konstant 92°C -> Unload
    VramConfig cfg; cfg.vram_poll_ms = 10; cfg.thermal_poll_ms = 10;
    VramMonitor mon(tele, cfg);

    std::atomic<int> unloads{0};
    mon.set_callback([&](VramAction a, const GpuSample&) {
        if (a == VramAction::ThermalUnload) ++unloads;
    });
    mon.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    mon.stop();
    check(mon.last_sample().valid, "Thread hat Samples geholt");
    check(unloads.load() >= 1, "Thread löste ThermalUnload aus");
}

int main() {
    std::cout << "=== Block-A Test: prefetch_stats + vram_monitor ===\n";
    test_prefetch();
    test_thermal();
    test_gaming_trim();
    test_monitor_thread();
    std::cout << "\n=== " << (g_fails == 0 ? "BESTANDEN" : "FEHLGESCHLAGEN")
              << " (" << g_fails << " Fehler) ===\n";
    return g_fails == 0 ? 0 : 1;
}

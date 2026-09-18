// testbed18_vram_floor.cpp — Proaktiver VRAM-Floor + Verify-before-load (Aufgabe 5.4, TB18).
//
// Kriterien:
//  - bei künstlich knappem VRAM wird der 3B-Load VERZÖGERT / KV getrimmt statt OOM,
//  - freier VRAM bleibt über dem Floor,
//  - kein Session-Abbruch.
#include "InferEngine/vram_monitor.h"

#include <cstdio>
#include <iostream>
#include <memory>

using namespace nova::infer;

namespace {
constexpr uint64_t GB = 1ull << 30;
constexpr uint64_t MB = 1ull << 20;

GpuSample sample_with_free(uint64_t free) {
    GpuSample s;
    s.vram_total_bytes = 10 * GB;
    s.vram_used_bytes = s.vram_total_bytes - free;
    s.temperature_c = 60.0;
    s.valid = true;
    return s;
}
bool has(const std::vector<VramAction>& v, VramAction a) {
    for (auto x : v) if (x == a) return true;
    return false;
}
}  // namespace

int main() {
    std::cout << "=== Testbed 18: VRAM-Floor + Verify-before-load ===\n";
    bool pass = true;

    VramConfig cfg;
    cfg.gaming_protection = false;   // Server: kein Gaming-Cap
    cfg.server_floor = true;
    cfg.floor_min_free = 500 * MB;
    VramState st;

    // (a) Floor-Zustandsmaschine: Unterschreitung -> FloorTrim (latched), Erholung -> reset.
    const auto a1 = evaluate_sample(sample_with_free(300 * MB), cfg, st, false, true);
    const auto a2 = evaluate_sample(sample_with_free(300 * MB), cfg, st, false, true);
    const auto a3 = evaluate_sample(sample_with_free(2 * GB),   cfg, st, false, true);
    const auto a4 = evaluate_sample(sample_with_free(300 * MB), cfg, st, false, true);
    std::printf("  FloorTrim: unter=%d wieder-unter(latched)=%d erholt=%d erneut=%d\n",
                has(a1, VramAction::FloorTrim), has(a2, VramAction::FloorTrim),
                has(a3, VramAction::FloorTrim), has(a4, VramAction::FloorTrim));
    pass &= has(a1, VramAction::FloorTrim);
    pass &= !has(a2, VramAction::FloorTrim);   // latched
    pass &= !has(a3, VramAction::FloorTrim);   // erholt -> reset
    pass &= has(a4, VramAction::FloorTrim);    // erneut auslösbar

    // (b) Verify-before-load: knapper VRAM -> erst KV-Trim, dann Load erlaubt.
    auto tele1 = std::make_shared<SimulatedTelemetry>();
    tele1->set_constant(sample_with_free(300 * MB));   // zu wenig für 1 GB Load
    VramLoadGuard g1(tele1, cfg.floor_min_free);
    bool trim_called = false;
    auto trim_ok = [&](uint64_t target) -> uint64_t {
        trim_called = true;   // simuliert KvCache::trim_to_bytes -> gibt VRAM frei
        (void)target; return 2 * GB;
    };
    const bool load1 = g1.acquire(1 * GB, trim_ok);
    std::printf("  Load(knapp+Trim): erlaubt=%s trim=%s free_nachher=%llu MB (Floor hält=%s)\n",
                load1 ? "ja" : "NEIN", trim_called ? "ja" : "nein",
                (unsigned long long)(g1.last_free() / MB),
                vram_can_load(g1.last_free(), 1 * GB, cfg.floor_min_free) ? "ja" : "NEIN");
    pass &= load1 && trim_called && g1.trimmed();
    pass &= vram_can_load(g1.last_free(), 1 * GB, cfg.floor_min_free);  // Floor hält nach Load

    // (c) Trim reicht NICHT -> Load verweigert (defer), KEIN OOM.
    auto tele2 = std::make_shared<SimulatedTelemetry>();
    tele2->set_constant(sample_with_free(300 * MB));
    VramLoadGuard g2(tele2, cfg.floor_min_free);
    auto trim_weak = [&](uint64_t) -> uint64_t { return 1200 * MB; };  // < need+floor (1,5 GB)
    std::string err;
    const bool load2 = g2.acquire(1 * GB, trim_weak, &err);
    std::printf("  Load(unmöglich): erlaubt=%s (defer, err='%s')\n", load2 ? "JA" : "nein", err.c_str());
    pass &= (load2 == false);        // deferred statt OOM
    pass &= !err.empty();

    // (d) Genug frei von Anfang an -> Load ohne Trim.
    auto tele3 = std::make_shared<SimulatedTelemetry>();
    tele3->set_constant(sample_with_free(3 * GB));
    VramLoadGuard g3(tele3, cfg.floor_min_free);
    const bool load3 = g3.acquire(1 * GB, trim_ok);
    pass &= load3 && !g3.trimmed();
    std::printf("  Load(genug frei): erlaubt=%s trim=%s\n", load3 ? "ja" : "NEIN",
                g3.trimmed() ? "ja" : "nein");

    std::cout << "\n=== Testbed 18: " << (pass ? "BESTANDEN" : "FEHLGESCHLAGEN") << " ===\n";
    return pass ? 0 : 1;
}

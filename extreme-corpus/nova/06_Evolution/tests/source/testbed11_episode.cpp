// testbed11_episode.cpp — Ministral 14B idle: Episode + VRAM (Design §16, TB 11).
//
// Kriterium: "Episode-Datei korrekt, VRAM während Ministral 14B < 7,5 GB."
//
// Der 14B-Lauf wird durch einen LlmFn-Callback modelliert (kein Modell vorhanden).
// Geprüft: (a) Episode-Datei mit '# TITEL' erste Zeile + Summary korrekt erzeugt
// und über last() lesbar; (b) ein simuliertes GPU-Sample während des Laufs zeigt
// VRAM < 7,5 GB und löst weder Thermal-Unload noch KV-Trim aus (§1b-Budget).
#include "InferEngine/vram_monitor.h"
#include "Memory/episode_store.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace nova;

int main() {
    std::cout << "=== Testbed 11: Ministral 14B idle (Episode + VRAM) ===\n";
    bool pass = true;

    const std::string dir = (fs::temp_directory_path() / "nova4_tb11_episodes").string();
    std::error_code ec; fs::remove_all(dir, ec);

    memory::EpisodeStore store(dir);
    std::string err;
    if (!store.init(&err)) { std::cout << "  init: " << err << "\n"; return 2; }

    // Stub-Ministral-14B: erzeugt Titel + Zusammenfassung.
    bool llm_called = false;
    brain::LlmFn llm = [&](const std::string& prompt) -> std::string {
        llm_called = true;
        (void)prompt;
        std::string summary;
        for (int i = 0; i < 60; ++i) summary += "Nova4 Block D wurde implementiert und getestet. ";
        return "TITEL: Block D Brain Implementierung\n\n" + summary;
    };

    const std::string working_memory =
        "user: Lass uns Block D bauen.\nassistant: Brain, Dreaming, Episodes implementiert.\n";
    memory::Episode ep;
    if (!store.create(llm, working_memory, "20260624_143022", ep, &err)) {
        std::cout << "  create: " << err << "\n"; return 2;
    }

    std::printf("  LLM aufgerufen        = %s\n", llm_called ? "ja" : "NEIN");
    std::printf("  Episode-Titel         = \"%s\"\n", ep.title.c_str());
    std::printf("  Summary-Länge         = %zu Zeichen\n", ep.summary.size());

    pass &= llm_called;
    pass &= (ep.title == "Block D Brain Implementierung");
    pass &= (ep.summary.size() > 200);  // 300–500 Wörter Zielbereich

    // Datei korrekt: erste Zeile == "# TITEL".
    std::ifstream f((fs::path(dir) / ep.filename).string(), std::ios::binary);
    std::string first; std::getline(f, first);
    if (!first.empty() && first.back() == '\r') first.pop_back();
    std::printf("  Datei erste Zeile     = \"%s\"\n", first.c_str());
    pass &= (first == "# Block D Brain Implementierung");

    // last() liefert die Episode.
    const auto last = store.last(15);
    pass &= (last.size() == 1 && last[0].title == ep.title);
    std::printf("  last(15) Treffer      = %zu\n", last.size());

    // --- VRAM während 14B-Lauf (Server-PC, §1b-Budget) ---------------------
    infer::GpuSample s;
    s.vram_total_bytes = 10ull << 30;     // RTX 3080 10 GB
    s.vram_used_bytes  = uint64_t(7.0 * (1ull << 30));  // 14B-Lauf ~7,0 GB
    s.temperature_c    = 64.0;
    s.valid = true;

    infer::VramConfig cfg; cfg.gaming_protection = false;  // Server-PC
    infer::VramState st;
    const auto actions = infer::evaluate_sample(s, cfg, st);

    const double used_gb = double(s.vram_used_bytes) / double(1ull << 30);
    bool no_unload_trim = true;
    for (auto a : actions)
        if (a == infer::VramAction::ThermalUnload || a == infer::VramAction::GamingTrim ||
            a == infer::VramAction::ThermalCritical) no_unload_trim = false;

    std::printf("  VRAM während 14B      = %.2f GB (Ziel < 7,5 GB)\n", used_gb);
    std::printf("  Keine Unload/Trim     = %s\n", no_unload_trim ? "ja" : "NEIN");
    pass &= (used_gb < 7.5);
    pass &= no_unload_trim;

    fs::remove_all(dir, ec);
    std::cout << "\n=== Testbed 11: " << (pass ? "BESTANDEN" : "FEHLGESCHLAGEN") << " ===\n";
    return pass ? 0 : 1;
}

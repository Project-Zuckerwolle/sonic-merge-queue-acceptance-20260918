// testbed2_chunk_read.cpp — Testbed 2: SSD/RAM Chunk-Read (Design §16, TB 2)
//
// BESTANDEN wenn: CRC stimmt für alle Chunks (beide Modi), CRC-Fehler werden
// erkannt. Bandbreiten-Zielwerte (>5 GB/s SSD, >10 GB/s RAM) werden gemessen
// und berichtet — auf dem Gaming-PC liest der OS-Cache frisch geschriebene
// Dateien aus RAM, daher ist die SSD-Zahl hier eine Cache-Obergrenze, kein
// roher NVMe-Durchsatz.
#include "InferEngine/chunk_streamer.h"
#include "ModelStore/chunk_validator.h"
#include "ModelStore/nv4_format.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace nova;

namespace {

struct Cfg { int chunks = 48, rows = 1024, cols = 2048; modelstore::QuantType qt = modelstore::QuantType::INT4; };

std::vector<std::string> generate_chunks(const fs::path& dir, const Cfg& c) {
    fs::create_directories(dir);
    std::vector<std::string> paths;
    std::mt19937 rng(0xBEEF);
    std::normal_distribution<float> dist(0.0f, 0.8f);
    const uint64_t n = uint64_t(c.rows) * c.cols;
    std::vector<float> data(static_cast<size_t>(n));

    for (int l = 0; l < c.chunks; ++l) {
        for (auto& v : data) v = dist(rng);
        auto q = modelstore::quantize_tensor(data.data(), n, c.qt);
        char name[64];
        std::snprintf(name, sizeof(name), "L%05u_C%05u.nv4", l, l);
        const std::string p = (dir / name).string();
        // compress=false: schneller + größeres I/O-Volumen für die Bandbreitenmessung.
        if (!modelstore::write_chunk(p, uint16_t(l), uint16_t(l), c.qt,
                                     uint32_t(c.rows), uint32_t(c.cols),
                                     q.scales, q.packed, /*compress=*/false)) {
            std::cerr << "FATAL: Schreiben fehlgeschlagen: " << p << "\n";
            std::exit(2);
        }
        paths.push_back(p);
    }
    return paths;
}

bool phase_validate_dir(const fs::path& dir) {
    auto fails = modelstore::validate_chunk_dir(dir.string(), /*only_failures=*/true);
    std::cout << "  Defekte Chunks: " << fails.size() << "\n";
    for (const auto& f : fails) std::cout << "    [BAD] " << f.path << " (" << f.detail << ")\n";
    return fails.empty();
}

bool phase_stream(infer::ChunkStreamer& s, const std::vector<std::string>& paths,
                  const char* label, double target_gibs) {
    using clock = std::chrono::steady_clock;

    // (a) Reine Lese-Bandbreite: nur read_raw, Puffer wiederverwendet,
    //     bester von mehreren Durchläufen (erster wärmt Cache/Allokation).
    std::vector<uint8_t> buf;
    uint64_t bytes = 0;
    double best = 1e30;
    for (int pass = 0; pass < 6; ++pass) {
        const auto t0 = clock::now();
        uint64_t b = 0;
        for (const auto& p : paths) {
            if (!s.read_raw(p, buf, nullptr)) { std::cout << "    -> FAIL: Lesefehler " << p << "\n"; return false; }
            b += buf.size();
        }
        const double dt = std::chrono::duration<double>(clock::now() - t0).count();
        best = (dt < best) ? dt : best;
        bytes = b;
    }
    const double gibs = (double(bytes) / (1024.0*1024.0*1024.0)) / best;

    // (b) CRC-Korrektheit über alle Chunks (nicht zeitkritisch).
    uint64_t bad = 0;
    s.stream_and_validate(paths, &bad, false);

    std::printf("  %-4s  Bytes=%.1f MiB  Lese-Zeit(best)=%.3f ms  -> %.2f GiB/s  (CRC-Fehler: %llu)\n",
                label, double(bytes) / (1024.0*1024.0), best * 1e3, gibs,
                (unsigned long long)bad);
    if (bad != 0) { std::cout << "    -> FAIL: CRC-Fehler\n"; return false; }
    if (gibs < target_gibs)
        std::printf("    -> Hinweis: unter Zielwert %.0f GiB/s (Hardware/Cache-abhängig, kein Hard-Fail)\n",
                    target_gibs);
    else
        std::printf("    -> Zielwert %.0f GiB/s erreicht\n", target_gibs);
    return true;
}

bool phase_corruption(const std::string& path) {
    // CRC ist OK -> ein Payload-Byte kippen -> muss als BAD_CRC erkannt werden.
    auto before = modelstore::validate_chunk_file(path);
    if (before.status != modelstore::ChunkStatus::OK) {
        std::cout << "    -> FAIL: Chunk war vor Korruption nicht OK\n"; return false;
    }
    std::vector<char> bytes;
    { std::ifstream f(path, std::ios::binary | std::ios::ate);
      const std::streamoff sz = f.tellg(); f.seekg(0);
      bytes.resize(size_t(sz)); f.read(bytes.data(), sz); }

    const size_t pos = bytes.size() - 1;  // letztes Byte (im Payload)
    const char orig = bytes[pos];
    bytes[pos] ^= 0xFF;
    { std::ofstream f(path, std::ios::binary | std::ios::trunc);
      f.write(bytes.data(), std::streamsize(bytes.size())); }

    auto after = modelstore::validate_chunk_file(path);
    bool detected = (after.status == modelstore::ChunkStatus::BAD_CRC);
    std::cout << "  Korruption injiziert -> Status: "
              << (detected ? "BAD_CRC erkannt (korrekt)" : "NICHT erkannt (FAIL)") << "\n";

    // Wiederherstellen
    bytes[pos] = orig;
    { std::ofstream f(path, std::ios::binary | std::ios::trunc);
      f.write(bytes.data(), std::streamsize(bytes.size())); }
    return detected;
}

}  // namespace

int main() {
    std::cout << "=== Testbed 2: SSD/RAM Chunk-Read ===\n";
    Cfg cfg;
    const fs::path dir = fs::temp_directory_path() / "nova4_tb2";
    std::error_code ec; fs::remove_all(dir, ec);

    std::cout << "[Gen] " << cfg.chunks << " Chunks (" << cfg.rows << "x" << cfg.cols
              << ", INT4, unkomprimiert)\n";
    auto paths = generate_chunks(dir, cfg);

    bool pass = true;

    std::cout << "[1] CRC-Prüfung aller Chunks im Verzeichnis (chunk_validator):\n";
    pass &= phase_validate_dir(dir);

    std::cout << "[2] SSD-Modus (OVERLAPPED I/O):\n";
    { infer::ChunkStreamer s(infer::StreamMode::SSD);
      pass &= phase_stream(s, paths, "SSD", 5.0); }

    std::cout << "[3] RAM-Modus (preload + memcpy):\n";
    { infer::ChunkStreamer s(infer::StreamMode::RAM);
      s.preload_all(paths);
      pass &= phase_stream(s, paths, "RAM", 10.0); }

    std::cout << "[4] CRC-Fehlererkennung:\n";
    pass &= phase_corruption(paths.front());

    fs::remove_all(dir, ec);

    std::cout << "\n=== Testbed 2: " << (pass ? "BESTANDEN" : "FEHLGESCHLAGEN") << " ===\n";
    return pass ? 0 : 1;
}

// testbed4_triple_buffer.cpp — Triple-Buffer-Pipeline (Design §16, TB 4).
//
// Der eigentliche TB 4 misst auf dem Server-PC per GPU-Profiler "Compute < 5ms
// idle zwischen Chunks". Hier wird die Pipeline-STRUKTUR host-seitig validiert:
//   Stream A (Load)    = chunk_streamer (OVERLAPPED I/O)
//   Stream B (Decomp)  = Zstd + INT->FP16 (decompressor_host)
//   Stream C (Compute) = simuliertes GEMM (FP16->float MAC, dominante Stufe)
//
// BESTANDEN wenn: (1) alle Chunks fehlerfrei durchlaufen, (2) Daten-Integrität
// — die Compute-Summe der Pipeline == sequenzielle Summe, (3) Overlap wirkt —
// Pipeline-Wall-Clock deutlich unter der seriellen Summe, Compute-Idle klein.
#include "InferEngine/chunk_streamer.h"
#include "InferEngine/decompressor.h"
#include "InferEngine/ring_buffer.h"
#include "ModelStore/nv4_format.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace nova;

namespace {

struct Cfg { int chunks = 24; uint32_t rows = 512, cols = 1024; };

std::vector<std::string> generate(const fs::path& dir, const Cfg& c) {
    fs::create_directories(dir);
    std::vector<std::string> paths;
    std::mt19937 rng(0xD00D);
    std::normal_distribution<float> dist(0.0f, 0.8f);
    const uint64_t n = uint64_t(c.rows) * c.cols;
    std::vector<float> data(static_cast<size_t>(n));
    for (int l = 0; l < c.chunks; ++l) {
        for (auto& v : data) v = dist(rng);
        auto q = modelstore::quantize_tensor(data.data(), n, modelstore::QuantType::INT2);
        char name[64]; std::snprintf(name, sizeof(name), "L%05u.nv4", l);
        const std::string p = (dir / name).string();
        modelstore::write_chunk(p, uint16_t(l), uint16_t(l), modelstore::QuantType::INT2,
                                c.rows, c.cols, q.scales, q.packed, /*compress=*/true);
        paths.push_back(p);
    }
    return paths;
}

// Simuliertes GEMM: reduziert die FP16-Gewichte zu einer Skalarsumme. Bewusst
// die teuerste Stufe (mehrere Durchläufe), damit Compute der Engpass ist und
// die I/O-/Decomp-Stufen dahinter "verschwinden".
double fake_gemm(const std::vector<uint16_t>& w) {
    double acc = 0.0;
    for (int pass = 0; pass < 4; ++pass)
        for (uint16_t h : w) acc += double(modelstore::half_to_float(h)) * 1.0001;
    return acc;
}

void print_stats(const char* label, const infer::PipelineStats& s) {
    std::printf("  %-10s total=%.2f ms  load=%.2f  decomp=%.2f  compute=%.2f  "
                "idle=%.2f  maxgap=%.3f ms\n",
                label, s.total_seconds * 1e3, s.load_busy_s * 1e3, s.decomp_busy_s * 1e3,
                s.compute_busy_s * 1e3, s.compute_idle_s * 1e3, s.max_compute_gap_ms);
}

}  // namespace

int main() {
    std::cout << "=== Testbed 4: Triple-Buffer-Pipeline ===\n";
    Cfg cfg;
    const fs::path dir = fs::temp_directory_path() / "nova4_tb4";
    std::error_code ec; fs::remove_all(dir, ec);
    std::cout << "[Gen] " << cfg.chunks << " Chunks (" << cfg.rows << "x" << cfg.cols
              << ", INT2+Zstd)\n";
    auto paths = generate(dir, cfg);

    infer::ChunkStreamer streamer(infer::StreamMode::SSD);

    // Stufen-Callbacks. Compute akkumuliert eine Gesamtsumme (Daten-Integrität).
    auto make_load = [&]() {
        return [&](int idx, std::vector<uint8_t>& raw) {
            return streamer.read_raw(paths[size_t(idx)], raw, nullptr);
        };
    };
    auto decomp = [&](int, std::vector<uint8_t>& raw, std::vector<uint16_t>& work) {
        modelstore::Chunk c;
        if (!modelstore::parse_chunk(raw.data(), raw.size(), c, nullptr)) return false;
        return infer::decompress_chunk_host(c, work, nullptr);
    };

    std::atomic<uint64_t> sum_bits_pipe{0}, sum_bits_seq{0};
    auto make_compute = [&](std::atomic<uint64_t>& sink) {
        return [&sink](int, std::vector<uint16_t>& work) {
            const double r = fake_gemm(work);
            uint64_t bits; std::memcpy(&bits, &r, sizeof(bits));
            // Reihenfolge-unabhängige Kombination (XOR der Bitmuster).
            sink.fetch_xor(bits);
            return true;
        };
    };

    infer::TripleBuffer tb(/*depth=*/3);
    std::string err;

    // (1) Pipeline (überlappt)
    auto sp = tb.run(cfg.chunks, make_load(), decomp, make_compute(sum_bits_pipe), &err);
    // (2) Seriell (Referenz)
    auto ss = tb.run_sequential(cfg.chunks, make_load(), decomp, make_compute(sum_bits_seq), &err);

    std::cout << "[Stats]\n";
    print_stats("pipeline", sp);
    print_stats("sequential", ss);

    bool pass = sp.ok && ss.ok && sp.chunks == uint64_t(cfg.chunks);

    // (2) Daten-Integrität: gleiche kombinierte Compute-Summe.
    const bool integrity = (sum_bits_pipe.load() == sum_bits_seq.load());
    std::cout << "[Integrität] pipeline == sequential: " << (integrity ? "JA" : "NEIN") << "\n";
    pass &= integrity;

    // (3) Overlap: Pipeline-Wall-Clock klar unter serieller Summe.
    const double serial_sum = ss.load_busy_s + ss.decomp_busy_s + ss.compute_busy_s;
    const double ratio = serial_sum > 0 ? sp.total_seconds / serial_sum : 1.0;
    std::printf("[Overlap] pipeline_total / serial_busy_sum = %.2f  (Ziel < 0.80)\n", ratio);
    pass &= (ratio < 0.80);

    fs::remove_all(dir, ec);
    std::cout << "\n=== Testbed 4: " << (pass ? "BESTANDEN" : "FEHLGESCHLAGEN") << " ===\n";
    return pass ? 0 : 1;
}

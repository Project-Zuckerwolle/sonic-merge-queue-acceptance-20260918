// ring_buffer.h — Triple-Buffer-Pipeline (Design §6.2, Testbed 4).
//
// Drei überlappende Stufen pro Chunk N:
//   Stream A (Load):    Quelle      -> input_buf   (async I/O / memcpy)
//   Stream B (Decomp):  input_buf   -> work_buf    (Zstd + INT->FP16)
//   Stream C (Compute): work_buf    -> activation  (GEMM); Gewichte freigeben
//
// Triple-Buffering (Tiefe 3): während Chunk N berechnet wird, dekomprimiert B
// Chunk N+1 und lädt A Chunk N+2. Ziel: Compute-Stufe nie warten lassen
// ("GPU < 5ms idle zwischen Chunks").
//
// Host-Backend: eine Thread pro Stufe, beschränkte Queues (Kapazität = Tiefe).
// Auf dem Server-PC werden die Stufen auf CUDA-Streams A/B/C mit Events
// abgebildet — gleiche Struktur, gleiche Invarianten.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace nova::infer {

struct PipelineStats {
    uint64_t chunks = 0;
    double   total_seconds   = 0.0;  // Wall-Clock der gesamten Pipeline
    double   load_busy_s     = 0.0;  // summierte Busy-Zeit Stream A
    double   decomp_busy_s   = 0.0;  // summierte Busy-Zeit Stream B
    double   compute_busy_s  = 0.0;  // summierte Busy-Zeit Stream C
    double   compute_idle_s  = 0.0;  // summierte Lücken der Compute-Stufe
    double   max_compute_gap_ms = 0.0;  // größte einzelne Compute-Lücke
    bool     ok = true;
};

class TripleBuffer {
public:
    // raw  = rohe Chunk-Bytes (Stream A Ausgabe)
    // work = dekomprimierte FP16-Gewichte (Stream B Ausgabe)
    using LoadFn    = std::function<bool(int idx, std::vector<uint8_t>& raw)>;
    using DecompFn  = std::function<bool(int idx, std::vector<uint8_t>& raw,
                                         std::vector<uint16_t>& work)>;
    using ComputeFn = std::function<bool(int idx, std::vector<uint16_t>& work)>;

    explicit TripleBuffer(int depth = 3) : depth_(depth < 1 ? 1 : depth) {}

    int depth() const { return depth_; }

    // Führt die Pipeline über num_chunks aus. Stufen laufen überlappt.
    PipelineStats run(int num_chunks, LoadFn load, DecompFn decomp,
                      ComputeFn compute, std::string* err = nullptr);

    // Referenz: dieselben Stufen rein sequenziell (zum Vergleich/zur Validierung).
    PipelineStats run_sequential(int num_chunks, LoadFn load, DecompFn decomp,
                                 ComputeFn compute, std::string* err = nullptr);

private:
    int depth_;
};

}  // namespace nova::infer

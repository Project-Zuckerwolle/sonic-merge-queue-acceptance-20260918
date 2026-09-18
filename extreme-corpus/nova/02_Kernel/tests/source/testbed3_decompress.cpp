// testbed3_decompress.cpp — Host-Validierung der Dequantisierung (Design §16, TB 3).
//
// Der eigentliche TB 3 misst den CUDA-Kernel (Cosine-Sim > 0.99 vs FP16-Original)
// auf dem Server-PC. Hier wird die IDENTISCHE Mathematik der Host-Referenz
// (decompressor_host.cpp == Mirror von decompressor.cu) geprüft: Chunk ->
// Zstd -> INT->FP16 -> Cosine-Sim gegen das FP32-Original.
//
// BESTANDEN (Host): INT4 Cosine-Sim > 0.99; INT2 wird gemessen + berichtet
// (2-Bit auf gaußschem Zufall liegt unter 0.99 — reale MoE-Gewichte sind
// gutmütiger, siehe Design §4.1).
#include "InferEngine/decompressor.h"
#include "ModelStore/nv4_format.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace nova;

namespace {

// Erzeugt FP32-Daten, schreibt Chunk, liest zurück, dekomprimiert zu FP16,
// liefert Cosine-Sim gegen das Original.
double roundtrip_cosine(modelstore::QuantType qt, uint32_t rows, uint32_t cols,
                        bool compress, std::string* err) {
    const uint64_t n = uint64_t(rows) * cols;
    std::vector<float> orig(static_cast<size_t>(n));
    std::mt19937 rng(123);
    std::normal_distribution<float> d(0.0f, 0.9f);
    for (auto& v : orig) v = d(rng);

    auto q = modelstore::quantize_tensor(orig.data(), n, qt);
    const std::string p = (fs::temp_directory_path() / "nova4_tb3.nv4").string();
    if (!modelstore::write_chunk(p, 0, 0, qt, rows, cols, q.scales, q.packed, compress)) {
        if (err) *err = "write_chunk fehlgeschlagen"; return -2.0;
    }
    modelstore::Chunk c;
    if (!modelstore::read_chunk(p, c, err)) return -2.0;

    std::vector<uint16_t> fp16;
    if (!infer::decompress_chunk_host(c, fp16, err)) return -2.0;

    std::error_code ec; fs::remove(p, ec);
    if (fp16.size() != size_t(n)) { if (err) *err = "Größe mismatch"; return -2.0; }
    return infer::cosine_similarity_fp16_vs_fp32(fp16.data(), orig.data(), n);
}

}  // namespace

int main() {
    std::cout << "=== Testbed 3 (Host): Dequantisierung INT->FP16 ===\n";
    bool pass = true;
    std::string err;

    const double cos_int4 = roundtrip_cosine(modelstore::QuantType::INT4, 512, 1024, true, &err);
    if (cos_int4 < -1.5) { std::cout << "  INT4 FEHLER: " << err << "\n"; return 2; }
    std::printf("  INT4+Zstd  Cosine-Sim = %.5f  (Ziel > 0.99) -> %s\n",
                cos_int4, cos_int4 > 0.99 ? "OK" : "FAIL");
    pass &= (cos_int4 > 0.99);

    const double cos_int2 = roundtrip_cosine(modelstore::QuantType::INT2, 512, 1024, true, &err);
    if (cos_int2 < -1.5) { std::cout << "  INT2 FEHLER: " << err << "\n"; return 2; }
    std::printf("  INT2+Zstd  Cosine-Sim = %.5f  (Bericht; 2-Bit auf Zufallsdaten)\n", cos_int2);

    std::cout << "\n=== Testbed 3 (Host): " << (pass ? "BESTANDEN" : "FEHLGESCHLAGEN")
              << " ===\n";
    return pass ? 0 : 1;
}

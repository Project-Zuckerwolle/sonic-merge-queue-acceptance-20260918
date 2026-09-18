// turboquant.h — TurboQuant KV-Cache-Kompression (Design §7, Testbed 6).
//
// TurboQuant (Google, ICLR 2026) komprimiert den KV-Cache online auf ~2,5 Bit/
// Wert. Drei Stufen (§7.1):
//   1. PolarQuant — zufällige orthogonale Rotation Q, danach gilt die Gauss-
//      Annahme (Kanäle annähernd normalverteilt). Q wird einmal aus einem Seed
//      erzeugt und liegt permanent im VRAM (hier: im TurboQuant-Objekt).
//   2. Outlier-Kanäle (~32/128) — 3-Bit Lloyd-Max-Quantisierung (per-Kanal-
//      Codebuch aus Kalibrierung).
//   3. Reguläre Kanäle — 2-Bit symmetrisch + 1-Bit QJL-Vorzeichenkorrektur.
//
// Die Mathematik ist host-seitig hier vollständig (turboquant_host.cpp); der
// CUDA-Pfad (turboquant.cu) spiegelt sie bit-kompatibel für den Server-PC.
//
// Dequant = Q^T * r_hat (Q orthogonal -> Inverse = Transponierte).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nova::infer {

struct TurboQuantConfig {
    int      head_dim     = 128;   // Gemma 4 KV-Head-Dim (§7.2)
    int      num_outliers = 32;    // 3-Bit Lloyd-Max-Kanäle (~32/128, §7.1)
    bool     qjl_correction = true;// 1-Bit QJL-Vorzeichenkorrektur für reguläre Kanäle
    uint64_t seed         = 0x6E6F76613471ull;  // "nova4q" — Rotation deterministisch
};

// Ein komprimierter Vektor (Länge head_dim). Layout kompakt, ~2,5–3 Bit/Wert.
struct CompressedVec {
    uint16_t              reg_scale_h = 0;   // FP16: 2-Bit-Skalierung reguläre Kanäle
    std::vector<uint8_t>  outlier_codes;     // je 3 Bit, gepackt (num_outliers Werte)
    std::vector<uint8_t>  reg_codes;         // je 2 Bit, gepackt (reguläre Kanäle)
    std::vector<uint8_t>  qjl_bits;          // je 1 Bit, gepackt (reguläre Kanäle), falls aktiv

    // Speicherbedarf dieses Vektors in Bytes (für Budget-Rechnung im KV-Manager).
    size_t bytes() const {
        return sizeof(uint16_t) + outlier_codes.size() + reg_codes.size() + qjl_bits.size();
    }
};

class TurboQuant {
public:
    explicit TurboQuant(TurboQuantConfig cfg);

    const TurboQuantConfig& config() const { return cfg_; }

    // Kalibrierung: bestimmt Outlier-Kanäle (höchste Varianz im rotierten Raum)
    // und 3-Bit-Lloyd-Max-Codebücher pro Outlier-Kanal aus Beispielvektoren.
    // vectors: num_vectors * head_dim FP32-Werte (zeilenweise). Ohne Kalibrierung
    // werden deterministische Defaults benutzt (erste num_outliers Kanäle, lineares
    // Codebuch) — funktioniert, aber schlechtere Qualität.
    void calibrate(const float* vectors, size_t num_vectors);

    // Online-Kompression eines head_dim-Vektors.
    CompressedVec compress(const float* vec) const;
    void          decompress(const CompressedVec& c, float* out) const;

    // Durchschnittliche Bit/Wert dieser Konfiguration (Diagnose).
    double bits_per_value() const;

    bool calibrated() const { return calibrated_; }

private:
    void rotate(const float* in, float* out) const;          // r = Q * in
    void rotate_transpose(const float* in, float* out) const;// v = Q^T * in

    TurboQuantConfig cfg_;
    std::vector<float> Q_;            // head_dim*head_dim, row-major, orthogonal
    std::vector<int>   outlier_idx_;  // sortierte Outlier-Kanal-Indizes
    std::vector<char>  is_outlier_;   // head_dim Flags
    // Lloyd-Max-Codebücher: outlier_idx_.size() * 8 Levels (FP32).
    std::vector<float> outlier_codebook_;
    int  num_regular_ = 0;
    bool calibrated_  = false;
};

#ifdef NOVA_HAVE_CUDA
// GPU-Pfad (turboquant.cu): batched Rotation r = Q * v (bzw. Q^T) — der
// dominante O(d^2)-Kostenanteil. in/out: num_vectors * head_dim FP32, Q row-major.
// transpose=false: r = Q*v (Kompression). transpose=true: v = Q^T*r (Dequant).
bool turboquant_rotate_batch_cuda(const float* Q, const float* in, float* out,
                                  int head_dim, int num_vectors, bool transpose,
                                  std::string* err = nullptr);
#endif

}  // namespace nova::infer

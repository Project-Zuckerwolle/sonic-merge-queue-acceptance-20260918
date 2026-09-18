// turboquant_host.cpp — Host-Referenz für turboquant.h (Design §7).
// Spiegelt die CUDA-Kernels aus turboquant.cu (gleiche Mathematik, bit-kompatibel).
#include "InferEngine/turboquant.h"

#include "ModelStore/nv4_format.h"  // float_to_half / half_to_float

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace nova::infer {

using modelstore::float_to_half;
using modelstore::half_to_float;

namespace {

// Packt/entpackt kleine Bit-Codes (2/3 Bit) in/aus einem Byte-Vektor.
void pack_codes(const std::vector<uint8_t>& codes, int bits, std::vector<uint8_t>& out) {
    const size_t total_bits = codes.size() * bits;
    out.assign((total_bits + 7) / 8, 0);
    for (size_t i = 0; i < codes.size(); ++i) {
        const size_t bitpos = i * bits;
        for (int b = 0; b < bits; ++b) {
            if (codes[i] & (1u << b)) {
                const size_t p = bitpos + b;
                out[p >> 3] |= uint8_t(1u << (p & 7));
            }
        }
    }
}

uint8_t get_code(const std::vector<uint8_t>& packed, size_t i, int bits) {
    const size_t bitpos = i * bits;
    uint8_t v = 0;
    for (int b = 0; b < bits; ++b) {
        const size_t p = bitpos + b;
        if (packed[p >> 3] & (1u << (p & 7))) v |= uint8_t(1u << b);
    }
    return v;
}

// Erzeugt eine orthogonale head_dim x head_dim Matrix per Gram-Schmidt auf
// gaußschen Spalten — deterministisch über seed (PolarQuant-Rotation §7.1).
void make_orthogonal(int d, uint64_t seed, std::vector<float>& Q) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<double> m(size_t(d) * d);
    for (auto& x : m) x = g(rng);
    // Modifiziertes Gram-Schmidt über die Zeilen (Zeile i = i-ter Basisvektor).
    for (int i = 0; i < d; ++i) {
        double* vi = &m[size_t(i) * d];
        for (int j = 0; j < i; ++j) {
            const double* vj = &m[size_t(j) * d];
            double dot = 0.0;
            for (int k = 0; k < d; ++k) dot += vi[k] * vj[k];
            for (int k = 0; k < d; ++k) vi[k] -= dot * vj[k];
        }
        double norm = 0.0;
        for (int k = 0; k < d; ++k) norm += vi[k] * vi[k];
        norm = std::sqrt(norm);
        if (norm < 1e-12) norm = 1.0;
        for (int k = 0; k < d; ++k) vi[k] /= norm;
    }
    Q.resize(size_t(d) * d);
    for (size_t i = 0; i < Q.size(); ++i) Q[i] = float(m[i]);
}

// Lloyd-Max (Max-Lloyd) Skalar-Quantisierer mit nlevels Stufen über samples.
// Liefert sortierte Repräsentanten. Iterativer k-means in 1D.
void lloyd_max(std::vector<float> samples, int nlevels, std::vector<float>& levels) {
    levels.assign(nlevels, 0.0f);
    if (samples.empty()) return;
    std::sort(samples.begin(), samples.end());
    // Initialisierung: Quantile.
    for (int i = 0; i < nlevels; ++i) {
        const double q = (i + 0.5) / nlevels;
        levels[i] = samples[std::min(samples.size() - 1,
                                     size_t(q * samples.size()))];
    }
    for (int iter = 0; iter < 25; ++iter) {
        std::vector<double> sum(nlevels, 0.0);
        std::vector<int>    cnt(nlevels, 0);
        for (float s : samples) {
            int best = 0; double bd = std::abs(s - levels[0]);
            for (int l = 1; l < nlevels; ++l) {
                const double d = std::abs(s - levels[l]);
                if (d < bd) { bd = d; best = l; }
            }
            sum[best] += s; cnt[best]++;
        }
        for (int l = 0; l < nlevels; ++l)
            if (cnt[l] > 0) levels[l] = float(sum[l] / cnt[l]);
    }
    std::sort(levels.begin(), levels.end());
}

}  // namespace

TurboQuant::TurboQuant(TurboQuantConfig cfg) : cfg_(cfg) {
    if (cfg_.head_dim <= 0) cfg_.head_dim = 128;
    if (cfg_.num_outliers < 0) cfg_.num_outliers = 0;
    if (cfg_.num_outliers > cfg_.head_dim) cfg_.num_outliers = cfg_.head_dim;
    make_orthogonal(cfg_.head_dim, cfg_.seed, Q_);

    // Default-Outlier (ohne Kalibrierung): erste num_outliers Kanäle.
    is_outlier_.assign(cfg_.head_dim, 0);
    outlier_idx_.clear();
    for (int i = 0; i < cfg_.num_outliers; ++i) { outlier_idx_.push_back(i); is_outlier_[i] = 1; }
    num_regular_ = cfg_.head_dim - cfg_.num_outliers;

    // Default-Codebuch: linear über [-3, 3] (Standardnormal-Spannweite).
    outlier_codebook_.assign(size_t(outlier_idx_.size()) * 8, 0.0f);
    for (size_t c = 0; c < outlier_idx_.size(); ++c)
        for (int l = 0; l < 8; ++l)
            outlier_codebook_[c * 8 + l] = -3.0f + 6.0f * (l + 0.5f) / 8.0f;
}

void TurboQuant::rotate(const float* in, float* out) const {
    const int d = cfg_.head_dim;
    for (int i = 0; i < d; ++i) {
        const float* row = &Q_[size_t(i) * d];
        float acc = 0.0f;
        for (int k = 0; k < d; ++k) acc += row[k] * in[k];
        out[i] = acc;
    }
}

void TurboQuant::rotate_transpose(const float* in, float* out) const {
    const int d = cfg_.head_dim;
    for (int k = 0; k < d; ++k) out[k] = 0.0f;
    for (int i = 0; i < d; ++i) {
        const float* row = &Q_[size_t(i) * d];
        const float xi = in[i];
        for (int k = 0; k < d; ++k) out[k] += row[k] * xi;
    }
}

void TurboQuant::calibrate(const float* vectors, size_t num_vectors) {
    const int d = cfg_.head_dim;
    if (num_vectors == 0) return;

    // Alle Vektoren rotieren, Per-Kanal-Statistik im rotierten Raum sammeln.
    std::vector<float> rot(size_t(num_vectors) * d);
    std::vector<double> mean(d, 0.0), var(d, 0.0);
    for (size_t v = 0; v < num_vectors; ++v) {
        rotate(vectors + v * d, &rot[v * d]);
        for (int k = 0; k < d; ++k) mean[k] += rot[v * d + k];
    }
    for (int k = 0; k < d; ++k) mean[k] /= num_vectors;
    for (size_t v = 0; v < num_vectors; ++v)
        for (int k = 0; k < d; ++k) {
            const double e = rot[v * d + k] - mean[k];
            var[k] += e * e;
        }
    for (int k = 0; k < d; ++k) var[k] /= num_vectors;

    // Outlier = Kanäle mit höchster Varianz.
    std::vector<int> order(d);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return var[a] > var[b]; });
    outlier_idx_.assign(order.begin(), order.begin() + cfg_.num_outliers);
    std::sort(outlier_idx_.begin(), outlier_idx_.end());
    is_outlier_.assign(d, 0);
    for (int c : outlier_idx_) is_outlier_[c] = 1;
    num_regular_ = d - cfg_.num_outliers;

    // 3-Bit Lloyd-Max-Codebuch pro Outlier-Kanal.
    outlier_codebook_.assign(size_t(outlier_idx_.size()) * 8, 0.0f);
    std::vector<float> col(num_vectors);
    for (size_t c = 0; c < outlier_idx_.size(); ++c) {
        const int ch = outlier_idx_[c];
        for (size_t v = 0; v < num_vectors; ++v) col[v] = rot[v * d + ch];
        std::vector<float> levels;
        lloyd_max(col, 8, levels);
        for (int l = 0; l < 8; ++l) outlier_codebook_[c * 8 + l] = levels[l];
    }
    calibrated_ = true;
}

CompressedVec TurboQuant::compress(const float* vec) const {
    const int d = cfg_.head_dim;
    std::vector<float> r(d);
    rotate(vec, r.data());

    CompressedVec out;

    // --- Outlier: 3-Bit, nächstes Codebuch-Level ---
    std::vector<uint8_t> ocodes(outlier_idx_.size());
    for (size_t c = 0; c < outlier_idx_.size(); ++c) {
        const float x = r[outlier_idx_[c]];
        const float* cb = &outlier_codebook_[c * 8];
        int best = 0; float bd = std::abs(x - cb[0]);
        for (int l = 1; l < 8; ++l) {
            const float dd = std::abs(x - cb[l]);
            if (dd < bd) { bd = dd; best = l; }
        }
        ocodes[c] = uint8_t(best);
    }
    pack_codes(ocodes, 3, out.outlier_codes);

    // --- Reguläre Kanäle: 2-Bit symmetrisch (mid-rise) + 1-Bit QJL ---
    float amax = 0.0f;
    for (int k = 0; k < d; ++k)
        if (!is_outlier_[k]) amax = std::max(amax, std::abs(r[k]));
    const float scale = amax > 0.0f ? amax / 1.5f : 1.0f;  // Levels {-1.5,-0.5,0.5,1.5}*scale
    out.reg_scale_h = float_to_half(scale);

    std::vector<uint8_t> rcodes; rcodes.reserve(num_regular_);
    std::vector<uint8_t> qbits;  qbits.reserve(num_regular_);
    const float lv[4] = {-1.5f, -0.5f, 0.5f, 1.5f};
    for (int k = 0; k < d; ++k) {
        if (is_outlier_[k]) continue;
        const float x = r[k];
        int best = 0; float bd = std::abs(x - lv[0] * scale);
        for (int l = 1; l < 4; ++l) {
            const float dd = std::abs(x - lv[l] * scale);
            if (dd < bd) { bd = dd; best = l; }
        }
        rcodes.push_back(uint8_t(best));
        if (cfg_.qjl_correction) {
            // QJL: Vorzeichen des Residuums -> 1 Bit Feinkorrektur (±scale/4).
            const float recon = lv[best] * scale;
            qbits.push_back(uint8_t(x - recon >= 0.0f ? 1 : 0));
        }
    }
    pack_codes(rcodes, 2, out.reg_codes);
    if (cfg_.qjl_correction) pack_codes(qbits, 1, out.qjl_bits);
    return out;
}

void TurboQuant::decompress(const CompressedVec& c, float* out) const {
    const int d = cfg_.head_dim;
    std::vector<float> r(d, 0.0f);

    for (size_t i = 0; i < outlier_idx_.size(); ++i) {
        const uint8_t code = get_code(c.outlier_codes, i, 3);
        r[outlier_idx_[i]] = outlier_codebook_[i * 8 + code];
    }

    const float scale = half_to_float(c.reg_scale_h);
    const float lv[4] = {-1.5f, -0.5f, 0.5f, 1.5f};
    size_t ri = 0;
    for (int k = 0; k < d; ++k) {
        if (is_outlier_[k]) continue;
        const uint8_t code = get_code(c.reg_codes, ri, 2);
        float val = lv[code] * scale;
        if (cfg_.qjl_correction && !c.qjl_bits.empty()) {
            const uint8_t qb = get_code(c.qjl_bits, ri, 1);
            val += (qb ? 1.0f : -1.0f) * (scale * 0.25f);
        }
        r[k] = val;
        ++ri;
    }

    rotate_transpose(r.data(), out);  // v_hat = Q^T r_hat
}

double TurboQuant::bits_per_value() const {
    const double outlier_bits = double(outlier_idx_.size()) * 3.0;
    const double reg_bits = double(num_regular_) * (cfg_.qjl_correction ? 3.0 : 2.0);
    const double scale_bits = 16.0;  // FP16 reg_scale je Vektor
    return (outlier_bits + reg_bits + scale_bits) / cfg_.head_dim;
}

}  // namespace nova::infer

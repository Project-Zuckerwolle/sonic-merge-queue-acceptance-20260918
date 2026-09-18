// fused_gemm_host.cpp — Host-Referenz für Fused INT3-Dequant-GEMM (②).
#include "InferEngine/fused_gemm.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace nova::infer {

namespace {
// 3 Bit an Bit-Offset schreiben/lesen (little-endian, Byte-straddelnd).
inline void put3(std::vector<uint8_t>& buf, size_t bitpos, uint8_t val) {
    for (int b = 0; b < 3; ++b)
        if ((val >> b) & 1u) buf[(bitpos + b) / 8] |= uint8_t(1u << ((bitpos + b) % 8));
}
inline uint8_t get3(const std::vector<uint8_t>& buf, size_t bitpos) {
    uint8_t v = 0;
    for (int b = 0; b < 3; ++b)
        if (buf[(bitpos + b) / 8] >> ((bitpos + b) % 8) & 1u) v |= uint8_t(1u << b);
    return v;
}
}  // namespace

Int3Matrix pack_int3(const float* W, int M, int K, int group) {
    Int3Matrix out;
    out.M = M; out.K = K; out.group = group;
    const int ng = out.groups_per_row();
    out.scales.assign(size_t(M) * ng, 0.0f);
    out.packed.assign((size_t(M) * K * 3 + 7) / 8, 0);

    for (int i = 0; i < M; ++i) {
        for (int g = 0; g < ng; ++g) {
            const int k0 = g * group;
            const int k1 = (k0 + group < K) ? k0 + group : K;
            float amax = 0.0f;
            for (int k = k0; k < k1; ++k) amax = std::max(amax, std::fabs(W[size_t(i) * K + k]));
            const float scale = amax > 0.0f ? amax / 4.0f : 1.0f;   // Level -4..3
            out.scales[size_t(i) * ng + g] = scale;
            for (int k = k0; k < k1; ++k) {
                int q = int(std::lround(W[size_t(i) * K + k] / scale));
                if (q < -4) q = -4; if (q > 3) q = 3;
                const uint8_t u = uint8_t(q + 4);   // signed -> 0..7
                put3(out.packed, (size_t(i) * K + k) * 3, u);
            }
        }
    }
    return out;
}

// Phase 2 FMT_BLOCK3: Gewichte bit-identisch zu pack_int3, Skalen zweistufig kodiert.
Int3Matrix pack_block3(const float* W, int M, int K, int group) {
    Int3Matrix out;
    out.M = M; out.K = K; out.group = group; out.fmt = SCALE_BLOCK3;
    const int ng  = out.groups_per_row();
    const int nsb = out.superblocks_per_row();
    out.packed.assign((size_t(M) * K * 3 + 7) / 8, 0);
    out.superscales.assign(size_t(M) * nsb, 0);
    out.subscales.assign(size_t(M) * nsb * 6, 0);   // 6 Byte je Superblock
    std::vector<float> gs(ng);
    for (int i = 0; i < M; ++i) {
        // 1. Per-Gruppe-Skala + Gewichte packen (identisch zu pack_int3 -> gleiche packed-Bytes).
        for (int g = 0; g < ng; ++g) {
            const int k0 = g * group;
            const int k1 = (k0 + group < K) ? k0 + group : K;
            float amax = 0.0f;
            for (int k = k0; k < k1; ++k) amax = std::max(amax, std::fabs(W[size_t(i) * K + k]));
            const float scale  = amax > 0.0f ? amax / 4.0f : 0.0f;   // Null-Gruppe -> 0
            gs[g] = scale;
            const float qscale = scale > 0.0f ? scale : 1.0f;
            for (int k = k0; k < k1; ++k) {
                int q = int(std::lround(W[size_t(i) * K + k] / qscale));
                if (q < -4) q = -4; if (q > 3) q = 3;
                put3(out.packed, (size_t(i) * K + k) * 3, uint8_t(q + 4));
            }
        }
        // 2. Zweistufige Kodierung der Gruppenskalen je Superblock (8 Gruppen).
        for (int sb = 0; sb < nsb; ++sb) {
            const int g0 = sb * 8;
            const int g1 = (g0 + 8 < ng) ? g0 + 8 : ng;
            float smax = 0.0f;
            for (int g = g0; g < g1; ++g) smax = std::max(smax, gs[g]);
            out.superscales[size_t(i) * nsb + sb] = b3_f32_to_f16(smax);
            const float inv = smax > 0.0f ? 63.0f / smax : 0.0f;
            for (int g = g0; g < g1; ++g) {
                int q6 = int(std::lround(gs[g] * inv));
                if (q6 < 0) q6 = 0; if (q6 > 63) q6 = 63;
                if (gs[g] > 0.0f && q6 == 0) q6 = 1;   // eine Nicht-Null-Gruppe nie auf 0 kippen
                b3_put6(out.subscales, (size_t(i) * nsb + sb) * 6, g - g0, uint8_t(q6));
            }
        }
    }
    return out;
}

void block3_expand_scales(const Int3Matrix& W, std::vector<float>& out) {
    const int ng = W.groups_per_row(), nsb = W.superblocks_per_row();
    out.assign(size_t(W.M) * ng, 0.0f);
    for (int i = 0; i < W.M; ++i)
        for (int sb = 0; sb < nsb; ++sb) {
            const float sup = b3_f16_to_f32(W.superscales[size_t(i) * nsb + sb]);
            const int g0 = sb * 8;
            const int g1 = (g0 + 8 < ng) ? g0 + 8 : ng;
            for (int g = g0; g < g1; ++g) {
                const uint8_t q6 = b3_get6(W.subscales, (size_t(i) * nsb + sb) * 6, g - g0);
                out[size_t(i) * ng + g] = sup / 63.0f * float(q6);
            }
        }
}

void fused_int3_gemv(const Int3Matrix& W, const float* x, float* y) {
    const int ng = W.groups_per_row();
    for (int i = 0; i < W.M; ++i) {
        double acc = 0.0;
        const size_t row_bit = size_t(i) * W.K * 3;
        for (int k = 0; k < W.K; ++k) {
            const int q = int(get3(W.packed, row_bit + size_t(k) * 3)) - 4;   // inline entpackt
            const float scale = W.scales[size_t(i) * ng + k / W.group];
            acc += double(q) * double(scale) * double(x[k]);                  // inline dequant·x
        }
        y[i] = float(acc);
    }
}

void fused_int3_gemm(const Int3Matrix& W, const float* X, int N, float* Y) {
    const int ng = W.groups_per_row();
    std::vector<float> acc(N);
    for (int i = 0; i < W.M; ++i) {
        for (int n = 0; n < N; ++n) acc[n] = 0.0f;
        const size_t row_bit = size_t(i) * W.K * 3;
        for (int k = 0; k < W.K; ++k) {
            const float qs = float(int(get3(W.packed, row_bit + size_t(k) * 3)) - 4)
                           * W.scales[size_t(i) * ng + k / W.group];   // Zeile 1× entpackt
            const float* xr = X + size_t(k) * N;
            for (int n = 0; n < N; ++n) acc[n] += qs * xr[n];
        }
        float* yr = Y + size_t(i) * N;
        for (int n = 0; n < N; ++n) yr[n] = acc[n];
    }
}

void reference_gemv(const float* W, int M, int K, const float* x, float* y) {
    for (int i = 0; i < M; ++i) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) acc += double(W[size_t(i) * K + k]) * double(x[k]);
        y[i] = float(acc);
    }
}

}  // namespace nova::infer

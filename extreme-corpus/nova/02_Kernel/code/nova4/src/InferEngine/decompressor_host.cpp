// decompressor_host.cpp — Host-Referenz für decompressor.h.
// Spiegelt die CUDA-Kernels aus decompressor.cu (gleiche Quant-Mathematik).
#include "InferEngine/decompressor.h"

#include <cmath>

namespace nova::infer {

using modelstore::QuantType;
using modelstore::GROUP_SIZE;
using modelstore::half_to_float;
using modelstore::float_to_half;

void dequantize_to_fp16_host(const uint8_t* packed, const uint8_t* scales,
                             uint64_t n, QuantType qt, uint16_t* out) {
    const int bias = (qt == QuantType::INT4) ? 8 : (qt == QuantType::INT3) ? 4 : 2;

    for (uint64_t i = 0; i < n; ++i) {
        const uint64_t g = i / GROUP_SIZE;
        const uint16_t sbits = uint16_t(scales[g * 2]) | (uint16_t(scales[g * 2 + 1]) << 8);
        const float scale = half_to_float(sbits);
        int code;
        if (qt == QuantType::INT4) {
            code = (packed[i >> 1] >> ((i & 1) * 4)) & 0xF;
        } else if (qt == QuantType::INT3) {
            const uint64_t bp = i * 3;   // byte-straddelnde 3-Bit-Codes
            code = 0;
            for (int b = 0; b < 3; ++b)
                if (packed[(bp + b) / 8] >> ((bp + b) % 8) & 1u) code |= (1 << b);
        } else {
            code = (packed[i >> 2] >> ((i & 3) * 2)) & 0x3;
        }
        out[i] = float_to_half(float(code - bias) * scale);
    }
}

bool decompress_chunk_host(const modelstore::Chunk& c,
                           std::vector<uint16_t>& fp16_out, std::string* err) {
    std::vector<uint8_t> packed;
    if (!modelstore::decompress_payload(c, packed, err)) return false;

    const uint64_t n = c.header.num_elements();
    const size_t need = modelstore::packed_bytes(n, modelstore::QuantType(c.header.quant_type));
    if (packed.size() < need) { if (err) *err = "Payload kürzer als erwartet"; return false; }

    fp16_out.resize(static_cast<size_t>(n));
    dequantize_to_fp16_host(packed.data(), c.scales.data(), n,
                            modelstore::QuantType(c.header.quant_type), fp16_out.data());
    return true;
}

double cosine_similarity_fp16_vs_fp32(const uint16_t* a_half, const float* b, uint64_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (uint64_t i = 0; i < n; ++i) {
        const double a = half_to_float(a_half[i]);
        dot += a * b[i];
        na  += a * a;
        nb  += double(b[i]) * b[i];
    }
    const double denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 0.0 ? dot / denom : 0.0;
}

}  // namespace nova::infer

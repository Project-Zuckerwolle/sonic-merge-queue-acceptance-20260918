// nv4_format.cpp — Implementierung der Nova-4-Storage-Formate.
#include "ModelStore/nv4_format.h"

#include <zstd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>

namespace nova::modelstore {

// ===========================================================================
// CRC32 (table-based, IEEE poly 0xEDB88320)
// ===========================================================================
namespace {
struct Crc32Table {
    std::array<uint32_t, 256> t{};
    Crc32Table() {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k)
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
    }
};
const Crc32Table g_crc;

// Little-endian Hilfsfunktionen
void put_u16(uint8_t* p, uint16_t v) { p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); }
void put_u32(uint8_t* p, uint32_t v) {
    p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); p[2] = uint8_t(v >> 16); p[3] = uint8_t(v >> 24);
}
uint16_t get_u16(const uint8_t* p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
uint32_t get_u32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
}  // namespace

uint32_t crc32(const void* data, size_t len, uint32_t seed) {
    uint32_t c = seed ^ 0xFFFFFFFFu;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i)
        c = g_crc.t[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

// ===========================================================================
// FP16 <-> FP32  (IEEE 754 half, round-to-nearest-even-ish)
// ===========================================================================
uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = int32_t((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;

    if (((x >> 23) & 0xFF) == 0xFF) {  // Inf/NaN
        return uint16_t(sign | 0x7C00u | (mant ? 0x200u : 0));
    }
    if (exp >= 0x1F) return uint16_t(sign | 0x7C00u);  // Overflow -> Inf
    if (exp <= 0) {                                     // Subnormal/Underflow
        if (exp < -10) return uint16_t(sign);
        mant |= 0x800000u;
        int shift = 14 - exp;
        uint32_t half_mant = mant >> shift;
        // Rundung
        if ((mant >> (shift - 1)) & 1) half_mant += 1;
        return uint16_t(sign | half_mant);
    }
    uint16_t half = uint16_t(sign | (uint32_t(exp) << 10) | (mant >> 13));
    if (mant & 0x1000u) half += 1;  // round half up
    return half;
}

float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t(h) & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {  // Subnormal
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
            mant &= 0x3FF;
            out = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

// ===========================================================================
// Quantisierung
// ===========================================================================
size_t num_groups(uint64_t n) { return size_t((n + GROUP_SIZE - 1) / GROUP_SIZE); }

size_t packed_bytes(uint64_t n, QuantType qt) {
    const int bits = (qt == QuantType::INT4) ? 4 : (qt == QuantType::INT3) ? 3 : 2;
    return size_t((n * bits + 7) / 8);
}

namespace {
// 3-Bit-Code an Bit-Offset schreiben/lesen (little-endian, byte-straddelnd).
inline void put3bits(std::vector<uint8_t>& buf, uint64_t bitpos, uint8_t code) {
    for (int b = 0; b < 3; ++b)
        if ((code >> b) & 1u) buf[(bitpos + b) / 8] |= uint8_t(1u << ((bitpos + b) % 8));
}
inline int get3bits(const std::vector<uint8_t>& buf, uint64_t bitpos) {
    int code = 0;
    for (int b = 0; b < 3; ++b)
        if (buf[(bitpos + b) / 8] >> ((bitpos + b) % 8) & 1u) code |= (1 << b);
    return code;
}
}  // namespace

QuantResult quantize_tensor(const float* data, uint64_t n, QuantType qt) {
    QuantResult r;
    const size_t groups = num_groups(n);
    r.scales.resize(groups * 2);
    r.packed.assign(packed_bytes(n, qt), 0);

    // Symmetrische Level pro Typ: qmax = positiver Maximalcode.
    float qmax; int lo, hi, bias;
    switch (qt) {
        case QuantType::INT4: qmax = 7.0f; lo = -8; hi = 7; bias = 8; break;
        case QuantType::INT3: qmax = 3.0f; lo = -4; hi = 3; bias = 4; break;
        default:              qmax = 1.0f; lo = -2; hi = 1; bias = 2; break;  // INT2
    }

    for (size_t g = 0; g < groups; ++g) {
        const uint64_t start = uint64_t(g) * GROUP_SIZE;
        const uint64_t end   = std::min<uint64_t>(start + GROUP_SIZE, n);

        float amax = 0.0f;
        for (uint64_t i = start; i < end; ++i)
            amax = std::max(amax, std::fabs(data[i]));

        // Skala so, dass amax auf qmax abgebildet wird (symmetrisch).
        float scale = (amax > 0.0f) ? (amax / qmax) : 1.0f;
        put_u16(&r.scales[g * 2], float_to_half(scale));
        const float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

        for (uint64_t i = start; i < end; ++i) {
            int q = int(std::lround(data[i] * inv));
            q = std::min(hi, std::max(lo, q));
            const uint8_t code = uint8_t(q + bias);
            if (qt == QuantType::INT4) {
                r.packed[i >> 1] |= uint8_t(code << ((i & 1) * 4));
            } else if (qt == QuantType::INT3) {
                put3bits(r.packed, i * 3, code);
            } else {  // INT2: 4 Codes pro Byte
                r.packed[i >> 2] |= uint8_t((code & 0x3) << ((i & 3) * 2));
            }
        }
    }
    return r;
}

std::vector<float> dequantize_tensor(const std::vector<uint8_t>& scales,
                                     const std::vector<uint8_t>& packed,
                                     uint64_t n, QuantType qt) {
    std::vector<float> out(static_cast<size_t>(n));
    const int bias = (qt == QuantType::INT4) ? 8 : (qt == QuantType::INT3) ? 4 : 2;

    for (uint64_t i = 0; i < n; ++i) {
        const size_t g = size_t(i / GROUP_SIZE);
        const float scale = half_to_float(get_u16(&scales[g * 2]));
        int code;
        if (qt == QuantType::INT4) {
            code = (packed[i >> 1] >> ((i & 1) * 4)) & 0xF;
        } else if (qt == QuantType::INT3) {
            code = get3bits(packed, i * 3);
        } else {
            code = (packed[i >> 2] >> ((i & 3) * 2)) & 0x3;
        }
        out[size_t(i)] = float(code - bias) * scale;
    }
    return out;
}

// ===========================================================================
// Header serialisieren / parsen
// ===========================================================================
size_t Nv4Header::scale_bytes() const {
    return num_groups(num_elements()) * 2;
}

bool Nv4Header::magic_ok() const {
    return std::memcmp(magic, NV4_MAGIC, 4) == 0;
}

void Nv4Header::serialize(uint8_t o[NV4_HEADER_SIZE]) const {
    std::memcpy(o + 0, magic, 4);
    put_u16(o + 4, version);
    put_u16(o + 6, layer_index);
    put_u16(o + 8, chunk_index);
    o[10] = quant_type;
    o[11] = flags;
    put_u32(o + 12, crc32);
    put_u32(o + 16, shape0);
    put_u32(o + 20, shape1);
    put_u32(o + 24, compressed_size);
    put_u32(o + 28, uncompressed_size);
}

bool Nv4Header::parse(const uint8_t in[NV4_HEADER_SIZE], Nv4Header& h) {
    std::memcpy(h.magic, in + 0, 4);
    h.version           = get_u16(in + 4);
    h.layer_index       = get_u16(in + 6);
    h.chunk_index       = get_u16(in + 8);
    h.quant_type        = in[10];
    h.flags             = in[11];
    h.crc32             = get_u32(in + 12);
    h.shape0            = get_u32(in + 16);
    h.shape1            = get_u32(in + 20);
    h.compressed_size   = get_u32(in + 24);
    h.uncompressed_size = get_u32(in + 28);
    return h.magic_ok();
}

void Nv4bHeader::serialize(uint8_t o[NV4B_HEADER_SIZE]) const {
    std::memcpy(o + 0, magic, 4);
    put_u16(o + 4, version);
    put_u32(o + 6, layers);
    put_u16(o + 10, hidden_dim);
    put_u32(o + 12, checksum);
}

bool Nv4bHeader::parse(const uint8_t in[NV4B_HEADER_SIZE], Nv4bHeader& h) {
    std::memcpy(h.magic, in + 0, 4);
    h.version    = get_u16(in + 4);
    h.layers     = get_u32(in + 6);
    h.hidden_dim = get_u16(in + 10);
    h.checksum   = get_u32(in + 12);
    return std::memcmp(h.magic, NV4B_MAGIC, 4) == 0;
}

// ===========================================================================
// Chunk schreiben / lesen
// ===========================================================================
bool write_chunk(const std::string& path,
                 uint16_t layer_index, uint16_t chunk_index, QuantType qt,
                 uint32_t shape0, uint32_t shape1,
                 const std::vector<uint8_t>& scales,
                 const std::vector<uint8_t>& quant_packed,
                 bool compress) {
    std::vector<uint8_t> payload;
    uint8_t flags = NV4_FLAG_NONE;

    if (compress) {
        const size_t bound = ZSTD_compressBound(quant_packed.size());
        payload.resize(bound);
        const size_t csz = ZSTD_compress(payload.data(), bound,
                                         quant_packed.data(), quant_packed.size(), 19);
        if (ZSTD_isError(csz)) return false;
        // Nur komprimiert speichern wenn es tatsächlich kleiner ist.
        if (csz < quant_packed.size()) {
            payload.resize(csz);
            flags |= NV4_FLAG_ZSTD;
        } else {
            payload = quant_packed;
        }
    } else {
        payload = quant_packed;
    }

    Nv4Header h{};
    std::memcpy(h.magic, NV4_MAGIC, 4);
    h.version           = NV4_VERSION;
    h.layer_index       = layer_index;
    h.chunk_index       = chunk_index;
    h.quant_type        = uint8_t(qt);
    h.flags             = flags;
    h.shape0            = shape0;
    h.shape1            = shape1;
    h.compressed_size   = uint32_t(payload.size());
    h.uncompressed_size = uint32_t(quant_packed.size());

    // CRC32 über Scales + Payload (alles nach dem Header).
    uint32_t c = crc32(scales.data(), scales.size(), 0);
    c = crc32(payload.data(), payload.size(), c);
    h.crc32 = c;

    uint8_t hdr[NV4_HEADER_SIZE];
    h.serialize(hdr);

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(hdr), NV4_HEADER_SIZE);
    if (!scales.empty())  f.write(reinterpret_cast<const char*>(scales.data()), scales.size());
    if (!payload.empty()) f.write(reinterpret_cast<const char*>(payload.data()), payload.size());
    return bool(f);
}

bool parse_chunk(const uint8_t* buf, size_t len, Chunk& out, std::string* err) {
    if (len < NV4_HEADER_SIZE) { if (err) *err = "Puffer < Header"; return false; }
    if (!Nv4Header::parse(buf, out.header)) { if (err) *err = "ungültiges Magic"; return false; }

    const size_t sbytes = out.header.scale_bytes();
    const size_t psize  = out.header.compressed_size;
    if (len < NV4_HEADER_SIZE + sbytes + psize) {
        if (err) *err = "Puffer zu kurz für Scales+Payload"; return false;
    }
    const uint8_t* p = buf + NV4_HEADER_SIZE;
    out.scales.assign(p, p + sbytes);
    p += sbytes;
    out.payload.assign(p, p + psize);
    return true;
}

bool read_chunk(const std::string& path, Chunk& out, std::string* err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { if (err) *err = "kann Datei nicht öffnen: " + path; return false; }
    const std::streamoff sz = f.tellg();
    if (sz < std::streamoff(NV4_HEADER_SIZE)) { if (err) *err = "Datei zu klein: " + path; return false; }
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    if (f.gcount() != sz) { if (err) *err = "Lesefehler: " + path; return false; }
    return parse_chunk(buf.data(), buf.size(), out, err);
}

bool decompress_payload(const Chunk& c, std::vector<uint8_t>& out, std::string* err) {
    if (!(c.header.flags & NV4_FLAG_ZSTD)) {
        out = c.payload;  // war nicht komprimiert
        return true;
    }
    out.resize(c.header.uncompressed_size);
    const size_t dsz = ZSTD_decompress(out.data(), out.size(),
                                       c.payload.data(), c.payload.size());
    if (ZSTD_isError(dsz) || dsz != c.header.uncompressed_size) {
        if (err) *err = std::string("Zstd-Dekomprimierung fehlgeschlagen: ") +
                        ZSTD_getErrorName(dsz);
        return false;
    }
    return true;
}

}  // namespace nova::modelstore

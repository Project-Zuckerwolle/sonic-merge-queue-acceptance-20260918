// gguf_reader.cpp — siehe gguf_reader.h
#include "ModelStore/gguf_reader.h"
#include "ModelStore/nv4_format.h"  // half_to_float

#include <cstring>
#include <fstream>

namespace nova::modelstore {

namespace {
constexpr uint32_t GGUF_MAGIC = 0x46554747u;  // "GGUF" little-endian

// GGUF Metadata-Value-Typen
enum GgufValType : uint32_t {
    GGUF_UINT8 = 0, GGUF_INT8, GGUF_UINT16, GGUF_INT16, GGUF_UINT32, GGUF_INT32,
    GGUF_FLOAT32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_UINT64, GGUF_INT64,
    GGUF_FLOAT64,
};

size_t type_size(uint32_t t) {
    switch (t) {
        case GGUF_UINT8: case GGUF_INT8: case GGUF_BOOL:   return 1;
        case GGUF_UINT16: case GGUF_INT16:                 return 2;
        case GGUF_UINT32: case GGUF_INT32: case GGUF_FLOAT32: return 4;
        case GGUF_UINT64: case GGUF_INT64: case GGUF_FLOAT64: return 8;
        default: return 0;  // STRING/ARRAY: variabel
    }
}

struct Cursor {
    std::ifstream& f;
    bool ok = true;
    template <typename T> T read() {
        T v{};
        f.read(reinterpret_cast<char*>(&v), sizeof(T));
        if (f.gcount() != std::streamsize(sizeof(T))) ok = false;
        return v;
    }
    std::string read_str() {
        uint64_t n = read<uint64_t>();
        if (!ok || n > (1u << 30)) { ok = false; return {}; }
        std::string s(size_t(n), '\0');
        if (n) {
            f.read(s.data(), std::streamsize(n));
            if (f.gcount() != std::streamsize(n)) ok = false;
        }
        return s;
    }
    void skip(std::streamoff n) { f.seekg(n, std::ios::cur); }
};

// Liest einen Metadata-Wert; gibt bei STRING optional den Text zurück.
bool read_value(Cursor& c, uint32_t vtype, std::string* str_out) {
    if (vtype == GGUF_STRING) {
        std::string s = c.read_str();
        if (str_out) *str_out = s;
        return c.ok;
    }
    if (vtype == GGUF_ARRAY) {
        uint32_t elem_type = c.read<uint32_t>();
        uint64_t count     = c.read<uint64_t>();
        if (!c.ok) return false;
        if (elem_type == GGUF_STRING) {
            for (uint64_t i = 0; i < count; ++i) { c.read_str(); if (!c.ok) return false; }
        } else if (elem_type == GGUF_ARRAY) {
            return false;  // verschachtelte Arrays nicht unterstützt
        } else {
            const size_t ts = type_size(elem_type);
            if (ts == 0) return false;
            c.skip(std::streamoff(ts * count));
        }
        return c.ok;
    }
    const size_t ts = type_size(vtype);
    if (ts == 0) return false;
    c.skip(std::streamoff(ts));
    return c.ok;
}
}  // namespace

uint64_t GgufTensorInfo::num_elements() const {
    uint64_t n = 1;
    for (uint64_t d : dims) n *= d;
    return dims.empty() ? 0 : n;
}

const char* GgufReader::ggml_type_name(uint32_t t) {
    switch (t) {
        case GGML_TYPE_F32:  return "F32";
        case GGML_TYPE_F16:  return "F16";
        case GGML_TYPE_Q4_0: return "Q4_0";
        case GGML_TYPE_Q4_1: return "Q4_1";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q2_K: return "Q2_K";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        default:             return "UNKNOWN";
    }
}

bool GgufReader::open(const std::string& path, std::string* err) {
    path_ = path;
    std::ifstream f(path, std::ios::binary);
    if (!f) { if (err) *err = "kann GGUF nicht öffnen: " + path; return false; }
    Cursor c{f};

    uint32_t magic = c.read<uint32_t>();
    if (!c.ok || magic != GGUF_MAGIC) {
        if (err) *err = "kein GGUF-Magic: " + path; return false;
    }
    version_ = c.read<uint32_t>();
    if (version_ < 2 || version_ > 3) {
        if (err) *err = "nicht unterstützte GGUF-Version: " + std::to_string(version_);
        return false;
    }
    uint64_t tensor_count = c.read<uint64_t>();
    uint64_t kv_count     = c.read<uint64_t>();
    if (!c.ok) { if (err) *err = "GGUF-Header beschädigt"; return false; }

    // Metadata-KV
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key   = c.read_str();
        uint32_t    vtype = c.read<uint32_t>();
        if (!c.ok) { if (err) *err = "Metadata-KV beschädigt bei #" + std::to_string(i); return false; }
        std::string sval;
        if (!read_value(c, vtype, &sval)) {
            if (err) *err = "kann Metadata-Wert nicht lesen: " + key; return false;
        }
        if (vtype == GGUF_STRING) meta_str_[key] = sval;
        if (key == "general.alignment") {
            // alignment ist UINT32 — wir haben es übersprungen; selten gesetzt.
        }
    }

    // Tensor-Infos
    tensors_.reserve(size_t(tensor_count));
    for (uint64_t i = 0; i < tensor_count; ++i) {
        GgufTensorInfo t;
        t.name = c.read_str();
        uint32_t ndims = c.read<uint32_t>();
        if (!c.ok || ndims > 8) { if (err) *err = "Tensor-Dims ungültig: " + t.name; return false; }
        t.dims.resize(ndims);
        for (uint32_t d = 0; d < ndims; ++d) t.dims[d] = c.read<uint64_t>();
        t.ggml_type = c.read<uint32_t>();
        t.offset    = c.read<uint64_t>();
        if (!c.ok) { if (err) *err = "Tensor-Info beschädigt: " + t.name; return false; }
        tensors_.push_back(std::move(t));
    }

    // Datenblock beginnt am nächsten alignment-Vielfachen nach den Infos.
    std::streamoff pos = f.tellg();
    if (pos < 0) { if (err) *err = "tellg fehlgeschlagen"; return false; }
    const uint64_t align = alignment_ ? alignment_ : 32;
    data_off_ = (uint64_t(pos) + align - 1) / align * align;
    return true;
}

namespace {
// ---------------------------------------------------------------------------
// ggml-Quant-Dequantisierung (kanonische Block-Layouts aus ggml). Reicht für
// die Ollama-Default-Quants Q4_K + Q6_K (Mistral-GGUF) sowie Q4_0/Q8_0.
// QK_K = 256 Werte pro Superblock, Q4_0/Q8_0 = 32 Werte pro Block.
// ---------------------------------------------------------------------------
constexpr int QK_K = 256;

inline uint16_t rd_u16(const uint8_t* p) {
    uint16_t v; std::memcpy(&v, p, 2); return v;  // GGUF ist little-endian
}

// 6-Bit-Scale/Min aus den 12 gepackten scales-Bytes (ggml get_scale_min_k4).
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else {
        d = uint8_t((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        m = uint8_t((q[j + 4] >>   4) | ((q[j - 0] >> 6) << 4));
    }
}

// Bytegröße eines quantisierten Tensors mit n Elementen (0 = nicht unterstützt).
uint64_t quant_tensor_bytes(uint32_t type, uint64_t n) {
    switch (type) {
        case GGML_TYPE_Q4_0: return (n / 32)  * 18;
        case GGML_TYPE_Q8_0: return (n / 32)  * 34;
        case GGML_TYPE_Q4_K: return (n / 256) * 144;
        case GGML_TYPE_Q6_K: return (n / 256) * 210;
        default:             return 0;
    }
}

uint64_t quant_block_elems(uint32_t type) {
    switch (type) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q8_0: return 32;
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q6_K: return 256;
        default:                                  return 0;
    }
}

// Dequantisiert n Elemente aus rohen Block-Bytes nach FP32.
bool dequant_blocks(uint32_t type, const uint8_t* src, uint64_t n, float* out) {
    switch (type) {
        case GGML_TYPE_Q4_0: {
            const uint64_t nb = n / 32;
            for (uint64_t i = 0; i < nb; ++i) {
                const uint8_t* b = src + i * 18;
                const float d = half_to_float(rd_u16(b));
                const uint8_t* qs = b + 2;
                float* y = out + i * 32;
                for (int j = 0; j < 16; ++j) {
                    y[j]      = ((qs[j] & 0x0F) - 8) * d;
                    y[j + 16] = ((qs[j] >>   4) - 8) * d;
                }
            }
            return true;
        }
        case GGML_TYPE_Q8_0: {
            const uint64_t nb = n / 32;
            for (uint64_t i = 0; i < nb; ++i) {
                const uint8_t* b = src + i * 34;
                const float d = half_to_float(rd_u16(b));
                const int8_t* qs = reinterpret_cast<const int8_t*>(b + 2);
                float* y = out + i * 32;
                for (int j = 0; j < 32; ++j) y[j] = qs[j] * d;
            }
            return true;
        }
        case GGML_TYPE_Q4_K: {
            const uint64_t nb = n / QK_K;
            for (uint64_t i = 0; i < nb; ++i) {
                const uint8_t* b = src + i * 144;
                const float d    = half_to_float(rd_u16(b));
                const float dmin = half_to_float(rd_u16(b + 2));
                const uint8_t* scales = b + 4;    // 12 Byte
                const uint8_t* q      = b + 16;   // 128 Byte (4-Bit-Quants)
                float* y = out + i * QK_K;
                int is = 0;
                for (int j = 0; j < QK_K; j += 64) {
                    uint8_t sc, m, sc2, m2;
                    get_scale_min_k4(is + 0, scales, sc,  m);
                    get_scale_min_k4(is + 1, scales, sc2, m2);
                    const float d1 = d * sc,  min1 = dmin * m;
                    const float d2 = d * sc2, min2 = dmin * m2;
                    for (int l = 0; l < 32; ++l) y[l]      = d1 * (q[l] & 0xF) - min1;
                    for (int l = 0; l < 32; ++l) y[l + 32] = d2 * (q[l] >> 4)  - min2;
                    q += 32; is += 2; y += 64;
                }
            }
            return true;
        }
        case GGML_TYPE_Q6_K: {
            const uint64_t nb = n / QK_K;
            for (uint64_t i = 0; i < nb; ++i) {
                const uint8_t* b  = src + i * 210;
                const uint8_t* ql = b;          // 128 Byte (untere 4 Bit)
                const uint8_t* qh = b + 128;    // 64 Byte  (obere 2 Bit)
                const int8_t*  sc = reinterpret_cast<const int8_t*>(b + 192);  // 16 Byte
                const float d = half_to_float(rd_u16(b + 208));
                float* y = out + i * QK_K;
                for (int seg = 0; seg < QK_K; seg += 128) {
                    for (int l = 0; l < 32; ++l) {
                        const int is = l / 16;
                        const int q1 = int((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int q3 = int((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int q4 = int((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        y[l +  0] = d * sc[is + 0] * q1;
                        y[l + 32] = d * sc[is + 2] * q2;
                        y[l + 64] = d * sc[is + 4] * q3;
                        y[l + 96] = d * sc[is + 6] * q4;
                    }
                    ql += 64; qh += 32; sc += 8; y += 128;
                }
            }
            return true;
        }
        default: return false;
    }
}
}  // namespace

bool GgufReader::read_tensor_f32(const GgufTensorInfo& t, std::vector<float>& out,
                                 std::string* err) {
    const uint64_t n = t.num_elements();
    out.resize(size_t(n));

    std::ifstream f(path_, std::ios::binary);
    if (!f) { if (err) *err = "kann GGUF erneut nicht öffnen"; return false; }
    f.seekg(std::streamoff(data_off_ + t.offset), std::ios::beg);

    if (t.ggml_type == GGML_TYPE_F32) {
        f.read(reinterpret_cast<char*>(out.data()), std::streamsize(n * 4));
        if (f.gcount() != std::streamsize(n * 4)) { if (err) *err = "F32-Daten zu kurz"; return false; }
        return true;
    }
    if (t.ggml_type == GGML_TYPE_F16) {
        std::vector<uint16_t> raw(static_cast<size_t>(n));
        f.read(reinterpret_cast<char*>(raw.data()), std::streamsize(n * 2));
        if (f.gcount() != std::streamsize(n * 2)) { if (err) *err = "F16-Daten zu kurz"; return false; }
        for (uint64_t i = 0; i < n; ++i) out[size_t(i)] = half_to_float(raw[size_t(i)]);
        return true;
    }

    // Quantisierte Quelltypen (Q4_K/Q6_K/Q4_0/Q8_0): roh lesen + dequantisieren.
    const uint64_t be = quant_block_elems(t.ggml_type);
    const uint64_t bytes = quant_tensor_bytes(t.ggml_type, n);
    if (be == 0 || bytes == 0) {
        if (err) *err = std::string("Tensor '") + t.name + "' hat ggml-Typ " +
                        ggml_type_name(t.ggml_type) +
                        " — Dequantisierung dieses Typs nicht implementiert.";
        return false;
    }
    if (n % be != 0) {
        if (err) *err = std::string("Tensor '") + t.name + "': Elementzahl " +
                        std::to_string(n) + " nicht durch Blockgröße " +
                        std::to_string(be) + " teilbar.";
        return false;
    }
    std::vector<uint8_t> raw(static_cast<size_t>(bytes));
    f.read(reinterpret_cast<char*>(raw.data()), std::streamsize(bytes));
    if (f.gcount() != std::streamsize(bytes)) { if (err) *err = "Quant-Daten zu kurz"; return false; }
    if (!dequant_blocks(t.ggml_type, raw.data(), n, out.data())) {
        if (err) *err = "Dequantisierung fehlgeschlagen für " + t.name;
        return false;
    }
    return true;
}

}  // namespace nova::modelstore

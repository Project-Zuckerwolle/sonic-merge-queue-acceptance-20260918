// nv4_format.h — Nova 4 Modell-Storage-Formate
//
// Zwei Formate (Design §5):
//   .nv4  Chunk-Streaming-Format (Gemma 4, Ministral 14B, Gemma 3 1B)
//         32-Byte Header + FP16-Scales + (Zstd-komprimierte) INT2/INT4-Rohdaten
//   .bin  Flat-INT4-Binary (Ministral 3B): 16-Byte Header + ein INT4-Block
//
// Alles Little-Endian. Quantisierung: symmetrisch pro Gruppe (GROUP_SIZE Werte),
// ein FP16-Skalierungsfaktor je Gruppe.
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace nova::modelstore {

// ---- Konstanten -----------------------------------------------------------
inline constexpr char     NV4_MAGIC[4]   = {'N', 'V', '4', 'C'};  // .nv4 Chunk
inline constexpr char     NV4B_MAGIC[4]  = {'N', 'V', '4', 'B'};  // .bin Flat-INT4
inline constexpr uint16_t NV4_VERSION    = 1;
inline constexpr uint32_t GROUP_SIZE     = 32;   // Werte pro FP16-Scale
inline constexpr size_t   NV4_HEADER_SIZE = 32;
inline constexpr size_t   NV4B_HEADER_SIZE = 16;

enum class QuantType : uint8_t {
    INT2 = 2,
    INT3 = 3,   // dense Modelle (Mistral 24B): bessere Qualität als INT2, kleiner als INT4
    INT4 = 4,
};

enum Nv4Flags : uint8_t {
    NV4_FLAG_NONE       = 0,
    NV4_FLAG_ZSTD       = 1 << 0,  // Payload ist Zstd-komprimiert
};

// ---- 32-Byte Chunk-Header (§5.2) ------------------------------------------
struct Nv4Header {
    char     magic[4];           // "NV4C"
    uint16_t version;
    uint16_t layer_index;
    uint16_t chunk_index;
    uint8_t  quant_type;         // QuantType
    uint8_t  flags;              // Nv4Flags
    uint32_t crc32;              // CRC32 über (Scales + Payload), alles nach dem Header
    uint32_t shape0;             // Tensor-Shape dim0
    uint32_t shape1;             // Tensor-Shape dim1 (1 für 1D)
    uint32_t compressed_size;    // Bytes des (ggf. komprimierten) Payloads im File
    uint32_t uncompressed_size;  // Bytes der gepackten INT-Rohdaten vor Zstd

    uint64_t num_elements() const {
        return static_cast<uint64_t>(shape0) * (shape1 ? shape1 : 1);
    }
    size_t scale_bytes() const;  // = num_groups * sizeof(fp16)

    void   serialize(uint8_t out[NV4_HEADER_SIZE]) const;
    static bool parse(const uint8_t in[NV4_HEADER_SIZE], Nv4Header& out);
    bool   magic_ok() const;
};

// ---- 16-Byte Flat-INT4-Header (§5.3) --------------------------------------
struct Nv4bHeader {
    char     magic[4];   // "NV4B"
    uint16_t version;
    uint32_t layers;     // packe Layers + Hidden-Dim in die 6 Shape-Bytes:
    uint16_t hidden_dim; //   layers(4) + hidden_dim(2) = 6 Byte
    uint32_t checksum;   // CRC32 über den INT4-Block

    void   serialize(uint8_t out[NV4B_HEADER_SIZE]) const;
    static bool parse(const uint8_t in[NV4B_HEADER_SIZE], Nv4bHeader& out);
};

// ---- Vollständiger gelesener Chunk -----------------------------------------
struct Chunk {
    Nv4Header             header;
    std::vector<uint8_t>  scales;   // FP16-Scales, roh (little-endian)
    std::vector<uint8_t>  payload;  // wie im File (ggf. Zstd-komprimiert)
};

// ---- CRC32 (IEEE 802.3, poly 0xEDB88320) ----------------------------------
uint32_t crc32(const void* data, size_t len, uint32_t seed = 0);

// ---- FP16 <-> FP32 ---------------------------------------------------------
uint16_t float_to_half(float f);
float    half_to_float(uint16_t h);

// ---- Quantisierung ---------------------------------------------------------
struct QuantResult {
    std::vector<uint8_t> scales;   // FP16, num_groups Einträge -> 2*num_groups Bytes
    std::vector<uint8_t> packed;   // gepackte INT2/INT4-Codes
};

// Quantisiert num_elements FP32-Werte symmetrisch pro Gruppe.
QuantResult quantize_tensor(const float* data, uint64_t num_elements, QuantType qt);

// Inverse: rekonstruiert FP32 aus Scales + gepackten Codes.
std::vector<float> dequantize_tensor(const std::vector<uint8_t>& scales,
                                     const std::vector<uint8_t>& packed,
                                     uint64_t num_elements, QuantType qt);

size_t packed_bytes(uint64_t num_elements, QuantType qt);
size_t num_groups(uint64_t num_elements);

// ---- Chunk schreiben / lesen ----------------------------------------------
// Schreibt einen .nv4-Chunk. quant_packed = unkomprimierte gepackte Codes,
// scales = FP16-Bytes. Bei compress=true wird das Payload Zstd-komprimiert.
bool write_chunk(const std::string& path,
                 uint16_t layer_index, uint16_t chunk_index, QuantType qt,
                 uint32_t shape0, uint32_t shape1,
                 const std::vector<uint8_t>& scales,
                 const std::vector<uint8_t>& quant_packed,
                 bool compress);

// Liest einen kompletten .nv4-Chunk (ohne Dekomprimierung des Payloads).
bool read_chunk(const std::string& path, Chunk& out, std::string* err = nullptr);

// Parst einen kompletten Chunk aus einem Speicherpuffer (für RAM-/SSD-Streaming).
bool parse_chunk(const uint8_t* buf, size_t len, Chunk& out, std::string* err = nullptr);

// Dekomprimiert das Payload eines Chunks (falls NV4_FLAG_ZSTD) zu gepackten Codes.
bool decompress_payload(const Chunk& c, std::vector<uint8_t>& out, std::string* err = nullptr);

}  // namespace nova::modelstore

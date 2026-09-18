// gguf_reader.h — Minimaler GGUF-Parser (Download-Quelle: `ollama pull`).
//
// Liest GGUF v2/v3 Struktur: Header, Metadata-KV, Tensor-Infos. Tensor-Daten
// werden für F32 und F16 nach FP32 dekodiert. Quantisierte ggml-Quelltypen
// (Q4_K etc.) werden erkannt aber (noch) nicht dekodiert — converter meldet
// das klar. converter.exe re-quantisiert dann zu INT2/INT4 (.nv4 / .bin).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace nova::modelstore {

// ggml-Tensortypen (Teilmenge — vollständige Liste in ggml.h)
enum GgmlType : uint32_t {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q6_K = 14,
};

struct GgufTensorInfo {
    std::string           name;
    std::vector<uint64_t> dims;        // ggml-Reihenfolge (dim0 schnellste)
    uint32_t              ggml_type = 0;
    uint64_t              offset    = 0;  // relativ zum Datenblock-Start
    uint64_t num_elements() const;
};

class GgufReader {
public:
    bool open(const std::string& path, std::string* err = nullptr);

    uint32_t version() const { return version_; }
    const std::vector<GgufTensorInfo>& tensors() const { return tensors_; }
    const std::map<std::string, std::string>& metadata_strings() const { return meta_str_; }

    // Liest Tensor-Daten nach FP32 (unterstützt F32 + F16). Fehler bei anderen Typen.
    bool read_tensor_f32(const GgufTensorInfo& t, std::vector<float>& out,
                         std::string* err = nullptr);

    static const char* ggml_type_name(uint32_t t);

private:
    std::string                        path_;
    uint32_t                           version_   = 0;
    uint64_t                           data_off_  = 0;   // Offset des Tensor-Datenblocks
    uint32_t                           alignment_ = 32;
    std::vector<GgufTensorInfo>        tensors_;
    std::map<std::string, std::string> meta_str_;
};

}  // namespace nova::modelstore

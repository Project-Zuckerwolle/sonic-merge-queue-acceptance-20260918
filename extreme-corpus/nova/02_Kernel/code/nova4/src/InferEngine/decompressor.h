// decompressor.h — Chunk-Dekomprimierung INT2/INT4(+Zstd) -> FP16 (Design §6.2).
//
// Pipeline-Stufe B ("Decomp"): Zstd-Dekomprimierung läuft CPU-seitig, die
// INT2/INT4 -> FP16 Dequantisierung läuft auf der GPU (decompressor.cu) bzw.
// als Host-Referenz (decompressor_host.cpp). Beide Pfade sind bit-kompatibel
// im Scale-Format (FP16 pro Gruppe, GROUP_SIZE Werte).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ModelStore/nv4_format.h"

namespace nova::infer {

// Reine Dequant-Stufe: gepackte Codes + FP16-Scales -> FP16-Ausgabe.
// (packed bereits Zstd-dekomprimiert.) out muss num_elements uint16_t fassen.
void dequantize_to_fp16_host(const uint8_t* packed, const uint8_t* scales,
                             uint64_t num_elements, modelstore::QuantType qt,
                             uint16_t* out);

// Komplett: Chunk (ggf. Zstd) -> FP16. Host-Referenz, Mirror der CUDA-Kernels.
bool decompress_chunk_host(const modelstore::Chunk& c,
                           std::vector<uint16_t>& fp16_out,
                           std::string* err = nullptr);

#ifdef NOVA_HAVE_CUDA
// GPU-Pfad (decompressor.cu): Zstd auf CPU, Dequant auf GPU.
bool decompress_chunk_cuda(const modelstore::Chunk& c,
                           std::vector<uint16_t>& fp16_out,
                           std::string* err = nullptr);

// Liefert true wenn ein nutzbares CUDA-Device vorhanden ist.
bool cuda_device_available();
#endif

// Cosine-Similarity zwischen FP16-Ausgabe und FP32-Original (Testbed 3).
double cosine_similarity_fp16_vs_fp32(const uint16_t* a_half, const float* b, uint64_t n);

}  // namespace nova::infer

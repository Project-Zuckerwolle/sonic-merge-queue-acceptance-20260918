// fused_gemm.h — Fused INT3-Dequant-GEMM (Kernel-Upgrade ②).
//
// Statt INT3 -> FP16 (VRAM) -> GEMM (Zweipass) dequantisiert der fused Kernel
// die INT3-Gewichte INLINE im Matmul und materialisiert nie eine FP16/FP32-
// Vollmatrix: spart den FP16-Work-Buffer (~2,6× größer) und VRAM-Bandbreite und
// schafft Platz im 10-GB-Budget. INT3 ist nicht HW-nativ — 3-Bit-Werte straddeln
// Byte-Grenzen und werden im Loop entpackt (das ist der Mehraufwand).
//
// Host-Referenz (hier, CPU) beweist die Numerik (Cosine-Sim > 0,99 vs FP32);
// der CUDA-Kernel (fused_gemm.cu, Server) spiegelt sie bit-kompatibel.
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace nova::infer {

// --- FMT_BLOCK3 Skalen-Kodier-Helfer (eine Definition für Host-Packer, MT-Packer, Expander). -----
// 6 Bit an (Superblock-Byte-Basis, Subindex 0..7): 8 Subskalen = 48 bit = 6 Byte.
inline void b3_put6(std::vector<uint8_t>& buf, size_t byte_base, int j, uint8_t v6) {
    const size_t bit = byte_base * 8 + size_t(j) * 6;
    for (int b = 0; b < 6; ++b)
        if ((v6 >> b) & 1u) buf[(bit + b) / 8] |= uint8_t(1u << ((bit + b) % 8));
}
inline uint8_t b3_get6(const std::vector<uint8_t>& buf, size_t byte_base, int j) {
    const size_t bit = byte_base * 8 + size_t(j) * 6;
    uint8_t v = 0;
    for (int b = 0; b < 6; ++b)
        if (buf[(bit + b) / 8] >> ((bit + b) % 8) & 1u) v |= uint8_t(1u << b);
    return v;
}
// FP32 -> IEEE-754-half (positive, endliche Skalen; round-to-nearest). Muss zur Device-__half passen.
inline uint16_t b3_f32_to_f16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = int32_t((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        mant |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t h = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) h += 1;
        return uint16_t(sign | h);
    } else if (exp >= 31) return uint16_t(sign | 0x7c00u);
    uint16_t h = uint16_t(sign | uint32_t(exp << 10) | (mant >> 13));
    if (mant & 0x1000u) h += 1;
    return h;
}
inline float b3_f16_to_f32(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) out = sign;
        else { exp = 127 - 15 + 1; while (!(mant & 0x400u)) { mant <<= 1; --exp; } mant &= 0x3ffu;
               out = sign | (exp << 23) | (mant << 13); }
    } else if (exp == 31) out = sign | 0x7f800000u | (mant << 13);
    else out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float f; std::memcpy(&f, &out, 4); return f;
}

// Skalen-Format einer Int3Matrix (Phase 2 / C1 Umbau).
enum ScaleFmt { SCALE_FP32 = 0, SCALE_BLOCK3 = 1 };

// INT3-gepackte Gewichtsmatrix (M×K, row-major) + per-Gruppe-Skalen.
struct Int3Matrix {
    int M = 0, K = 0, group = 32;
    int bits = 3;                  // Quantisierungs-Bit-Breite (24B: 3; 3B-Draft optional 3..6, Phase 3)
    std::vector<uint8_t> packed;   // M*K bits-Werte, dicht gepackt (Byte-straddelnd)
    std::vector<float>   scales;   // M * ceil(K/group)  (nur SCALE_FP32)
    // --- Phase 2 FMT_BLOCK3: zweistufige Skalen, ~0,25 bit/Gewicht statt 1,0 (FP32). --------------
    // Superblock = 256 Gewichte (8 Gruppen à 32): 1 FP16-Superskala. Subblock = 32 Gewichte: 1
    // 6-Bit-Subskala (bit-gepackt, 8 Subskalen = 6 Byte/Superblock). Effektive Gruppenskala =
    // superscale/63 * subscale_q6. Nur der CUDA-Streaming-Pfad (24B) nutzt dieses Format; scales[]
    // bleibt dann leer. Die 3-Bit-Gewichte (packed) sind bit-identisch zum FP32-Pfad.
    int fmt = SCALE_FP32;
    std::vector<uint16_t> superscales;   // __half bits, M * ceil(ng/8)   (SCALE_BLOCK3)
    std::vector<uint8_t>  subscales;     // 6-Bit gepackt, je Superblock 6 Byte (SCALE_BLOCK3)
    // Optionale VRAM-Residenz (nur 3B-Draft, User-Entscheidung): entpacktes INT8 [M*K] + Skalen
    // permanent im VRAM. Gesetzt ⇒ der CUDA-Pfad überspringt H2D + Entpacken (nur Pass-2-GEMV/GEMM).
    // null ⇒ 24B-Streaming (H2D packed pro Token). `mutable`, da nur GPU-Cache, nicht logischer Zustand.
    mutable void* d_unpacked = nullptr;   // int8_t*, M*K, resident
    mutable void* d_scales   = nullptr;   // float*, M*ng, resident
    mutable bool  host_pinned = false;    // Phase 4: packed/scales via cudaHostRegister gepinnt (einmalig)
    int groups_per_row() const { return (K + group - 1) / group; }
    int superblocks_per_row() const { return (groups_per_row() + 7) / 8; }   // 8 Gruppen/Superblock
};

// Quantisiert W (M×K FP32) zu INT3 (symmetrisch, Level -4..3) + per-Gruppe-Skalen.
Int3Matrix pack_int3(const float* W, int M, int K, int group = 32);

// Wie pack_int3, aber zweistufige Skalen (SCALE_BLOCK3). Gewichte bit-identisch zu pack_int3;
// nur die Skalen werden kompakt (FP16-Super + 6-Bit-Sub) kodiert. group muss 32 sein.
Int3Matrix pack_block3(const float* W, int M, int K, int group = 32);

// Host: expandiert die zweistufigen Skalen einer SCALE_BLOCK3-Matrix zu FP32 [M*ng] (für Referenz/Test).
void block3_expand_scales(const Int3Matrix& W, std::vector<float>& out_scales);

// Fused Dequant-GEMV: y = dequant(Wq) · x — dequantisiert INLINE, nie eine
// Vollmatrix materialisierend. y: M, x: K.
void fused_int3_gemv(const Int3Matrix& W, const float* x, float* y);

// Batched Fused INT3-GEMM: Y[M×N] = dequant(W[M×K]) · X[K×N].
// EINMAL und wendet sie auf alle N Spalten an -> amortisiert den VRAM-Read über N
// Positionen. Basis für Spec-Decoding-Verify (24B prüft N Draft-Tokens in 1 Pass).
void fused_int3_gemm(const Int3Matrix& W, const float* X, int N, float* Y);

// Referenz-GEMV auf FP32 (Numerik-Vergleich).
void reference_gemv(const float* W, int M, int K, const float* x, float* y);

#ifdef NOVA_HAVE_CUDA
// CUDA-GEMV: entpackt INT3→INT8 in einen VRAM-Scratch (koalescierte Writes) und rechnet dann warp-
// koalesciert (ein Warp je Zeile). Bit-nah zu fused_int3_gemv (Host) — Cosine>0,99 in test_fused_gemm.
// Ist W VRAM-resident (d_unpacked gesetzt), entfallen H2D + Entpacken (nur Pass-2).
void fused_int3_gemv_cuda(const Int3Matrix& W, const float* x, float* y);
// Batched Variante (N Spalten in einem Weight-Read/Entpacken). N <= 32, sonst Host-Fallback.
void fused_int3_gemm_cuda(const Int3Matrix& W, const float* X, int N, float* Y);

// Macht W VRAM-resident: entpackt einmalig INT3→INT8 ins VRAM + lädt Skalen resident (für das 3B-Draft).
// Idempotent (no-op wenn schon resident). W.packed/scales im RAM bleiben unverändert.
void int3_residentize(const Int3Matrix& W);
void int3_free_resident(const Int3Matrix& W);

// GEMV komplett auf Device-Zeigern: y_dev = dequant(W)·x_dev. W MUSS resident sein (d_unpacked gesetzt);
// x_dev/y_dev sind Device-Ptr. Kein H2D/D2H — für den on-GPU-Forward (gpu_forward.cu). Async auf `stream`.
void fused_int3_gemv_dev(const Int3Matrix& W, const float* x_dev, float* y_dev, void* stream = nullptr);

// Streamed-device: streamt+entpackt das INT3-Gewicht H2D (Layer-Streaming BLEIBT, W NICHT resident),
// rechnet dann auf DEVICE-Aktivierungen x_dev/y_dev — kein Aktivierungs-Round-Trip. Für den on-GPU-
// Forward des gestreamten 24B (nimmt die CPU aus dem Loop). N-Spalten-Variante: N<=32 (gemm_i8 acc[32]).
void fused_int3_gemv_streamed_dev(const Int3Matrix& W, const float* x_dev, float* y_dev, void* stream = nullptr);
void fused_int3_gemm_streamed_dev(const Int3Matrix& W, const float* X_dev, int N, float* Y_dev, void* stream = nullptr);

// --- Instrumentierung (Messplan Phase 2): CUDA-Event-Zeiten je Stufe des Streaming-Pfads. -------
// Standard AUS → kein Overhead, Numerik unverändert (Korrektheits-Gates bleiben grün). Trennt sauber
// Transfer (H2D packed/scales/x + D2H y) von Compute (unpack-Kernel + gemm/gemv-Kernel).
struct GemmInstr {
    double h2d_packed_ms = 0, h2d_scales_ms = 0, h2d_x_ms = 0, unpack_ms = 0, kernel_ms = 0, d2h_ms = 0;
    double h2d_packed_bytes = 0, h2d_scales_bytes = 0, h2d_x_bytes = 0, d2h_bytes = 0;
    long   calls = 0, unpack_calls = 0, kernel_calls = 0;   // calls = gemv+gemm-Aufrufe im Streaming-Pfad
    double transfer_ms() const { return h2d_packed_ms + h2d_scales_ms + h2d_x_ms + d2h_ms; }
    double compute_ms()  const { return unpack_ms + kernel_ms; }
};
void      fused_instr_enable(bool on);
bool      fused_instr_enabled();
void      fused_instr_reset();
GemmInstr fused_instr_snapshot();
// Phase 3 (NOVA_STREAM_OVERLAP): Pin-Buchhaltung + Overlap-Status für Log/Diagnose.
void      fused_pin_stats(long& ok, long& fail, long& already);
bool      fused_stream_overlap_enabled();
#endif

}  // namespace nova::infer

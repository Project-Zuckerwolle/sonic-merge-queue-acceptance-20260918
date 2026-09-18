// gemm_int3.h — GEMM/GEMV auf GEPACKT-RESIDENTEN Gewichten + die restlichen
// Layer-Kernel des Forward-Pfads.
//
// ================= WARUM PACKED-RESIDENT ====================================
// Nova 4 kannte zwei Modi, und beide sind für ein residentes 24B unbrauchbar:
//
//   A) STREAMING (fused_gemm.cu:308 resolve_device, :401 resolve_device_pipe).
//      Jeder einzelne Matmul kopiert sein Gewicht per hipMemcpy H2D und
//      entpackt es mit unpack_int3_kernel in einen INT8-Scratch. Auf der
//      Zielmaschine (RX 7900 XT, 20 GB) ist das schlicht unnötig — das Modell
//      passt. Das ganze Gerüst (StreamSlot :378, try_pin :49, die Flags
//      NOVA_PINNED_STREAM / NOVA_STREAM_OVERLAP / NOVA_STREAM_DEBUG, die
//      Transfer-Instrumentierung GemmInstr.h2d_*) entfällt ersatzlos.
//
//   B) RESIDENT via int3_residentize (fused_gemm.cu:494). Drei harte Fehler:
//      1. Es entpackt nach INT8, also 1 BYTE JE GEWICHT. Für 20,1 G Parameter
//         sind das 23,6 GB — die 3-Bit-Quantisierung wird damit vollständig
//         wieder aufgegeben, und es passt nicht in 20 GB VRAM.
//      2. Es alloziert die gepackte Kopie (:498) VOR dem Freigeben (:506), der
//         Spitzenbedarf ist also packed + unpacked gleichzeitig (~32 GB).
//      3. Es liest W.scales (:508-509) — bei SCALE_BLOCK3 ist dieser Vektor
//         LEER (fused_gemm.h:77). Für jedes BLOCK3-Gewicht lädt es also einen
//         0-Byte-Puffer und rechnet danach mit undefinierten Skalen. Kein
//         Fehler, keine Meldung — nur falsche Zahlen.
//      Gebaut wurde es für das 3B-Draft, das es nicht mehr gibt.
//
// NEU: Die Gewichte bleiben GEPACKT im VRAM. Der GEMM-Kernel liest direkt aus
// `packed` und dekodiert die BLOCK3-Skalen (FP16-Superskala je 256 Gewichte +
// 6-Bit-Subskala je 32) inline. Es gibt keinen INT8-Scratch, keine Expansion
// der Skalen nach FP32 (die allein 2,5 GB kosten würde: 20,1 G / 32 Gruppen
// x 4 Byte) und keinen Transfer im Decode-Pfad.
//
// ================= AKTIVIERUNGS-LAYOUT ======================================
// Alle Aktivierungen sind ZEILENWEISE [N][features]. Nova 4 benutzte spalten-
// weise [features][N] (gpu_forward.cu:123, fused_gemm.cu:187) — dort liest ein
// Block, der über die Merkmale läuft, mit Stride N*4 Byte und erzeugt je Lane
// eine eigene Speichertransaktion. Zeilenweise ist derselbe Zugriff koalesciert.
//
// ================= REGISTERBILANZ ===========================================
// gemm_i8_kernel (fused_gemm.cu:139) hält `float acc[32]` je Thread — 32 VGPR
// allein für den Akkumulator, plus Zeiger und Skalen. Auf gfx1100 ist das die
// Schwelle, ab der der Compiler in den Scratch spillt (und Spilling in einer
// bandbreitengebundenen Schleife ist teurer als der halbe Kernel). Das Tiling
// hier ist so gewählt, dass der Akkumulator 16 VGPR belegt: siehe die Bilanz
// im Kernel-Kommentar in gemm_int3.hip.
// STAND IM BAUM: NICHT GEBAUT — gemm_int3.hip liegt im Baum, wird hier aber von
// keinem Übersetzer angefasst. KORREKTUR: bis zum ersten Vollbau stand hier
// "die zugehörige .hip-Datei liegt nicht im Baum". Das stimmt nicht mehr —
// gemm_int3.hip ist da (gemv_packed_kernel, quantize_act_kernel,
// gemm_packed_dp4a_kernel, rmsnorm_kernel, rope_kernel, kv_quant_kernel,
// embed_gather_kernel). Was fehlt, ist etwas anderes und Schlimmeres: OHNE HIP
// SDK übersetzt sie NIEMAND. CMakeLists.txt schickt die .hip-Dateien über
// add_custom_command durch hipcc, und dieser Zweig läuft nur bei
// NOVA_HIP_FOUND. Auf einer Maschine ohne SDK — dem Normalfall bis zur
// Auslieferung — ist dieser Kernel also nicht bloß ungetestet, sondern nie
// syntaktisch geprüft worden.
// WAS HEUTE STATTDESSEN PASSIERT: ohne NOVA_HAVE_HIP liefern die Launcher
// sauber "hip.not_compiled" (nova_hip_compat.h:181), die MockEngine trägt den
// gesamten Baum. MIT NOVA_HAVE_HIP fehlt das Symbol beim Linken — und das ist
// die richtige Reihenfolge: ein Fehler, den man sieht, statt eines Aufrufs,
// der still nichts rechnet.
#pragma once

#include "Core/nova_core.h"
#include "InferEngine/nova_hip_compat.h"

#include <cstddef>
#include <cstdint>

namespace nova::infer {

// Skalen-Format, identisch zu fused_gemm.h:66.
enum ScaleFmt : int { SCALE_FP32 = 0, SCALE_BLOCK3 = 1 };

// PUFFER-NACHLAUF: die Entpackroutine liest die 32er-Gruppe als zwei 32-Bit-
// Worte; das zweite kann bis zu 4 Byte hinter dem letzten Element liegen. Jeder
// packed-Puffer im VRAM MUSS mit diesem Nachlauf alloziert werden. Eine Abfrage
// im innersten Loop wäre teurer als 16 Byte je Matrix.
inline constexpr size_t weight_tail_bytes() { return 16; }

// Sicht auf eine gepackte, VRAM-residente Gewichtsmatrix M x K (row-major).
struct WeightView {
    const void* packed = nullptr;       // const unsigned char*
    const void* scales = nullptr;       // const float*,          nur SCALE_FP32
    const void* superscales = nullptr;  // const unsigned short*, nur SCALE_BLOCK3
    const void* subscales = nullptr;    // const unsigned char*,  nur SCALE_BLOCK3
    int M = 0, K = 0;
    int group = 32;
    int bits = 3;
    int fmt = SCALE_FP32;

    int ng()  const { return (K + group - 1) / group; }
    int nsb() const { return (ng() + 7) / 8; }
    bool valid() const {
        return packed && M > 0 && K > 0 && group == 32 && (bits >= 2 && bits <= 6) &&
               (fmt == SCALE_FP32 ? scales != nullptr
                                  : (superscales != nullptr && subscales != nullptr));
    }
};

// Host-seitige Byte-Größen (auch ohne HIP nutzbar — der Loader braucht sie).
inline size_t weight_packed_bytes(int M, int K, int bits) {
    return ((size_t)M * (size_t)K * (size_t)bits + 7) / 8;
}

// ---------------------------------------------------------------------------
// Launcher. Alle asynchron auf `stream` (hipStream_t als void*), alle liefern
// Status. Ohne HIP übersetzt liefern sie einen Fehler statt Linkfehler.
// ---------------------------------------------------------------------------
struct GemvParams {
    WeightView  w;
    const void* x = nullptr;   // const float*, [K]
    void*       y = nullptr;   // float*, [M]
    void*       stream = nullptr;
};

struct GemmParams {
    WeightView  w;
    const void* x = nullptr;      // const float*, [N][K] zeilenweise
    void*       y = nullptr;      // float*, [N][M] zeilenweise
    int         N = 0;
    void*       x8 = nullptr;     // signed char*, [N][K]  Scratch
    void*       act_scale = nullptr;  // float*, [N]       Scratch
    void*       stream = nullptr;
};

struct RmsNormParams {
    const void* x = nullptr;   // [N][H]
    const void* g = nullptr;   // [H]
    void*       out = nullptr; // [N][H]
    int N = 0, H = 0;
    float eps = 1e-5f;
    void* stream = nullptr;
};

struct RopeParams {
    void* v = nullptr;         // [N][nHeads*hd], in-place
    int   N = 0, n_heads = 0, hd = 0;
    int   pos0 = 0;            // Position der Zeile 0
    const void* pos_arr = nullptr;   // optional const int*, [N] (Baum-Positionen)
    float theta = 1e9f;
    void* stream = nullptr;
};

struct RotateParams {
    void*       v = nullptr;   // [N][nHeads*hd], in-place
    const void* r = nullptr;   // const float*, [hd][hd]
    int N = 0, n_heads = 0, hd = 0;
    void* stream = nullptr;
};

struct KvQuantParams {
    const void* src = nullptr;   // const float*, [N][kvd] zeilenweise
    void*       codes = nullptr;
    void*       meta = nullptr;
    int N = 0, pos0 = 0, nKV = 0, hd = 0;
    int bits = 3, qjl = 0;
    void* stream = nullptr;
};

struct EmbedParams {
    WeightView  w;               // token_embd, M = vocab, K = hidden
    const void* tokens = nullptr; // const int*, [N] (Device)
    void*       out = nullptr;    // float*, [N][K]
    int         N = 0;
    void*       stream = nullptr;
};

struct ElemParams {
    void*       a = nullptr;
    const void* b = nullptr;
    size_t      n = 0;
    void*       stream = nullptr;
};

#ifdef NOVA_HAVE_HIP
Status gemv_packed(const GemvParams& p);
Status gemm_packed(const GemmParams& p);
Status rmsnorm(const RmsNormParams& p);
Status rope(const RopeParams& p);
Status rotate_heads(const RotateParams& p);
Status kv_quant(const KvQuantParams& p);
Status embed_gather(const EmbedParams& p);
Status silu_mul(const ElemParams& p);      // a = silu(a) * b
Status add_inplace(const ElemParams& p);   // a = a + b
// Werkzeuge für Verifikation und Konverter — nicht im Decode-Pfad.
Status unpack_nbit(const void* packed, size_t total, int bits, int offset,
                   void* out_i8, void* stream);
Status expand_block3_scales(const void* super, const void* sub, int M, int ng, int nsb,
                            void* out_f32, void* stream);
#else
inline Status gemv_packed(const GemvParams&)  { return gpu::no_hip("gemv_packed"); }
inline Status gemm_packed(const GemmParams&)  { return gpu::no_hip("gemm_packed"); }
inline Status rmsnorm(const RmsNormParams&)   { return gpu::no_hip("rmsnorm"); }
inline Status rope(const RopeParams&)         { return gpu::no_hip("rope"); }
inline Status rotate_heads(const RotateParams&) { return gpu::no_hip("rotate_heads"); }
inline Status kv_quant(const KvQuantParams&)  { return gpu::no_hip("kv_quant"); }
inline Status embed_gather(const EmbedParams&) { return gpu::no_hip("embed_gather"); }
inline Status silu_mul(const ElemParams&)     { return gpu::no_hip("silu_mul"); }
inline Status add_inplace(const ElemParams&)  { return gpu::no_hip("add_inplace"); }
inline Status unpack_nbit(const void*, size_t, int, int, void*, void*) {
    return gpu::no_hip("unpack_nbit");
}
inline Status expand_block3_scales(const void*, const void*, int, int, int, void*, void*) {
    return gpu::no_hip("expand_block3_scales");
}
#endif

}  // namespace nova::infer

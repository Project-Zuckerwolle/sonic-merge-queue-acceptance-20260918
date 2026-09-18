// gpu_forward.h — Kompletter inkrementeller Forward auf der GPU für ein VRAM-residentes Modell.
//
// Motivation (gemessen, siehe Memory nova-c1-speed-diagnosis): der CPU-Forward ruft pro Token 182
// einzelne GEMV-Aufrufe mit synchronem cudaMemcpy — er pingpongt PRO MATMUL CPU↔GPU. Beim Spec-
// Decoding (K Draft-Pässe/Schritt) sind das >1000 Roundtrips/Schritt, reiner Launch/Sync-Overhead.
// GpuForward hält Aktivierungen + KV-Cache permanent im VRAM: pro Token geht nur token_id rein,
// der greedy-argmax raus. Nur EIN D2H (der argmax-Index) pro Token.
//
// Repliziert real_inference.cpp::forward_step EXAKT (Embedding-Gather, RMSNorm, plain-RoPE interleaved,
// GQA-Attention, SwiGLU) — validiert gegen die CPU-forward_step-Referenz (test_real_inference Teil E).
// Gedacht fürs residente 3B-Draft (Phase 1); dieselbe Maschinerie trägt später das gestreamte 24B.
#pragma once

#include "InferEngine/fused_gemm.h"
#include "InferEngine/mistral3_config.h"

#include <string>
#include <vector>

namespace nova::infer {

struct Int3Weights;
struct Int3Matrix;

#ifdef NOVA_HAVE_CUDA

// Zwischenspeicher der Matrix-Zeiger je Layer (Map-Lookup nur bei init). Member, damit mehrere
// GpuForward-Instanzen (residentes 3B-Draft UND gestreamtes 24B) gleichzeitig existieren können.
struct GpuForwardWeights {
    const Int3Matrix *embd = nullptr, *out = nullptr;
    std::vector<const Int3Matrix*> q, k, v, o, gate, up, down;
};

class GpuForward {
public:
    ~GpuForward();
    // Bereitet den on-GPU-Forward vor. streamed=false: W residentisieren (3B-Draft), Matmuls via
    // fused_int3_gemv_dev. streamed=true: W NICHT residentisieren (24B bleibt layer-streamed), Matmuls
    // via fused_int3_gemm_streamed_dev — Gewicht streamt H2D pro Layer, Aktivierungen bleiben resident.
    // Norms hochladen, Aktivierungs- + KV-Puffer im VRAM (KV bis max_seq Positionen). Idempotent-safe.
    bool init(const Int3Weights& w, int max_seq, std::string* err = nullptr, bool streamed = false,
              int kv_bits = 0,    // kv_bits 0=FP32; 2/3/4=affine quantisierter residenter KV (verlustbehaftet)
              bool want_batch = false);   // Phase 6A-Wide: batched Puffer auch für den residenten Draft (Baum)
    bool ready() const { return ready_; }

    void reset();                              // KV leeren (neue Sequenz), seq_=0
    // Ein Token forward @pos; greedy-argmax für pos+1 zurück. host_logits (optional, V floats) nur
    // für Validierung — kostet einen zusätzlichen D2H.
    int  step_argmax(int token_id, int pos, float* host_logits = nullptr);
    // Wie step_argmax, aber schreibt die vollen V Logits nach host_logits (für Spec-Verify) — 1 D2H.
    void step_logits(int token_id, int pos, float* host_logits);
    // Batched Spec-Verify (nur streamed): N Tokens an Positionen start_pos..start_pos+N-1 in EINEM
    // Weight-Read/Layer (gestreamt). Schreibt [N][V] Logits nach host_out (Zeile n = Position start_pos+n).
    // N <= MAX_BATCH. Nutzt denselben KV-Cache wie step_logits (gemeinsame Sequenz).
    // tree_pos/tree_mask (Host, optional, Phase 6A): Baum-Verify — per-Knoten RoPE-Position (Tiefe) +
    // Ahnen-Maske [N*N]. nullptr => lineares kausales Verhalten (unverändert).
    void forward_batch_gpu(const int* tokens, int N, int start_pos, float* host_out,
                           const int* tree_pos = nullptr, const unsigned char* tree_mask = nullptr);
    void trim(int len);                        // KV auf len Positionen zurückschneiden (Spec-Rollback)
    void step_top2(int token_id, int pos, int out2[2]);       // Phase 6A: Forward + Top-2-Token (Baum-Draft)
    void gather_kv_block(int from, int to, int len);          // Phase 6A: KV-Block kompaktieren (Tree-Accept)
    // Phase 6B EAGLE: 3 Layer-Hidden [3][H][N] (column-major je Tap) nach hid_host [3*H*N] exportieren.
    void set_hidden_capture(bool on) { hid_capture_ = on; }
    void forward_batch_hidden(const int* tokens, int N, int start_pos, float* host_out, float* hid_host);
    int  seq_len() const { return seq_; }
    void free();

    static constexpr int MAX_BATCH = 512;      // Phase 4: N-gekachelter gemm_i8 (32-Spalten je Block) -> N>32
                                               // Phase 6B: 512 für gepackten EAGLE-Dump (volle Stream-Auslastung)

private:
    void run_layers(int token_id, int pos);    // gemeinsamer per-Layer-Forward -> d_logits
    void run_layers_batch(const int* tokens, int N, int start_pos,
                          const int* d_pos = nullptr, const unsigned char* d_mask = nullptr);  // batched -> d_logits_b [feature][N]
    const Int3Weights* w_ = nullptr;
    Mistral3Config cfg_;
    int  max_seq_ = 0, seq_ = 0;
    bool ready_ = false;
    bool streamed_ = false;
    // Perf Phase 3.2: dedizierter Compute-Stream (als void*, damit der Header keinen CUDA-Header
    // braucht). Der gesamte on-GPU-Forward läuft darauf statt auf dem Legacy-Null-Stream (Stream 0),
    // Voraussetzung für echten H2D/Compute-Overlap (Phase 3.3).
    void* stream_ = nullptr;
    GpuForwardWeights gw_;                      // Gewichts-Zeiger dieser Instanz
    std::vector<float> h_embed_;                // Host-Scratch fürs Embedding (streamed: dequant_row -> H2D)
    std::vector<float> h_embed_b_;              // Host-Scratch fürs batched Embedding [H*N]
    // Batched Aktivierungs-Puffer [feature][N], N<=MAX_BATCH (nur streamed alloziert).
    float *d_xb=nullptr, *d_xnb=nullptr, *d_qb=nullptr, *d_kb=nullptr, *d_vb=nullptr,
          *d_ctxb=nullptr, *d_tmpb=nullptr, *d_gb=nullptr, *d_ub=nullptr, *d_logitsb=nullptr;

    // Aktivierungs-Puffer (VRAM, pro Schritt wiederverwendet).
    float *d_x=nullptr, *d_xn=nullptr, *d_q=nullptr, *d_k=nullptr, *d_v=nullptr,
          *d_ctx=nullptr, *d_tmp=nullptr, *d_g=nullptr, *d_u=nullptr, *d_logits=nullptr;
    int   *d_arg=nullptr;                       // argmax-Index (1 int)
    // Phase 6A Tree-Verify: per-Knoten RoPE-Positionen + Ahnen-Maske (VRAM, nur streamed alloziert).
    int           *d_treepos_ = nullptr;        // [MAX_BATCH]
    unsigned char *d_treemask_ = nullptr;       // [MAX_BATCH*MAX_BATCH]
    // Phase 6B EAGLE-3: 3 Layer-Hidden-Taps (low/mid/high) für den Draft-Kopf/das Training.
    float *d_hid_[3] = {nullptr, nullptr, nullptr};   // [H*MAX_BATCH] je Tap
    int    hidtap_[3] = {0, 0, 0};                     // Tap-Layer-Indizes
    bool   hid_capture_ = false;                       // an -> run_layers_batch schreibt die Taps
    // Norms resident (pro Layer + final), VRAM.
    std::vector<float*> d_attn_norm_, d_ffn_norm_;
    float* d_out_norm_ = nullptr;
    // KV-Cache resident. kv_bits_==0: FP32 [max_seq * kv_dim] pro Layer (keys nach RoPE).
    std::vector<float*> d_kcache_, d_vcache_;
    // kv_bits_>0: quantisiert. Pro Layer packed codes [max_seq * kvd*bits/8] + meta (scale,zero FP16
    // je Head/Token). Attention liest einen gemeinsamen 1-Layer-FP32-Scratch (dequant just-in-time).
    int kv_bits_ = 0;
    std::vector<unsigned char*> d_kcode_, d_vcode_;    // packed codes je Layer
    std::vector<unsigned short*> d_kmeta_, d_vmeta_;   // (scale,zero) FP16 je Head/Token, je Layer
    float *d_kdq_ = nullptr, *d_vdq_ = nullptr;        // gemeinsamer FP32-Dequant-Scratch (1 Layer)
    float *d_rot_ = nullptr;                            // orthogonale Rotation [hd*hd] (Hadamard) — verteilt
                                                        // K-Ausreißer, damit per-Block-Quant funktioniert
};

#endif  // NOVA_HAVE_CUDA

}  // namespace nova::infer

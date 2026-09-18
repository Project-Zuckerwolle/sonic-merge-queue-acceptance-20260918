// real_inference.h — Echte INT3-RAM-Engine für das mistral3-Textmodell (C1).
//
// Lädt die TEXT-Gewichte (blk.N.*, token_embd, output_norm, output) aus einer GGUF
// PER NAMEN, quantisiert sie beim Start zu INT3 (Int3Matrix, fused_gemm.h — cosine>0.99
// vs FP32 validiert) und hält sie RAM-resident (~9 GB statt 15 GB, keine Disk-Reads pro
// Token). Der Forward ist inkrementell mit KV-Cache (kein O(N²)-Recompute) — die zwei
// 220-s/Token-Killer sind damit beseitigt. Mathematik identisch zu mistral3_forward.cpp
// (RMSNorm/RoPE interleaved/GQA h÷grp/scale 1/√hd/SwiGLU), nur inkrementell.
//
// Vision-Tensoren (v.*, mm.*) werden ignoriert — reiner Textpfad.
#pragma once

#include "InferEngine/fused_gemm.h"
#include "InferEngine/inference.h"
#include "InferEngine/mistral3_config.h"
#include "InferEngine/tekken.h"
#ifdef NOVA_HAVE_CUDA
#include "InferEngine/gpu_forward.h"
#endif

#include <deque>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace nova::infer {

// Matmuls auf GPU (fused_int3_gemv_cuda) an/aus. Unter NOVA_HAVE_CUDA Default an,
// sonst immer CPU. Nur für Benchmark/Diagnose; die Ergebnisse sind bit-kompatibel.
void real_inference_use_gpu(bool on);
// Fertigstellung Phase 1: setzt einen NOVA_*-Flag-Wert aus der config.json (Vorrang vor Env).
// VOR RealInference::load()/load_draft() aufrufen (die ss_*-Helfer lesen einmalig).
void real_inference_set_flag(const std::string& key, const std::string& value);
bool real_inference_gpu();

// RAM-residente INT3-Gewichte eines mistral3-Textmodells.
struct Int3Weights {
    Mistral3Config cfg;
    bool interleaved = true;                       // RoPE-Konvention (wie Referenz)
    // Pro-Layer + global: Matmul-Gewichte als INT3 (M=out, K=in), Norms als F32.
    std::map<std::string, Int3Matrix>       mats;  // z.B. "blk.7.attn_q", "token_embd", "output"
    std::map<std::string, std::vector<float>> norms; // z.B. "blk.7.attn_norm", "output_norm"

    const Int3Matrix* mat(const std::string& n) const {
        auto it = mats.find(n); return it == mats.end() ? nullptr : &it->second;
    }
    const std::vector<float>* norm(const std::string& n) const {
        auto it = norms.find(n); return it == norms.end() ? nullptr : &it->second;
    }
    size_t approx_bytes() const;
};

// Load-Stage-Timing (Messplan Phase 1): getrennt read_tensor_f32 (SSD-Read + GGUF-Dequant Q4/Q6→F32)
// vs pack_int3_mt (INT3-Quantisierung). Wird bei jedem load_int3_from_gguf zurückgesetzt.
struct LoadTiming { double read_ms = 0, pack_ms = 0, open_ms = 0; long tensors = 0; };
LoadTiming last_load_timing();

// Lädt die Textgewichte aus einer GGUF und quantisiert sie zu INT3 (Int3Weights).
// group = Quant-Gruppengröße (32). progress: optionaler Callback (name, i, total).
bool load_int3_from_gguf(const std::string& gguf_path, const Mistral3Config& cfg,
                         Int3Weights& out, int group = 32, bool rope_interleaved = true,
                         std::string* err = nullptr,
                         const std::function<void(const std::string&, int, int)>& progress = {},
                         bool block3 = false,     // Phase 2: 24B mit FMT_BLOCK3 packen (nur Streaming)
                         int bits = 3,            // Phase 3: Quant-Bit-Breite (3B-Draft 3..6)
                         int attn_bits = 0);      // Mixed (§4.1): Attention-Layer mit attn_bits (>bits) packen

// Dequantisiert eine einzelne Zeile o einer Int3Matrix nach FP32 (für Embedding-Gather).
void dequant_row(const Int3Matrix& W, int o, float* out);

// Inkrementeller KV-gecachter Forward-Schritt: verarbeitet EIN Token an Position pos,
// aktualisiert den KV-Cache und liefert die Logits (Größe vocab). Reine Funktion über
// den State — von RealInference UND dem Korrektheits-Test genutzt.
struct KvCacheF32 {
    // pro Layer: keys/values als flaches [pos][kv_dim] FP32 (nach RoPE für keys).
    std::vector<std::vector<float>> k;   // [n_layers][ (pos+1) * kv_dim ]
    std::vector<std::vector<float>> v;
    int seq_len = 0;
    void reset(int n_layers) { k.assign(n_layers, {}); v.assign(n_layers, {}); seq_len = 0; }
};

// Ein Token forward (inkrementell). tokens: nur zum Embedden von token_id nötig.
std::vector<float> forward_step(const Int3Weights& w, int token_id, int pos, KvCacheF32& kv);

// Batched Forward: N Tokens ab start_pos in EINEM Weight-Read (Gewichte je Layer 1×
// gelesen, auf alle N Positionen angewandt). Hängt N KV-Einträge an, liefert flach
// [N][vocab] (Zeile n = Logits an Position start_pos+n). Basis für Spec-Decoding-Verify.
std::vector<float> forward_batch(const Int3Weights& w, const int* tokens, int N,
                                 int start_pos, KvCacheF32& kv);

// KV-Cache auf `len` Positionen zurückschneiden (Spec-Rollback verworfener Draft-Tokens).
void kv_trim(KvCacheF32& kv, int len, const Mistral3Config& cfg);

// ---- Engine hinter IInference -------------------------------------------------
class RealInference : public IInference {
public:
    struct Options { bool greedy = true; int max_new = 512; };

    // Lädt Gewichte (GGUF->INT3, RAM) + Tokenizer. Danach einsatzbereit.
    bool load(const std::string& gguf_path, const Mistral3Config& cfg,
              const std::string& tekken_dir, Options opt = {}, std::string* err = nullptr);

    // Injiziert vorquantisierte Gewichte (für Tests ohne GGUF).
    void set_weights(Int3Weights w) { w_ = std::move(w); loaded_ = true; }

    // Optionales Draft-Modell (3B) für Spec-Decoding. Ohne -> plain greedy.
    bool load_draft(const std::string& gguf_path, const Mistral3Config& cfg, std::string* err = nullptr);
    void set_draft_weights(Int3Weights w) { draft_cfg_ = w.cfg; draft_w_ = std::move(w); has_draft_ = true; }
    void set_spec_k(int k) { spec_k_ = k; }
    // Perf Phase 2: Prefix-Cache-Override für Tests/Harness (-1=Flag NOVA_PREFIX_CACHE, 0=aus, 1=an).
    void set_prefix_cache(bool on) { pc_override_ = on ? 1 : 0; }
    int  last_prefix_reuse() const { return last_reuse_; }   // reused KV-Positionen im letzten begin()
    bool has_draft() const { return has_draft_; }
    double last_accept_rate() const { return spec_steps_ ? double(spec_accepted_) / spec_steps_ : 0.0; }
    long spec_steps() const { return spec_steps_; }        // Anzahl Verify-Pässe (für Per-Verify-Messung)
    long spec_accepted() const { return spec_accepted_; }

    void begin(const GenRequest& req) override;   // Prompt prefillen (KV aufbauen)
    std::string next_token() override;            // ein Dekodier-Schritt; "" = Ende
    int next_token_id();                          // wie next_token, aber Token-ID (-1 = Ende)

    bool ready() const { return loaded_; }
    const Int3Weights& weights() const { return w_; }
    int  dump_eagle(const std::string& corpus_path, const std::string& out_path);  // Phase 6B: EAGLE-Trainingsdaten

private:
    int sample(const std::vector<float>& logits) const;   // greedy/argmax

    Int3Weights w_;
    Tekken      tok_;
    Options     opt_;
    bool        loaded_ = false;

    KvCacheF32  kv_;
    int         pos_ = 0;          // aktuelle Sequenzlänge (Target)
    int         produced_ = 0;
    bool        done_ = false;
    std::vector<float> pending_;   // Target-Logits an Position pos_ (nächstes Token)
    bool        have_pending_ = false;

    // Spec-Decoding (3B Draft + 24B Target, greedy). has_draft_ = false -> plain greedy.
    Int3Weights        draft_w_;
    Mistral3Config     draft_cfg_;
    KvCacheF32         draft_kv_;
    std::vector<float> draft_pending_;   // Draft-Logits an Position pos_
    bool               has_draft_ = false;
    int                spec_k_ = 8;      // Draft-Länge pro Schritt (K-Sweep-Peak, Accept~2,5)
    std::deque<int>    emit_queue_;      // akzeptierte Tokens, noch zu emittieren
    long               spec_steps_ = 0, spec_accepted_ = 0;
    double             acc_ema_ = -1.0;   // Perf Phase 6: EMA der Accept-Länge für adaptives K
    int                ss_pend_tok_ = -1;  // Phase 1 single-stream: aufgeschobener Token
                                           // (KV noch nicht geschrieben), Kopf des nächsten Batch
    // Perf Phase 2 (NOVA_PREFIX_CACHE): Token-Sequenz + geschriebene Target-KV-Länge des letzten
    // begin(); für Longest-Common-Prefix-Reuse des residenten KV über Turns.
    std::vector<int>   pc_ids_;
    int                pc_tgt_n_ = 0;
    bool               pc_valid_ = false;
    int                pc_override_ = -1;   // -1=Flag, 0=aus, 1=an (Test-Override)
    int                last_reuse_ = 0;     // reused KV-Länge im letzten begin() (Diagnose)
#ifdef NOVA_HAVE_CUDA
    // On-GPU-Draft (Phase 1): das residente 3B läuft komplett auf der GPU (kein Per-Matmul-Roundtrip).
    // draft_next_ ersetzt draft_pending_ — nur der greedy-argmax fürs nächste Draft-Token.
    GpuForward         draft_gpu_;
    bool               draft_gpu_active_ = false;
    int                draft_next_ = 0;
    // Foundation (NOVA_GPU_FWD24): das gestreamte 24B-Target läuft on-GPU (Aktivierungen+KV resident,
    // Gewichte gestreamt) -> CPU aus dem Loop. Wenn aktiv, ersetzt es forward_step/forward_batch/kv_trim
    // fürs Target.
    GpuForward         tgt_gpu_;
    bool               tgt_gpu_active_ = false;
#endif

    // Target-Forward-Wrapper: on-GPU (tgt_gpu_) wenn aktiv, sonst CPU-Glue (forward_step/-batch/kv_trim).
    std::vector<float> tgt_forward_step(int tok, int pos);
    std::vector<float> tgt_forward_batch(const int* toks, int N, int start_pos);
    void               tgt_kv_trim(int len);

    void spec_step();   // ein Spec-Schritt: füllt emit_queue_
};

}  // namespace nova::infer

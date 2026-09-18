// mistral3_config.h — Architektur-Parameter der Nova-Modelle (GGUF-Arch "mistral3").
//
// Werte aus den GGUF-Metadaten verifiziert (siehe Memory nova-model-architecture).
// Beide Modelle: GQA (32 Q / 8 KV Heads), head_dim 128, SwiGLU-FFN, RMSNorm,
// separater output.weight (nicht tied), vocab 131072, bos=1 eos=2. Full-Attention
// (24B, sliding_window=null). Die Vision-Teile werden ignoriert (nur Text-Pfad).
#pragma once

#include <cstdint>

namespace nova::infer {

struct RopeScaling {          // nur 3B nutzt YaRN; 24B: none
    bool   yarn = false;
    float  factor = 1.0f;
    int    orig_context = 0;
    float  beta_fast = 32.0f;
    float  beta_slow = 1.0f;
};

struct Mistral3Config {
    int   n_layers      = 0;
    int   n_heads       = 0;    // Q-Heads
    int   n_kv_heads    = 0;    // GQA
    int   head_dim      = 0;
    int   hidden        = 0;    // embedding_length
    int   ffn           = 0;    // feed_forward_length (SwiGLU intermediate)
    int   vocab         = 0;
    float rope_theta    = 0.0f;
    float rms_eps       = 1e-5f;
    int   bos_id        = 1;
    int   eos_id        = 2;
    RopeScaling rope;

    int q_dim()  const { return n_heads    * head_dim; }   // 24B: 4096
    int kv_dim() const { return n_kv_heads * head_dim; }   // 24B/3B: 1024
    int kv_group() const { return n_heads / n_kv_heads; }  // 4

    // Mistral Small 24B (chat/brain/apex).
    static Mistral3Config m24b() {
        Mistral3Config c;
        c.n_layers = 40; c.n_heads = 32; c.n_kv_heads = 8; c.head_dim = 128;
        c.hidden = 5120; c.ffn = 32768; c.vocab = 131072;
        c.rope_theta = 1e9f; c.rms_eps = 1e-5f;
        return c;
    }
    // ministral-3:3b (draft/subagent) — ~3.8B, YaRN-RoPE.
    static Mistral3Config m3b() {
        Mistral3Config c;
        c.n_layers = 26; c.n_heads = 32; c.n_kv_heads = 8; c.head_dim = 128;
        c.hidden = 3072; c.ffn = 9216; c.vocab = 131072;
        c.rope_theta = 1e6f; c.rms_eps = 1e-5f;
        c.rope.yarn = true; c.rope.factor = 16.0f; c.rope.orig_context = 16384;
        c.rope.beta_fast = 32.0f; c.rope.beta_slow = 1.0f;
        return c;
    }
};

// GGUF-Tensornamen (Sprachmodell) — verifiziert gegen das 24B-Inventar:
//   token_embd.weight            [hidden, vocab]      Q4_K
//   blk.{i}.attn_norm.weight     [hidden]             F32   (RMSNorm-Gain)
//   blk.{i}.attn_q.weight        [hidden, q_dim]      Q4_K
//   blk.{i}.attn_k.weight        [hidden, kv_dim]     Q4_K
//   blk.{i}.attn_v.weight        [hidden, kv_dim]     Q6_K
//   blk.{i}.attn_output.weight   [q_dim, hidden]      Q4_K
//   blk.{i}.ffn_norm.weight      [hidden]             F32
//   blk.{i}.ffn_gate.weight      [hidden, ffn]        Q4_K
//   blk.{i}.ffn_up.weight        [hidden, ffn]        Q4_K
//   blk.{i}.ffn_down.weight      [ffn, hidden]        Q6_K
//   output_norm.weight           [hidden]             F32
//   output.weight                [hidden, vocab]      Q6_K  (nicht tied)

}  // namespace nova::infer

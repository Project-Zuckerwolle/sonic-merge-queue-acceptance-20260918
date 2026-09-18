// kv_manager.h — TurboQuant KV-Cache + H2O-Fallback (Design §7, Testbed 6).
//
// Nur 5 der 30 Gemma-4-Layer wachsen mit dem Kontext (globale Attention);
// die 25 Sliding-Window-Layer sind fix (§7.2). Dieser Manager hält die
// wachsenden Layer: pro Token, pro Layer, pro KV-Head je ein K- und ein
// V-Vektor (head_dim), TurboQuant-komprimiert (~2,5–3 Bit/Wert).
//
// Budget (§7.2): ~3,1 KB/Token bei 5 Layern, 8 KV-Heads, head_dim 128.
// Gaming-PC 1 GB -> ~302K Token; Server-PC 3 GB -> ~947K Token.
//
// H2O-Fallback (§7.3): bei VRAM-Druck (Gaming-Schwelle) oder Apex-Start wird
// der Cache auf ein Byte-Budget getrimmt. Verworfen werden die Tokens mit der
// geringsten Priorität = akkumulierte Attention-Masse (Heavy-Hitter-Oracle),
// die jüngsten `recent_window` Tokens sind geschützt (Recency).
#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "InferEngine/turboquant.h"

namespace nova::infer {

struct KvConfig {
    int growing_layers = 5;    // globale Attention-Layer (§7.2)
    int kv_heads       = 8;
    int head_dim       = 128;
    int recent_window  = 64;   // H2O: jüngste Tokens nie verwerfen (Recency)
};

// Komprimierter KV-Eintrag eines Tokens über alle wachsenden Layer/Heads.
struct KvToken {
    int                       position = 0;       // ursprüngliche Sequenzposition
    double                    importance = 0.0;   // akkumulierte Attention-Masse (H2O)
    std::vector<CompressedVec> keys;              // growing_layers * kv_heads
    std::vector<CompressedVec> values;            // growing_layers * kv_heads
    size_t bytes() const;
};

class KvCache {
public:
    KvCache(KvConfig cfg, const TurboQuantConfig& tq);

    // Hängt einen Token an. k/v: growing_layers*kv_heads*head_dim FP32
    // (Layout: [layer][head][dim]). Liefert den Index im Cache.
    int append(const float* k, const float* v);

    // Rekonstruiert K bzw. V eines Tokens (Layout wie bei append).
    void reconstruct_keys(int token_slot, float* out) const;
    void reconstruct_values(int token_slot, float* out) const;

    // H2O: registriert Attention-Gewichte des aktuellen Query-Schritts auf die
    // gespeicherten Tokens (weights.size() == size()). Erhöht importance.
    void record_attention(const std::vector<float>& weights);

    // Trimmt den Cache auf <= budget_bytes (H2O). Liefert Anzahl verworfener Tokens.
    // Der gepinnte Prefix (pin_prefix) wird nie verworfen.
    int  trim_to_bytes(size_t budget_bytes);

    // --- KV-Prefix-Cache (KV bounded-per-turn) ---------------------------------
    // Pinnt die ersten n_tokens (stabiler System-Prefix: persona/identity/hot/
    // daily_briefing) — zwischen Turns NICHT neu prefillen, nie evicten.
    void pin_prefix(int n_tokens);
    // Verwirft nur den wechselnden Suffix (Working Memory + BM25 + Turn); der
    // Prefix-KV bleibt erhalten. Vor dem Prefill des nächsten Turns aufrufen.
    void reset_dynamic();
    int  prefix_len() const { return prefix_pinned_; }

    size_t size()  const { return tokens_.size(); }
    size_t bytes() const { return total_bytes_; }
    size_t bytes_per_token_estimate() const;  // §7.2-Schätzung für Budget-Planung

    const KvToken& token(size_t i) const { return tokens_[i]; }
    const TurboQuant& quantizer() const { return tq_; }

private:
    int vectors_per_token() const { return cfg_.growing_layers * cfg_.kv_heads; }

    KvConfig            cfg_;
    TurboQuant          tq_;
    std::deque<KvToken> tokens_;
    size_t              total_bytes_ = 0;
    int                 next_position_ = 0;
    int                 prefix_pinned_ = 0;   // gepinnte Prefix-Tokens (nie evicten)
};

}  // namespace nova::infer

// kv_manager.cpp — Implementierung von kv_manager.h (Design §7.2/§7.3).
#include "InferEngine/kv_manager.h"

#include <algorithm>
#include <vector>

namespace nova::infer {

size_t KvToken::bytes() const {
    size_t b = sizeof(int) + sizeof(double);
    for (const auto& c : keys)   b += c.bytes();
    for (const auto& c : values) b += c.bytes();
    return b;
}

KvCache::KvCache(KvConfig cfg, const TurboQuantConfig& tq)
    : cfg_(cfg), tq_(tq) {}

int KvCache::append(const float* k, const float* v) {
    const int vpt = vectors_per_token();
    const int d   = cfg_.head_dim;

    KvToken t;
    t.position = next_position_++;
    t.keys.reserve(vpt);
    t.values.reserve(vpt);
    for (int i = 0; i < vpt; ++i) {
        t.keys.push_back(tq_.compress(k + size_t(i) * d));
        t.values.push_back(tq_.compress(v + size_t(i) * d));
    }
    total_bytes_ += t.bytes();
    tokens_.push_back(std::move(t));
    return int(tokens_.size()) - 1;
}

void KvCache::reconstruct_keys(int slot, float* out) const {
    const int vpt = vectors_per_token();
    const int d   = cfg_.head_dim;
    const KvToken& t = tokens_[slot];
    for (int i = 0; i < vpt; ++i) tq_.decompress(t.keys[i], out + size_t(i) * d);
}

void KvCache::reconstruct_values(int slot, float* out) const {
    const int vpt = vectors_per_token();
    const int d   = cfg_.head_dim;
    const KvToken& t = tokens_[slot];
    for (int i = 0; i < vpt; ++i) tq_.decompress(t.values[i], out + size_t(i) * d);
}

void KvCache::record_attention(const std::vector<float>& weights) {
    const size_t n = std::min(weights.size(), tokens_.size());
    for (size_t i = 0; i < n; ++i) tokens_[i].importance += weights[i];
}

int KvCache::trim_to_bytes(size_t budget_bytes) {
    if (total_bytes_ <= budget_bytes) return 0;

    // Schutz: gepinnter Prefix (unten) + jüngste recent_window Tokens (oben).
    const int n = int(tokens_.size());
    const int lo = std::min(prefix_pinned_, n);
    const int protected_from = std::max(lo, n - cfg_.recent_window);

    // Kandidaten = mittlere Tokens, aufsteigend nach importance (Heavy-Hitter behalten).
    std::vector<int> cand;
    cand.reserve(protected_from > lo ? protected_from - lo : 0);
    for (int i = lo; i < protected_from; ++i) cand.push_back(i);
    std::sort(cand.begin(), cand.end(),
              [&](int a, int b) { return tokens_[a].importance < tokens_[b].importance; });

    std::vector<char> drop(n, 0);
    size_t bytes = total_bytes_;
    int dropped = 0;
    for (int idx : cand) {
        if (bytes <= budget_bytes) break;
        drop[idx] = 1;
        bytes -= tokens_[idx].bytes();
        ++dropped;
    }
    if (dropped == 0) return 0;

    std::deque<KvToken> kept;
    for (int i = 0; i < n; ++i)
        if (!drop[i]) kept.push_back(std::move(tokens_[i]));
    tokens_.swap(kept);
    total_bytes_ = bytes;
    return dropped;
}

void KvCache::pin_prefix(int n_tokens) {
    if (n_tokens < 0) n_tokens = 0;
    if (n_tokens > int(tokens_.size())) n_tokens = int(tokens_.size());
    prefix_pinned_ = n_tokens;
}

void KvCache::reset_dynamic() {
    // Nur den Suffix (Working Memory + BM25 + Turn) verwerfen; Prefix-KV behalten.
    while (int(tokens_.size()) > prefix_pinned_) {
        total_bytes_ -= tokens_.back().bytes();
        tokens_.pop_back();
    }
    next_position_ = prefix_pinned_;
}

size_t KvCache::bytes_per_token_estimate() const {
    // §7.2: 5 Layer x 2 (K+V) x 8 Heads x 128 Dim x 0,3125 Byte = 3,1 KB.
    // Hier aus der tatsächlichen TurboQuant-Bitrate gerechnet.
    const double bits_per_val = tq_.bits_per_value();
    const double vals = double(vectors_per_token()) * 2.0 * cfg_.head_dim;
    return size_t(vals * bits_per_val / 8.0);
}

}  // namespace nova::infer

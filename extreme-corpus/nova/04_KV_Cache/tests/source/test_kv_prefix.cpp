// test_kv_prefix.cpp — KV-Prefix-Cache (KV bounded-per-turn, Kernel-Task ④/KV).
//
// Beweist die KV-Strategie ohne Qualitätsverlust:
//  - pin_prefix() hält den stabilen System-Prefix; reset_dynamic() verwirft NUR
//    den wechselnden Suffix (Working Memory + BM25 + Turn), Prefix-KV bleibt,
//  - trim_to_bytes (H2O) evictet unter Druck nie den gepinnten Prefix,
//  - Positionen laufen nach reset_dynamic korrekt weiter (Re-Prefill des Suffix).
#include "InferEngine/kv_manager.h"

#include <cstdio>
#include <iostream>
#include <set>
#include <vector>

using namespace nova::infer;

namespace {
bool has_position(const KvCache& kv, int pos) {
    for (size_t i = 0; i < kv.size(); ++i) if (kv.token(i).position == pos) return true;
    return false;
}
}  // namespace

int main() {
    std::cout << "=== KV-Prefix-Cache (bounded-per-turn) ===\n";

    KvConfig kc; kc.growing_layers = 1; kc.kv_heads = 1; kc.head_dim = 4; kc.recent_window = 2;
    TurboQuantConfig tq; tq.head_dim = 4; tq.num_outliers = 1;
    KvCache kv(kc, tq);

    const std::vector<float> k(4, 0.5f), v(4, -0.25f);  // vpt*head_dim = 1*4

    // Prefix: 3 stabile Tokens (persona/identity/hot), dann pinnen.
    for (int i = 0; i < 3; ++i) kv.append(k.data(), v.data());
    kv.pin_prefix(3);
    // Dynamischer Suffix: 6 Tokens (Working Memory + BM25 + Turn).
    for (int i = 0; i < 6; ++i) kv.append(k.data(), v.data());

    bool pass = true;
    pass &= (kv.size() == 9);
    pass &= (kv.prefix_len() == 3);
    std::printf("  Nach Turn 1: size=%zu, prefix_len=%d\n", kv.size(), kv.prefix_len());

    // Turn-Ende: nur Suffix verwerfen, Prefix behalten.
    kv.reset_dynamic();
    pass &= (kv.size() == 3);
    for (int p = 0; p < 3; ++p) pass &= has_position(kv, p);
    std::printf("  Nach reset_dynamic: size=%zu (Prefix 0,1,2 erhalten)\n", kv.size());

    // Turn 2: neuen Suffix prefillen — Positionen laufen ab 3 weiter.
    for (int i = 0; i < 4; ++i) kv.append(k.data(), v.data());
    pass &= (kv.size() == 7);
    pass &= has_position(kv, 3) && has_position(kv, 6);   // Suffix 3..6
    std::printf("  Nach Turn 2: size=%zu (Suffix 3..6)\n", kv.size());

    // H2O unter maximalem Druck: Prefix (0,1,2) + Recency (5,6) müssen überleben.
    const int dropped = kv.trim_to_bytes(0);
    std::set<int> present;
    for (size_t i = 0; i < kv.size(); ++i) present.insert(kv.token(i).position);
    std::printf("  trim_to_bytes(0): %d verworfen, verbleibend = {", dropped);
    for (int p : present) std::printf("%d ", p);
    std::printf("}\n");

    for (int p : {0, 1, 2}) pass &= (present.count(p) == 1);   // Prefix NIE evictet
    for (int p : {5, 6}) pass &= (present.count(p) == 1);      // Recency geschützt
    pass &= (present.count(3) == 0 && present.count(4) == 0);  // mittlere Tokens evictet

    std::cout << "\n=== KV-Prefix-Cache: " << (pass ? "BESTANDEN" : "FEHLGESCHLAGEN") << " ===\n";
    return pass ? 0 : 1;
}

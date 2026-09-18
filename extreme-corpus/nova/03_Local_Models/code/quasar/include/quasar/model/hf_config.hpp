#pragma once
//
// quasar — model/hf_config.hpp
//
// aufzaehlt, in *einer* Struktur -- und zwei Befuellern:
//
//   read_hf_config(pfad)      HuggingFace config.json
//   params_from_gguf(datei)   Metadaten einer GGUF-Datei
//
// Zwei Befueller, weil die Quellen die Felder verschieden nennen.
//
// welches. Es wird nichts auf 0 gesetzt und nichts geraten.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace quasar::gguf {
class GgufFile;
}

namespace quasar {

// RoPE-Skalierung. `type` ist leer, wenn die Quelle keine angibt.
struct RopeScaling {
  std::string type;                          // "linear", "dynamic", "yarn", "llama3", ...
  double factor = 1.0;
  std::uint32_t original_context_length = 0;  // 0 = nicht angegeben
  double low_frequency_factor = 0.0;
  double high_frequency_factor = 0.0;
  double attention_factor = 0.0;
  double beta_fast = 0.0;
  double beta_slow = 0.0;

  bool present() const noexcept { return !type.empty(); }
};

struct ModelParams {
  std::string architecture;   // "llama", "mistral3", "gemma3", ...
  std::uint32_t n_layers = 0;
  std::uint32_t n_embd = 0;   // hidden size
  std::uint32_t n_heads = 0;
  std::uint32_t n_heads_kv = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t n_vocab = 0;
  std::uint32_t n_ff = 0;     // feed-forward / intermediate size
  std::uint32_t context_length = 0;
  float rope_theta = 0.0f;
  RopeScaling rope_scaling;
  float norm_epsilon = 0.0f;

  // Nachvollziehbar machen, was aus der Datei kam und was abgeleitet wurde.
  bool head_dim_derived = false;    // head_dim = n_embd / n_heads
  bool n_heads_kv_derived = false;  // kein GQA-Feld -> n_heads_kv = n_heads
  std::string source;               // Pfad bzw. Herkunft, fuer Fehlermeldungen

  // Wirft, wenn eine Groesse in sich nicht stimmt (0-Werte, n_heads_kv teilt
  // n_heads nicht, ...). Wird von beiden Befuellern am Ende aufgerufen.
  void validate() const;

  std::string describe() const;
};

// `path` darf die config.json selbst oder das Verzeichnis sein, das sie
// enthaelt. Multimodale Konfigurationen legen die Sprachparameter unter
// "text_config" ab; dort wird ebenfalls gesucht.
ModelParams read_hf_config(const std::string& path);

ModelParams params_from_gguf(const gguf::GgufFile& file);

}  // namespace quasar

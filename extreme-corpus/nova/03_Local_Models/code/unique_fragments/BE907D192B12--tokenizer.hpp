#pragma once
//
// quasar — tokenizer/tokenizer.hpp
//
// SentencePiece-Variante, Unigram, WordPiece) mit den zugehoerigen
// Normalisierer- und Vorzerleger-Ketten. Quellen: HuggingFace `tokenizer.json`
// und das in GGUF eingebettete Vokabular.
//
// werden **exakt** reproduziert. Die Referenz stammt aus der Python-Bibliothek
// `tokenizers`, nicht aus quasar.
//
// Was dieser Tokenizer bewusst NICHT tut: Offsets im Ausgangstext mitfuehren.
// `detokenize`, keine Zeichenbereiche. Das spart die gesamte

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace quasar::gguf {
class GgufFile;
}

namespace quasar::tok {

using TokenId = std::uint32_t;

// Ein Sondertoken bzw. nachtraeglich hinzugefuegtes Token.
struct AddedToken {
  std::string content;
  TokenId id = 0;
  bool special = false;
  bool single_word = false;
  bool lstrip = false;
  bool rstrip = false;
  bool normalized = true;
};

struct SpecialIds {
  std::optional<TokenId> bos;
  std::optional<TokenId> eos;
  std::optional<TokenId> unk;
  std::optional<TokenId> pad;
  bool add_bos = false;
  bool add_eos = false;
};

// Vorwaertsdeklarationen der inneren Bausteine. Ihre Definition steht in
// src/tokenizer/components.hpp und geht den Aufrufer nichts an.
class Normalizer;
class PreTokenizer;
class Model;
class Decoder;
class PostProcessor;
class AddedVocabulary;

class Tokenizer {
 public:
  Tokenizer();
  ~Tokenizer();
  Tokenizer(Tokenizer&&) noexcept;
  Tokenizer& operator=(Tokenizer&&) noexcept;
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  // --- Aufbau -------------------------------------------------------------
  // HuggingFace tokenizer.json. Wirft quasar::Error bei allem, was nicht
  // eindeutig ist -- unbekannter Bausteintyp, fehlendes Pflichtfeld, kaputtes
  // Vokabular. Kein stiller Rueckfall auf einen anderen Algorithmus.
  static Tokenizer from_json_file(const std::string& path);
  static Tokenizer from_json_text(std::string_view text, std::string_view origin);

  // In GGUF eingebettetes Vokabular (tokenizer.ggml.*).
  static Tokenizer from_gguf(const gguf::GgufFile& g);

  // Die Rohbeschreibung eines Tokenizers, wie GGUF sie liefert. Genau das
  // "Vollstaendig eingebettet -- zur Laufzeit ist keine Fremddatei noetig").
  // Deshalb ist es eine eigene, einfach serialisierbare Struktur und nicht
  // der fertige Tokenizer.
  struct VocabData {
    std::string family;                    // tokenizer.ggml.model, z. B. "gpt2"
    std::string pre;                       // tokenizer.ggml.pre, z. B. "llama-bpe"
    std::vector<std::string> tokens;
    std::vector<std::int32_t> token_types;  // leer erlaubt
    std::vector<std::string> merges;        // je "a b"
    SpecialIds special;
  };
  static VocabData extract_vocab(const gguf::GgufFile& g);
  static Tokenizer from_vocab_data(const VocabData& data, std::string_view origin);

  // --- Benutzung ----------------------------------------------------------
  std::vector<TokenId> encode(std::string_view text, bool add_special_tokens) const;
  // Zusaetzlich die Token-Strings -- nur fuer Tests und Fehlersuche.
  std::vector<TokenId> encode_with_tokens(std::string_view text,
                                          bool add_special_tokens,
                                          std::vector<std::string>& tokens_out) const;

  std::string decode(std::span<const TokenId> ids, bool skip_special_tokens) const;

  std::size_t vocab_size() const;
  std::optional<TokenId> token_to_id(std::string_view token) const;
  std::optional<std::string> id_to_token(TokenId id) const;

  const SpecialIds& special_ids() const noexcept { return special_; }

  // Kurzbeschreibung der geladenen Kette -- fuer `quasar-convert --info`
  // und fuer Fehlermeldungen.
  std::string describe() const;

  // --- Zwischenstufen, nur fuer Tests ------------------------------------
  // Damit ein Unterschied in den IDs lokalisierbar ist statt nur zaehlbar.
  std::string normalize_only(std::string_view text) const;
  std::vector<std::string> pre_tokenize_only(std::string_view normalized) const;

 private:
  friend class TokenizerBuilder;

  std::unique_ptr<Normalizer> normalizer_;
  std::unique_ptr<PreTokenizer> pre_tokenizer_;
  std::unique_ptr<Model> model_;
  std::unique_ptr<Decoder> decoder_;
  std::unique_ptr<PostProcessor> post_processor_;
  std::unique_ptr<AddedVocabulary> added_;
  SpecialIds special_;
  std::string description_;
};

}  // namespace quasar::tok

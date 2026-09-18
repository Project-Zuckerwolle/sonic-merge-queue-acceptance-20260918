#pragma once
//
// quasar — src/tokenizer/components.hpp
//
// Die inneren Bausteine der Tokenizer-Kette. Nicht oeffentlich: der Aufrufer
// sieht nur `quasar::tok::Tokenizer`.
//
// Aufbau wie bei HuggingFace `tokenizers`, weil die Referenzdaten von dort
// stammen und jede Abweichung in der Zerlegung eine Abweichung in den IDs ist:
//
//   Text
//    -> AddedVocabulary   zerlegt an hinzugefuegten/Sondertoken
//    -> Normalizer        NFC/NFKC, Lowercase, StripAccents, Replace, ...
//    -> PreTokenizer      ByteLevel, Metaspace, Whitespace, Bert, Split, ...
//    -> Model             BPE | Unigram | WordPiece
//    -> PostProcessor     Sondertoken anfuegen
//
// Rueckweg:
//   IDs -> Token-Strings -> Decoder-Kette -> Text
//
// Anders als HuggingFace fuehren wir **keine** Offsets mit (siehe
// tokenizer.hpp). Ein Normalisierer ist deshalb schlicht string -> string.

#include <array>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "quasar/core/error.hpp"
#include "quasar/tokenizer/tokenizer.hpp"

namespace quasar::json {
class Value;
}

namespace quasar::tok {

// Laenge des UTF-8-Zeichens, das bei `i` beginnt. Bei einem ungueltigen
// Startbyte 1 -- hier wird immer auf bereits geprueftem UTF-8 gearbeitet, und
// ein Fortschritt von 1 verhindert eine Endlosschleife.
inline std::size_t utf8_len(std::string_view s, std::size_t i) noexcept {
  const auto b = static_cast<unsigned char>(s[i]);
  std::size_t n = 1;
  if ((b >> 5) == 0x6) n = 2;
  else if ((b >> 4) == 0xE) n = 3;
  else if ((b >> 3) == 0x1E) n = 4;
  const std::size_t rest = s.size() - i;
  return n < rest ? n : rest;
}

// --- Normalisierer --------------------------------------------------------

class Normalizer {
 public:
  virtual ~Normalizer() = default;
  virtual std::string normalize(std::string s) const = 0;
  virtual std::string name() const = 0;
};

std::unique_ptr<Normalizer> make_normalizer(const json::Value& v, std::string_view where);

// --- Vorzerleger ----------------------------------------------------------
//
// Ein Vorzerleger nimmt ein Stueck und liefert null oder mehr Stuecke. Ketten
// wirken auf jedes Stueck einzeln.

class PreTokenizer {
 public:
  virtual ~PreTokenizer() = default;
  virtual void pre_tokenize(std::string piece, std::vector<std::string>& out) const = 0;

  // Wie oben, aber mit der Angabe, ob dieses Stueck **am Anfang des
  // urspruenglichen Textes** beginnt (Byte-Offset 0).
  //
  // Genau ein Vorzerleger braucht das: Metaspace mit `prepend_scheme: "first"`
  // (Befund S14). Die Referenz prueft dort `normalized.offsets_original().0 == 0`
  // -- also nicht "erstes Stueck", sondern "beginnt bei Offset 0". Der
  // Unterschied ist messbar: bei `" Hallo Welt"` entfernt WhitespaceSplit das
  // fuehrende Leerzeichen, das erste Stueck beginnt bei Offset 1, und die
  // Referenz stellt **nichts** voran.
  //
  // Alle anderen Vorzerleger ignorieren die Angabe -- die Vorgabe reicht den
  // Aufruf an die zweistellige Form durch, sodass keine Umsetzung angefasst
  // werden muss, die es nicht angeht.
  virtual void pre_tokenize(std::string piece, std::vector<std::string>& out,
                            bool at_original_start) const {
    (void)at_original_start;
    pre_tokenize(std::move(piece), out);
  }

  virtual std::string name() const = 0;

  std::vector<std::string> run(std::string piece) const {
    std::vector<std::string> out;
    pre_tokenize(std::move(piece), out, /*at_original_start=*/true);
    return out;
  }
};

std::unique_ptr<PreTokenizer> make_pre_tokenizer(const json::Value& v,
                                                 std::string_view where);

// --- Vokabular ------------------------------------------------------------
//
// Token-String <-> ID. Die Strings liegen einmal im Speicher; die Suche laeuft
// ueber eine Hashtabelle auf `string_view` in diesen Speicher.

class Vocab {
 public:
  // Legt `token` bei `id` ab. Ein zweites Mal derselbe Text behaelt die erste
  // ID (HuggingFace-Vokabulare enthalten vereinzelt Dubletten); dieselbe ID
  // zweimal ist dagegen ein Fehler und bricht ab.
  void add(std::string token, TokenId id);

  std::optional<TokenId> id_of(std::string_view token) const {
    auto it = to_id_.find(token);
    return it == to_id_.end() ? std::nullopt : std::optional<TokenId>(it->second);
  }
  bool contains(std::string_view token) const { return to_id_.find(token) != to_id_.end(); }
  const std::string* token_of(TokenId id) const {
    if (id >= to_token_.size()) return nullptr;
    return to_token_[id];
  }
  std::size_t size() const noexcept { return to_token_.size(); }
  const std::deque<std::string>& all() const noexcept { return storage_; }

 private:
  struct SvHash {
    using is_transparent = void;
    std::size_t operator()(std::string_view s) const noexcept {
      return std::hash<std::string_view>{}(s);
    }
  };
  struct SvEq {
    using is_transparent = void;
    bool operator()(std::string_view a, std::string_view b) const noexcept { return a == b; }
  };

  std::deque<std::string> storage_;  // stabile Adressen fuer die string_view
  std::unordered_map<std::string_view, TokenId, SvHash, SvEq> to_id_;
  std::vector<const std::string*> to_token_;
};

// --- Modell ---------------------------------------------------------------

class Model {
 public:
  virtual ~Model() = default;
  // Zerlegt EIN vorzerlegtes Stueck.
  virtual void tokenize(std::string_view piece, std::vector<TokenId>& ids,
                        std::vector<std::string>* tokens) const = 0;
  virtual const Vocab& vocab() const = 0;
  virtual std::string name() const = 0;
};

std::unique_ptr<Model> make_model(const json::Value& v, std::string_view where);

// Bauteile fuer den GGUF-Weg. Dort gibt es kein tokenizer.json, sondern
// `tokenizer.ggml.*`; die Kette wird aus dem Modellnamen und dem
// `tokenizer.ggml.pre`-Feld zusammengesetzt.
struct BpeSpec {
  std::vector<std::pair<std::string, std::string>> merges;
  std::optional<std::string> unk_token;
  std::optional<std::string> continuing_subword_prefix;
  std::optional<std::string> end_of_word_suffix;
  bool fuse_unk = false;
  bool byte_fallback = false;
  bool ignore_merges = false;
};
std::unique_ptr<Model> make_bpe_model(Vocab vocab, BpeSpec spec, std::string_view where);
std::unique_ptr<Model> make_unigram_model(std::vector<std::pair<std::string, double>> pieces,
                                          std::optional<TokenId> unk_id,
                                          bool byte_fallback, std::string_view where);
std::unique_ptr<Model> make_wordpiece_model(Vocab vocab, std::string unk_token,
                                            std::string prefix, std::size_t max_chars,
                                            std::string_view where);

std::unique_ptr<PreTokenizer> make_byte_level_pre(bool add_prefix_space, bool use_regex);
std::unique_ptr<PreTokenizer> make_llama3_pre();
std::unique_ptr<PreTokenizer> make_tekken_pre();
std::unique_ptr<Decoder> make_byte_level_decoder();
std::unique_ptr<Decoder> make_sentencepiece_decoder();

// --- Dekodierer -----------------------------------------------------------

class Decoder {
 public:
  virtual ~Decoder() = default;
  virtual std::vector<std::string> decode_chain(std::vector<std::string> tokens) const = 0;
  virtual std::string name() const = 0;
};

std::unique_ptr<Decoder> make_decoder(const json::Value& v, std::string_view where);

// --- Nachbearbeitung ------------------------------------------------------

class PostProcessor {
 public:
  virtual ~PostProcessor() = default;
  virtual void process(std::vector<TokenId>& ids, std::vector<std::string>* tokens) const = 0;
  virtual std::string name() const = 0;
};

std::unique_ptr<PostProcessor> make_post_processor(const json::Value& v,
                                                   std::string_view where);

// Fuer den GGUF-Weg: dort gibt es keine Vorlage, sondern nur die Schalter
// `add_bos_token` / `add_eos_token`.
class Model;
class AddedVocabulary;
std::unique_ptr<PostProcessor> make_bos_eos_processor(std::optional<TokenId> bos,
                                                      std::optional<TokenId> eos,
                                                      const Model& model,
                                                      const AddedVocabulary& added);

// --- Hinzugefuegte Token --------------------------------------------------
//
// HuggingFace zerlegt zweistufig: erst an den nicht-normalisierten
// hinzugefuegten Token auf dem Rohtext, dann -- je Reststueck -- normalisieren
// und an den normalisierten hinzugefuegten Token zerlegen. Diese Reihenfolge
// ist nicht beliebig; sie entscheidet ueber die IDs.

class AddedVocabulary {
 public:
  void add(AddedToken t);

  // Baut die Suchreihenfolge auf (laengste zuerst).
  //
  // `normalizer` wird gebraucht, weil hinzugefuegte Token mit
  // `normalized: true` **in ihrer normalisierten Form** gesucht werden. Bei
  // Llama-2 stellt der Normalisierer ein U+2581 voran; das Sondertoken `<s>`
  // wird deshalb als `_<s>` gesucht, nicht als `<s>`. Wer das uebersieht,
  // bekommt ein zusaetzliches Leerzeichen-Token vor jedem Sondertoken.
  void finish(const Normalizer* normalizer);

  struct Piece {
    std::string text;           // Rohtext, wenn !id; sonst der Treffertext
    std::optional<TokenId> id;  // gesetzt -> fertiges Token
  };

  // Zerlegt `text` an den Token mit `normalized == want_normalized`.
  void split(std::string_view text, bool want_normalized,
             std::vector<Piece>& out) const;

  const std::vector<AddedToken>& tokens() const noexcept { return tokens_; }
  bool is_special(TokenId id) const;
  const std::string* content_of(TokenId id) const;
  std::optional<TokenId> id_of(std::string_view content) const;
  bool empty() const noexcept { return tokens_.empty(); }

 private:
  std::vector<AddedToken> tokens_;
  // Die Zeichenkette, nach der tatsaechlich gesucht wird. Fuer
  // `normalized: true` ist das die normalisierte Fassung von `content`.
  std::vector<std::string> patterns_;
  // Indizes in `tokens_`, sortiert nach Laenge absteigend, getrennt nach
  // normalisiert/nicht normalisiert -- und zusaetzlich nach erstem Byte
  // vorsortiert. Ohne diesen Eimer laeuft die Suche bei einem Modell mit ein
  // paar tausend Sondertoken pro Byteposition ueber alle davon.
  std::array<std::vector<std::size_t>, 256> bucket_raw_;
  std::array<std::vector<std::size_t>, 256> bucket_norm_;
  bool any_raw_ = false;
  bool any_norm_ = false;
  std::map<TokenId, std::size_t> by_id_;
  std::map<std::string, TokenId, std::less<>> by_content_;
};

// --- gemeinsame Hilfen ----------------------------------------------------

// Byte <-> sichtbares Zeichen, wie GPT-2 es macht (256 Bytes auf 256
// druckbare Codepunkte). Wird von ByteLevel-Vorzerleger und -Dekodierer
// gebraucht.
const std::array<char32_t, 256>& byte_to_char();
// Umkehrung; liefert -1, wenn der Codepunkt kein Byte-Zeichen ist.
int char_to_byte(char32_t c);

// Der Vorzerleger-Regex von GPT-2, von Hand ausprogrammiert:
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// Oniguruma-Semantik: leftmost-first, Alternativen in Reihenfolge.
// Nachgemessen an der Referenzbibliothek: \s ist die Unicode-Eigenschaft
// White_Space, \p{L} und \p{N} sind die Unicode-Hauptkategorien.
void gpt2_split(std::string_view s, std::vector<std::string>& out);

// Der Regex von Llama-3 / `tokenizer.ggml.pre == "llama-bpe"`:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
void llama3_split(std::string_view s, std::vector<std::string>& out);

// SentencePiece-Charmap (`Precompiled`-Normalisierer). Traegt einen
// Darts-Doppelfeld-Trie plus einen Block mit Ersetzungszeichenketten.
class Precompiled {
 public:
  static Precompiled from_base64(std::string_view b64, std::string_view where);
  std::string normalize(std::string_view s) const;

 private:
  std::optional<std::string_view> lookup(std::string_view s) const;
  std::optional<std::string_view> decode_at(std::uint32_t offset) const;
  std::vector<std::uint32_t> trie_;
  std::string normalized_;
};

std::string base64_decode(std::string_view in, std::string_view where);

// Setzt die Teile zu einem Tokenizer zusammen. Einzige Stelle, die an die
// privaten Felder von `Tokenizer` darf (dort als friend eingetragen).
class TokenizerBuilder {
 public:
  struct Parts {
    std::unique_ptr<Normalizer> normalizer;
    std::unique_ptr<PreTokenizer> pre_tokenizer;
    std::unique_ptr<Model> model;
    std::unique_ptr<Decoder> decoder;
    std::unique_ptr<PostProcessor> post_processor;
    std::unique_ptr<AddedVocabulary> added;
    SpecialIds special;
  };
  static Tokenizer assemble(Parts p, std::string_view origin);
  static Tokenizer from_json(const json::Value& root, std::string_view origin);
};

}  // namespace quasar::tok

// quasar — src/tokenizer/tokenizer.cpp
//
// Zusammenbau und Ablauf. Die Reihenfolge ist die der Referenzbibliothek und
// nicht verhandelbar:
//
//   1. an den NICHT normalisierten hinzugefuegten Token zerlegen (Rohtext)
//   2. je Reststueck: normalisieren, dann an den NORMALISIERTEN hinzugefuegten
//      Token zerlegen
//   3. je Reststueck: vorzerlegen, dann das Modell anwenden
//   4. bei add_special_tokens: nachbearbeiten
//
// Wer 1 und 2 vertauscht, bekommt bei jedem Text mit Sondertoken andere IDs.

#include <algorithm>
#include <cstring>

#include "quasar/core/error.hpp"
#include "quasar/core/json.hpp"
#include "quasar/core/mapped_file.hpp"
#include "quasar/core/unicode.hpp"
#include "tokenizer/components.hpp"

namespace quasar::tok {

namespace uc = quasar::unicode;

namespace {
// Steht an Byteposition `i` (bzw. dem Zeichen, das dort endet) ein
// alphanumerisches Zeichen? Fuer `single_word` bei hinzugefuegten Token.
bool alnum_before(std::string_view s, std::size_t end) {
  // Befund S12: der Rueckwaertslauf hielt auf dem **ersten** Fortsetzungsbyte an
  // und ging nie bis zum Startbyte zurueck; `if (start == end) --start;` griff
  // nur fuer ASCII. Bei "\xC3\xA9" (é) blieb `start` auf 1 stehen, und
  // `to_codepoints("\xA9")` warf "freistehendes Fortsetzungsbyte" -- `encode()`
  // brach mitten in einer gueltigen Eingabe ab, sobald vor einem
  // `single_word`-Token ein Nicht-ASCII-Zeichen stand.
  //
  // Richtig ist: erst ueber alle Fortsetzungsbytes zurueck, **dann** ueber das
  // Startbyte.
  if (end == 0) return false;
  std::size_t start = end;
  while (start > 0 && (static_cast<unsigned char>(s[start - 1]) & 0xC0) == 0x80) --start;
  if (start == 0) return false;  // nur Fortsetzungsbytes -- kaputte Eingabe
  --start;                       // das Startbyte selbst
  const auto cps = uc::to_codepoints(s.substr(start, end - start));
  return !cps.empty() && uc::is_alphanumeric(cps.back());
}

// Laenge des Leerraum-Zeichens, das bei `i` beginnt -- 0, wenn dort keines
// steht. Befund G17: `lstrip`/`rstrip` schnitten nur ASCII-Leerraum ab, die
// Referenz benutzt `char::is_whitespace` (Property White_Space). Ein NBSP vor
// einem Added-Token blieb damit stehen und erzeugte ein zusaetzliches Stueck.
std::size_t whitespace_len_at(std::string_view s, std::size_t i) {
  if (i >= s.size()) return 0;
  const std::size_t n = utf8_len(s, i);
  if (n == 0 || i + n > s.size()) return 0;
  const std::string_view part = s.substr(i, n);
  if (!uc::is_valid_utf8(part)) return 0;
  const auto cps = uc::to_codepoints(part);
  return (!cps.empty() && uc::is_whitespace(cps.front())) ? n : 0;
}

// Dasselbe rueckwaerts: Laenge des Leerraum-Zeichens, das unmittelbar **vor**
// `end` endet -- 0, wenn dort keines steht.
std::size_t whitespace_len_before(std::string_view s, std::size_t end) {
  if (end == 0) return 0;
  std::size_t start = end;
  while (start > 0 && (static_cast<unsigned char>(s[start - 1]) & 0xC0) == 0x80) --start;
  if (start == 0) return 0;
  --start;
  const std::string_view part = s.substr(start, end - start);
  if (!uc::is_valid_utf8(part)) return 0;
  const auto cps = uc::to_codepoints(part);
  return (!cps.empty() && uc::is_whitespace(cps.front())) ? end - start : 0;
}

bool alnum_at(std::string_view s, std::size_t i) {
  if (i >= s.size()) return false;
  const auto cps = uc::to_codepoints(s.substr(i, utf8_len(s, i)));
  return !cps.empty() && uc::is_alphanumeric(cps.front());
}
}  // namespace

// =========================================================================
// AddedVocabulary
// =========================================================================

void AddedVocabulary::add(AddedToken t) {
  by_id_[t.id] = tokens_.size();
  tokens_.push_back(std::move(t));
}

void AddedVocabulary::finish(const Normalizer* normalizer) {
  for (auto& b : bucket_raw_) b.clear();
  for (auto& b : bucket_norm_) b.clear();
  any_raw_ = any_norm_ = false;
  by_content_.clear();
  patterns_.assign(tokens_.size(), std::string());

  for (std::size_t i = 0; i < tokens_.size(); ++i) {
    const AddedToken& t = tokens_[i];
    by_content_.emplace(t.content, t.id);
    patterns_[i] = (t.normalized && normalizer) ? normalizer->normalize(t.content)
                                                : t.content;
    if (patterns_[i].empty()) continue;
    const auto first = static_cast<unsigned char>(patterns_[i][0]);
    if (t.normalized) {
      bucket_norm_[first].push_back(i);
      any_norm_ = true;
    } else {
      bucket_raw_[first].push_back(i);
      any_raw_ = true;
    }
  }
  // Laengste zuerst -- sonst schluckt ein kurzes Sondertoken ein laengeres.
  const auto by_len = [this](std::size_t a, std::size_t b) {
    if (patterns_[a].size() != patterns_[b].size()) {
      return patterns_[a].size() > patterns_[b].size();
    }
    return patterns_[a] < patterns_[b];
  };
  for (auto& b : bucket_raw_) std::sort(b.begin(), b.end(), by_len);
  for (auto& b : bucket_norm_) std::sort(b.begin(), b.end(), by_len);
}

bool AddedVocabulary::is_special(TokenId id) const {
  auto it = by_id_.find(id);
  return it != by_id_.end() && tokens_[it->second].special;
}

const std::string* AddedVocabulary::content_of(TokenId id) const {
  auto it = by_id_.find(id);
  return it == by_id_.end() ? nullptr : &tokens_[it->second].content;
}

std::optional<TokenId> AddedVocabulary::id_of(std::string_view content) const {
  auto it = by_content_.find(content);
  return it == by_content_.end() ? std::nullopt : std::optional<TokenId>(it->second);
}

void AddedVocabulary::split(std::string_view text, bool want_normalized,
                            std::vector<Piece>& out) const {
  const auto& buckets = want_normalized ? bucket_norm_ : bucket_raw_;
  const bool any = want_normalized ? any_norm_ : any_raw_;
  if (!any || text.empty()) {
    if (!text.empty()) out.push_back(Piece{std::string(text), std::nullopt});
    return;
  }

  std::size_t i = 0;
  std::string pending;
  while (i < text.size()) {
    // Laengster Treffer an dieser Stelle. Der Eimer ist bereits nach Laenge
    // sortiert, der erste Treffer ist also der laengste.
    const AddedToken* hit = nullptr;
    const std::string* hit_pattern = nullptr;
    for (std::size_t idx : buckets[static_cast<unsigned char>(text[i])]) {
      const AddedToken& t = tokens_[idx];
      const std::string& pat = patterns_[idx];
      if (text.compare(i, pat.size(), pat) != 0) continue;
      if (t.single_word) {
        const bool left_ok = i == 0 || !alnum_before(text, i);
        const std::size_t after = i + pat.size();
        const bool right_ok = after >= text.size() || !alnum_at(text, after);
        if (!left_ok || !right_ok) continue;
      }
      hit = &t;
      hit_pattern = &pat;
      break;
    }
    if (!hit) {
      pending.push_back(text[i]);
      ++i;
      continue;
    }

    std::size_t start = i;
    std::size_t stop = i + hit_pattern->size();
    // Befund G17: hier standen vier ASCII-Zeichen. Die Referenz schneidet
    // Leerraum nach `char::is_whitespace` ab, also nach der Unicode-Eigenschaft
    // White_Space -- ein NBSP (U+00A0) vor einem `lstrip`-Token blieb bei uns
    // stehen und erzeugte ein zusaetzliches Stueck (`ids=[1,3]` statt `[3]`).
    if (hit->lstrip) {
      while (std::size_t n = whitespace_len_before(pending, pending.size())) {
        pending.erase(pending.size() - n);
      }
    }
    if (hit->rstrip) {
      while (std::size_t n = whitespace_len_at(text, stop)) stop += n;
    }
    (void)start;
    if (!pending.empty()) {
      out.push_back(Piece{std::move(pending), std::nullopt});
      pending.clear();
    }
    // Der Treffertext ist das, was hier tatsaechlich stand -- bei
    // normalisierten Token also die normalisierte Fassung. Genau diese
    // meldet auch die Referenzbibliothek als Token-String.
    out.push_back(Piece{*hit_pattern, hit->id});
    i = stop;
  }
  if (!pending.empty()) out.push_back(Piece{std::move(pending), std::nullopt});
}

// =========================================================================
// Tokenizer
// =========================================================================

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

std::size_t Tokenizer::vocab_size() const {
  std::size_t n = model_ ? model_->vocab().size() : 0;
  if (added_) {
    for (const auto& t : added_->tokens()) n = std::max<std::size_t>(n, t.id + 1);
  }
  return n;
}

std::optional<TokenId> Tokenizer::token_to_id(std::string_view token) const {
  if (added_) {
    if (auto id = added_->id_of(token)) return id;
  }
  return model_ ? model_->vocab().id_of(token) : std::nullopt;
}

std::optional<std::string> Tokenizer::id_to_token(TokenId id) const {
  if (added_) {
    if (const std::string* c = added_->content_of(id)) return *c;
  }
  if (model_) {
    if (const std::string* t = model_->vocab().token_of(id)) return *t;
  }
  return std::nullopt;
}

std::string Tokenizer::normalize_only(std::string_view text) const {
  return normalizer_ ? normalizer_->normalize(std::string(text)) : std::string(text);
}

std::vector<std::string> Tokenizer::pre_tokenize_only(std::string_view normalized) const {
  if (!pre_tokenizer_) return {std::string(normalized)};
  return pre_tokenizer_->run(std::string(normalized));
}

std::vector<TokenId> Tokenizer::encode(std::string_view text, bool add_special) const {
  std::vector<std::string> ignored;
  return encode_with_tokens(text, add_special, ignored);
}

std::vector<TokenId> Tokenizer::encode_with_tokens(std::string_view text, bool add_special,
                                                   std::vector<std::string>& tokens) const {
  if (!model_) throw_error("Tokenizer: kein Modell geladen");
  tokens.clear();
  std::vector<TokenId> ids;

  // Schritt 1: an den nicht normalisierten hinzugefuegten Token zerlegen.
  std::vector<AddedVocabulary::Piece> raw;
  if (added_) added_->split(text, /*want_normalized=*/false, raw);
  else if (!text.empty()) raw.push_back({std::string(text), std::nullopt});

  const auto emit_added = [&](TokenId id, std::string text) {
    ids.push_back(id);
    if (text.empty()) {
      const std::string* c = added_ ? added_->content_of(id) : nullptr;
      if (c) text = *c;
    }
    tokens.push_back(std::move(text));
  };

  // Befund S14: Metaspace mit `prepend_scheme: "first"` haengt daran, ob ein
  // Stueck am **Anfang des urspruenglichen Textes** beginnt. Das ist beim
  // ersten Reststueck der Fall und danach nie wieder.
  bool at_original_start = true;

  for (auto& piece : raw) {
    if (piece.id) {
      at_original_start = false;
      emit_added(*piece.id, std::move(piece.text));
      continue;
    }
    // Schritt 2: normalisieren, dann an den normalisierten Token zerlegen.
    std::string norm = normalizer_ ? normalizer_->normalize(std::move(piece.text))
                                   : std::move(piece.text);
    std::vector<AddedVocabulary::Piece> parts;
    if (added_) added_->split(norm, /*want_normalized=*/true, parts);
    else if (!norm.empty()) parts.push_back({std::move(norm), std::nullopt});

    for (auto& p : parts) {
      const bool at_start = at_original_start;
      at_original_start = false;  // ab hier ist nichts mehr am Textanfang
      if (p.id) {
        emit_added(*p.id, std::move(p.text));
        continue;
      }
      // Schritt 3: vorzerlegen, dann das Modell.
      std::vector<std::string> pieces;
      if (pre_tokenizer_) {
        // Befund S14: Metaspace(prepend_scheme="first") haengt daran, ob das
        // Stueck am Anfang des urspruenglichen Textes beginnt -- nicht daran,
        // ob es das erste einer beliebigen Zerlegung ist.
        pre_tokenizer_->pre_tokenize(std::move(p.text), pieces, at_start);
      } else if (!p.text.empty()) {
        pieces.push_back(std::move(p.text));
      }
      for (const auto& s : pieces) model_->tokenize(s, ids, &tokens);
    }
  }

  // Schritt 4.
  if (add_special && post_processor_) post_processor_->process(ids, &tokens);
  return ids;
}

std::string Tokenizer::decode(std::span<const TokenId> ids, bool skip_special) const {
  std::vector<std::string> tokens;
  tokens.reserve(ids.size());
  for (TokenId id : ids) {
    if (skip_special && added_ && added_->is_special(id)) continue;
    if (auto t = id_to_token(id)) tokens.push_back(std::move(*t));
    // Unbekannte IDs faellt die Referenzbibliothek stillschweigend weg
    // (filter_map ueber id_to_token) -- hier ebenso.
  }
  if (!decoder_) {
    std::string out;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      if (i) out.push_back(' ');
      out += tokens[i];
    }
    return out;
  }
  const auto chain = decoder_->decode_chain(std::move(tokens));
  std::string out;
  for (const auto& t : chain) out += t;
  return out;
}

std::string Tokenizer::describe() const { return description_; }

// =========================================================================
// Aufbau aus tokenizer.json
// =========================================================================

Tokenizer TokenizerBuilder::assemble(Parts p, std::string_view origin) {
  (void)origin;
  Tokenizer t;
  t.normalizer_ = std::move(p.normalizer);
  t.pre_tokenizer_ = std::move(p.pre_tokenizer);
  t.model_ = std::move(p.model);
  t.decoder_ = std::move(p.decoder);
  t.post_processor_ = std::move(p.post_processor);
  t.added_ = std::move(p.added);
  t.special_ = p.special;
  if (!t.model_) throw_error(origin, ": kein Modell -- der Tokenizer waere unbrauchbar");
  if (t.added_) t.added_->finish(t.normalizer_.get());
  t.description_ = std::string("model=") + t.model_->name() +
                   " normalizer=" + (t.normalizer_ ? t.normalizer_->name() : "-") +
                   " pre=" + (t.pre_tokenizer_ ? t.pre_tokenizer_->name() : "-") +
                   " decoder=" + (t.decoder_ ? t.decoder_->name() : "-") +
                   " post=" + (t.post_processor_ ? t.post_processor_->name() : "-") +
                   " vocab=" + std::to_string(t.vocab_size());
  return t;
}

Tokenizer TokenizerBuilder::from_json(const json::Value& root, std::string_view origin) {
  static const json::Value null_value;
  const auto deref = [&](const char* key) -> const json::Value& {
    const json::Value* v = root.get(key);
    return v ? *v : null_value;
  };

  Parts p;
  p.normalizer = make_normalizer(deref("normalizer"), std::string(origin) + ".normalizer");
  p.pre_tokenizer =
      make_pre_tokenizer(deref("pre_tokenizer"), std::string(origin) + ".pre_tokenizer");
  p.model = make_model(deref("model"), std::string(origin) + ".model");
  p.decoder = make_decoder(deref("decoder"), std::string(origin) + ".decoder");
  p.post_processor =
      make_post_processor(deref("post_processor"), std::string(origin) + ".post_processor");

  p.added = std::make_unique<AddedVocabulary>();
  if (const json::Value* a = root.get("added_tokens"); a && a->is_array()) {
    for (const json::Value& v : a->as_array("added_tokens")) {
      AddedToken t2;
      t2.content = v.require("content", "added_tokens").as_string("content");
      t2.id = static_cast<TokenId>(v.require("id", "added_tokens").as_int("id"));
      const auto flag = [&](const char* k, bool def) {
        const json::Value* q = v.get(k);
        return (q && !q->is_null()) ? q->as_bool(k) : def;
      };
      t2.special = flag("special", false);
      t2.single_word = flag("single_word", false);
      t2.lstrip = flag("lstrip", false);
      t2.rstrip = flag("rstrip", false);
      t2.normalized = flag("normalized", !t2.special);
      p.added->add(std::move(t2));
    }
  }
  return assemble(std::move(p), origin);
}

Tokenizer Tokenizer::from_json_text(std::string_view text, std::string_view origin) {
  const json::Value root = json::parse(text, origin);
  return TokenizerBuilder::from_json(root, origin);
}

Tokenizer Tokenizer::from_json_file(const std::string& path) {
  MappedFile f(path);
  return from_json_text(std::string_view(reinterpret_cast<const char*>(f.data()), f.size()),
                        path);
}

}  // namespace quasar::tok

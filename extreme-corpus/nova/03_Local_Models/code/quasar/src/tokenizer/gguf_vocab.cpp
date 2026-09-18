// quasar — src/tokenizer/gguf_vocab.cpp
//
// "Quellen: in GGUF eingebettetes Vokabular; HuggingFace tokenizer.json").
//
// GGUF beschreibt den Tokenizer nicht als Kette, sondern ueber zwei Felder:
//   tokenizer.ggml.model  Familie: "gpt2" (byte-level BPE), "llama" (SPM), ...
//   tokenizer.ggml.pre    welche Zerlegungsregel: "default", "llama-bpe", ...
// Daraus wird hier die Kette gebaut.
//
// Zweistufig: `extract_vocab` holt die Rohdaten aus dem GGUF,
// `from_vocab_data` baut daraus die Kette. Der Zwischenschritt ist genau das,
// was `format/` in den Tokenizer-Abschnitt des Quasar Blocks schreibt --
//
// Unterstuetzt ist derzeit **nur** family == "gpt2". Das ist die Familie des
// model="gpt2", pre="llama-bpe"). Fuer die anderen Familien bricht quasar mit
// klarer Meldung ab, statt eine Zerlegung zu raten -- ein falsch geratener
// Tokenizer erzeugt Ausgaben, die wie Modellfehler aussehen.

#include <algorithm>

#include "quasar/model/gguf.hpp"
#include "tokenizer/components.hpp"

namespace quasar::tok {

namespace {

// GGUF-Tokentypen (llama.cpp: llama_token_type)
constexpr std::int32_t kTypeUnknown = 2;
constexpr std::int32_t kTypeControl = 3;
constexpr std::int32_t kTypeUserDefined = 4;

std::string key_for(std::string_view suffix) {
  return std::string("tokenizer.ggml.") + std::string(suffix);
}

}  // namespace

Tokenizer::VocabData Tokenizer::extract_vocab(const gguf::GgufFile& g) {
  const std::string where = g.path() + " (tokenizer.ggml.*)";
  const gguf::Metadata& md = g.metadata();

  VocabData d;
  d.family = std::string(md.require(key_for("model"), where).as_string("tokenizer.ggml.model"));
  d.pre = "default";
  if (const gguf::MetaValue* p = md.find(key_for("pre"))) {
    d.pre = std::string(p->as_string("tokenizer.ggml.pre"));
  }

  std::vector<std::string_view> tokens;
  md.require(key_for("tokens"), where).read_strings(tokens, "tokenizer.ggml.tokens");
  d.tokens.reserve(tokens.size());
  for (std::string_view t : tokens) d.tokens.emplace_back(t);

  if (const gguf::MetaValue* t = md.find(key_for("token_type"))) {
    t->read_numbers(d.token_types, "tokenizer.ggml.token_type");
    if (d.token_types.size() != d.tokens.size()) {
      throw_error(where, ": token_type hat ", d.token_types.size(),
                  " Eintraege, tokens aber ", d.tokens.size());
    }
  }

  if (const gguf::MetaValue* m = md.find(key_for("merges"))) {
    std::vector<std::string_view> lines;
    m->read_strings(lines, "tokenizer.ggml.merges");
    d.merges.reserve(lines.size());
    for (std::string_view s : lines) d.merges.emplace_back(s);
  }

  const auto id_of = [&](const char* suffix) -> std::optional<TokenId> {
    if (const gguf::MetaValue* v = md.find(key_for(suffix))) {
      return static_cast<TokenId>(v->as_u64(suffix));
    }
    return std::nullopt;
  };
  const auto flag_of = [&](const char* suffix, bool def) {
    if (const gguf::MetaValue* v = md.find(key_for(suffix))) return v->as_bool(suffix);
    return def;
  };
  d.special.bos = id_of("bos_token_id");
  d.special.eos = id_of("eos_token_id");
  d.special.unk = id_of("unknown_token_id");
  d.special.pad = id_of("padding_token_id");
  d.special.add_bos = flag_of("add_bos_token", false);
  d.special.add_eos = flag_of("add_eos_token", false);
  return d;
}

Tokenizer Tokenizer::from_vocab_data(const VocabData& d, std::string_view origin) {
  if (d.family != "gpt2") {
    throw_error(origin, ": Tokenizer-Familie \"", d.family,
                "\" wird noch nicht unterstuetzt. quasar liest derzeit nur die "
                "byte-level-BPE-Familie (\"gpt2\"). Fuer die anderen Familien bitte "
                "tokenizer.json benutzen -- geraten wird hier nicht.");
  }
  if (d.tokens.empty()) throw_error(origin, ": das Vokabular ist leer");

  Vocab vocab;
  for (std::size_t i = 0; i < d.tokens.size(); ++i) {
    vocab.add(d.tokens[i], static_cast<TokenId>(i));
  }

  BpeSpec spec;
  spec.ignore_merges = d.pre == "tekken";
  spec.merges.reserve(d.merges.size());
  for (std::size_t i = 0; i < d.merges.size(); ++i) {
    const std::string& s = d.merges[i];
    const std::size_t sp = s.find(' ');
    if (sp == std::string::npos) {
      throw_error(origin, ": Merge Nr. ", i, " (\"", s, "\") hat kein Trennzeichen");
    }
    spec.merges.emplace_back(s.substr(0, sp), s.substr(sp + 1));
  }

  TokenizerBuilder::Parts parts;
  parts.model = make_bpe_model(std::move(vocab), std::move(spec), origin);

  if (d.pre == "llama3" || d.pre == "llama-bpe" || d.pre == "llama-v3") {
    parts.pre_tokenizer = make_llama3_pre();
  } else if (d.pre == "tekken") {
    parts.pre_tokenizer = make_tekken_pre();
  } else if (d.pre == "default" || d.pre == "gpt-2" || d.pre == "gpt2") {
    parts.pre_tokenizer = make_byte_level_pre(/*add_prefix_space=*/false, /*use_regex=*/true);
  } else {
    throw_error(origin, ": Zerlegungsregel \"", d.pre,
                "\" ist quasar nicht bekannt. Sie entscheidet ueber die Token-IDs und "
                "wird nicht geraten. Bekannt: default, gpt-2, llama-bpe/llama3, tekken.");
  }
  parts.decoder = make_byte_level_decoder();

  auto added = std::make_unique<AddedVocabulary>();
  for (std::size_t i = 0; i < d.token_types.size(); ++i) {
    const std::int32_t t = d.token_types[i];
    if (t != kTypeControl && t != kTypeUserDefined && t != kTypeUnknown) continue;
    if (d.tokens[i].empty()) continue;
    AddedToken a;
    a.content = d.tokens[i];
    a.id = static_cast<TokenId>(i);
    a.special = (t == kTypeControl || t == kTypeUnknown);
    a.normalized = false;  // Rohtext-Vergleich, wie llama.cpp es macht
    added->add(std::move(a));
  }
  parts.added = std::move(added);
  parts.special = d.special;

  // GGUF beschreibt keine Vorlage, sondern nur die beiden Schalter.
  if (d.special.add_bos || d.special.add_eos) {
    parts.post_processor = make_bos_eos_processor(
        d.special.add_bos ? d.special.bos : std::nullopt,
        d.special.add_eos ? d.special.eos : std::nullopt, *parts.model, *parts.added);
  }
  return TokenizerBuilder::assemble(std::move(parts), origin);
}

Tokenizer Tokenizer::from_gguf(const gguf::GgufFile& g) {
  return from_vocab_data(extract_vocab(g), g.path() + " (tokenizer.ggml.*)");
}

}  // namespace quasar::tok

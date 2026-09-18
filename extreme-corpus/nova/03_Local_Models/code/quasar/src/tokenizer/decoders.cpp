// quasar — src/tokenizer/decoders.cpp
//
// Rueckweg: Token-Strings -> Text. Und die Nachbearbeitung, die beim
// Kodieren Sondertoken anfuegt.
//
// Dekodieren ist bei den meisten Tokenizern nicht verlustfrei (Lowercase,
// StripAccents, Metaspace). Der Test prueft deshalb nicht gegen den
// Eingabetext, sondern gegen das, was die Referenzbibliothek dekodiert.

#include <algorithm>

#include "quasar/core/json.hpp"
#include "quasar/core/unicode.hpp"
#include "tokenizer/components.hpp"

namespace quasar::tok {
namespace {

namespace uc = quasar::unicode;

// --- ByteLevel ------------------------------------------------------------

class ByteLevelDec final : public Decoder {
 public:
  std::vector<std::string> decode_chain(std::vector<std::string> tokens) const override {
    std::string bytes;
    for (const auto& t : tokens) {
      std::string mapped;
      bool ok = true;
      for (std::size_t i = 0; i < t.size();) {
        const std::size_t n = utf8_len(t, i);
        const auto cps = uc::to_codepoints(t.substr(i, n));
        if (cps.size() != 1) { ok = false; break; }
        const int b = char_to_byte(cps[0]);
        if (b < 0) { ok = false; break; }
        mapped.push_back(static_cast<char>(static_cast<unsigned char>(b)));
        i += n;
      }
      // Wie in der Referenzbibliothek: laesst sich ein Token nicht vollstaendig
      // zurueckbilden, gehen seine Rohbytes durch.
      bytes += ok ? mapped : t;
    }
    return {replace_invalid_utf8(bytes)};
  }
  std::string name() const override { return "ByteLevel"; }

 private:
  // Entspricht `String::from_utf8_lossy`.
  //
  // Befund M20: die frueher hier stehende Fassung erzeugte **pro Byte** ein
  // U+FFFD. Rust ersetzt dagegen jede *maximale ungueltige Teilfolge* durch
  // **ein** U+FFFD (Unicode "maximal subpart"). Gemessen am echten
  // gpt2-Vokabular: das Emoji U+1F600 hat die ids [47249, 222];
  // `decode([47249])` ergibt in der Referenz `efbfbd`, bei uns
  // Abschnitt 7 (`session.detokenize(token)` je Token).
  //
  // Umgesetzt ist die Regel von `std::str::from_utf8`: an einer Fehlerstelle
  // ist `error_len` die Zahl der Bytes der maximalen Teilfolge (1, 2 oder 3);
  // endet die Eingabe mitten in einer Folge, gilt der ganze Rest als **eine**
  // Teilfolge.
  static std::string replace_invalid_utf8(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    std::size_t i = 0;
    while (i < in.size()) {
      const std::size_t n = utf8_len(in, i);
      if (n > 0 && i + n <= in.size() &&
          uc::is_valid_utf8(std::string_view(in.data() + i, n))) {
        out.append(in, i, n);
        i += n;
        continue;
      }
      out += "\xEF\xBF\xBD";  // U+FFFD -- genau eines je Teilfolge
      i += maximal_subpart_len(in, i);
    }
    return out;
  }

  // Wie viele Bytes gehoeren zur maximalen Teilfolge, die bei `i` beginnt?
  // Immer mindestens 1, damit der Lauf endet.
  static std::size_t maximal_subpart_len(const std::string& in, std::size_t i) {
    const auto byte = [&](std::size_t k) { return static_cast<unsigned char>(in[k]); };
    const unsigned char lead = byte(i);
    // Freistehendes Fortsetzungsbyte oder unmoegliches Startbyte: ein Byte.
    if (lead < 0xC2 || lead > 0xF4) return 1;

    // Zweites Byte: der erlaubte Bereich haengt vom Startbyte ab
    // (Unicode-Standard, Tabelle 3-7).
    unsigned char lo = 0x80, hi = 0xBF;
    if (lead == 0xE0) lo = 0xA0;
    else if (lead == 0xED) hi = 0x9F;
    else if (lead == 0xF0) lo = 0x90;
    else if (lead == 0xF4) hi = 0x8F;
    if (i + 1 >= in.size()) return 1;
    if (byte(i + 1) < lo || byte(i + 1) > hi) return 1;
    if (lead < 0xE0) return 2;  // 2-Byte-Folge, hier also schon vollstaendig

    if (i + 2 >= in.size()) return 2;
    if (byte(i + 2) < 0x80 || byte(i + 2) > 0xBF) return 2;
    if (lead < 0xF0) return 3;

    return 3;  // viertes Byte fehlt oder ist ungueltig
  }
};

// --- Metaspace ------------------------------------------------------------

class MetaspaceDec final : public Decoder {
 public:
  MetaspaceDec(std::string rep, bool prepend) : rep_(std::move(rep)), prepend_(prepend) {}
  std::vector<std::string> decode_chain(std::vector<std::string> tokens) const override {
    std::vector<std::string> out;
    out.reserve(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      const std::string& t = tokens[i];
      std::string s;
      std::size_t j = 0;
      while (j < t.size()) {
        if (t.compare(j, rep_.size(), rep_) == 0) {
          // Im ERSTEN Token faellt das Ersatzzeichen ganz weg, sonst wird es
          // zum Leerzeichen. (Genau so in der Referenzbibliothek.)
          if (!(i == 0 && prepend_)) s.push_back(' ');
          j += rep_.size();
        } else {
          s.push_back(t[j]);
          ++j;
        }
      }
      out.push_back(std::move(s));
    }
    return out;
  }
  std::string name() const override { return "Metaspace"; }

 private:
  std::string rep_;
  bool prepend_;
};

// --- WordPiece ------------------------------------------------------------

std::string wordpiece_cleanup(std::string s) {
  static const std::pair<const char*, const char*> rules[] = {
      {" .", "."},   {" ?", "?"},    {" !", "!"},   {" ,", ","},
      {" ' ", "'"},  {" n't", "n't"},{" 'm", "'m"}, {" do not", " don't"},
      {" 's", "'s"}, {" 've", "'ve"},{" 're", "'re"}};
  for (const auto& [from, to] : rules) {
    const std::string f = from, t = to;
    std::string out;
    out.reserve(s.size());
    std::size_t i = 0;
    while (i < s.size()) {
      if (s.compare(i, f.size(), f) == 0) {
        out += t;
        i += f.size();
      } else {
        out.push_back(s[i]);
        ++i;
      }
    }
    s.swap(out);
  }
  return s;
}

class WordPieceDec final : public Decoder {
 public:
  WordPieceDec(std::string prefix, bool cleanup)
      : prefix_(std::move(prefix)), cleanup_(cleanup) {}
  std::vector<std::string> decode_chain(std::vector<std::string> tokens) const override {
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      if (i != 0) {
        if (tokens[i].starts_with(prefix_)) tokens[i].erase(0, prefix_.size());
        else tokens[i] = " " + tokens[i];
      }
      if (cleanup_) tokens[i] = wordpiece_cleanup(std::move(tokens[i]));
    }
    return tokens;
  }
  std::string name() const override { return "WordPiece"; }

 private:
  std::string prefix_;
  bool cleanup_;
};

// --- Replace / ByteFallback / Fuse / Strip -------------------------------

class ReplaceDec final : public Decoder {
 public:
  ReplaceDec(std::string pat, std::string content)
      : pat_(std::move(pat)), content_(std::move(content)) {}
  std::vector<std::string> decode_chain(std::vector<std::string> tokens) const override {
    if (pat_.empty()) return tokens;
    for (auto& t : tokens) {
      std::string out;
      out.reserve(t.size());
      std::size_t i = 0;
      while (i < t.size()) {
        if (t.compare(i, pat_.size(), pat_) == 0) {
          out += content_;
          i += pat_.size();
        } else {
          out.push_back(t[i]);
          ++i;
        }
      }
      t.swap(out);
    }
    return tokens;
  }
  std::string name() const override { return "Replace"; }

 private:
  std::string pat_, content_;
};

class ByteFallbackDec final : public Decoder {
 public:
  std::vector<std::string> decode_chain(std::vector<std::string> tokens) const override {
    std::vector<std::string> out;
    std::string pending;
    const auto flush = [&] {
      if (pending.empty()) return;
      if (uc::is_valid_utf8(pending)) {
        out.push_back(pending);
      } else {
        for (std::size_t k = 0; k < pending.size(); ++k) out.push_back("\xEF\xBF\xBD");
      }
      pending.clear();
    };
    for (auto& t : tokens) {
      std::optional<unsigned char> b;
      if (t.size() == 6 && t.starts_with("<0x") && t.back() == '>') {
        const auto hexval = [](char c) -> int {
          if (c >= '0' && c <= '9') return c - '0';
          if (c >= 'A' && c <= 'F') return c - 'A' + 10;
          if (c >= 'a' && c <= 'f') return c - 'a' + 10;
          return -1;
        };
        const int hi = hexval(t[3]), lo = hexval(t[4]);
        if (hi >= 0 && lo >= 0) b = static_cast<unsigned char>(hi * 16 + lo);
      }
      if (b) {
        pending.push_back(static_cast<char>(*b));
      } else {
        flush();
        out.push_back(std::move(t));
      }
    }
    flush();
    return out;
  }
  std::string name() const override { return "ByteFallback"; }
};

class FuseDec final : public Decoder {
 public:
  std::vector<std::string> decode_chain(std::vector<std::string> tokens) const override {
    std::string all;
    for (const auto& t : tokens) all += t;
    return {std::move(all)};
  }
  std::string name() const override { return "Fuse"; }
};

class StripDec final : public Decoder {
 public:
  StripDec(std::string content, std::size_t start, std::size_t stop)
      : c_(std::move(content)), start_(start), stop_(stop) {}
  std::vector<std::string> decode_chain(std::vector<std::string> tokens) const override {
    if (c_.empty()) return tokens;
    for (auto& t : tokens) {
      std::size_t a = 0;
      while (a < start_ && t.size() - a >= c_.size() && t.compare(a, c_.size(), c_) == 0) {
        a += c_.size();
        ++a;
        --a;  // c_ ist hier immer ein einzelnes Zeichen; a wurde schon erhoeht
        break;
      }
      // Einfache, wortgetreue Umsetzung: bis zu `start_` Vorkommen vorne,
      // bis zu `stop_` Vorkommen hinten.
      std::size_t front = 0, n_front = 0;
      while (n_front < start_ && t.size() - front >= c_.size() &&
             t.compare(front, c_.size(), c_) == 0) {
        front += c_.size();
        ++n_front;
      }
      std::size_t back = t.size(), n_back = 0;
      while (n_back < stop_ && back >= front + c_.size() &&
             t.compare(back - c_.size(), c_.size(), c_) == 0) {
        back -= c_.size();
        ++n_back;
      }
      t = t.substr(front, back - front);
    }
    return tokens;
  }
  std::string name() const override { return "Strip"; }

 private:
  std::string c_;
  std::size_t start_, stop_;
};

class BpeDec final : public Decoder {
 public:
  explicit BpeDec(std::string suffix) : suffix_(std::move(suffix)) {}
  std::vector<std::string> decode_chain(std::vector<std::string> tokens) const override {
    // Befund G15: das Suffix wurde in **jedem** Token durch ein Leerzeichen
    // ersetzt -- auch im letzten. `["ab</w>", "cd</w>"]` ergab dadurch "ab cd "
    // statt "ab cd". Die Referenz ersetzt das Suffix im letzten Token durch
    // nichts und in allen anderen durch ein Leerzeichen; das Leerzeichen
    // gehoert an die Fugen, nicht ans Ende.
    if (suffix_.empty()) return tokens;
    for (std::size_t k = 0; k < tokens.size(); ++k) {
      const bool last = k + 1 == tokens.size();
      std::string& t = tokens[k];
      std::string out;
      out.reserve(t.size());
      std::size_t i = 0;
      while (i < t.size()) {
        if (t.compare(i, suffix_.size(), suffix_) == 0) {
          if (!last) out.push_back(' ');
          i += suffix_.size();
        } else {
          out.push_back(t[i]);
          ++i;
        }
      }
      t.swap(out);
    }
    return tokens;
  }
  std::string name() const override { return "BPEDecoder"; }

 private:
  std::string suffix_;
};

class SequenceDec final : public Decoder {
 public:
  explicit SequenceDec(std::vector<std::unique_ptr<Decoder>> v) : v_(std::move(v)) {}
  std::vector<std::string> decode_chain(std::vector<std::string> tokens) const override {
    for (const auto& d : v_) tokens = d->decode_chain(std::move(tokens));
    return tokens;
  }
  std::string name() const override {
    std::string s = "Sequence[";
    for (std::size_t i = 0; i < v_.size(); ++i) {
      if (i) s += ", ";
      s += v_[i]->name();
    }
    return s + "]";
  }

 private:
  std::vector<std::unique_ptr<Decoder>> v_;
};

}  // namespace

std::unique_ptr<Decoder> make_byte_level_decoder() {
  return std::make_unique<ByteLevelDec>();
}

std::unique_ptr<Decoder> make_sentencepiece_decoder() {
  std::vector<std::unique_ptr<Decoder>> v;
  v.push_back(std::make_unique<ReplaceDec>("\xE2\x96\x81", " "));
  v.push_back(std::make_unique<ByteFallbackDec>());
  v.push_back(std::make_unique<FuseDec>());
  v.push_back(std::make_unique<StripDec>(" ", 1, 0));
  return std::make_unique<SequenceDec>(std::move(v));
}

std::unique_ptr<Decoder> make_decoder(const json::Value& v, std::string_view where) {
  if (v.is_null()) return nullptr;
  const std::string& type = v.require("type", where).as_string("decoder.type");

  auto opt_bool = [&](const char* key, bool def) {
    const json::Value* p = v.get(key);
    return (p && !p->is_null()) ? p->as_bool(key) : def;
  };

  if (type == "ByteLevel") return std::make_unique<ByteLevelDec>();
  if (type == "ByteFallback") return std::make_unique<ByteFallbackDec>();
  if (type == "Fuse") return std::make_unique<FuseDec>();
  if (type == "WordPiece") {
    std::string prefix = "##";
    if (const json::Value* p = v.get("prefix"); p && !p->is_null()) {
      prefix = p->as_string("prefix");
    }
    return std::make_unique<WordPieceDec>(prefix, opt_bool("cleanup", true));
  }
  if (type == "Metaspace") {
    std::string rep = "\xE2\x96\x81";
    if (const json::Value* p = v.get("replacement"); p && !p->is_null()) {
      rep = p->as_string("replacement");
    }
    bool prepend = true;
    if (const json::Value* p = v.get("prepend_scheme"); p && !p->is_null()) {
      prepend = p->as_string("prepend_scheme") != "never";
    } else if (const json::Value* q = v.get("add_prefix_space"); q && !q->is_null()) {
      prepend = q->as_bool("add_prefix_space");
    }
    return std::make_unique<MetaspaceDec>(std::move(rep), prepend);
  }
  if (type == "Replace") {
    const json::Value& pat = v.require("pattern", where);
    const json::Value* str = pat.get("String");
    if (!str) throw_error(where, ": Replace-Dekodierer mit Regex-Muster wird nicht unterstuetzt");
    return std::make_unique<ReplaceDec>(str->as_string("pattern.String"),
                                        v.require("content", where).as_string("content"));
  }
  if (type == "Strip") {
    std::string content = " ";
    if (const json::Value* p = v.get("content"); p && !p->is_null()) {
      content = p->as_string("content");
    }
    const auto num = [&](const char* key) -> std::size_t {
      const json::Value* p = v.get(key);
      return (p && !p->is_null()) ? static_cast<std::size_t>(p->as_int(key)) : 0;
    };
    return std::make_unique<StripDec>(content, num("start"), num("stop"));
  }
  if (type == "BPEDecoder") {
    std::string suffix = "</w>";
    if (const json::Value* p = v.get("suffix"); p && !p->is_null()) {
      suffix = p->as_string("suffix");
    }
    return std::make_unique<BpeDec>(suffix);
  }
  if (type == "Sequence") {
    const json::Array& a = v.require("decoders", where).as_array("decoders");
    std::vector<std::unique_ptr<Decoder>> parts;
    for (std::size_t i = 0; i < a.size(); ++i) {
      std::string sub = std::string(where) + ".decoders[" + std::to_string(i) + "]";
      if (auto d = make_decoder(a[i], sub)) parts.push_back(std::move(d));
    }
    return std::make_unique<SequenceDec>(std::move(parts));
  }

  throw_error(where, ": unbekannter Dekodierer-Typ \"", type,
              "\". quasar kennt: ByteLevel, ByteFallback, Fuse, WordPiece, Metaspace, "
              "Replace, Strip, BPEDecoder, Sequence.");
}

// =========================================================================
// Nachbearbeitung
// =========================================================================

namespace {

// TemplateProcessing, aber nur fuer eine einzelne Folge (`single`). quasar
// Text. Ein `pair`-Template wird deshalb gelesen und ignoriert.
class TemplateProcessing final : public PostProcessor {
 public:
  struct Item {
    bool is_sequence;
    std::vector<TokenId> ids;
    std::vector<std::string> tokens;
  };
  explicit TemplateProcessing(std::vector<Item> single) : single_(std::move(single)) {}

  void process(std::vector<TokenId>& ids, std::vector<std::string>* tokens) const override {
    std::vector<TokenId> out;
    std::vector<std::string> out_tok;
    for (const auto& it : single_) {
      if (it.is_sequence) {
        out.insert(out.end(), ids.begin(), ids.end());
        if (tokens) out_tok.insert(out_tok.end(), tokens->begin(), tokens->end());
      } else {
        out.insert(out.end(), it.ids.begin(), it.ids.end());
        if (tokens) out_tok.insert(out_tok.end(), it.tokens.begin(), it.tokens.end());
      }
    }
    ids.swap(out);
    if (tokens) tokens->swap(out_tok);
  }
  std::string name() const override { return "TemplateProcessing"; }

 private:
  std::vector<Item> single_;
};

class NoopProcessor final : public PostProcessor {
 public:
  explicit NoopProcessor(std::string n) : n_(std::move(n)) {}
  void process(std::vector<TokenId>&, std::vector<std::string>*) const override {}
  std::string name() const override { return n_; }

 private:
  std::string n_;
};

class FixedPairProcessor final : public PostProcessor {
 public:
  FixedPairProcessor(std::vector<TokenId> pre, std::vector<std::string> pre_t,
                     std::vector<TokenId> post, std::vector<std::string> post_t,
                     std::string n)
      : pre_(std::move(pre)),
        post_(std::move(post)),
        pre_t_(std::move(pre_t)),
        post_t_(std::move(post_t)),
        n_(std::move(n)) {}
  void process(std::vector<TokenId>& ids, std::vector<std::string>* tokens) const override {
    std::vector<TokenId> out = pre_;
    out.insert(out.end(), ids.begin(), ids.end());
    out.insert(out.end(), post_.begin(), post_.end());
    ids.swap(out);
    if (tokens) {
      std::vector<std::string> t = pre_t_;
      t.insert(t.end(), tokens->begin(), tokens->end());
      t.insert(t.end(), post_t_.begin(), post_t_.end());
      tokens->swap(t);
    }
  }
  std::string name() const override { return n_; }

 private:
  std::vector<TokenId> pre_, post_;
  std::vector<std::string> pre_t_, post_t_;
  std::string n_;
};

class SequenceProcessor final : public PostProcessor {
 public:
  explicit SequenceProcessor(std::vector<std::unique_ptr<PostProcessor>> v)
      : v_(std::move(v)) {}
  void process(std::vector<TokenId>& ids, std::vector<std::string>* tokens) const override {
    for (const auto& p : v_) p->process(ids, tokens);
  }
  std::string name() const override { return "Sequence"; }

 private:
  std::vector<std::unique_ptr<PostProcessor>> v_;
};

}  // namespace

std::unique_ptr<PostProcessor> make_post_processor(const json::Value& v,
                                                   std::string_view where) {
  if (v.is_null()) return nullptr;
  const std::string& type = v.require("type", where).as_string("post_processor.type");

  if (type == "ByteLevel") {
    // Aendert nur Offsets, keine IDs. quasar fuehrt keine Offsets mit.
    return std::make_unique<NoopProcessor>("ByteLevel");
  }

  if (type == "TemplateProcessing") {
    const json::Object& specials =
        v.require("special_tokens", where).as_object("special_tokens");
    const json::Array& single = v.require("single", where).as_array("single");

    std::vector<TemplateProcessing::Item> items;
    for (std::size_t i = 0; i < single.size(); ++i) {
      if (const json::Value* seq = single[i].get("Sequence")) {
        (void)seq;
        items.push_back({true, {}, {}});
        continue;
      }
      const json::Value* st = single[i].get("SpecialToken");
      if (!st) throw_error(where, ": Eintrag Nr. ", i, " in \"single\" ist weder "
                                  "\"Sequence\" noch \"SpecialToken\"");
      const std::string& id = st->require("id", "SpecialToken").as_string("id");
      const json::Value* entry = specials.find(id);
      if (!entry) {
        throw_error(where, ": Sondertoken \"", id,
                    "\" aus der Vorlage steht nicht in \"special_tokens\"");
      }
      TemplateProcessing::Item item{false, {}, {}};
      for (const json::Value& x : entry->require("ids", "special_tokens").as_array("ids")) {
        item.ids.push_back(static_cast<TokenId>(x.as_int("id")));
      }
      for (const json::Value& x :
           entry->require("tokens", "special_tokens").as_array("tokens")) {
        item.tokens.push_back(x.as_string("token"));
      }
      items.push_back(std::move(item));
    }
    return std::make_unique<TemplateProcessing>(std::move(items));
  }

  if (type == "BertProcessing" || type == "RobertaProcessing") {
    const json::Array& sep = v.require("sep", where).as_array("sep");
    const json::Array& cls = v.require("cls", where).as_array("cls");
    if (sep.size() != 2 || cls.size() != 2) {
      throw_error(where, ": sep/cls muessen [Token, ID] sein");
    }
    return std::make_unique<FixedPairProcessor>(
        std::vector<TokenId>{static_cast<TokenId>(cls[1].as_int("cls-ID"))},
        std::vector<std::string>{cls[0].as_string("cls-Token")},
        std::vector<TokenId>{static_cast<TokenId>(sep[1].as_int("sep-ID"))},
        std::vector<std::string>{sep[0].as_string("sep-Token")}, type);
  }

  if (type == "Sequence") {
    const json::Array& a = v.require("processors", where).as_array("processors");
    std::vector<std::unique_ptr<PostProcessor>> parts;
    for (std::size_t i = 0; i < a.size(); ++i) {
      std::string sub = std::string(where) + ".processors[" + std::to_string(i) + "]";
      if (auto p = make_post_processor(a[i], sub)) parts.push_back(std::move(p));
    }
    return std::make_unique<SequenceProcessor>(std::move(parts));
  }

  throw_error(where, ": unbekannter Nachbearbeiter-Typ \"", type,
              "\". quasar kennt: TemplateProcessing, ByteLevel, BertProcessing, "
              "RobertaProcessing, Sequence.");
}

std::unique_ptr<PostProcessor> make_bos_eos_processor(std::optional<TokenId> bos,
                                                      std::optional<TokenId> eos,
                                                      const Model& model,
                                                      const AddedVocabulary& added) {
  const auto text_of = [&](TokenId id) -> std::string {
    if (const std::string* c = added.content_of(id)) return *c;
    if (const std::string* t = model.vocab().token_of(id)) return *t;
    return {};
  };
  std::vector<TokenId> pre, post;
  std::vector<std::string> pre_t, post_t;
  if (bos) {
    pre.push_back(*bos);
    pre_t.push_back(text_of(*bos));
  }
  if (eos) {
    post.push_back(*eos);
    post_t.push_back(text_of(*eos));
  }
  return std::make_unique<FixedPairProcessor>(std::move(pre), std::move(pre_t),
                                              std::move(post), std::move(post_t),
                                              "BOS/EOS (aus GGUF)");
}

}  // namespace quasar::tok

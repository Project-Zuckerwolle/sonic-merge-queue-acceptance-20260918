// quasar — src/tokenizer/normalizers.cpp
//
// Referenzbibliothek; jede Abweichung hier ist eine Abweichung in den IDs.

#include <algorithm>

#include "quasar/core/json.hpp"
#include "quasar/core/unicode.hpp"
#include "tokenizer/components.hpp"

namespace quasar::tok {
namespace {

namespace uc = quasar::unicode;

// --- NFC / NFD / NFKC / NFKD ---------------------------------------------

class NormalizeForm final : public Normalizer {
 public:
  enum class Form { NFC, NFD, NFKC, NFKD };
  explicit NormalizeForm(Form f) : f_(f) {}
  std::string normalize(std::string s) const override {
    switch (f_) {
      case Form::NFC: return uc::nfc(s);
      case Form::NFD: return uc::nfd(s);
      case Form::NFKC: return uc::nfkc(s);
      case Form::NFKD: return uc::nfkd(s);
    }
    return s;
  }
  std::string name() const override {
    switch (f_) {
      case Form::NFC: return "NFC";
      case Form::NFD: return "NFD";
      case Form::NFKC: return "NFKC";
      case Form::NFKD: return "NFKD";
    }
    return "?";
  }

 private:
  Form f_;
};

// --- Lowercase ------------------------------------------------------------

class Lowercase final : public Normalizer {
 public:
  std::string normalize(std::string s) const override { return uc::lowercase(s); }
  std::string name() const override { return "Lowercase"; }
};

// --- StripAccents ---------------------------------------------------------
//
// Achtung: die Referenzbibliothek normalisiert hier NICHT vorher nach NFD.
// Sie entfernt schlicht alle Zeichen der Kategorie Mn. Deshalb steht in den
// Ketten, die das benutzen, ein NFKD davor.

class StripAccents final : public Normalizer {
 public:
  std::string normalize(std::string s) const override {
    const auto cps = uc::to_codepoints(s);
    std::vector<char32_t> out;
    out.reserve(cps.size());
    for (char32_t c : cps) {
      if (uc::category(c) != uc::Category::Mn) out.push_back(c);
    }
    return uc::to_utf8(out);
  }
  std::string name() const override { return "StripAccents"; }
};

// --- Strip ----------------------------------------------------------------

class Strip final : public Normalizer {
 public:
  Strip(bool left, bool right) : left_(left), right_(right) {}
  std::string normalize(std::string s) const override {
    const auto cps = uc::to_codepoints(s);
    std::size_t a = 0, b = cps.size();
    if (left_) while (a < b && uc::is_whitespace(cps[a])) ++a;
    if (right_) while (b > a && uc::is_whitespace(cps[b - 1])) --b;
    return uc::to_utf8(std::vector<char32_t>(cps.begin() + static_cast<std::ptrdiff_t>(a),
                                             cps.begin() + static_cast<std::ptrdiff_t>(b)));
  }
  std::string name() const override { return "Strip"; }

 private:
  bool left_, right_;
};

// --- Replace --------------------------------------------------------------

class ReplaceString final : public Normalizer {
 public:
  ReplaceString(std::string pat, std::string content)
      : pat_(std::move(pat)), content_(std::move(content)) {}
  std::string normalize(std::string s) const override {
    if (pat_.empty()) return s;
    std::string out;
    out.reserve(s.size());
    std::size_t i = 0;
    while (i < s.size()) {
      if (s.compare(i, pat_.size(), pat_) == 0) {
        out += content_;
        i += pat_.size();
      } else {
        out.push_back(s[i]);
        ++i;
      }
    }
    return out;
  }
  std::string name() const override { return "Replace"; }

 private:
  std::string pat_, content_;
};

// --- Prepend --------------------------------------------------------------

class Prepend final : public Normalizer {
 public:
  explicit Prepend(std::string p) : p_(std::move(p)) {}
  // Eine leere Kette bleibt leer -- so macht es die Referenzbibliothek. Ohne
  // diese Ausnahme wird aus dem leeren Text ein "_"-Token.
  std::string normalize(std::string s) const override { return s.empty() ? s : p_ + s; }
  std::string name() const override { return "Prepend"; }

 private:
  std::string p_;
};

// --- Nmt ------------------------------------------------------------------

class Nmt final : public Normalizer {
 public:
  std::string normalize(std::string s) const override {
    const auto cps = uc::to_codepoints(s);
    std::vector<char32_t> out;
    out.reserve(cps.size());
    for (char32_t c : cps) {
      // Wie in der Referenzbibliothek: diese Zeichen fallen weg, ...
      if ((c >= 0x0001 && c <= 0x0008) || c == 0x000B ||
          (c >= 0x000E && c <= 0x001F) || c == 0x007F || c == 0x008F || c == 0x009F) {
        continue;
      }
      // ... diese werden zum Leerzeichen.
      //
      // Befund M21: hier stand `c >= 0x2000 && c <= 0x200F` -- zehn Zeichen zu
      // viel. Die Referenz ersetzt nur U+200B..U+200F; U+2000..U+200A
      // (EN QUAD bis HAIR SPACE) bleiben unveraendert. Gemessen an
      // " a b": Referenz e2808061e2808162, quasar 20612062.
      if (c == 0x0009 || c == 0x000A || c == 0x000C || c == 0x000D || c == 0x1680 ||
          (c >= 0x200B && c <= 0x200F) || c == 0x2028 || c == 0x2029 || c == 0x2581 ||
          c == 0xFEFF || c == 0xFFFD) {
        out.push_back(U' ');
        continue;
      }
      out.push_back(c);
    }
    return uc::to_utf8(out);
  }
  std::string name() const override { return "Nmt"; }
};

// --- BertNormalizer -------------------------------------------------------

bool is_bert_control(char32_t c) {
  if (c == U'\t' || c == U'\n' || c == U'\r') return false;
  const auto cat = uc::category(c);
  // Die Referenzbibliothek zaehlt hier Cc, Cf, Cn, Co und Cs.
  return cat == uc::Category::Cc || cat == uc::Category::Cf ||
         cat == uc::Category::Cn || cat == uc::Category::Co ||
         cat == uc::Category::Cs;
}

bool is_bert_whitespace(char32_t c) {
  if (c == U'\t' || c == U'\n' || c == U'\r') return true;
  return uc::is_whitespace(c);
}

bool is_chinese_char(char32_t c) {
  return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) ||
         (c >= 0x20000 && c <= 0x2A6DF) || (c >= 0x2A700 && c <= 0x2B73F) ||
         (c >= 0x2B740 && c <= 0x2B81F) || (c >= 0x2B920 && c <= 0x2CEAF) ||
         (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x2F800 && c <= 0x2FA1F);
}

class BertNormalizer final : public Normalizer {
 public:
  BertNormalizer(bool clean_text, bool handle_chinese, std::optional<bool> strip_accents,
                 bool lowercase)
      : clean_(clean_text),
        chinese_(handle_chinese),
        strip_(strip_accents),
        lower_(lowercase) {}

  std::string normalize(std::string s) const override {
    auto cps = uc::to_codepoints(s);

    if (clean_) {
      std::vector<char32_t> out;
      out.reserve(cps.size());
      for (char32_t c : cps) {
        if (c == 0 || c == 0xFFFD || is_bert_control(c)) continue;
        out.push_back(is_bert_whitespace(c) ? U' ' : c);
      }
      cps.swap(out);
    }

    if (chinese_) {
      std::vector<char32_t> out;
      out.reserve(cps.size() * 2);
      for (char32_t c : cps) {
        if (is_chinese_char(c)) {
          out.push_back(U' ');
          out.push_back(c);
          out.push_back(U' ');
        } else {
          out.push_back(c);
        }
      }
      cps.swap(out);
    }

    // Vorgabe: strip_accents folgt lowercase, wenn nicht ausdruecklich gesetzt.
    if (strip_.value_or(lower_)) {
      const std::string decomposed = uc::nfd(uc::to_utf8(cps));
      const auto d = uc::to_codepoints(decomposed);
      std::vector<char32_t> out;
      out.reserve(d.size());
      for (char32_t c : d) {
        if (uc::category(c) != uc::Category::Mn) out.push_back(c);
      }
      cps.swap(out);
    }

    std::string text = uc::to_utf8(cps);
    if (lower_) text = uc::lowercase(text);
    return text;
  }

  std::string name() const override { return "BertNormalizer"; }

 private:
  bool clean_, chinese_;
  std::optional<bool> strip_;
  bool lower_;
};

// --- Precompiled ----------------------------------------------------------

class PrecompiledNormalizer final : public Normalizer {
 public:
  explicit PrecompiledNormalizer(Precompiled p) : p_(std::move(p)) {}
  std::string normalize(std::string s) const override { return p_.normalize(s); }
  std::string name() const override { return "Precompiled"; }

 private:
  Precompiled p_;
};

// --- Sequence -------------------------------------------------------------

class SequenceNormalizer final : public Normalizer {
 public:
  explicit SequenceNormalizer(std::vector<std::unique_ptr<Normalizer>> v)
      : v_(std::move(v)) {}
  std::string normalize(std::string s) const override {
    for (const auto& n : v_) s = n->normalize(std::move(s));
    return s;
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
  std::vector<std::unique_ptr<Normalizer>> v_;
};

}  // namespace

std::unique_ptr<Normalizer> make_normalizer(const json::Value& v, std::string_view where) {
  if (v.is_null()) return nullptr;
  const std::string& type = v.require("type", where).as_string("normalizer.type");

  auto opt_bool = [&](const char* key, bool def) {
    const json::Value* p = v.get(key);
    return (p && !p->is_null()) ? p->as_bool(key) : def;
  };
  if (type == "NFC") return std::make_unique<NormalizeForm>(NormalizeForm::Form::NFC);
  if (type == "NFD") return std::make_unique<NormalizeForm>(NormalizeForm::Form::NFD);
  if (type == "NFKC") return std::make_unique<NormalizeForm>(NormalizeForm::Form::NFKC);
  if (type == "NFKD") return std::make_unique<NormalizeForm>(NormalizeForm::Form::NFKD);
  if (type == "Lowercase") return std::make_unique<Lowercase>();
  if (type == "StripAccents") return std::make_unique<StripAccents>();
  if (type == "Nmt") return std::make_unique<Nmt>();

  if (type == "Strip") {
    return std::make_unique<Strip>(opt_bool("strip_left", true), opt_bool("strip_right", true));
  }
  if (type == "Prepend") {
    return std::make_unique<Prepend>(v.require("prepend", where).as_string("prepend"));
  }
  if (type == "Replace") {
    const json::Value& pat = v.require("pattern", where);
    const json::Value* str = pat.get("String");
    if (!str) {
      const json::Value* re = pat.get("Regex");
      throw_error(where, ": Replace mit einem Regex-Muster",
                  re ? std::string(" (\"" + re->as_string("Regex") + "\")") : std::string(),
                  " wird noch nicht unterstuetzt. quasar bricht hier ab, statt "
                  "ersatzweise etwas anderes zu tun -- ein falsch normalisierter "
                  "Text erzeugt falsche Token-IDs, die wie Modellfehler aussehen.");
    }
    return std::make_unique<ReplaceString>(str->as_string("pattern.String"),
                                           v.require("content", where).as_string("content"));
  }
  if (type == "BertNormalizer") {
    std::optional<bool> strip;
    if (const json::Value* p = v.get("strip_accents"); p && !p->is_null()) {
      strip = p->as_bool("strip_accents");
    }
    return std::make_unique<BertNormalizer>(opt_bool("clean_text", true),
                                            opt_bool("handle_chinese_chars", true), strip,
                                            opt_bool("lowercase", true));
  }
  if (type == "Precompiled") {
    const std::string& b64 = v.require("precompiled_charsmap", where)
                                 .as_string("precompiled_charsmap");
    return std::make_unique<PrecompiledNormalizer>(Precompiled::from_base64(b64, where));
  }
  if (type == "Sequence") {
    const json::Array& a = v.require("normalizers", where).as_array("normalizers");
    std::vector<std::unique_ptr<Normalizer>> parts;
    parts.reserve(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
      std::string sub = std::string(where) + ".normalizers[" + std::to_string(i) + "]";
      if (auto n = make_normalizer(a[i], sub)) parts.push_back(std::move(n));
    }
    return std::make_unique<SequenceNormalizer>(std::move(parts));
  }

  // Bewusst kein Rueckfall auf "nichts tun": ein uebergangener Normalisierer
  throw_error(where, ": unbekannter Normalisierer-Typ \"", type,
              "\". quasar kennt: NFC, NFD, NFKC, NFKD, Lowercase, StripAccents, "
              "Strip, Prepend, Replace, Nmt, BertNormalizer, Precompiled, Sequence.");
}

}  // namespace quasar::tok

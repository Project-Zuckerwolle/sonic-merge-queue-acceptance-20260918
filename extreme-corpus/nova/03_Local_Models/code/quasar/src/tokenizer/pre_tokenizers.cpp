// quasar — src/tokenizer/pre_tokenizers.cpp
//
// Vorzerleger und die beiden von Hand ausprogrammierten Zerlegungsregeln
// (GPT-2 und Llama-3). Diese Regeln stehen in der Referenzbibliothek als
// Oniguruma-Ausdruecke; hier sind sie ausgeschrieben, weil quasar keine
// Paketmanager-Abhaengigkeit) und weil die Semantik ohnehin exakt
// nachgebildet werden muss.
//
//   \s      = Unicode-Eigenschaft White_Space
//   \p{L}   = Unicode-Hauptkategorie L
//   \p{N}   = Unicode-Hauptkategorie N
// Alternativen werden in ihrer Reihenfolge probiert (leftmost-first), die
// Quantoren sind gierig mit Rueckzug.

#include <array>
#include <mutex>

#include "quasar/core/json.hpp"
#include "quasar/core/unicode.hpp"
#include "tokenizer/components.hpp"

namespace quasar::tok {

namespace uc = quasar::unicode;

// --- Byte <-> sichtbares Zeichen -----------------------------------------

const std::array<char32_t, 256>& byte_to_char() {
  static const std::array<char32_t, 256> table = [] {
    std::array<char32_t, 256> t{};
    std::array<bool, 256> direct{};
    for (int b = '!'; b <= '~'; ++b) direct[static_cast<std::size_t>(b)] = true;
    for (int b = 0xA1; b <= 0xAC; ++b) direct[static_cast<std::size_t>(b)] = true;
    for (int b = 0xAE; b <= 0xFF; ++b) direct[static_cast<std::size_t>(b)] = true;
    char32_t next = 256;
    for (std::size_t b = 0; b < 256; ++b) {
      t[b] = direct[b] ? static_cast<char32_t>(b) : next++;
    }
    return t;
  }();
  return table;
}

int char_to_byte(char32_t c) {
  static const std::array<int, 512> rev = [] {
    std::array<int, 512> r{};
    r.fill(-1);
    const auto& t = byte_to_char();
    for (int b = 0; b < 256; ++b) r[static_cast<std::size_t>(t[static_cast<std::size_t>(b)])] = b;
    return r;
  }();
  return c < 512 ? rev[static_cast<std::size_t>(c)] : -1;
}

namespace {

inline bool is_ws(char32_t c) { return uc::is_whitespace(c); }
inline bool is_L(char32_t c) { return uc::is_letter(c); }
inline bool is_N(char32_t c) { return uc::is_number(c); }
inline bool is_other(char32_t c) { return !is_ws(c) && !is_L(c) && !is_N(c); }

// Laenge der Kontraktion an Position i, 0 wenn keine.
// Reihenfolge wie im Ausdruck: 's 't 're 've 'm 'll 'd
std::size_t match_contraction(const std::vector<char32_t>& c, std::size_t i,
                              bool case_insensitive) {
  if (i >= c.size() || c[i] != U'\'') return 0;
  auto eq = [&](std::size_t k, char32_t lower) {
    if (k >= c.size()) return false;
    char32_t x = c[k];
    if (case_insensitive && x >= U'A' && x <= U'Z') x = x - U'A' + U'a';
    return x == lower;
  };
  if (eq(i + 1, U's')) return 2;
  if (eq(i + 1, U't')) return 2;
  if (eq(i + 1, U'r') && eq(i + 2, U'e')) return 3;
  if (eq(i + 1, U'v') && eq(i + 2, U'e')) return 3;
  if (eq(i + 1, U'm')) return 2;
  if (eq(i + 1, U'l') && eq(i + 2, U'l')) return 3;
  if (eq(i + 1, U'd')) return 2;
  return 0;
}

// ` ?PRED+` -- ein optionales Leerzeichen, dann mindestens ein PRED-Zeichen.
template <typename Pred>
std::size_t match_opt_space_run(const std::vector<char32_t>& c, std::size_t i, Pred pred) {
  const std::size_t n = c.size();
  // Erst mit Leerzeichen versuchen (gierig), dann ohne (Rueckzug).
  for (int take_space = (i < n && c[i] == U' ') ? 1 : 0; take_space >= 0; --take_space) {
    std::size_t j = i + static_cast<std::size_t>(take_space);
    if (j < n && pred(c[j])) {
      while (j < n && pred(c[j])) ++j;
      return j - i;
    }
  }
  return 0;
}

// `\s+(?!\S)` -- der Leerraumlauf ohne sein letztes Zeichen, falls danach
// etwas anderes als Leerraum kommt; sonst der ganze Lauf.
std::size_t match_ws_not_followed(const std::vector<char32_t>& c, std::size_t i) {
  const std::size_t n = c.size();
  if (i >= n || !is_ws(c[i])) return 0;
  std::size_t k = i;
  while (k < n && is_ws(c[k])) ++k;
  if (k == n) return k - i;
  const std::size_t len = k - i - 1;
  return len >= 1 ? len : 0;
}

std::size_t match_ws_run(const std::vector<char32_t>& c, std::size_t i) {
  const std::size_t n = c.size();
  if (i >= n || !is_ws(c[i])) return 0;
  std::size_t k = i;
  while (k < n && is_ws(c[k])) ++k;
  return k - i;
}

void emit(const std::vector<char32_t>& c, std::size_t from, std::size_t len,
          std::vector<std::string>& out) {
  if (len == 0) return;
  std::string s;
  for (std::size_t k = from; k < from + len; ++k) uc::append_utf8(s, c[k]);
  out.push_back(std::move(s));
}

}  // namespace

void gpt2_split(std::string_view s, std::vector<std::string>& out) {
  const auto c = uc::to_codepoints(s);
  const std::size_t n = c.size();
  std::size_t i = 0;
  while (i < n) {
    std::size_t m = match_contraction(c, i, /*case_insensitive=*/false);
    if (!m) m = match_opt_space_run(c, i, is_L);
    if (!m) m = match_opt_space_run(c, i, is_N);
    if (!m) m = match_opt_space_run(c, i, is_other);
    if (!m) m = match_ws_not_followed(c, i);
    if (!m) m = match_ws_run(c, i);
    if (!m) {
      // Kann nicht vorkommen: jedes Zeichen faellt in genau eine der Klassen
      // \s, \p{L}, \p{N} oder den Rest. Wenn doch, ist die Annahme falsch --
      // dann laut abbrechen statt still ein Zeichen zu verschlucken.
      throw_error("gpt2_split: kein Zweig greift bei Codepunkt U+",
                  hex(static_cast<std::uint64_t>(c[i])), " an Position ", i);
    }
    emit(c, i, m, out);
    i += m;
  }
}

void llama3_split(std::string_view s, std::vector<std::string>& out) {
  // (?i:'s|'t|'re|'ve|'m|'ll|'d)
  // |[^\r\n\p{L}\p{N}]?\p{L}+
  // |\p{N}{1,3}
  // | ?[^\s\p{L}\p{N}]+[\r\n]*
  // |\s*[\r\n]+
  // |\s+(?!\S)
  // |\s+
  const auto c = uc::to_codepoints(s);
  const std::size_t n = c.size();
  auto is_crlf = [](char32_t x) { return x == U'\r' || x == U'\n'; };

  std::size_t i = 0;
  while (i < n) {
    std::size_t m = match_contraction(c, i, /*case_insensitive=*/true);

    if (!m) {  // [^\r\n\p{L}\p{N}]? \p{L}+
      for (int take = 1; take >= 0 && !m; --take) {
        std::size_t j = i;
        if (take) {
          if (i >= n || is_crlf(c[i]) || is_L(c[i]) || is_N(c[i])) continue;
          j = i + 1;
        }
        if (j < n && is_L(c[j])) {
          while (j < n && is_L(c[j])) ++j;
          m = j - i;
        }
      }
    }
    if (!m) {  // \p{N}{1,3}
      std::size_t j = i;
      while (j < n && j < i + 3 && is_N(c[j])) ++j;
      m = j - i;
    }
    if (!m) {  //  ?[^\s\p{L}\p{N}]+[\r\n]*
      for (int take_space = (i < n && c[i] == U' ') ? 1 : 0; take_space >= 0 && !m;
           --take_space) {
        std::size_t j = i + static_cast<std::size_t>(take_space);
        if (j < n && is_other(c[j])) {
          while (j < n && is_other(c[j])) ++j;
          while (j < n && is_crlf(c[j])) ++j;
          m = j - i;
        }
      }
    }
    if (!m) {  // \s*[\r\n]+   (gierig mit Rueckzug)
      std::size_t k = i;
      while (k < n && is_ws(c[k])) ++k;
      for (std::size_t star = k; star >= i; --star) {
        if (star < n && is_crlf(c[star])) {
          std::size_t j = star;
          while (j < n && is_crlf(c[j])) ++j;
          m = j - i;
          break;
        }
        if (star == i) break;
      }
    }
    if (!m) m = match_ws_not_followed(c, i);
    if (!m) m = match_ws_run(c, i);
    if (!m) {
      throw_error("llama3_split: kein Zweig greift bei Codepunkt U+",
                  hex(static_cast<std::uint64_t>(c[i])), " an Position ", i);
    }
    emit(c, i, m, out);
    i += m;
  }
}

void tekken_split(std::string_view s, std::vector<std::string>& out) {
  // runtime regex dependency.  The two letter alternatives preserve the
  // upper/title-to-lower boundary used for camel-case and acronyms.
  const auto c = uc::to_codepoints(s);
  const std::size_t n = c.size();
  const auto is_crlf = [](char32_t x) { return x == U'\r' || x == U'\n'; };
  const auto upperish = [](char32_t x) {
    const auto cat = uc::category(x);
    return cat == uc::Category::Lu || cat == uc::Category::Lt ||
           cat == uc::Category::Lm || cat == uc::Category::Lo || uc::is_mark(x);
  };
  const auto lowerish = [](char32_t x) {
    const auto cat = uc::category(x);
    return cat == uc::Category::Ll || cat == uc::Category::Lm ||
           cat == uc::Category::Lo || uc::is_mark(x);
  };
  const auto optional_prefix = [&](std::size_t i) {
    return i < n && !is_crlf(c[i]) && !is_L(c[i]) && !is_N(c[i]) ? i + 1 : i;
  };

  std::size_t i = 0;
  while (i < n) {
    std::size_t m = 0;
    const std::size_t begin = optional_prefix(i);

    // upperish* lowerish+.  The lower run ends at the next uppercase letter;
    // consuming the complete L* run first incorrectly merged camelCase and,
    // worse, left a lowercase-leading suffix with no matching branch.
    std::size_t split = begin;
    while (split < n && upperish(c[split])) ++split;
    std::size_t finish = split;
    while (finish < n && lowerish(c[finish])) ++finish;
    // Lm/Lo/M occur in both classes. Regex greediness backtracks enough of
    // the upper repetition to provide the required lower repetition.
    while (finish == split && split > begin && lowerish(c[split - 1])) {
      --split;
      finish = split;
      while (finish < n && lowerish(c[finish])) ++finish;
    }
    if (finish > split) m = finish - i;

    // upperish+ lowerish*.
    if (!m) {
      std::size_t upper_end = begin;
      while (upper_end < n && upperish(c[upper_end])) ++upper_end;
      if (upper_end > begin) {
        std::size_t lower_end = upper_end;
        while (lower_end < n && lowerish(c[lower_end])) ++lower_end;
        m = lower_end - i;
      }
    }
    if (m) m += match_contraction(c, i + m, /*case_insensitive=*/true);
    if (!m && is_N(c[i])) m = 1;  // \p{N}
    if (!m) {  //  ?[^\s\p{L}\p{N}]+[\r\n/]*
      for (int take_space = c[i] == U' ' ? 1 : 0; take_space >= 0 && !m; --take_space) {
        std::size_t j = i + static_cast<std::size_t>(take_space);
        if (j < n && is_other(c[j])) {
          while (j < n && is_other(c[j])) ++j;
          while (j < n && (is_crlf(c[j]) || c[j] == U'/')) ++j;
          m = j - i;
        }
      }
    }
    if (!m) {  // \s*[\r\n]+
      std::size_t k = i;
      while (k < n && is_ws(c[k])) ++k;
      for (std::size_t star = k; star >= i; --star) {
        if (star < n && is_crlf(c[star])) {
          std::size_t j = star;
          while (j < n && is_crlf(c[j])) ++j;
          m = j - i;
          break;
        }
        if (star == i) break;
      }
    }
    if (!m) m = match_ws_not_followed(c, i);
    if (!m) m = match_ws_run(c, i);
    if (!m) throw_error("tekken_split: no branch for codepoint U+",
                        hex(static_cast<std::uint64_t>(c[i])), " at position ", i);
    emit(c, i, m, out);
    i += m;
  }
}

namespace {

std::string to_byte_chars(std::string_view s) {
  const auto& t = byte_to_char();
  std::string out;
  out.reserve(s.size() * 2);
  for (unsigned char b : s) uc::append_utf8(out, t[b]);
  return out;
}

// --- ByteLevel ------------------------------------------------------------

class ByteLevelPre final : public PreTokenizer {
 public:
  ByteLevelPre(bool add_prefix_space, bool use_regex)
      : add_prefix_(add_prefix_space), use_regex_(use_regex) {}

  void pre_tokenize(std::string piece, std::vector<std::string>& out) const override {
    if (add_prefix_ && !piece.starts_with(' ')) piece.insert(piece.begin(), ' ');
    if (!use_regex_) {
      if (!piece.empty()) out.push_back(to_byte_chars(piece));
      return;
    }
    std::vector<std::string> parts;
    gpt2_split(piece, parts);
    for (auto& p : parts) out.push_back(to_byte_chars(p));
  }
  std::string name() const override { return "ByteLevel"; }

 private:
  bool add_prefix_, use_regex_;
};

// Nur die Byte-Abbildung, ohne eigene Zerlegung -- so wird ByteLevel in
// tokenizer.json von Llama-3-artigen Modellen hinter einem `Split` benutzt.
class ByteMapOnly final : public PreTokenizer {
 public:
  void pre_tokenize(std::string piece, std::vector<std::string>& out) const override {
    if (!piece.empty()) out.push_back(to_byte_chars(piece));
  }
  std::string name() const override { return "ByteLevel(nur Abbildung)"; }
};

// --- Metaspace ------------------------------------------------------------

enum class PrependScheme { Always, Never, First };

class Metaspace final : public PreTokenizer {
 public:
  Metaspace(std::string replacement, PrependScheme scheme, bool split)
      : rep_(std::move(replacement)), scheme_(scheme), split_(split) {}

  void pre_tokenize(std::string piece, std::vector<std::string>& out) const override {
    pre_tokenize(std::move(piece), out, /*at_original_start=*/true);
  }
  void pre_tokenize(std::string piece, std::vector<std::string>& out,
                    bool at_original_start) const override {
    // 1. Leerzeichen durch das Ersatzzeichen ersetzen
    std::string s;
    s.reserve(piece.size() + rep_.size());
    for (char ch : piece) {
      if (ch == ' ') s += rep_;
      else s.push_back(ch);
    }
    // 2. voranstellen
    //
    // Befund S14: hier stand `scheme_ != PrependScheme::Never`. Der Enum-Wert
    // `First` existierte, hatte aber kein eigenes Verhalten -- "first" wirkte
    // wie "always". Bei `Sequence[WhitespaceSplit, Metaspace(prepend_scheme=
    // "first")]` und der Eingabe "Hallo Welt drei" liefert die Referenz
    // `_Hallo | Welt | drei`, quasar lieferte `_Hallo | _Welt | _drei`: jedes
    // Wort ab dem zweiten bekommt eine andere ID. "first" steht in
    const bool prepend = scheme_ == PrependScheme::Always ||
                         (scheme_ == PrependScheme::First && at_original_start);
    if (prepend && !s.starts_with(rep_)) s = rep_ + s;
    // 3. am Ersatzzeichen trennen, das Zeichen bleibt beim folgenden Stueck
    if (!split_) {
      if (!s.empty()) out.push_back(std::move(s));
      return;
    }
    std::size_t i = 0;
    while (i < s.size()) {
      std::size_t next = s.find(rep_, i + (i == 0 && s.compare(0, rep_.size(), rep_) == 0
                                               ? rep_.size()
                                               : 0));
      if (i == 0 && s.compare(0, rep_.size(), rep_) == 0) {
        next = s.find(rep_, rep_.size());
      } else {
        next = s.find(rep_, i + 1);
      }
      if (next == std::string::npos) {
        if (i < s.size()) out.push_back(s.substr(i));
        break;
      }
      if (next > i) out.push_back(s.substr(i, next - i));
      i = next;
    }
  }
  std::string name() const override { return "Metaspace"; }

 private:
  std::string rep_;
  PrependScheme scheme_;
  bool split_;
};

// --- WhitespaceSplit ------------------------------------------------------

class WhitespaceSplit final : public PreTokenizer {
 public:
  void pre_tokenize(std::string piece, std::vector<std::string>& out) const override {
    const auto c = uc::to_codepoints(piece);
    std::size_t i = 0;
    while (i < c.size()) {
      while (i < c.size() && is_ws(c[i])) ++i;
      std::size_t j = i;
      while (j < c.size() && !is_ws(c[j])) ++j;
      if (j > i) emit(c, i, j - i, out);
      i = j;
    }
  }
  std::string name() const override { return "WhitespaceSplit"; }
};

// --- Whitespace (\w+|[^\w\s]+, isoliert) ---------------------------------

class WhitespacePre final : public PreTokenizer {
 public:
  void pre_tokenize(std::string piece, std::vector<std::string>& out) const override {
    const auto c = uc::to_codepoints(piece);
    auto is_w = [](char32_t x) { return uc::is_alphanumeric(x) || x == U'_'; };
    std::size_t i = 0;
    while (i < c.size()) {
      if (is_w(c[i])) {
        std::size_t j = i;
        while (j < c.size() && is_w(c[j])) ++j;
        emit(c, i, j - i, out);
        i = j;
      } else if (!is_ws(c[i])) {
        std::size_t j = i;
        while (j < c.size() && !is_w(c[j]) && !is_ws(c[j])) ++j;
        emit(c, i, j - i, out);
        i = j;
      } else {
        ++i;
      }
    }
  }
  std::string name() const override { return "Whitespace"; }
};

// --- BertPreTokenizer -----------------------------------------------------

bool is_bert_punc(char32_t c) {
  if (c < 128) {
    return (c >= U'!' && c <= U'/') || (c >= U':' && c <= U'@') ||
           (c >= U'[' && c <= U'`') || (c >= U'{' && c <= U'~');
  }
  return uc::is_punctuation(c);
}

class BertPre final : public PreTokenizer {
 public:
  void pre_tokenize(std::string piece, std::vector<std::string>& out) const override {
    const auto c = uc::to_codepoints(piece);
    std::size_t i = 0;
    while (i < c.size()) {
      while (i < c.size() && is_ws(c[i])) ++i;
      std::size_t j = i;
      while (j < c.size() && !is_ws(c[j])) ++j;
      // Wort [i, j) an Satzzeichen zerlegen, Satzzeichen bleiben als
      // eigenstaendige Stuecke stehen.
      std::size_t a = i;
      while (a < j) {
        if (is_bert_punc(c[a])) {
          emit(c, a, 1, out);
          ++a;
          continue;
        }
        std::size_t b = a;
        while (b < j && !is_bert_punc(c[b])) ++b;
        emit(c, a, b - a, out);
        a = b;
      }
      i = j;
    }
  }
  std::string name() const override { return "BertPreTokenizer"; }
};

// --- Punctuation / Digits / CharDelimiterSplit ---------------------------

class PunctuationPre final : public PreTokenizer {
 public:
  void pre_tokenize(std::string piece, std::vector<std::string>& out) const override {
    const auto c = uc::to_codepoints(piece);
    std::size_t i = 0;
    while (i < c.size()) {
      if (is_bert_punc(c[i])) {
        emit(c, i, 1, out);
        ++i;
        continue;
      }
      std::size_t j = i;
      while (j < c.size() && !is_bert_punc(c[j])) ++j;
      emit(c, i, j - i, out);
      i = j;
    }
  }
  std::string name() const override { return "Punctuation"; }
};

class DigitsPre final : public PreTokenizer {
 public:
  explicit DigitsPre(bool individual) : individual_(individual) {}
  void pre_tokenize(std::string piece, std::vector<std::string>& out) const override {
    const auto c = uc::to_codepoints(piece);
    std::size_t i = 0;
    while (i < c.size()) {
      const bool d = is_N(c[i]);
      if (d && individual_) {
        emit(c, i, 1, out);
        ++i;
        continue;
      }
      std::size_t j = i;
      while (j < c.size() && is_N(c[j]) == d) ++j;
      emit(c, i, j - i, out);
      i = j;
    }
  }
  std::string name() const override { return "Digits"; }

 private:
  bool individual_;
};

class CharDelimiterSplit final : public PreTokenizer {
 public:
  explicit CharDelimiterSplit(char32_t d) : d_(d) {}
  void pre_tokenize(std::string piece, std::vector<std::string>& out) const override {
    const auto c = uc::to_codepoints(piece);
    std::size_t i = 0;
    while (i < c.size()) {
      std::size_t j = i;
      while (j < c.size() && c[j] != d_) ++j;
      if (j > i) emit(c, i, j - i, out);
      i = j + 1;
    }
  }
  std::string name() const override { return "CharDelimiterSplit"; }

 private:
  char32_t d_;
};

// --- Split mit Zeichenketten-Muster --------------------------------------

class SplitString final : public PreTokenizer {
 public:
  enum class Behavior { Removed, Isolated, MergedWithPrevious, MergedWithNext, Contiguous };
  SplitString(std::string pat, Behavior b, bool invert)
      : pat_(std::move(pat)), b_(b), invert_(invert) {}

  void pre_tokenize(std::string piece, std::vector<std::string>& out) const override {
    if (invert_) {
      throw_error("Split: invert=true wird noch nicht unterstuetzt");
    }
    if (pat_.empty()) {
      if (!piece.empty()) out.push_back(std::move(piece));
      return;
    }
    std::vector<std::pair<std::string, bool>> parts;  // Text, ist_Treffer
    std::size_t i = 0;
    std::string cur;
    while (i < piece.size()) {
      if (piece.compare(i, pat_.size(), pat_) == 0) {
        if (!cur.empty()) parts.emplace_back(std::move(cur), false);
        cur.clear();
        parts.emplace_back(pat_, true);
        i += pat_.size();
      } else {
        cur.push_back(piece[i]);
        ++i;
      }
    }
    if (!cur.empty()) parts.emplace_back(std::move(cur), false);

    switch (b_) {
      case Behavior::Removed:
        for (auto& [t, hit] : parts)
          if (!hit && !t.empty()) out.push_back(std::move(t));
        break;
      case Behavior::Isolated:
        for (auto& [t, hit] : parts)
          if (!t.empty()) out.push_back(std::move(t));
        break;
      case Behavior::MergedWithPrevious: {
        std::string acc;
        for (auto& [t, hit] : parts) {
          if (hit) {
            acc += t;
            if (!acc.empty()) out.push_back(acc);
            acc.clear();
          } else {
            if (!acc.empty()) out.push_back(acc);
            acc = t;
          }
        }
        if (!acc.empty()) out.push_back(acc);
        break;
      }
      case Behavior::MergedWithNext: {
        std::string acc;
        for (auto& [t, hit] : parts) {
          if (hit) {
            if (!acc.empty()) out.push_back(acc);
            acc = t;
          } else {
            acc += t;
          }
        }
        if (!acc.empty()) out.push_back(acc);
        break;
      }
      case Behavior::Contiguous: {
        std::string acc;
        bool acc_hit = false;
        for (auto& [t, hit] : parts) {
          if (!acc.empty() && hit != acc_hit) {
            out.push_back(acc);
            acc.clear();
          }
          acc += t;
          acc_hit = hit;
        }
        if (!acc.empty()) out.push_back(acc);
        break;
      }
    }
  }
  std::string name() const override { return "Split"; }

 private:
  std::string pat_;
  Behavior b_;
  bool invert_;
};

// --- Sequence -------------------------------------------------------------

class SequencePre final : public PreTokenizer {
 public:
  explicit SequencePre(std::vector<std::unique_ptr<PreTokenizer>> v) : v_(std::move(v)) {}
  void pre_tokenize(std::string piece, std::vector<std::string>& out) const override {
    pre_tokenize(std::move(piece), out, /*at_original_start=*/true);
  }
  void pre_tokenize(std::string piece, std::vector<std::string>& out,
                    bool at_original_start) const override {
    // Befund S14: die Angabe "beginnt bei Offset 0" gehoert durch die Kette
    // weitergereicht -- Metaspace mit `prepend_scheme: "first"` haengt daran.
    // Vorher gab es sie nicht, und "first" verhielt sich wie "always": in
    // `Sequence[WhitespaceSplit, Metaspace(first)]` bekam jedes Wort ab dem
    // zweiten eine andere ID.
    //
    // Ein Ausgabestueck beginnt genau dann bei Offset 0, wenn sein Eingabestueck
    // weggenommen hat. Das Letzte ist am Praefix ablesbar: WhitespaceSplit
    // liefert echte Teilstuecke, und schneidet es das fuehrende Leerzeichen weg,
    // ist das erste Ausgabestueck kein Praefix mehr.
    std::vector<std::string> cur{std::move(piece)};
    std::vector<char> at_zero{static_cast<char>(at_original_start)};
    std::vector<std::string> next;
    std::vector<char> next_at_zero;
    for (const auto& p : v_) {
      next.clear();
      next_at_zero.clear();
      for (std::size_t i = 0; i < cur.size(); ++i) {
        const std::size_t before = next.size();
        const bool flag = at_zero[i] != 0;
        const std::string original = cur[i];  // Kopie fuer den Praefixtest
        p->pre_tokenize(std::move(cur[i]), next, flag);
        for (std::size_t k = before; k < next.size(); ++k) {
          const bool starts_here =
              (k == before) && flag && original.starts_with(next[k]);
          next_at_zero.push_back(static_cast<char>(starts_here));
        }
      }
      cur.swap(next);
      at_zero.swap(next_at_zero);
    }
    for (auto& c : cur) out.push_back(std::move(c));
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
  std::vector<std::unique_ptr<PreTokenizer>> v_;
};

SplitString::Behavior parse_behavior(const std::string& s, std::string_view where) {
  if (s == "Removed") return SplitString::Behavior::Removed;
  if (s == "Isolated") return SplitString::Behavior::Isolated;
  if (s == "MergedWithPrevious") return SplitString::Behavior::MergedWithPrevious;
  if (s == "MergedWithNext") return SplitString::Behavior::MergedWithNext;
  if (s == "Contiguous") return SplitString::Behavior::Contiguous;
  throw_error(where, ": unbekanntes Split-Verhalten \"", s, "\"");
}

}  // namespace

std::unique_ptr<PreTokenizer> make_pre_tokenizer(const json::Value& v,
                                                 std::string_view where) {
  if (v.is_null()) return nullptr;
  const std::string& type = v.require("type", where).as_string("pre_tokenizer.type");

  auto opt_bool = [&](const char* key, bool def) {
    const json::Value* p = v.get(key);
    return (p && !p->is_null()) ? p->as_bool(key) : def;
  };

  if (type == "ByteLevel") {
    return std::make_unique<ByteLevelPre>(opt_bool("add_prefix_space", true),
                                          opt_bool("use_regex", true));
  }
  if (type == "ByteLevelNoRegex") return std::make_unique<ByteMapOnly>();
  if (type == "WhitespaceSplit") return std::make_unique<WhitespaceSplit>();
  if (type == "Whitespace") return std::make_unique<WhitespacePre>();
  if (type == "BertPreTokenizer") return std::make_unique<BertPre>();
  if (type == "Punctuation") return std::make_unique<PunctuationPre>();
  if (type == "Digits") {
    return std::make_unique<DigitsPre>(opt_bool("individual_digits", false));
  }
  if (type == "CharDelimiterSplit") {
    const std::string& d = v.require("delimiter", where).as_string("delimiter");
    const auto cps = uc::to_codepoints(d);
    if (cps.size() != 1) throw_error(where, ": CharDelimiterSplit braucht genau ein Zeichen");
    return std::make_unique<CharDelimiterSplit>(cps[0]);
  }
  if (type == "Metaspace") {
    std::string rep = "\xE2\x96\x81";  // U+2581
    if (const json::Value* p = v.get("replacement"); p && !p->is_null()) {
      rep = p->as_string("replacement");
    }
    PrependScheme scheme = PrependScheme::Always;
    if (const json::Value* p = v.get("prepend_scheme"); p && !p->is_null()) {
      const std::string& s = p->as_string("prepend_scheme");
      if (s == "always") scheme = PrependScheme::Always;
      else if (s == "never") scheme = PrependScheme::Never;
      else if (s == "first") scheme = PrependScheme::First;
      else throw_error(where, ": unbekanntes prepend_scheme \"", s, "\"");
    } else if (const json::Value* q = v.get("add_prefix_space"); q && !q->is_null()) {
      scheme = q->as_bool("add_prefix_space") ? PrependScheme::Always : PrependScheme::Never;
    }
    return std::make_unique<Metaspace>(std::move(rep), scheme, opt_bool("split", true));
  }
  if (type == "Split") {
    const json::Value& pat = v.require("pattern", where);
    const json::Value* str = pat.get("String");
    if (!str) {
      const json::Value* re = pat.get("Regex");
      const std::string p = re ? re->as_string("Regex") : std::string("(unbekannt)");
      throw_error(where,
                  ": Split mit Regex-Muster wird noch nicht unterstuetzt. Muster: ", p,
                  ". quasar bricht ab, statt ersatzweise anders zu zerlegen -- eine "
                  "falsche Zerlegung erzeugt falsche Token-IDs.");
    }
    const std::string& beh = v.require("behavior", where).as_string("behavior");
    return std::make_unique<SplitString>(str->as_string("pattern.String"),
                                         parse_behavior(beh, where),
                                         opt_bool("invert", false));
  }
  if (type == "Sequence") {
    const json::Array& a = v.require("pretokenizers", where).as_array("pretokenizers");
    std::vector<std::unique_ptr<PreTokenizer>> parts;
    for (std::size_t i = 0; i < a.size(); ++i) {
      std::string sub = std::string(where) + ".pretokenizers[" + std::to_string(i) + "]";
      if (auto p = make_pre_tokenizer(a[i], sub)) parts.push_back(std::move(p));
    }
    return std::make_unique<SequencePre>(std::move(parts));
  }

  throw_error(where, ": unbekannter Vorzerleger-Typ \"", type,
              "\". quasar kennt: ByteLevel, Metaspace, Whitespace, WhitespaceSplit, "
              "BertPreTokenizer, Punctuation, Digits, CharDelimiterSplit, Split, Sequence.");
}

// Fuer den GGUF-Weg gebaute Vorzerleger (dort gibt es kein tokenizer.json).
std::unique_ptr<PreTokenizer> make_byte_level_pre(bool add_prefix_space, bool use_regex) {
  return std::make_unique<ByteLevelPre>(add_prefix_space, use_regex);
}

class Llama3Pre final : public PreTokenizer {
 public:
  void pre_tokenize(std::string piece, std::vector<std::string>& out) const override {
    std::vector<std::string> parts;
    llama3_split(piece, parts);
    for (auto& p : parts) out.push_back(to_byte_chars(p));
  }
  std::string name() const override { return "Llama3ByteLevel"; }
};

std::unique_ptr<PreTokenizer> make_llama3_pre() { return std::make_unique<Llama3Pre>(); }

class TekkenPre final : public PreTokenizer {
 public:
  void pre_tokenize(std::string piece, std::vector<std::string>& out) const override {
    std::vector<std::string> parts;
    tekken_split(piece, parts);
    for (auto& p : parts) out.push_back(to_byte_chars(p));
  }
  std::string name() const override { return "TekkenByteLevel"; }
};

std::unique_ptr<PreTokenizer> make_tekken_pre() { return std::make_unique<TekkenPre>(); }

}  // namespace quasar::tok

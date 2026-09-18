// quasar — src/tokenizer/models.cpp
//
//
// Die Reihenfolge, in der BPE Paare verschmilzt, und die Art, wie Unigram
// unbekannte Stuecke bewertet, sind nicht "im Wesentlichen so" nachzubauen,
// sondern genau. Ein Unterschied von einem Token sieht spaeter aus wie ein

#include <algorithm>
#include <cstdio>
#include <limits>
#include <queue>

#include "quasar/core/json.hpp"
#include "quasar/core/unicode.hpp"
#include "tokenizer/components.hpp"

namespace quasar::tok {

void Vocab::add(std::string token, TokenId id) {
  if (auto it = to_id_.find(token); it != to_id_.end()) {
    // Gleicher Text, andere ID: die erste gewinnt (so macht es auch die
    // Referenzbibliothek). Rueckwaerts wird die neue ID trotzdem eingetragen.
    if (id >= to_token_.size()) to_token_.resize(id + 1, nullptr);
    if (to_token_[id] == nullptr) {
      storage_.push_back(std::move(token));
      to_token_[id] = &storage_.back();
    }
    return;
  }
  if (id >= to_token_.size()) to_token_.resize(id + 1, nullptr);
  if (to_token_[id] != nullptr) {
    throw_error("Vokabular: ID ", id, " ist doppelt vergeben (\"", *to_token_[id],
                "\" und \"", token, "\")");
  }
  storage_.push_back(std::move(token));
  to_token_[id] = &storage_.back();
  to_id_.emplace(std::string_view(storage_.back()), id);
}

namespace {

namespace uc = quasar::unicode;

// =========================================================================
// BPE
// =========================================================================
//
// Symbolkette als doppelt verkettete Liste in einem Vektor; die Paare stehen
// in einer Vorrangwarteschlange nach (Rang, Position). Genau so arbeitet die
// Referenzbibliothek -- eine naive Schleife "suche das global beste Paar"
// liefert bei gleichem Rang eine andere Reihenfolge und damit andere IDs.

struct Symbol {
  TokenId c = 0;
  int prev = -1;
  int next = -1;
  std::uint32_t len = 0;  // 0 = entfernt
};

struct MergeEntry {
  std::uint32_t rank;
  TokenId new_id;
};

struct PendingMerge {
  std::size_t pos;
  std::uint32_t rank;
  TokenId new_id;
  // Kleinster Rang zuerst; bei gleichem Rang die kleinere Position.
  bool operator<(const PendingMerge& o) const {
    if (rank != o.rank) return rank > o.rank;
    return pos > o.pos;
  }
};

struct PairHash {
  std::size_t operator()(const std::pair<TokenId, TokenId>& p) const noexcept {
    return (static_cast<std::size_t>(p.first) << 32) ^ p.second;
  }
};

class BpeModel final : public Model {
 public:
  BpeModel(Vocab v, BpeSpec spec, std::string_view where) : vocab_(std::move(v)) {
    unk_ = spec.unk_token;
    prefix_ = spec.continuing_subword_prefix;
    suffix_ = spec.end_of_word_suffix;
    fuse_unk_ = spec.fuse_unk;
    byte_fallback_ = spec.byte_fallback;
    ignore_merges_ = spec.ignore_merges;

    if (unk_ && !vocab_.contains(*unk_)) {
      throw_error(where, ": unk_token \"", *unk_, "\" steht nicht im Vokabular");
    }
    merges_.reserve(spec.merges.size() * 2);
    std::uint32_t rank = 0;
    for (auto& [a, b] : spec.merges) {
      const auto ia = vocab_.id_of(a);
      const auto ib = vocab_.id_of(b);
      if (!ia || !ib) {
        // Ein Merge, dessen Teile nicht im Vokabular stehen, ist unbrauchbar.
        // Die Referenzbibliothek bricht hier ab -- wir auch.
        throw_error(where, ": Merge Nr. ", rank, " (\"", a, "\" + \"", b,
                    "\") verweist auf ein Token, das nicht im Vokabular steht");
      }
      const std::string joined = a + b;
      const auto in = vocab_.id_of(joined);
      if (!in) {
        throw_error(where, ": Merge Nr. ", rank, " ergibt \"", joined,
                    "\", das nicht im Vokabular steht");
      }
      merges_.emplace(std::pair<TokenId, TokenId>{*ia, *ib}, MergeEntry{rank, *in});
      ++rank;
    }
  }

  void tokenize(std::string_view piece, std::vector<TokenId>& ids,
                std::vector<std::string>* tokens) const override {
    if (piece.empty()) return;

    if (ignore_merges_) {
      if (auto id = vocab_.id_of(piece)) {
        push(ids, tokens, *id);
        return;
      }
    }

    std::vector<Symbol> sym;
    sym.reserve(piece.size());

    // --- Startsymbole: je ein Zeichen, mit Prefix/Suffix ------------------
    std::optional<std::pair<TokenId, std::uint32_t>> unk_run;
    const auto push_sym = [&](TokenId id, std::uint32_t len) {
      const int idx = static_cast<int>(sym.size());
      sym.push_back(Symbol{id, idx - 1, idx + 1, len});
    };

    std::size_t i = 0;
    while (i < piece.size()) {
      const std::size_t clen = utf8_len(piece, i);
      const std::string_view ch = piece.substr(i, clen);
      const bool is_first = (i == 0);
      const bool is_last = (i + clen >= piece.size());
      i += clen;

      std::string s(ch);
      if (!is_first && prefix_) s = *prefix_ + s;
      if (is_last && suffix_) s = s + *suffix_;
      const auto byte_len = static_cast<std::uint32_t>(ch.size());

      if (auto id = vocab_.id_of(s)) {
        if (unk_run) {
          push_sym(unk_run->first, unk_run->second);
          unk_run.reset();
        }
        push_sym(*id, byte_len);
        continue;
      }

      if (byte_fallback_) {
        bool all = true;
        std::vector<TokenId> bytes;
        bytes.reserve(s.size());
        for (unsigned char b : s) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "<0x%02X>", b);
          const auto id = vocab_.id_of(buf);
          if (!id) { all = false; break; }
          bytes.push_back(*id);
        }
        if (all) {
          if (unk_run) {
            push_sym(unk_run->first, unk_run->second);
            unk_run.reset();
          }
          for (TokenId b : bytes) push_sym(b, 1);
          continue;
        }
      }

      if (unk_) {
        const TokenId uid = *vocab_.id_of(*unk_);
        if (unk_run && fuse_unk_) {
          unk_run->second += byte_len;
        } else if (unk_run) {
          push_sym(unk_run->first, unk_run->second);
          unk_run = {uid, byte_len};
        } else {
          unk_run = {uid, byte_len};
        }
      }
      // Ohne unk_token faellt ein unbekanntes Zeichen weg -- genau wie in der
      // Referenzbibliothek.
    }
    if (unk_run) push_sym(unk_run->first, unk_run->second);

    if (sym.empty()) return;
    sym.back().next = -1;

    // --- Verschmelzen -----------------------------------------------------
    std::priority_queue<PendingMerge> queue;
    for (std::size_t k = 0; k + 1 < sym.size(); ++k) {
      auto it = merges_.find({sym[k].c, sym[k + 1].c});
      if (it != merges_.end()) {
        queue.push(PendingMerge{k, it->second.rank, it->second.new_id});
      }
    }

    while (!queue.empty()) {
      const PendingMerge top = queue.top();
      queue.pop();
      if (sym[top.pos].len == 0) continue;
      if (sym[top.pos].next == -1) continue;
      const std::size_t next_pos = static_cast<std::size_t>(sym[top.pos].next);
      if (next_pos >= sym.size() || sym[next_pos].len == 0) continue;

      // Abgelaufener Eintrag?
      auto it = merges_.find({sym[top.pos].c, sym[next_pos].c});
      if (it == merges_.end() || it->second.new_id != top.new_id) continue;

      const Symbol right = sym[next_pos];
      sym[top.pos].c = top.new_id;
      sym[top.pos].len += right.len;
      sym[top.pos].next = right.next;
      sym[next_pos].len = 0;
      if (right.next > -1 && static_cast<std::size_t>(right.next) < sym.size()) {
        sym[static_cast<std::size_t>(right.next)].prev = static_cast<int>(top.pos);
      }

      if (sym[top.pos].prev >= 0) {
        const std::size_t p = static_cast<std::size_t>(sym[top.pos].prev);
        auto m = merges_.find({sym[p].c, sym[top.pos].c});
        if (m != merges_.end()) {
          queue.push(PendingMerge{p, m->second.rank, m->second.new_id});
        }
      }
      if (sym[top.pos].next > -1 &&
          static_cast<std::size_t>(sym[top.pos].next) < sym.size()) {
        const std::size_t nx = static_cast<std::size_t>(sym[top.pos].next);
        auto m = merges_.find({sym[top.pos].c, sym[nx].c});
        if (m != merges_.end()) {
          queue.push(PendingMerge{top.pos, m->second.rank, m->second.new_id});
        }
      }
    }

    for (const Symbol& s : sym) {
      if (s.len == 0) continue;
      push(ids, tokens, s.c);
    }
  }

  const Vocab& vocab() const override { return vocab_; }
  std::string name() const override { return "BPE"; }

 private:
  void push(std::vector<TokenId>& ids, std::vector<std::string>* tokens, TokenId id) const {
    ids.push_back(id);
    if (tokens) {
      const std::string* t = vocab_.token_of(id);
      tokens->push_back(t ? *t : std::string());
    }
  }

  Vocab vocab_;
  std::unordered_map<std::pair<TokenId, TokenId>, MergeEntry, PairHash> merges_;
  std::optional<std::string> unk_, prefix_, suffix_;
  bool fuse_unk_ = false, byte_fallback_ = false, ignore_merges_ = false;
};

// =========================================================================
// WordPiece
// =========================================================================

class WordPieceModel final : public Model {
 public:
  WordPieceModel(Vocab v, std::string unk, std::string prefix, std::size_t max_chars,
                 std::string_view where)
      : vocab_(std::move(v)),
        unk_(std::move(unk)),
        prefix_(std::move(prefix)),
        max_chars_(max_chars) {
    const auto id = vocab_.id_of(unk_);
    if (!id) throw_error(where, ": unk_token \"", unk_, "\" steht nicht im Vokabular");
    unk_id_ = *id;
  }

  void tokenize(std::string_view piece, std::vector<TokenId>& ids,
                std::vector<std::string>* tokens) const override {
    if (piece.empty()) return;

    std::size_t char_count = 0;
    for (std::size_t i = 0; i < piece.size(); i += utf8_len(piece, i)) ++char_count;
    if (char_count > max_chars_) {
      push(ids, tokens, unk_id_);
      return;
    }

    std::vector<TokenId> sub;
    std::vector<std::string> sub_tok;
    std::size_t start = 0;
    bool bad = false;
    while (start < piece.size()) {
      std::size_t end = piece.size();
      std::optional<TokenId> found;
      std::string found_tok;
      while (start < end) {
        std::string s;
        if (start > 0) s = prefix_;
        s.append(piece.substr(start, end - start));
        if (auto id = vocab_.id_of(s)) {
          found = *id;
          found_tok = std::move(s);
          break;
        }
        // Ein Zeichen vom Ende wegnehmen.
        std::size_t last = start;
        for (std::size_t k = start; k < end; k += utf8_len(piece, k)) last = k;
        if (last == start) { end = start; break; }
        end = last;
      }
      if (!found) { bad = true; break; }
      sub.push_back(*found);
      sub_tok.push_back(std::move(found_tok));
      start = end;
    }

    if (bad) {
      push(ids, tokens, unk_id_);
      return;
    }
    for (std::size_t k = 0; k < sub.size(); ++k) {
      ids.push_back(sub[k]);
      if (tokens) tokens->push_back(sub_tok[k]);
    }
  }

  const Vocab& vocab() const override { return vocab_; }
  std::string name() const override { return "WordPiece"; }

 private:
  void push(std::vector<TokenId>& ids, std::vector<std::string>* tokens, TokenId id) const {
    ids.push_back(id);
    if (tokens) {
      const std::string* t = vocab_.token_of(id);
      tokens->push_back(t ? *t : std::string());
    }
  }

  Vocab vocab_;
  std::string unk_, prefix_;
  std::size_t max_chars_ = 100;
  TokenId unk_id_ = 0;
};

// =========================================================================
// Unigram
// =========================================================================
//
// Viterbi ueber alle Vokabular-Treffer an jeder Position. Unbekannte Zeichen
// bekommen `min_score - 10.0` (die Strafe stammt aus SentencePiece und ist in
// der Referenzbibliothek dieselbe Konstante).

constexpr double kUnkPenalty = 10.0;

class UnigramModel final : public Model {
 public:
  UnigramModel(std::vector<std::pair<std::string, double>> pieces,
               std::optional<TokenId> unk_id, bool byte_fallback, std::string_view where)
      : scores_(pieces.size()), unk_id_(unk_id), byte_fallback_(byte_fallback) {
    // Befund S13: hier stand `(void)byte_fallback_;` -- der Wert wurde geparst,
    // gespeichert und nie benutzt. Er wird jetzt in `tokenize()` ausgewertet.
    min_score_ = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < pieces.size(); ++i) {
      scores_[i] = pieces[i].second;
      min_score_ = std::min(min_score_, pieces[i].second);
      vocab_.add(pieces[i].first, static_cast<TokenId>(i));
    }
    if (pieces.empty()) throw_error(where, ": Unigram-Vokabular ist leer");
    if (unk_id_ && *unk_id_ >= pieces.size()) {
      throw_error(where, ": unk_id ", *unk_id_, " liegt ausserhalb des Vokabulars (",
                  pieces.size(), " Eintraege)");
    }
    // Praefix-Suche ueber eine sortierte Liste der Vokabulareintraege.
    sorted_.reserve(pieces.size());
    for (std::size_t i = 0; i < pieces.size(); ++i) {
      sorted_.emplace_back(std::string_view(*vocab_.token_of(static_cast<TokenId>(i))),
                           static_cast<TokenId>(i));
    }
    std::sort(sorted_.begin(), sorted_.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
  }

  void tokenize(std::string_view s, std::vector<TokenId>& ids,
                std::vector<std::string>* tokens) const override {
    if (s.empty()) return;

    struct Node {
      TokenId id = 0;
      double score = 0.0;
      std::size_t starts_at = 0;
      bool set = false;
    };
    const std::size_t n = s.size();
    std::vector<Node> best(n + 1);
    best[0].set = true;
    const double unk_score = min_score_ - kUnkPenalty;

    std::size_t at = 0;
    while (at < n) {
      const double till_here = best[at].score;
      if (!best[at].set) {
        // Diese Position ist nicht erreichbar -- kann nicht vorkommen, weil
        // jede Position ueber den Unk-Zweig erreichbar ist.
        throw_error("Unigram: Position ", at, " ist nicht erreichbar");
      }
      const std::size_t mblen = utf8_len(s, at);
      bool has_single = false;

      for_each_prefix(s.substr(at), [&](std::size_t len, TokenId id) {
        const std::size_t key = at + len;
        const double cand = scores_[id] + till_here;
        Node& t = best[key];
        if (!t.set || cand > t.score) {
          t.set = true;
          t.score = cand;
          t.starts_at = at;
          t.id = id;
        }
        if (len == mblen) has_single = true;
      });

      if (!has_single) {
        if (!unk_id_) {
          throw_error("Unigram: unbekanntes Zeichen an Position ", at,
                      " und kein unk_id im Modell");
        }
        Node& t = best[at + mblen];
        const double cand = unk_score + till_here;
        if (!t.set || cand > t.score) {
          t.set = true;
          t.score = cand;
          t.starts_at = at;
          t.id = *unk_id_;
        }
      }
      at += mblen;
    }

    // Rueckwaerts abwickeln; aufeinanderfolgende Unk-Stuecke werden
    // zusammengefasst (fuse_unk ist bei Unigram immer an).
    std::vector<std::string> rev;
    std::vector<std::string> unk_acc;
    std::size_t ends = n;
    while (ends > 0) {
      const Node& node = best[ends];
      if (!node.set) throw_error("Unigram: Rueckweg bricht bei ", ends, " ab");
      const std::size_t st = node.starts_at;
      const std::string_view seg = s.substr(st, ends - st);
      if (unk_id_ && node.id == *unk_id_) {
        unk_acc.emplace_back(seg);
      } else {
        if (!unk_acc.empty()) {
          std::string joined;
          for (auto it = unk_acc.rbegin(); it != unk_acc.rend(); ++it) joined += *it;
          rev.push_back(std::move(joined));
          unk_acc.clear();
        }
        rev.emplace_back(seg);
      }
      ends = st;
    }
    if (!unk_acc.empty()) {
      std::string joined;
      for (auto it = unk_acc.rbegin(); it != unk_acc.rend(); ++it) joined += *it;
      rev.push_back(std::move(joined));
    }
    std::reverse(rev.begin(), rev.end());

    for (auto& t : rev) {
      const auto id = vocab_.id_of(t);
      if (id) {
        ids.push_back(*id);
        if (tokens) tokens->push_back(std::move(t));
        continue;
      }
      // Befund S13: `byte_fallback` wurde aus dem Modell gelesen, gespeichert
      // und mit `(void)byte_fallback_;` ausdruecklich weggeworfen -- ein stiller
      // ausdruecklich als Anforderung nennt. Gemessen an einem Unigram mit
      // `unk_id=0`, `byte_fallback=true` und dem Vokabular
      // `<unk>/a/<0xC3>/<0xA9>`: die Referenz liefert fuer "aé" ids [1,2,3],
      // quasar lieferte [1,0]. Betrifft SentencePiece-Unigram (T5-/Gemma-Familie).
      //
      // Regel wie in der Referenz: nur wenn **alle** Bytes des Stuecks als
      // `<0xXX>` im Vokabular stehen, wird es byteweise zerlegt. Sonst bleibt es
      // bei `<unk>` -- ein halber Fallback waere schlimmer als keiner.
      if (byte_fallback_) {
        std::vector<TokenId> byte_ids;
        byte_ids.reserve(t.size());
        bool all_present = true;
        for (unsigned char b : t) {
          char name[8];
          std::snprintf(name, sizeof name, "<0x%02X>", b);
          const auto byte_id = vocab_.id_of(std::string_view(name));
          if (!byte_id) {
            all_present = false;
            break;
          }
          byte_ids.push_back(*byte_id);
        }
        if (all_present && !byte_ids.empty()) {
          for (TokenId bid : byte_ids) {
            ids.push_back(bid);
            if (tokens) {
              const std::string* text = vocab_.token_of(bid);
              tokens->push_back(text ? *text : std::string());
            }
          }
          continue;
        }
      }
      if (!unk_id_) throw_error("Unigram: Token \"", t, "\" unbekannt, kein unk_id");
      ids.push_back(*unk_id_);
      if (tokens) tokens->push_back(std::move(t));
    }
  }

  const Vocab& vocab() const override { return vocab_; }
  std::string name() const override { return "Unigram"; }

 private:
  // Alle Vokabulareintraege, die Praefix von `rest` sind.
  template <typename F>
  void for_each_prefix(std::string_view rest, F&& f) const {
    // Untere Schranke fuer den ersten Eintrag, der mit rest[0] beginnt.
    auto it = std::lower_bound(sorted_.begin(), sorted_.end(), rest.substr(0, 1),
                               [](const auto& a, std::string_view b) { return a.first < b; });
    for (; it != sorted_.end(); ++it) {
      if (it->first.empty()) continue;
      if (it->first[0] != rest[0]) break;
      if (it->first.size() <= rest.size() &&
          rest.compare(0, it->first.size(), it->first) == 0) {
        f(it->first.size(), it->second);
      }
    }
  }

  Vocab vocab_;
  std::vector<double> scores_;
  std::vector<std::pair<std::string_view, TokenId>> sorted_;
  std::optional<TokenId> unk_id_;
  bool byte_fallback_ = false;
  double min_score_ = 0.0;
};

}  // namespace

// --- Fabriken -------------------------------------------------------------

std::unique_ptr<Model> make_bpe_model(Vocab vocab, BpeSpec spec, std::string_view where) {
  return std::make_unique<BpeModel>(std::move(vocab), std::move(spec), where);
}
std::unique_ptr<Model> make_unigram_model(std::vector<std::pair<std::string, double>> pieces,
                                          std::optional<TokenId> unk_id, bool byte_fallback,
                                          std::string_view where) {
  return std::make_unique<UnigramModel>(std::move(pieces), unk_id, byte_fallback, where);
}
std::unique_ptr<Model> make_wordpiece_model(Vocab vocab, std::string unk_token,
                                            std::string prefix, std::size_t max_chars,
                                            std::string_view where) {
  return std::make_unique<WordPieceModel>(std::move(vocab), std::move(unk_token),
                                          std::move(prefix), max_chars, where);
}

namespace {

Vocab read_vocab_object(const json::Value& v, std::string_view where) {
  const json::Object& o = v.as_object(where);
  Vocab vocab;
  for (const auto& [k, val] : o.items()) {
    vocab.add(k, static_cast<TokenId>(val.as_int("vocab-Eintrag")));
  }
  return vocab;
}

std::vector<std::pair<std::string, std::string>> read_merges(const json::Value& v,
                                                             std::string_view where) {
  const json::Array& a = v.as_array(where);
  std::vector<std::pair<std::string, std::string>> out;
  out.reserve(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].is_array()) {
      // Neues Format: ["a", "b"]
      const json::Array& p = a[i].as_array("merge");
      if (p.size() != 2) throw_error(where, ": Merge Nr. ", i, " hat ", p.size(),
                                     " Teile statt 2");
      out.emplace_back(p[0].as_string("merge[0]"), p[1].as_string("merge[1]"));
    } else {
      // Altes Format: "a b" -- getrennt am ERSTEN Leerzeichen.
      const std::string& s = a[i].as_string("merge");
      const std::size_t sp = s.find(' ');
      if (sp == std::string::npos) {
        throw_error(where, ": Merge Nr. ", i, " (\"", s, "\") hat kein Trennzeichen");
      }
      out.emplace_back(s.substr(0, sp), s.substr(sp + 1));
    }
  }
  return out;
}

}  // namespace

std::unique_ptr<Model> make_model(const json::Value& v, std::string_view where) {
  if (v.is_null()) throw_error(where, ": das Feld \"model\" fehlt");

  // `type` fehlt in aelteren tokenizer.json (bei drei von vier
  // Referenzdateien). Dann wird der Typ aus der Struktur erschlossen -- aber
  // nur eindeutig, sonst Abbruch.
  std::string type;
  if (const json::Value* t = v.get("type"); t && !t->is_null()) {
    type = t->as_string("model.type");
  } else if (v.get("max_input_chars_per_word") || v.get("continuing_subword_prefix")) {
    if (v.get("merges")) type = "BPE";
    else type = "WordPiece";
  } else if (v.get("merges")) {
    type = "BPE";
  } else if (v.get("unk_id") && v.get("vocab") && v.require("vocab", where).is_array()) {
    type = "Unigram";
  } else {
    throw_error(where,
                ": das Feld \"model.type\" fehlt und der Typ laesst sich nicht "
                "eindeutig erschliessen. quasar raet hier nicht.");
  }

  auto opt_str = [&](const char* key) -> std::optional<std::string> {
    const json::Value* p = v.get(key);
    if (!p || p->is_null()) return std::nullopt;
    return p->as_string(key);
  };
  auto opt_bool = [&](const char* key, bool def) {
    const json::Value* p = v.get(key);
    return (p && !p->is_null()) ? p->as_bool(key) : def;
  };

  if (type == "BPE") {
    Vocab vocab = read_vocab_object(v.require("vocab", where), "model.vocab");
    BpeSpec spec;
    spec.merges = read_merges(v.require("merges", where), "model.merges");
    spec.unk_token = opt_str("unk_token");
    spec.continuing_subword_prefix = opt_str("continuing_subword_prefix");
    spec.end_of_word_suffix = opt_str("end_of_word_suffix");
    // Leerer Prefix/Suffix bedeutet "keiner" (gpt2 schreibt "").
    if (spec.continuing_subword_prefix && spec.continuing_subword_prefix->empty())
      spec.continuing_subword_prefix.reset();
    if (spec.end_of_word_suffix && spec.end_of_word_suffix->empty())
      spec.end_of_word_suffix.reset();
    spec.fuse_unk = opt_bool("fuse_unk", false);
    spec.byte_fallback = opt_bool("byte_fallback", false);
    spec.ignore_merges = opt_bool("ignore_merges", false);
    if (const json::Value* d = v.get("dropout"); d && !d->is_null()) {
      throw_error(where, ": BPE-Dropout wird nicht unterstuetzt (es macht die "
                         "Zerlegung zufaellig und damit unpruefbar)");
    }
    return make_bpe_model(std::move(vocab), std::move(spec), where);
  }

  if (type == "WordPiece") {
    Vocab vocab = read_vocab_object(v.require("vocab", where), "model.vocab");
    const std::string unk = opt_str("unk_token").value_or("[UNK]");
    const std::string prefix = opt_str("continuing_subword_prefix").value_or("##");
    std::size_t maxc = 100;
    if (const json::Value* p = v.get("max_input_chars_per_word"); p && !p->is_null()) {
      maxc = static_cast<std::size_t>(p->as_int("max_input_chars_per_word"));
    }
    return make_wordpiece_model(std::move(vocab), unk, prefix, maxc, where);
  }

  if (type == "Unigram") {
    const json::Array& a = v.require("vocab", where).as_array("model.vocab");
    std::vector<std::pair<std::string, double>> pieces;
    pieces.reserve(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
      const json::Array& p = a[i].as_array("Unigram-Eintrag");
      if (p.size() != 2) {
        throw_error(where, ": Unigram-Eintrag Nr. ", i, " hat ", p.size(), " Teile statt 2");
      }
      pieces.emplace_back(p[0].as_string("piece"), p[1].as_double("score"));
    }
    std::optional<TokenId> unk;
    if (const json::Value* p = v.get("unk_id"); p && !p->is_null()) {
      unk = static_cast<TokenId>(p->as_int("unk_id"));
    }
    return make_unigram_model(std::move(pieces), unk, opt_bool("byte_fallback", false), where);
  }

  throw_error(where, ": unbekannter Modelltyp \"", type,
              "\". quasar kennt BPE, WordPiece und Unigram.");
}

}  // namespace quasar::tok

// quasar — src/tokenizer/precompiled.cpp
//
// Der `Precompiled`-Normalisierer aus SentencePiece. In `tokenizer.json` steht
// er als base64-Blob mit folgendem Aufbau:
//
//   [uint32 trie_size (little endian)]
//   [trie_size Byte: Darts-clone Doppelfeld-Trie, uint32-Einheiten]
//   [Rest: nullterminierte Ersatzzeichenketten, hintereinander]
//
// Gesucht wird der laengste Praefix der Eingabe, der im Trie steht; sein Wert
// ist ein Offset in den Ersetzungsblock.
//
// Die Referenzbibliothek wendet das **auf erweiterten Graphem-Clustern** an
// und erst danach zeichenweise. Das ist nicht schoen, aber es ist das
// beobachtbare Verhalten, und die Referenzdaten stammen daher.

#include <cstring>

#include "quasar/core/unicode.hpp"
#include "tokenizer/components.hpp"

namespace quasar::tok {

namespace uc = quasar::unicode;

std::string base64_decode(std::string_view in, std::string_view where) {
  static const auto table = [] {
    std::array<signed char, 256> t{};
    t.fill(-1);
    const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) t[static_cast<unsigned char>(a[i])] = static_cast<signed char>(i);
    t[static_cast<unsigned char>('-')] = 62;  // URL-sichere Schreibweise
    t[static_cast<unsigned char>('_')] = 63;
    return t;
  }();

  std::string out;
  out.reserve(in.size() / 4 * 3);
  std::uint32_t acc = 0;
  int bits = 0;
  for (std::size_t i = 0; i < in.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(in[i]);
    if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
    const signed char v = table[c];
    if (v < 0) {
      throw_error(where, ": ungueltiges Zeichen '", static_cast<char>(c),
                  "' (Byte ", static_cast<int>(c), ") an Position ", i, " im base64-Blob");
    }
    acc = (acc << 6) | static_cast<std::uint32_t>(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<char>((acc >> bits) & 0xFF));
    }
  }
  return out;
}

namespace {

// Darts-clone Einheiten-Zugriffe (darts.h, Fassung 0.32).
inline bool unit_has_leaf(std::uint32_t u) { return ((u >> 8) & 1) == 1; }
inline std::uint32_t unit_value(std::uint32_t u) { return u & 0x7FFFFFFFu; }
inline std::uint32_t unit_label(std::uint32_t u) { return u & 0x800000FFu; }
inline std::uint32_t unit_offset(std::uint32_t u) {
  return (u >> 10) << ((u & 0x200) >> 6);
}

}  // namespace

Precompiled Precompiled::from_base64(std::string_view b64, std::string_view where) {
  const std::string blob = base64_decode(b64, where);
  if (blob.size() < 4) {
    throw_error(where, ": precompiled_charsmap ist nur ", blob.size(),
                " Byte gross -- zu klein fuer den Kopf");
  }
  std::uint32_t trie_bytes = 0;
  std::memcpy(&trie_bytes, blob.data(), 4);
  if (trie_bytes % 4 != 0 || static_cast<std::size_t>(trie_bytes) + 4 > blob.size()) {
    throw_error(where, ": precompiled_charsmap gibt ", trie_bytes,
                " Byte Trie an, hat aber nur ", blob.size() - 4, " Byte Nutzdaten");
  }

  Precompiled p;
  p.trie_.resize(trie_bytes / 4);
  std::memcpy(p.trie_.data(), blob.data() + 4, trie_bytes);
  p.normalized_.assign(blob.data() + 4 + trie_bytes, blob.size() - 4 - trie_bytes);
  return p;
}

// Achtung, hier steckt eine Eigenheit der Referenzbibliothek:
//
// SentencePiece selbst nimmt den **laengsten** Praefix-Treffer und ersetzt
// genau diesen. Die Referenzbibliothek nimmt den **ersten** (kuerzesten)
// Treffer und ersetzt damit das **ganze** uebergebene Stueck -- auch wenn der
// Treffer nur ein Teil davon war. Im Quelltext steht dazu sinngemaess, man
// wisse auch nicht warum, aber jede andere Fassung breche die Tests.
//
// Beobachtbare Folge: "\r\n" ist ein Graphem-Cluster, im Trie steht "\r",
// und heraus kommt **ein** Leerzeichen statt zweier. Genau das steht in den
// Referenzdaten (Fall `zeilenumbruch_crlf`). Wer hier den laengsten Treffer
// nimmt und volle Abdeckung verlangt, bekommt zwei Leerzeichen.
std::optional<std::string_view> Precompiled::lookup(std::string_view s) const {
  if (trie_.empty()) return std::nullopt;
  std::size_t node_pos = 0;
  std::uint32_t unit = trie_[node_pos];
  node_pos ^= unit_offset(unit);

  for (std::size_t i = 0; i < s.size(); ++i) {
    const auto key = static_cast<std::uint32_t>(static_cast<unsigned char>(s[i]));
    node_pos ^= key;
    if (node_pos >= trie_.size()) break;
    unit = trie_[node_pos];
    if (unit_label(unit) != key) break;
    node_pos ^= unit_offset(unit);
    if (node_pos >= trie_.size()) break;
    if (unit_has_leaf(unit)) return decode_at(unit_value(trie_[node_pos]));
  }
  return std::nullopt;
}

std::optional<std::string_view> Precompiled::decode_at(std::uint32_t offset) const {
  if (offset >= normalized_.size()) return std::nullopt;
  const std::size_t end = normalized_.find('\0', offset);
  const std::size_t stop = end == std::string::npos ? normalized_.size() : end;
  return std::string_view(normalized_).substr(offset, stop - offset);
}

std::string Precompiled::normalize(std::string_view s) const {
  if (s.empty()) return {};

  // Die Referenzbibliothek geht ueber erweiterte Graphem-Cluster: erst wird
  // der ganze Cluster nachgeschlagen (nur wenn er kuerzer als 6 Byte ist),
  // sonst jedes Zeichen einzeln.
  const auto bounds = uc::grapheme_boundaries(s);
  std::string out;
  out.reserve(s.size());

  for (std::size_t g = 0; g + 1 < bounds.size(); ++g) {
    const std::string_view cluster = s.substr(bounds[g], bounds[g + 1] - bounds[g]);

    if (cluster.size() < 6) {
      if (const auto r = lookup(cluster)) {
        out.append(*r);
        continue;
      }
    }
    for (std::size_t i = 0; i < cluster.size();) {
      const std::size_t n = utf8_len(cluster, i);
      const std::string_view ch = cluster.substr(i, n);
      if (const auto r = lookup(ch)) out.append(*r);
      else out.append(ch);
      i += n;
    }
  }
  return out;
}

}  // namespace quasar::tok

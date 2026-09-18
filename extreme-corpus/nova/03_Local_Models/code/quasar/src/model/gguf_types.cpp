// quasar — src/model/gguf_types.cpp
//
// Die beiden Tabellen des GGUF-Formats: Metadaten-Werttypen und ggml-Tensor-
// typen. Sie stehen getrennt vom Leser, weil sie reine Daten sind und weil
//
// Blockgroessen aus ggml-common.h. Wer hier eine Zahl aendert, aendert die
// berechnete Bytezahl jedes Tensors -- deshalb steht neben jeder Zeile, woraus
// sie sich zusammensetzt.

#include <array>

#include "quasar/core/error.hpp"
#include "quasar/model/gguf.hpp"

namespace quasar::gguf {
namespace {

using GgmlTypeRow = GgmlTypeEntry;

// QK_K = 256. `half` = 2 Byte.
//
// Befund S6 -- **die Widerlegung aus Runde 1 war falsch** (Auftrag C,
// Pruefrunde 2). Hier stand, die Sache sei "nicht eindeutig belegbar", weil das
// Python-Paket `gguf` 0.19.0 in `gguf/constants.py`
//
//     GGMLQuantizationType.Q8_1: (32, 4 + 4 + 32)      -> 40 Byte
//
// rechnet. Sie ist entscheidbar, und 36 ist die Zahl:
//
//   - `ggml-common.h` beschreibt `block_q8_1` als `ggml_half d; ggml_half s;
//     int8_t qs[QK8_1];` und sichert die Groesse mit
//     `static_assert(sizeof(block_q8_1) == 2*sizeof(ggml_half) + QK8_1)` = 36
//     ab -- im **ausgefuehrten** Code. Das ist die Struktur, mit der die
//     Referenzumsetzung tatsaechlich rechnet.
//   - Der Eintrag in `gguf/constants.py` stammt aus der Zeit vor der Umstellung
//     von `float` auf `ggml_half` und wird nie ausgefuehrt. Nachgeprueft am
//     Paket selbst:
//     Q8_1 hat in `gguf.quants` weder eine Quantisierung noch eine
//     Dequantisierung. Es ist toter Code, und deshalb ist der Fehler stehen
//     geblieben.
//
// Eine Zahl, die ein `static_assert` im ausgefuehrten Code absichert, schlaegt
// eine Zahl in einer Tabelle, die niemand anfasst.
//
// **Die Massnahme bleibt trotzdem**: Q8_1 ist aus der Tabelle genommen und steht
// in `kDisputedTypes`. Nicht weil die Zahl unklar waere, sondern weil der Typ in
// ausgelieferten GGUF-Dateien nicht vorkommt (er ist ein Zwischenformat fuer
// Skalarprodukte) und quasar ihn nie zu lesen hat. Ablehnen ist sicherer als
// eine Zahl zu fuehren, die kein Testfall je betritt -- und die Meldung nennt
// den Sachverhalt beim Namen, statt still 4 Byte des Nachbartensors mitzulesen.
constexpr std::array<GgmlTypeRow, 32> kGgmlTypes{{
    {0, {"F32", 1, 4}},
    {1, {"F16", 1, 2}},
    {2, {"Q4_0", 32, 18}},        // half d + 16 Byte Nibbles
    {3, {"Q4_1", 32, 20}},        // half d + half m + 16
    // 4 und 5 (Q4_2, Q4_3) sind aus ggml entfernt und fehlen absichtlich.
    {6, {"Q5_0", 32, 22}},        // half d + u32 qh + 16
    {7, {"Q5_1", 32, 24}},        // half d + half m + u32 qh + 16
    {8, {"Q8_0", 32, 34}},        // half d + 32
    // 9 (Q8_1) fehlt absichtlich -- siehe kDisputedTypes.
    {10, {"Q2_K", 256, 84}},      // 16 scales + 64 qs + 2 half
    {11, {"Q3_K", 256, 110}},     // 32 hmask + 64 qs + 12 scales + half
    {12, {"Q4_K", 256, 144}},     // 2 half + 12 scales + 128 qs
    {13, {"Q5_K", 256, 176}},     // 2 half + 12 scales + 32 qh + 128 qs
    {14, {"Q6_K", 256, 210}},     // 128 ql + 64 qh + 16 scales + half
    {15, {"Q8_K", 256, 292}},     // float d + 256 qs + 32 Byte bsums
    {16, {"IQ2_XXS", 256, 66}},
    {17, {"IQ2_XS", 256, 74}},
    {18, {"IQ3_XXS", 256, 98}},
    {19, {"IQ1_S", 256, 50}},
    {20, {"IQ4_NL", 32, 18}},
    {21, {"IQ3_S", 256, 110}},
    {22, {"IQ2_S", 256, 82}},
    {23, {"IQ4_XS", 256, 136}},
    {24, {"I8", 1, 1}},
    {25, {"I16", 1, 2}},
    {26, {"I32", 1, 4}},
    {27, {"I64", 1, 8}},
    {28, {"F64", 1, 8}},
    {29, {"IQ1_M", 256, 56}},
    {30, {"BF16", 1, 2}},
    // 31..33 (Q4_0_4_4/_4_8/_8_8) sind aus ggml entfernt und fehlen absichtlich.
    {34, {"TQ1_0", 256, 54}},     // 48 qs + 4 qh + half
    {35, {"TQ2_0", 256, 66}},     // 64 qs + half
}};

// Typen, deren Blockgroesse zwischen den Quellen strittig ist. Sie werden
// entweder "heile Datei gilt als beschaedigt" oder -- schlimmer -- "es werden
// stillschweigend Bytes des Nachbartensors gelesen".
struct DisputedType {
  std::uint32_t id;
  const char* name;
  const char* why;
};
constexpr std::array<DisputedType, 1> kDisputedTypes{{
    {9, "Q8_1",
     "ggml-common.h sichert `sizeof(block_q8_1) == 2*sizeof(ggml_half) + QK8_1` = 36 "
     "Byte mit einem static_assert ab; die 40 Byte (4+4+32) aus gguf/constants.py sind "
     "ein toter Rest aus der Zeit vor ggml_half und werden nie ausgefuehrt. Die Zahl "
     "ist damit 36. quasar liest den Typ trotzdem nicht: Q8_1 ist ein Zwischenformat "
     "fuer Skalarprodukte und kommt in ausgelieferten GGUF-Dateien nicht vor, also "
     "betritt kein Testfall diesen Zweig"},
}};

}  // namespace

std::span<const GgmlTypeEntry> all_ggml_types() noexcept {
  return std::span<const GgmlTypeEntry>(kGgmlTypes.data(), kGgmlTypes.size());
}

const GgmlTypeTraits* find_ggml_type(std::uint32_t id) noexcept {
  for (const GgmlTypeRow& row : kGgmlTypes) {
    if (row.id == id) return &row.traits;
  }
  return nullptr;
}

const GgmlTypeTraits& require_ggml_type(std::uint32_t id, std::string_view what) {
  const GgmlTypeTraits* traits = find_ggml_type(id);
  if (!traits) {
    for (const DisputedType& row : kDisputedTypes) {
      if (row.id != id) continue;
      throw_error("GGUF: der Tensortyp ", id, " (", row.name, ") bei ", what,
                  " hat keine eindeutige Blockgroesse: ", row.why,
                  ". quasar liest ihn deshalb nicht, statt eine der beiden Zahlen zu "
                  "raten -- eine falsche Blockgroesse liest still Bytes des "
                  "Nachbartensors mit.");
    }
    throw_error("GGUF: unbekannter ggml-Tensortyp ", id, " bei ", what,
                ". quasar kennt die Typen 0..3, 6..8, 10..30, 34 und 35; die uebrigen "
                "Kennungen sind aus ggml entfernt, strittig oder neuer als dieser "
                "Leser. Kein Raten -- der Tensor waere sonst falsch gross.");
  }
  return *traits;
}

std::span<const std::uint32_t> disputed_ggml_types() noexcept {
  static const std::array<std::uint32_t, kDisputedTypes.size()> ids = [] {
    std::array<std::uint32_t, kDisputedTypes.size()> out{};
    for (std::size_t i = 0; i < kDisputedTypes.size(); ++i) out[i] = kDisputedTypes[i].id;
    return out;
  }();
  return {ids.data(), ids.size()};
}

std::string_view ggml_type_name(GgmlType type) noexcept {
  const GgmlTypeTraits* traits = find_ggml_type(static_cast<std::uint32_t>(type));
  return traits ? traits->name : std::string_view{"?"};
}

std::string_view value_type_name(ValueType type) noexcept {
  switch (type) {
    case ValueType::UInt8: return "u8";
    case ValueType::Int8: return "i8";
    case ValueType::UInt16: return "u16";
    case ValueType::Int16: return "i16";
    case ValueType::UInt32: return "u32";
    case ValueType::Int32: return "i32";
    case ValueType::Float32: return "f32";
    case ValueType::Bool: return "bool";
    case ValueType::String: return "string";
    case ValueType::Array: return "array";
    case ValueType::UInt64: return "u64";
    case ValueType::Int64: return "i64";
    case ValueType::Float64: return "f64";
  }
  return "?";
}

std::uint64_t scalar_type_size(ValueType type) noexcept {
  switch (type) {
    case ValueType::UInt8:
    case ValueType::Int8:
    case ValueType::Bool: return 1;
    case ValueType::UInt16:
    case ValueType::Int16: return 2;
    case ValueType::UInt32:
    case ValueType::Int32:
    case ValueType::Float32: return 4;
    case ValueType::UInt64:
    case ValueType::Int64:
    case ValueType::Float64: return 8;
    case ValueType::String:
    case ValueType::Array: return 0;
  }
  return 0;
}

}  // namespace quasar::gguf

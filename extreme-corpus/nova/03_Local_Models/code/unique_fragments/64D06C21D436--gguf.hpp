#pragma once
//
// quasar — model/gguf.hpp
//
// GGUF-Leser ueber MappedFile. SPEC Abschnitt 3: "`model/` — GGUF-Leser
// (`mmap`), HF-safetensors-Leser, Metadaten, Tensor-Verzeichnis".
//
// Der Leser kopiert nichts. Zeichenketten und Arrays bleiben in der gemappten
// Datei liegen; herausgereicht werden `std::string_view` und Spannen. Das
// Vokabular eines 24B-Modells hat 131 072 Eintraege und die Merge-Liste
// 269 443 — die einzeln in `std::string` zu kopieren waere reine Verschwendung.
//
// SPEC Abschnitt 10, "Kein stiller Rueckfall": jede Laenge und jeder Offset
// wird gegen die Dateigroesse geprueft, ueberlaufsicher. Version 1 wird nicht
// geraten, sondern abgelehnt (dort sind die Laengenfelder u32 statt u64).

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "quasar/core/mapped_file.hpp"

namespace quasar::gguf {

// ---------------------------------------------------------------------------
// Werttypen der Metadaten (gguf.md, 13 Stueck)
// ---------------------------------------------------------------------------
enum class ValueType : std::uint32_t {
  UInt8 = 0,
  Int8 = 1,
  UInt16 = 2,
  Int16 = 3,
  UInt32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  UInt64 = 10,
  Int64 = 11,
  Float64 = 12,
};

inline constexpr std::uint32_t kValueTypeCount = 13;

// Name des Werttyps. Wirft bei unbekanntem Typ nicht -- gibt "?" zurueck;
// abgelehnt wird beim Parsen, nicht beim Benennen.
std::string_view value_type_name(ValueType type) noexcept;

// Bytes eines Skalartyps. 0 fuer String und Array (nicht fest).
std::uint64_t scalar_type_size(ValueType type) noexcept;

// ---------------------------------------------------------------------------
// Tensortypen (ggml)
// ---------------------------------------------------------------------------
enum class GgmlType : std::uint32_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  IQ2_XXS = 16,
  IQ2_XS = 17,
  IQ3_XXS = 18,
  IQ1_S = 19,
  IQ4_NL = 20,
  IQ3_S = 21,
  IQ2_S = 22,
  IQ4_XS = 23,
  I8 = 24,
  I16 = 25,
  I32 = 26,
  I64 = 27,
  F64 = 28,
  IQ1_M = 29,
  BF16 = 30,
  TQ1_0 = 34,
  TQ2_0 = 35,
};

struct GgmlTypeTraits {
  std::string_view name;
  std::uint64_t block_elements;  // Elemente je Block
  std::uint64_t block_bytes;     // Bytes je Block
};

// Eine Zeile der Tabelle samt Kennung.
struct GgmlTypeEntry {
  std::uint32_t id;
  GgmlTypeTraits traits;
};

// nullptr, wenn die Kennung nicht in der Tabelle steht.
const GgmlTypeTraits* find_ggml_type(std::uint32_t id) noexcept;

// Die vollstaendige Tabelle. `model.gguf` braucht sie, um die Pruefung gegen die
// Referenz in **beide** Richtungen zu fuehren: bis zur zweiten Pruefrunde lief
// sie nur ueber die Typen der Referenz, sodass ein Eintrag in quasars Tabelle,
// den die Referenz nicht kennt, nie verglichen wurde (Auftrag C, Pruefrunde 2).
std::span<const GgmlTypeEntry> all_ggml_types() noexcept;

// Wie oben, bricht aber ab statt nullptr zu liefern.
const GgmlTypeTraits& require_ggml_type(std::uint32_t id, std::string_view what);

// Kennungen, deren Blockgroesse zwischen den Quellen strittig ist und die quasar
// deshalb bewusst **nicht** liest (Befund S6, derzeit nur Q8_1). Sie stehen in
// keiner Tabelle; `require_ggml_type` nennt den Widerspruch beim Namen.
// `model.gguf` benutzt diese Liste, um die Referenztabelle aus dem Python-Paket
// `gguf` richtig gegen quasars Tabelle zu halten.
std::span<const std::uint32_t> disputed_ggml_types() noexcept;

std::string_view ggml_type_name(GgmlType type) noexcept;

// ---------------------------------------------------------------------------
// Ein Metadaten-Wert
// ---------------------------------------------------------------------------
// Skalare werden beim Parsen ausgelesen. Zeichenketten und Arrays bleiben in
// der Datei; der Wert merkt sich nur, wo sie stehen.
class MetaValue {
 public:
  MetaValue() = default;

  ValueType type() const noexcept { return raw_.type; }
  bool is_array() const noexcept { return raw_.type == ValueType::Array; }

  // --- Skalare ------------------------------------------------------------
  // `what` erscheint in der Fehlermeldung, wenn der Typ nicht passt.
  bool as_bool(std::string_view what) const;
  std::uint64_t as_u64(std::string_view what) const;
  std::int64_t as_i64(std::string_view what) const;
  double as_double(std::string_view what) const;
  float as_float(std::string_view what) const;
  std::string_view as_string(std::string_view what) const;

  // --- Arrays -------------------------------------------------------------
  ValueType element_type(std::string_view what) const;
  std::uint64_t size(std::string_view what) const;  // Elementzahl

  // Zeichenketten-Array: Sicht in die gemappte Datei, ohne Kopie.
  std::string_view string_at(std::uint64_t index, std::string_view what) const;
  void read_strings(std::vector<std::string_view>& out, std::string_view what) const;

  // Zahlen-Array: fuellt den Zielvektor direkt, ein Durchgang, keine Boxen.
  void read_numbers(std::vector<float>& out, std::string_view what) const;
  void read_numbers(std::vector<double>& out, std::string_view what) const;
  void read_numbers(std::vector<std::int32_t>& out, std::string_view what) const;
  void read_numbers(std::vector<std::uint32_t>& out, std::string_view what) const;
  void read_numbers(std::vector<std::int64_t>& out, std::string_view what) const;
  void read_numbers(std::vector<std::uint64_t>& out, std::string_view what) const;

  // Verschachteltes Array (Array von Arrays).
  const MetaValue& element_at(std::uint64_t index, std::string_view what) const;

  // Kompakte Darstellung fuer describe() und Fehlermeldungen. Lange Arrays
  // werden als "typ[n]" mit den ersten Eintraegen abgekuerzt.
  std::string to_text(std::uint64_t max_elements = 8) const;

  // --- nur fuer den Leser --------------------------------------------------
  struct Raw {
    ValueType type = ValueType::UInt8;
    ValueType element_type = ValueType::UInt8;
    const std::byte* base = nullptr;   // Dateianfang
    std::uint64_t scalar = 0;          // Bitmuster des Skalars
    std::uint64_t offset = 0;          // String: Bytes; Array: erstes Element
    std::uint64_t length = 0;          // String: Bytes;  Array: Elementzahl
    std::shared_ptr<const std::vector<std::uint64_t>> string_offsets;
    std::shared_ptr<const std::vector<MetaValue>> children;
  };
  explicit MetaValue(Raw raw) : raw_(std::move(raw)) {}

 private:
  template <typename T>
  void fill_numbers(std::vector<T>& out, std::string_view what) const;

  void require_array(std::string_view what) const;
  std::string_view read_string_at(std::uint64_t byte_offset) const;

  Raw raw_;
};

// ---------------------------------------------------------------------------
// Metadaten-Tabelle: Einfuegereihenfolge bleibt erhalten
// ---------------------------------------------------------------------------
class Metadata {
 public:
  struct Entry {
    std::string key;
    MetaValue value;
  };

  void add(std::string key, MetaValue value);

  const std::vector<Entry>& entries() const noexcept { return entries_; }
  std::size_t size() const noexcept { return entries_.size(); }
  const MetaValue* find(std::string_view key) const;
  const MetaValue& require(std::string_view key, std::string_view origin) const;
  bool contains(std::string_view key) const { return find(key) != nullptr; }

 private:
  std::vector<Entry> entries_;
  std::map<std::string, std::size_t, std::less<>> index_;
};

// ---------------------------------------------------------------------------
// Ein Tensor im Verzeichnis
// ---------------------------------------------------------------------------
struct TensorInfo {
  std::string name;
  std::vector<std::uint64_t> dims;  // ggml-Reihenfolge (schnellste Achse zuerst)
  GgmlType type = GgmlType::F32;
  std::uint64_t offset_in_data = 0;  // relativ zum Datenanfang
  std::uint64_t n_elements = 0;
  std::uint64_t n_bytes = 0;

  std::string_view type_name() const noexcept { return ggml_type_name(type); }
  std::string describe() const;
};

// ---------------------------------------------------------------------------
// Die Datei
// ---------------------------------------------------------------------------
class GgufFile {
 public:
  GgufFile() = default;
  GgufFile(const GgufFile&) = delete;
  GgufFile& operator=(const GgufFile&) = delete;
  GgufFile(GgufFile&&) = default;
  GgufFile& operator=(GgufFile&&) = default;

  // Oeffnet, mappt und liest den gesamten Kopf. Wirft quasar::Error bei jeder
  // Unstimmigkeit -- es gibt keinen halbgelesenen Zustand.
  static GgufFile open(const std::string& path);

  std::uint32_t version() const noexcept { return version_; }
  std::uint64_t alignment() const noexcept { return alignment_; }
  std::uint64_t data_offset() const noexcept { return data_offset_; }
  std::uint64_t header_end() const noexcept { return header_end_; }
  std::uint64_t file_size() const noexcept { return file_.size(); }
  const std::string& path() const noexcept { return path_; }

  const Metadata& metadata() const noexcept { return metadata_; }
  const std::vector<TensorInfo>& tensors() const noexcept { return tensors_; }

  const TensorInfo* find_tensor(std::string_view name) const;
  const TensorInfo& require_tensor(std::string_view name) const;
  std::span<const std::byte> tensor_bytes(const TensorInfo& info) const;

  // --- typisierter Metadaten-Zugriff --------------------------------------
  // get_*     : fehlt der Schluessel -> leer. Falscher Typ -> Abbruch.
  // get_*_or  : fehlt der Schluessel -> Vorgabe.
  // require_* : fehlt der Schluessel -> Abbruch mit Schluesselnamen.
  std::optional<std::uint32_t> get_u32(std::string_view key) const;
  std::optional<std::uint64_t> get_u64(std::string_view key) const;
  std::optional<std::int64_t> get_i64(std::string_view key) const;
  std::optional<float> get_f32(std::string_view key) const;
  std::optional<bool> get_bool(std::string_view key) const;
  std::optional<std::string_view> get_string(std::string_view key) const;

  std::uint32_t get_u32_or(std::string_view key, std::uint32_t fallback) const;
  std::uint64_t get_u64_or(std::string_view key, std::uint64_t fallback) const;
  std::int64_t get_i64_or(std::string_view key, std::int64_t fallback) const;
  float get_f32_or(std::string_view key, float fallback) const;
  bool get_bool_or(std::string_view key, bool fallback) const;
  std::string_view get_string_or(std::string_view key, std::string_view fallback) const;

  std::uint32_t require_u32(std::string_view key) const;
  std::uint64_t require_u64(std::string_view key) const;
  std::int64_t require_i64(std::string_view key) const;
  float require_f32(std::string_view key) const;
  bool require_bool(std::string_view key) const;
  std::string_view require_string(std::string_view key) const;

  // general.architecture -- Pflichtfeld, deshalb eigener Zugriff.
  std::string_view architecture() const;

  // Kompakte Textbeschreibung fuer `quasar-convert --info`.
  std::string describe() const;

 private:
  void parse();

  MappedFile file_;
  std::string path_;
  std::uint32_t version_ = 0;
  std::uint64_t alignment_ = 32;
  std::uint64_t header_end_ = 0;  // erstes Byte nach dem Tensor-Verzeichnis
  std::uint64_t data_offset_ = 0;
  Metadata metadata_;
  std::vector<TensorInfo> tensors_;
  std::map<std::string, std::size_t, std::less<>> tensor_index_;
};

}  // namespace quasar::gguf

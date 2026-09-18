// quasar — src/model/gguf.cpp
//
// Der GGUF-Leser. Er liest den Kopf vollstaendig beim Oeffnen und kopiert
// dabei nichts ausser den Namen (Metadaten-Schluessel, Tensornamen).
//
// uebernommen, ohne gegen die Dateigroesse geprueft zu sein, und zwar
// ueberlaufsicher (erst `x > groesse - y` pruefen, nie `x + y > groesse`
// rechnen). Eine Datei, die einen Tensor ausserhalb ihrer selbst beschreibt,
// bricht ab -- sie stuerzt nicht ab und liefert erst recht nichts.

#include "quasar/model/gguf.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>

#include "gguf_detail.hpp"
#include "quasar/core/error.hpp"
#include "quasar/core/unicode.hpp"

namespace quasar::gguf {
namespace {

constexpr std::uint64_t kMagic = 0x46554747ULL;  // 'G','G','U','F' little-endian
constexpr int kMaxArrayDepth = 32;
constexpr std::uint32_t kMaxTensorDims = 4;  // GGML_MAX_DIMS

// Kleinstmoegliche Byteszahl eines Eintrags. Nur dafuer da, absurde Zaehler
// aus einer kaputten Datei abzufangen, bevor irgendetwas reserviert wird.
constexpr std::uint64_t kMinKvBytes = 8 + 4;       // Schluessellaenge + Typ
constexpr std::uint64_t kMinTensorBytes = 8 + 4 + 4 + 8;  // Name, n_dims, Typ, Offset

using detail::load_le;

// Laufender Zeiger mit Grenzpruefung. Jede Bewegung geht hierdurch.
class Cursor {
 public:
  Cursor(const std::byte* base, std::uint64_t size, std::string_view path)
      : base_(base), size_(size), path_(path) {}

  std::uint64_t pos() const noexcept { return pos_; }
  const std::byte* base() const noexcept { return base_; }

  void need(std::uint64_t n, std::string_view what) const {
    if (n > size_ - pos_) {
      throw_error("GGUF ", path_, " ist abgeschnitten: ", what, " braucht ", n,
                  " Byte ab Offset ", pos_, ", die Datei hat aber nur ", size_,
                  " Byte (es fehlen ", n - (size_ - pos_), ").");
    }
  }

  const std::byte* take(std::uint64_t n, std::string_view what) {
    need(n, what);
    const std::byte* p = base_ + pos_;
    pos_ += n;
    return p;
  }

  void skip(std::uint64_t n, std::string_view what) {
    need(n, what);
    pos_ += n;
  }

  std::uint32_t u32(std::string_view what) {
    return static_cast<std::uint32_t>(load_le(take(4, what), 4));
  }
  std::uint64_t u64(std::string_view what) {
    return load_le(take(8, what), 8);
  }

  // GGUF-Zeichenkette: u64 Laenge, dann die Bytes. Rueckgabe zeigt in die
  // gemappte Datei -- nichts wird kopiert.
  std::string_view string(std::string_view what) {
    const std::uint64_t len = u64(what);
    need(len, what);
    const char* p = reinterpret_cast<const char*>(base_ + pos_);
    pos_ += len;
    const std::string_view text(p, static_cast<std::size_t>(len));
    // Befund G21: ungueltiges UTF-8 wurde still angenommen. Die als Gegenprobe
    // gedachte Python-Referenz dekodiert mit errors="strict" und wuerde werfen --
    // Leser und Referenz waren an dieser Stelle also nicht deckungsgleich, und
    // die Abweichung faellt erst auf, wenn sie beide dieselbe Datei sehen.
    // GGUF schreibt UTF-8 vor; was das nicht ist, wird nicht ersatzweise anders
    if (!unicode::is_valid_utf8(text)) {
      throw_error("GGUF: ", what, " ist keine gueltige UTF-8-Zeichenkette (", len,
                  " Byte). GGUF schreibt UTF-8 vor -- quasar deutet die Bytes nicht "
                  "ersatzweise anders.");
    }
    return text;
  }

  std::uint64_t remaining() const noexcept { return size_ - pos_; }

 private:
  const std::byte* base_;
  std::uint64_t size_;
  std::string_view path_;
  std::uint64_t pos_ = 0;
};

ValueType check_value_type(std::uint32_t raw, std::string_view what,
                           std::string_view path) {
  if (raw >= kValueTypeCount) {
    throw_error("GGUF ", path, ": unbekannter GGUF-Werttyp ", raw, " bei ", what,
                ". Gueltig sind 0..12.");
  }
  return static_cast<ValueType>(raw);
}

MetaValue parse_value(Cursor& cursor, ValueType type, int depth,
                      std::string_view what, std::string_view path);

MetaValue parse_array(Cursor& cursor, int depth, std::string_view what,
                      std::string_view path) {
  const ValueType element = check_value_type(cursor.u32(what), what, path);
  const std::uint64_t count = cursor.u64(what);

  MetaValue::Raw raw;
  raw.type = ValueType::Array;
  raw.element_type = element;
  raw.base = cursor.base();
  raw.length = count;

  if (element == ValueType::Array) {
    if (depth + 1 >= kMaxArrayDepth) {
      throw_error("GGUF ", path, ": Array-Verschachtelung tiefer als ",
                  kMaxArrayDepth, " bei ", what, " -- abgelehnt.");
    }
    // Jedes Element kostet mindestens Typ (4) + Zahl (8).
    if (count > cursor.remaining() / 12) {
      throw_error("GGUF ", path, ": Array ", what, " gibt ", count,
                  " verschachtelte Arrays an, dafuer reichen die restlichen ",
                  cursor.remaining(), " Byte nicht.");
    }
    auto children = std::make_shared<std::vector<MetaValue>>();
    children->reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
      children->push_back(parse_value(cursor, ValueType::Array, depth + 1, what, path));
    }
    raw.children = children;
    raw.offset = 0;
    return MetaValue(std::move(raw));
  }

  if (element == ValueType::String) {
    if (count > cursor.remaining() / 8) {
      throw_error("GGUF ", path, ": Array ", what, " gibt ", count,
                  " Zeichenketten an, dafuer reichen die restlichen ",
                  cursor.remaining(), " Byte nicht (je 8 Byte Laenge mindestens).");
    }
    auto offsets = std::make_shared<std::vector<std::uint64_t>>();
    offsets->reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
      offsets->push_back(cursor.pos());
      (void)cursor.string(what);
    }
    raw.string_offsets = offsets;
    raw.offset = offsets->empty() ? cursor.pos() : offsets->front();
    return MetaValue(std::move(raw));
  }

  const std::uint64_t element_size = scalar_type_size(element);
  if (element_size == 0) {
    throw_error("GGUF ", path, ": Array ", what, " hat den Elementtyp ",
                value_type_name(element), ", der keine feste Groesse hat.");
  }
  if (count > cursor.remaining() / element_size) {
    throw_error("GGUF ", path, ": Array ", what, " gibt ", count, " Elemente vom Typ ",
                value_type_name(element), " an (", count * element_size,
                " Byte), die Datei hat ab Offset ", cursor.pos(), " aber nur noch ",
                cursor.remaining(), " Byte.");
  }
  raw.offset = cursor.pos();
  cursor.skip(count * element_size, what);
  return MetaValue(std::move(raw));
}

MetaValue parse_value(Cursor& cursor, ValueType type, int depth,
                      std::string_view what, std::string_view path) {
  if (type == ValueType::Array) return parse_array(cursor, depth, what, path);

  MetaValue::Raw raw;
  raw.type = type;
  raw.base = cursor.base();

  if (type == ValueType::String) {
    const std::uint64_t len_at = cursor.pos();
    const std::string_view s = cursor.string(what);
    raw.offset = len_at + 8;
    raw.length = s.size();
    return MetaValue(std::move(raw));
  }

  const std::uint64_t size = scalar_type_size(type);
  raw.scalar = load_le(cursor.take(size, what), static_cast<std::size_t>(size));
  return MetaValue(std::move(raw));
}

}  // namespace

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

void Metadata::add(std::string key, MetaValue value) {
  const auto it = index_.find(key);
  if (it != index_.end()) {
    throw_error("GGUF: der Metadaten-Schluessel '", key, "' kommt zweimal vor.");
  }
  index_.emplace(key, entries_.size());
  entries_.push_back(Entry{std::move(key), std::move(value)});
}

const MetaValue* Metadata::find(std::string_view key) const {
  const auto it = index_.find(key);
  if (it == index_.end()) return nullptr;
  return &entries_[it->second].value;
}

const MetaValue& Metadata::require(std::string_view key, std::string_view origin) const {
  const MetaValue* value = find(key);
  if (!value) {
    throw_error("GGUF ", origin, ": der Pflicht-Metadatenschluessel '", key,
                "' fehlt. Kein Ersatzwert -- ohne ihn ist das Modell nicht "
                "beschrieben.");
  }
  return *value;
}

// ---------------------------------------------------------------------------
// TensorInfo
// ---------------------------------------------------------------------------

std::string TensorInfo::describe() const {
  std::string out = name;
  out.append(" [");
  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (i) out.append("x");
    out.append(std::to_string(dims[i]));
  }
  out.append("] ");
  out.append(type_name());
  out.append(" off=");
  out.append(std::to_string(offset_in_data));
  out.append(" bytes=");
  out.append(std::to_string(n_bytes));
  return out;
}

// ---------------------------------------------------------------------------
// GgufFile
// ---------------------------------------------------------------------------

GgufFile GgufFile::open(const std::string& path) {
  GgufFile out;
  out.path_ = path;
  out.file_.open(path);
  out.parse();
  return out;
}

void GgufFile::parse() {
  const std::uint64_t size = file_.size();
  Cursor cursor(file_.data(), size, path_);

  const std::uint64_t magic = load_le(cursor.take(4, "Magic"), 4);
  if (magic != kMagic) {
    throw_error("Keine GGUF-Datei: ", path_, " beginnt mit ",
                hex_bytes(file_.data(), std::min<std::size_t>(4, file_.size())),
                " statt 'GGUF' (47 47 55 46).");
  }

  version_ = cursor.u32("Version");
  if (version_ == 1) {
    throw_error("GGUF ", path_, ": Version 1 wird nicht unterstuetzt. In Version 1 "
                "sind Laengen- und Zaehlerfelder u32 statt u64; ein Leser fuer "
                "Version 2/3 wuerde die Datei falsch zerlegen. quasar raet hier "
                "nicht -- bitte mit einem aktuellen Werkzeug neu konvertieren.");
  }
  if (version_ != 2 && version_ != 3) {
    throw_error("GGUF ", path_, ": Version ", version_, " wird nicht unterstuetzt. "
                "quasar liest Version 2 und 3.");
  }

  const std::uint64_t tensor_count = cursor.u64("Tensorzahl");
  const std::uint64_t kv_count = cursor.u64("Metadatenzahl");
  if (kv_count > cursor.remaining() / kMinKvBytes) {
    throw_error("GGUF ", path_, ": der Kopf gibt ", kv_count, " Metadaten-Eintraege an, "
                "dafuer reichen die restlichen ", cursor.remaining(), " Byte nicht.");
  }
  if (tensor_count > size / kMinTensorBytes) {
    throw_error("GGUF ", path_, ": der Kopf gibt ", tensor_count, " Tensoren an, dafuer "
                "reicht die Dateigroesse von ", size, " Byte nicht.");
  }

  // --- Metadaten ----------------------------------------------------------
  for (std::uint64_t i = 0; i < kv_count; ++i) {
    const std::string_view key = cursor.string("Metadaten-Schluessel");
    const std::string key_text(key);
    const ValueType type =
        check_value_type(cursor.u32(key_text), key_text, path_);
    metadata_.add(key_text, parse_value(cursor, type, 0, key_text, path_));
  }

  // --- Tensor-Verzeichnis --------------------------------------------------
  tensors_.reserve(static_cast<std::size_t>(tensor_count));
  for (std::uint64_t i = 0; i < tensor_count; ++i) {
    TensorInfo info;
    info.name = std::string(cursor.string("Tensorname"));

    const std::uint32_t n_dims = cursor.u32(info.name);
    if (n_dims == 0 || n_dims > kMaxTensorDims) {
      throw_error("GGUF ", path_, ": Tensor '", info.name, "' gibt ", n_dims,
                  " Dimensionen an, erlaubt sind 1 bis ", kMaxTensorDims, ".");
    }
    info.dims.reserve(n_dims);
    std::uint64_t elements = 1;
    for (std::uint32_t d = 0; d < n_dims; ++d) {
      const std::uint64_t dim = cursor.u64(info.name);
      if (dim == 0) {
        throw_error("GGUF ", path_, ": Tensor '", info.name, "' hat die Dimension ", d,
                    " mit Laenge 0.");
      }
      if (dim > std::numeric_limits<std::uint64_t>::max() / elements) {
        throw_error("GGUF ", path_, ": Tensor '", info.name,
                    "': das Produkt der Dimensionen laeuft ueber.");
      }
      elements *= dim;
      info.dims.push_back(dim);
    }

    const std::uint32_t type_id = cursor.u32(info.name);
    const GgmlTypeTraits& traits = require_ggml_type(type_id, info.name);
    info.type = static_cast<GgmlType>(type_id);
    info.offset_in_data = cursor.u64(info.name);
    info.n_elements = elements;

    if (elements % traits.block_elements != 0) {
      throw_error("GGUF ", path_, ": Tensor '", info.name, "' hat ", elements,
                  " Elemente, das ist kein Vielfaches der Blockgroesse ",
                  traits.block_elements, " des Typs ", traits.name, ".");
    }
    const std::uint64_t blocks = elements / traits.block_elements;
    if (traits.block_bytes != 0 &&
        blocks > std::numeric_limits<std::uint64_t>::max() / traits.block_bytes) {
      throw_error("GGUF ", path_, ": Tensor '", info.name,
                  "': die Bytezahl laeuft ueber (", blocks, " Bloecke a ",
                  traits.block_bytes, " Byte).");
    }
    info.n_bytes = blocks * traits.block_bytes;

    if (tensor_index_.find(info.name) != tensor_index_.end()) {
      throw_error("GGUF ", path_, ": der Tensor '", info.name, "' kommt zweimal vor.");
    }
    tensor_index_.emplace(info.name, tensors_.size());
    tensors_.push_back(std::move(info));
  }

  header_end_ = cursor.pos();

  // --- Ausrichtung und Datenanfang ----------------------------------------
  alignment_ = 32;
  if (const MetaValue* value = metadata_.find("general.alignment")) {
    alignment_ = value->as_u64("general.alignment");
  }
  if (alignment_ == 0 || (alignment_ & (alignment_ - 1)) != 0) {
    throw_error("GGUF ", path_, ": general.alignment ist ", alignment_,
                ", das ist keine Zweierpotenz.");
  }
  if (alignment_ > size) {
    throw_error("GGUF ", path_, ": general.alignment ist ", alignment_,
                ", groesser als die Datei (", size, " Byte).");
  }
  data_offset_ = (header_end_ + alignment_ - 1) & ~(alignment_ - 1);
  if (data_offset_ > size) {
    throw_error("GGUF ", path_, ": der Datenanfang laege bei ", data_offset_,
                ", die Datei hat aber nur ", size, " Byte.");
  }

  // --- Tensorbereiche gegen die Dateigroesse ------------------------------
  const std::uint64_t data_size = size - data_offset_;
  for (const TensorInfo& info : tensors_) {
    if (info.offset_in_data % alignment_ != 0) {
      throw_error("GGUF ", path_, ": Tensor '", info.name, "' liegt bei Offset ",
                  info.offset_in_data, ", das ist nicht auf ", alignment_,
                  " ausgerichtet.");
    }
    if (info.offset_in_data > data_size || info.n_bytes > data_size - info.offset_in_data) {
      throw_error("GGUF ", path_, ": Tensor '", info.name, "' beschreibt den Bereich ",
                  info.offset_in_data, "..", info.offset_in_data + info.n_bytes,
                  " im Datenteil, der aber nur ", data_size,
                  " Byte gross ist (Datei ", size, " Byte, Datenanfang ", data_offset_,
                  "). Die Datei ist unvollstaendig oder beschaedigt.");
    }
  }

  // --- Ueberlappung --------------------------------------------------------
  // Befund S7: geprueft wurde bis hierher nur, ob **jeder einzelne** Tensor in
  // den Datenteil passt. Ob sich zwei denselben Bereich teilen, fragte niemand.
  // Belegt: ein GGUF, in dem `blk.1.attn_q.weight` den Offset 0 bekommt (also
  // den Bereich von `token_embd.weight`), lief ohne ein Wort durch -- der
  // Konverter quantisierte fuer Layer 1 die Embedding-Bytes und schriebe ein
  // stumm falsches Modell.
  //
  // `safetensors.cpp` macht genau diese Pruefung und nennt sie im Kopf selbst
  // "die, die kaputte Shard-Dateien tatsaechlich findet". Hier fehlte sie.
  //
  // Leere Tensoren (n_bytes == 0) koennen nicht ueberlappen und bleiben draussen.
  {
    std::vector<const TensorInfo*> ordered;
    ordered.reserve(tensors_.size());
    for (const TensorInfo& info : tensors_) {
      if (info.n_bytes > 0) ordered.push_back(&info);
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const TensorInfo* a, const TensorInfo* b) {
                return a->offset_in_data < b->offset_in_data;
              });
    for (std::size_t i = 1; i < ordered.size(); ++i) {
      const TensorInfo& prev = *ordered[i - 1];
      const TensorInfo& cur = *ordered[i];
      const std::uint64_t prev_end = prev.offset_in_data + prev.n_bytes;
      if (cur.offset_in_data < prev_end) {
        throw_error("GGUF ", path_, ": die Tensoren '", prev.name, "' (",
                    prev.offset_in_data, "..", prev_end, ") und '", cur.name, "' (",
                    cur.offset_in_data, "..", cur.offset_in_data + cur.n_bytes,
                    ") ueberlappen sich um ", prev_end - cur.offset_in_data,
                    " Byte. Die Datei ist beschaedigt -- quasar wuerde sonst fuer den "
                    "einen Tensor die Gewichte des anderen quantisieren.");
      }
    }
  }
}

const TensorInfo* GgufFile::find_tensor(std::string_view name) const {
  const auto it = tensor_index_.find(name);
  if (it == tensor_index_.end()) return nullptr;
  return &tensors_[it->second];
}

const TensorInfo& GgufFile::require_tensor(std::string_view name) const {
  const TensorInfo* info = find_tensor(name);
  if (!info) {
    throw_error("GGUF ", path_, ": der Tensor '", name, "' fehlt. Die Datei hat ",
                tensors_.size(), " Tensoren.");
  }
  return *info;
}

std::span<const std::byte> GgufFile::tensor_bytes(const TensorInfo& info) const {
  return file_.slice(data_offset_ + info.offset_in_data, info.n_bytes, info.name);
}

std::optional<std::uint32_t> GgufFile::get_u32(std::string_view key) const {
  const MetaValue* value = metadata_.find(key);
  if (!value) return std::nullopt;
  const std::uint64_t v = value->as_u64(key);
  if (v > std::numeric_limits<std::uint32_t>::max()) {
    throw_error("GGUF ", path_, ": '", key, "' ist ", v, " und passt nicht in u32.");
  }
  return static_cast<std::uint32_t>(v);
}

std::optional<std::uint64_t> GgufFile::get_u64(std::string_view key) const {
  const MetaValue* value = metadata_.find(key);
  if (!value) return std::nullopt;
  return value->as_u64(key);
}

std::optional<std::int64_t> GgufFile::get_i64(std::string_view key) const {
  const MetaValue* value = metadata_.find(key);
  if (!value) return std::nullopt;
  return value->as_i64(key);
}

std::optional<float> GgufFile::get_f32(std::string_view key) const {
  const MetaValue* value = metadata_.find(key);
  if (!value) return std::nullopt;
  return value->as_float(key);
}

std::optional<bool> GgufFile::get_bool(std::string_view key) const {
  const MetaValue* value = metadata_.find(key);
  if (!value) return std::nullopt;
  return value->as_bool(key);
}

std::optional<std::string_view> GgufFile::get_string(std::string_view key) const {
  const MetaValue* value = metadata_.find(key);
  if (!value) return std::nullopt;
  return value->as_string(key);
}

std::uint32_t GgufFile::get_u32_or(std::string_view key, std::uint32_t fallback) const {
  return get_u32(key).value_or(fallback);
}
std::uint64_t GgufFile::get_u64_or(std::string_view key, std::uint64_t fallback) const {
  return get_u64(key).value_or(fallback);
}
std::int64_t GgufFile::get_i64_or(std::string_view key, std::int64_t fallback) const {
  return get_i64(key).value_or(fallback);
}
float GgufFile::get_f32_or(std::string_view key, float fallback) const {
  return get_f32(key).value_or(fallback);
}
bool GgufFile::get_bool_or(std::string_view key, bool fallback) const {
  return get_bool(key).value_or(fallback);
}
std::string_view GgufFile::get_string_or(std::string_view key,
                                         std::string_view fallback) const {
  return get_string(key).value_or(fallback);
}

std::uint32_t GgufFile::require_u32(std::string_view key) const {
  const std::uint64_t v = metadata_.require(key, path_).as_u64(key);
  if (v > std::numeric_limits<std::uint32_t>::max()) {
    throw_error("GGUF ", path_, ": '", key, "' ist ", v, " und passt nicht in u32.");
  }
  return static_cast<std::uint32_t>(v);
}
std::uint64_t GgufFile::require_u64(std::string_view key) const {
  return metadata_.require(key, path_).as_u64(key);
}
std::int64_t GgufFile::require_i64(std::string_view key) const {
  return metadata_.require(key, path_).as_i64(key);
}
float GgufFile::require_f32(std::string_view key) const {
  return metadata_.require(key, path_).as_float(key);
}
bool GgufFile::require_bool(std::string_view key) const {
  return metadata_.require(key, path_).as_bool(key);
}
std::string_view GgufFile::require_string(std::string_view key) const {
  return metadata_.require(key, path_).as_string(key);
}

std::string_view GgufFile::architecture() const {
  return require_string("general.architecture");
}

std::string GgufFile::describe() const {
  std::string out = "GGUF ";
  out.append(path_);
  out.append("\n  Version ");
  out.append(std::to_string(version_));
  out.append(", Ausrichtung ");
  out.append(std::to_string(alignment_));
  out.append(", Kopfende ");
  out.append(std::to_string(header_end_));
  out.append(", Datenanfang ");
  out.append(std::to_string(data_offset_));
  out.append(", Dateigroesse ");
  out.append(std::to_string(file_.size()));
  out.append("\n  Metadaten ");
  out.append(std::to_string(metadata_.size()));
  out.append(", Tensoren ");
  out.append(std::to_string(tensors_.size()));

  if (const MetaValue* arch = metadata_.find("general.architecture")) {
    out.append("\n  Architektur ");
    out.append(arch->as_string("general.architecture"));
  }

  // Verteilung nach Tensortyp -- die Kennzahl, die beim Konvertieren zaehlt.
  std::vector<std::pair<std::string_view, std::pair<std::uint64_t, std::uint64_t>>> by_type;
  for (const TensorInfo& info : tensors_) {
    const std::string_view name = info.type_name();
    auto it = std::find_if(by_type.begin(), by_type.end(),
                           [&](const auto& e) { return e.first == name; });
    if (it == by_type.end()) {
      by_type.push_back({name, {info.n_elements, info.n_bytes}});
    } else {
      it->second.first += info.n_elements;
      it->second.second += info.n_bytes;
    }
  }
  std::sort(by_type.begin(), by_type.end(),
            [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
  out.append("\n  Tensortypen:");
  for (const auto& entry : by_type) {
    out.append("\n    ");
    out.append(entry.first);
    out.append(" ");
    out.append(std::to_string(entry.second.first));
    out.append(" Elemente, ");
    out.append(std::to_string(entry.second.second));
    out.append(" Byte");
  }
  out.push_back('\n');
  return out;
}

}  // namespace quasar::gguf

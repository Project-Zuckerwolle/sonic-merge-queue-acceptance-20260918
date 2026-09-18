#pragma once
//
// quasar — model/safetensors.hpp
//
// HuggingFace-safetensors ueber MappedFile. Aufbau der Datei:
//
//   [u64 Kopflaenge][JSON-Kopf][Rohdaten]
//
// Der Kopf bildet Tensornamen auf {dtype, shape, data_offsets:[a,b]} ab, wobei
// a und b relativ zum Anfang des Datenteils gelten. Zusaetzlich darf der
// Schluessel "__metadata__" mit einem flachen Zeichenketten-Objekt vorkommen.
//
//   - Kopflaenge und Datenbereich liegen in der Datei,
//   - b >= a,
//   - (b - a) passt genau zu shape x Elementgroesse,
//   - keine zwei Tensoren ueberlappen.
// Die Ueberlappungspruefung ist die, die kaputte Shard-Dateien tatsaechlich
// findet -- die einzelnen Groessen stimmen dort meist.
//
// Ein Verzeichnis mit mehreren `*.safetensors` (mit oder ohne
// `model.safetensors.index.json`) wird als ein einziges logisches
// Tensor-Verzeichnis gelesen: `SafetensorsSet`.

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "quasar/core/json.hpp"
#include "quasar/core/mapped_file.hpp"

namespace quasar::safetensors {

// ---------------------------------------------------------------------------
// Datentypen
// ---------------------------------------------------------------------------
enum class DType : std::uint8_t {
  BOOL,
  U8,
  I8,
  F8_E5M2,
  F8_E4M3,
  I16,
  U16,
  F16,
  BF16,
  I32,
  U32,
  F32,
  I64,
  U64,
  F64,
};

struct DTypeTraits {
  std::string_view name;   // wie im JSON-Kopf
  std::uint64_t bytes;     // Bytes je Element
};

const DTypeTraits& dtype_traits(DType type) noexcept;
std::string_view dtype_name(DType type) noexcept;

// leer, wenn der Name nicht in der Tabelle steht.
std::optional<DType> find_dtype(std::string_view name) noexcept;

// ---------------------------------------------------------------------------
// Ein Tensor
// ---------------------------------------------------------------------------
struct TensorInfo {
  std::string name;
  DType dtype = DType::F32;
  std::vector<std::uint64_t> shape;  // Zeilen-Hauptordnung wie bei HuggingFace
  std::uint64_t begin = 0;           // Offset im Datenteil
  std::uint64_t end = 0;             // exklusiv
  std::uint64_t n_elements = 0;
  std::uint64_t n_bytes = 0;
  std::size_t shard = 0;  // Index der Datei innerhalb eines SafetensorsSet

  std::string_view type_name() const noexcept { return dtype_name(dtype); }
  std::string describe() const;
};

// ---------------------------------------------------------------------------
// Eine Datei
// ---------------------------------------------------------------------------
class SafetensorsFile {
 public:
  SafetensorsFile() = default;
  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;
  SafetensorsFile(SafetensorsFile&&) = default;
  SafetensorsFile& operator=(SafetensorsFile&&) = default;

  static SafetensorsFile open(const std::string& path);

  const std::string& path() const noexcept { return path_; }
  std::uint64_t file_size() const noexcept { return file_.size(); }
  std::uint64_t header_length() const noexcept { return header_length_; }
  std::uint64_t data_offset() const noexcept { return data_offset_; }
  std::uint64_t data_size() const noexcept { return file_.size() - data_offset_; }

  const std::vector<TensorInfo>& tensors() const noexcept { return tensors_; }
  const TensorInfo* find_tensor(std::string_view name) const;
  const TensorInfo& require_tensor(std::string_view name) const;
  std::span<const std::byte> tensor_bytes(const TensorInfo& info) const;

  // Inhalt von "__metadata__". Leeres Objekt, wenn der Schluessel fehlt.
  const json::Object& file_metadata() const noexcept { return file_metadata_; }

  std::string describe() const;

 private:
  void parse();

  MappedFile file_;
  std::string path_;
  std::uint64_t header_length_ = 0;
  std::uint64_t data_offset_ = 0;
  std::vector<TensorInfo> tensors_;
  std::map<std::string, std::size_t, std::less<>> index_;
  json::Object file_metadata_;
};

// ---------------------------------------------------------------------------
// Ein Verzeichnis: mehrere Shards als ein Verzeichnis
// ---------------------------------------------------------------------------
class SafetensorsSet {
 public:
  SafetensorsSet() = default;
  SafetensorsSet(const SafetensorsSet&) = delete;
  SafetensorsSet& operator=(const SafetensorsSet&) = delete;
  SafetensorsSet(SafetensorsSet&&) = default;
  SafetensorsSet& operator=(SafetensorsSet&&) = default;

  // Verzeichnis mit `model.safetensors.index.json` (dann ist der Index
  // massgeblich) oder ohne (dann werden alle `*.safetensors` genommen).
  // Auch eine einzelne Datei ist zulaessig.
  static SafetensorsSet open(const std::string& path);

  const std::string& root() const noexcept { return root_; }
  std::size_t shard_count() const noexcept { return shards_.size(); }
  const SafetensorsFile& shard(std::size_t i) const;
  const std::string& index_path() const noexcept { return index_path_; }

  const std::vector<TensorInfo>& tensors() const noexcept { return tensors_; }
  const TensorInfo* find_tensor(std::string_view name) const;
  const TensorInfo& require_tensor(std::string_view name) const;
  std::span<const std::byte> tensor_bytes(const TensorInfo& info) const;

  std::string describe() const;

 private:
  void add_shard(const std::string& path);

  std::string root_;
  std::string index_path_;
  std::vector<std::unique_ptr<SafetensorsFile>> shards_;
  std::vector<TensorInfo> tensors_;
  std::map<std::string, std::size_t, std::less<>> index_;
};

}  // namespace quasar::safetensors

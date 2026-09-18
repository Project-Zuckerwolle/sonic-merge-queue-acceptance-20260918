// quasar — src/model/safetensors.cpp

#include "quasar/model/safetensors.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>

#include "quasar/core/error.hpp"
#include "quasar/core/utf8_path.hpp"

namespace quasar::safetensors {
namespace {

namespace fs = std::filesystem;
using quasar::core::path_to_utf8;

// Die safetensors-Spezifikation begrenzt den Kopf auf 100 MB. Groesser heisst:
// kaputte oder feindliche Datei, nicht "grosses Modell".
constexpr std::uint64_t kMaxHeaderBytes = 100ull * 1024 * 1024;

struct DTypeRow {
  DType type;
  DTypeTraits traits;
};

constexpr std::array<DTypeRow, 15> kDTypes{{
    {DType::BOOL, {"BOOL", 1}},
    {DType::U8, {"U8", 1}},
    {DType::I8, {"I8", 1}},
    {DType::F8_E5M2, {"F8_E5M2", 1}},
    {DType::F8_E4M3, {"F8_E4M3", 1}},
    {DType::I16, {"I16", 2}},
    {DType::U16, {"U16", 2}},
    {DType::F16, {"F16", 2}},
    {DType::BF16, {"BF16", 2}},
    {DType::I32, {"I32", 4}},
    {DType::U32, {"U32", 4}},
    {DType::F32, {"F32", 4}},
    {DType::I64, {"I64", 8}},
    {DType::U64, {"U64", 8}},
    {DType::F64, {"F64", 8}},
}};

std::string known_dtype_names() {
  std::string out;
  for (const DTypeRow& row : kDTypes) {
    if (!out.empty()) out.append(", ");
    out.append(row.traits.name);
  }
  return out;
}

std::uint64_t load_le64(const std::byte* p) noexcept {
  std::uint64_t v = 0;
  for (std::size_t i = 0; i < 8; ++i) {
    v |= static_cast<std::uint64_t>(static_cast<unsigned char>(p[i])) << (8 * i);
  }
  return v;
}

[[noreturn]] void throw_missing_field(std::string_view path, std::string_view tensor,
                                      std::string_view field) {
  throw_error("safetensors ", path, ": der Eintrag fuer Tensor '", tensor,
              "' hat kein Feld '", field, "'.");
}

bool has_suffix(std::string_view text, std::string_view suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

const DTypeTraits& dtype_traits(DType type) noexcept {
  for (const DTypeRow& row : kDTypes) {
    if (row.type == type) return row.traits;
  }
  return kDTypes[0].traits;  // nicht erreichbar: DType kommt nur aus der Tabelle
}

std::string_view dtype_name(DType type) noexcept { return dtype_traits(type).name; }

std::optional<DType> find_dtype(std::string_view name) noexcept {
  for (const DTypeRow& row : kDTypes) {
    if (row.traits.name == name) return row.type;
  }
  return std::nullopt;
}

std::string TensorInfo::describe() const {
  std::string out = name;
  out.append(" [");
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i) out.append("x");
    out.append(std::to_string(shape[i]));
  }
  out.append("] ");
  out.append(type_name());
  out.append(" ");
  out.append(std::to_string(begin));
  out.append("..");
  out.append(std::to_string(end));
  out.append(" (");
  out.append(std::to_string(n_bytes));
  out.append(" Byte)");
  return out;
}

// ---------------------------------------------------------------------------
// SafetensorsFile
// ---------------------------------------------------------------------------

SafetensorsFile SafetensorsFile::open(const std::string& path) {
  SafetensorsFile out;
  out.path_ = path;
  out.file_.open(path);
  out.parse();
  return out;
}

void SafetensorsFile::parse() {
  const std::uint64_t size = file_.size();
  if (size < 8) {
    throw_error("safetensors ", path_, ": die Datei hat nur ", size,
                " Byte, allein die Kopflaenge braucht 8.");
  }
  header_length_ = load_le64(file_.data());
  if (header_length_ == 0) {
    throw_error("safetensors ", path_, ": die Kopflaenge ist 0.");
  }
  if (header_length_ > size - 8) {
    throw_error("safetensors ", path_, ": die Kopflaenge ist ", header_length_,
                " Byte, die Datei hat nach dem Laengenfeld aber nur noch ", size - 8,
                " Byte.");
  }
  if (header_length_ > kMaxHeaderBytes) {
    throw_error("safetensors ", path_, ": die Kopflaenge ist ", header_length_,
                " Byte und damit groesser als die zulaessigen ", kMaxHeaderBytes,
                " Byte.");
  }
  data_offset_ = 8 + header_length_;
  const std::uint64_t data_size = size - data_offset_;

  const std::span<const std::byte> header =
      file_.slice(8, header_length_, "safetensors-Kopf");
  const json::Value root = json::parse(
      std::string_view(reinterpret_cast<const char*>(header.data()), header.size()),
      path_);
  const json::Object& object = root.as_object("safetensors-Kopf");

  for (const auto& item : object.items()) {
    if (item.first == "__metadata__") {
      file_metadata_ = item.second.as_object("__metadata__");
      continue;
    }

    TensorInfo info;
    info.name = item.first;
    const json::Object& entry = item.second.as_object(info.name);

    const json::Value* dtype_value = entry.find("dtype");
    if (!dtype_value) throw_missing_field(path_, info.name, "dtype");
    const std::string& dtype_text = dtype_value->as_string(info.name + ".dtype");
    const std::optional<DType> dtype = find_dtype(dtype_text);
    if (!dtype) {
      throw_error("safetensors ", path_, ": Tensor '", info.name,
                  "' hat den unbekannten dtype '", dtype_text, "'. Bekannt sind: ",
                  known_dtype_names(), ".");
    }
    info.dtype = *dtype;
    const std::uint64_t element_bytes = dtype_traits(info.dtype).bytes;

    const json::Value* shape_value = entry.find("shape");
    if (!shape_value) throw_missing_field(path_, info.name, "shape");
    const json::Array& shape = shape_value->as_array(info.name + ".shape");
    std::uint64_t elements = 1;
    info.shape.reserve(shape.size());
    for (std::size_t d = 0; d < shape.size(); ++d) {
      const std::int64_t dim = shape[d].as_int(info.name + ".shape");
      if (dim < 0) {
        throw_error("safetensors ", path_, ": Tensor '", info.name, "' hat in shape[",
                    d, "] die negative Laenge ", dim, ".");
      }
      const std::uint64_t udim = static_cast<std::uint64_t>(dim);
      if (udim != 0 && elements > std::numeric_limits<std::uint64_t>::max() / udim) {
        throw_error("safetensors ", path_, ": Tensor '", info.name,
                    "': das Produkt der Dimensionen laeuft ueber.");
      }
      elements *= udim;
      info.shape.push_back(udim);
    }
    info.n_elements = elements;
    if (elements != 0 &&
        elements > std::numeric_limits<std::uint64_t>::max() / element_bytes) {
      throw_error("safetensors ", path_, ": Tensor '", info.name,
                  "': die Bytezahl laeuft ueber.");
    }
    info.n_bytes = elements * element_bytes;

    const json::Value* offsets_value = entry.find("data_offsets");
    if (!offsets_value) throw_missing_field(path_, info.name, "data_offsets");
    const json::Array& offsets = offsets_value->as_array(info.name + ".data_offsets");
    if (offsets.size() != 2) {
      throw_error("safetensors ", path_, ": Tensor '", info.name, "' hat ",
                  offsets.size(), " data_offsets statt genau 2.");
    }
    const std::int64_t a = offsets[0].as_int(info.name + ".data_offsets[0]");
    const std::int64_t b = offsets[1].as_int(info.name + ".data_offsets[1]");
    if (a < 0 || b < 0) {
      throw_error("safetensors ", path_, ": Tensor '", info.name,
                  "' hat negative data_offsets [", a, ", ", b, "].");
    }
    info.begin = static_cast<std::uint64_t>(a);
    info.end = static_cast<std::uint64_t>(b);
    if (info.end < info.begin) {
      throw_error("safetensors ", path_, ": Tensor '", info.name,
                  "' hat data_offsets [", info.begin, ", ", info.end,
                  "] -- das Ende liegt vor dem Anfang.");
    }
    if (info.begin > data_size || info.end > data_size) {
      throw_error("safetensors ", path_, ": Tensor '", info.name,
                  "' beschreibt den Bereich ", info.begin, "..", info.end,
                  ", der Datenteil ist aber nur ", data_size,
                  " Byte gross (Datei ", size, " Byte, Datenanfang ", data_offset_,
                  ").");
    }
    if (info.end - info.begin != info.n_bytes) {
      throw_error("safetensors ", path_, ": Tensor '", info.name, "' belegt ",
                  info.end - info.begin, " Byte, seine Form (", info.describe(),
                  ") braucht aber ", info.n_bytes, " Byte.");
    }

    if (index_.find(info.name) != index_.end()) {
      throw_error("safetensors ", path_, ": der Tensor '", info.name,
                  "' kommt zweimal im Kopf vor.");
    }
    index_.emplace(info.name, tensors_.size());
    tensors_.push_back(std::move(info));
  }

  // --- Ueberlappung ---------------------------------------------------------
  // Leere Tensoren (begin == end) koennen nicht ueberlappen und stoeren die
  // Ordnung, deshalb bleiben sie draussen.
  std::vector<const TensorInfo*> ordered;
  ordered.reserve(tensors_.size());
  for (const TensorInfo& info : tensors_) {
    if (info.end > info.begin) ordered.push_back(&info);
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const TensorInfo* a, const TensorInfo* b) { return a->begin < b->begin; });
  for (std::size_t i = 1; i < ordered.size(); ++i) {
    const TensorInfo& prev = *ordered[i - 1];
    const TensorInfo& cur = *ordered[i];
    if (cur.begin < prev.end) {
      throw_error("safetensors ", path_, ": die Tensoren '", prev.name, "' (",
                  prev.begin, "..", prev.end, ") und '", cur.name, "' (", cur.begin,
                  "..", cur.end, ") ueberlappen sich um ", prev.end - cur.begin,
                  " Byte. Die Datei ist beschaedigt.");
    }
  }
}

const TensorInfo* SafetensorsFile::find_tensor(std::string_view name) const {
  const auto it = index_.find(name);
  if (it == index_.end()) return nullptr;
  return &tensors_[it->second];
}

const TensorInfo& SafetensorsFile::require_tensor(std::string_view name) const {
  const TensorInfo* info = find_tensor(name);
  if (!info) {
    throw_error("safetensors ", path_, ": der Tensor '", name, "' fehlt. Die Datei hat ",
                tensors_.size(), " Tensoren.");
  }
  return *info;
}

std::span<const std::byte> SafetensorsFile::tensor_bytes(const TensorInfo& info) const {
  return file_.slice(data_offset_ + info.begin, info.n_bytes, info.name);
}

std::string SafetensorsFile::describe() const {
  std::string out = "safetensors ";
  out.append(path_);
  out.append("\n  Kopflaenge ");
  out.append(std::to_string(header_length_));
  out.append(", Datenanfang ");
  out.append(std::to_string(data_offset_));
  out.append(", Dateigroesse ");
  out.append(std::to_string(file_.size()));
  out.append("\n  Tensoren ");
  out.append(std::to_string(tensors_.size()));
  out.append(", __metadata__-Eintraege ");
  out.append(std::to_string(file_metadata_.size()));
  out.push_back('\n');
  return out;
}

// ---------------------------------------------------------------------------
// SafetensorsSet
// ---------------------------------------------------------------------------

SafetensorsSet SafetensorsSet::open(const std::string& path) {
  SafetensorsSet out;
  out.root_ = path;

  const fs::path root = quasar::core::utf8_path(path);
  std::error_code ec;
  if (fs::is_regular_file(root, ec)) {
    out.add_shard(path);
    return out;
  }
  if (!fs::is_directory(root, ec)) {
    throw_error("safetensors: '", path,
                "' ist weder eine Datei noch ein Verzeichnis.");
  }

  const fs::path index = root / "model.safetensors.index.json";
  if (fs::is_regular_file(index, ec)) {
    out.index_path_ = path_to_utf8(index);
    const json::Value root_value = json::parse_file(out.index_path_);
    const json::Value& map_value =
        root_value.require("weight_map", "safetensors-Index");
    const json::Object& weight_map = map_value.as_object("weight_map");

    // Dateien in stabiler Reihenfolge oeffnen, damit die Shard-Indizes
    // reproduzierbar sind.
    std::vector<std::string> files;
    for (const auto& item : weight_map.items()) {
      const std::string& file = item.second.as_string("weight_map." + item.first);
      if (std::find(files.begin(), files.end(), file) == files.end()) {
        files.push_back(file);
      }
    }
    std::sort(files.begin(), files.end());
    for (const std::string& file : files) {
      const fs::path shard = root / quasar::core::utf8_path(file);
      if (!fs::is_regular_file(shard, ec)) {
        throw_error("safetensors-Index ", out.index_path_, ": die Datei '", file,
                    "' wird verwendet, existiert aber nicht: ", path_to_utf8(shard));
      }
      out.add_shard(path_to_utf8(shard));
    }

    // Der Index muss zu den Dateien passen -- in beide Richtungen.
    std::size_t mapped = 0;
    for (const auto& item : weight_map.items()) {
      const std::string& file = item.second.as_string("weight_map." + item.first);
      const TensorInfo* info = out.find_tensor(item.first);
      if (!info) {
        throw_error("safetensors-Index ", out.index_path_, ": der Tensor '", item.first,
                    "' steht im Index, aber in keiner der Dateien.");
      }
      const std::string& actual = out.shards_[info->shard]->path();
      if (!has_suffix(actual, file)) {
        throw_error("safetensors-Index ", out.index_path_, ": der Tensor '", item.first,
                    "' soll in '", file, "' liegen, gefunden wurde er in '", actual,
                    "'.");
      }
      ++mapped;
    }
    if (mapped != out.tensors_.size()) {
      throw_error("safetensors-Index ", out.index_path_, ": der Index nennt ", mapped,
                  " Tensoren, die Dateien enthalten aber ", out.tensors_.size(), ".");
    }
    return out;
  }

  // Kein Index: alle *.safetensors des Verzeichnisses, nach Namen sortiert.
  std::vector<std::string> files;
  for (const fs::directory_entry& entry : fs::directory_iterator(root)) {
    if (!entry.is_regular_file()) continue;
    const std::string name = path_to_utf8(entry.path().filename());
    if (has_suffix(name, ".safetensors")) files.push_back(path_to_utf8(entry.path()));
  }
  if (files.empty()) {
    throw_error("safetensors: im Verzeichnis ", path,
                " liegt weder model.safetensors.index.json noch eine *.safetensors-Datei.");
  }
  std::sort(files.begin(), files.end());
  for (const std::string& file : files) out.add_shard(file);
  return out;
}

void SafetensorsSet::add_shard(const std::string& path) {
  auto shard = std::make_unique<SafetensorsFile>(SafetensorsFile::open(path));
  const std::size_t shard_index = shards_.size();
  for (const TensorInfo& info : shard->tensors()) {
    if (index_.find(info.name) != index_.end()) {
      const TensorInfo& other = tensors_[index_.find(info.name)->second];
      throw_error("safetensors: der Tensor '", info.name, "' kommt zweimal vor -- in ",
                  shards_[other.shard]->path(), " und in ", path, ".");
    }
    TensorInfo copy = info;
    copy.shard = shard_index;
    index_.emplace(copy.name, tensors_.size());
    tensors_.push_back(std::move(copy));
  }
  shards_.push_back(std::move(shard));
}

const SafetensorsFile& SafetensorsSet::shard(std::size_t i) const {
  if (i >= shards_.size()) {
    throw_error("safetensors: Shard-Index ", i, " liegt ausserhalb der ", shards_.size(),
                " geoeffneten Dateien.");
  }
  return *shards_[i];
}

const TensorInfo* SafetensorsSet::find_tensor(std::string_view name) const {
  const auto it = index_.find(name);
  if (it == index_.end()) return nullptr;
  return &tensors_[it->second];
}

const TensorInfo& SafetensorsSet::require_tensor(std::string_view name) const {
  const TensorInfo* info = find_tensor(name);
  if (!info) {
    throw_error("safetensors ", root_, ": der Tensor '", name, "' fehlt. Das "
                "Verzeichnis hat ", tensors_.size(), " Tensoren in ", shards_.size(),
                " Datei(en).");
  }
  return *info;
}

std::span<const std::byte> SafetensorsSet::tensor_bytes(const TensorInfo& info) const {
  const SafetensorsFile& file = shard(info.shard);
  return file.tensor_bytes(file.require_tensor(info.name));
}

std::string SafetensorsSet::describe() const {
  std::string out = "safetensors-Verzeichnis ";
  out.append(root_);
  if (!index_path_.empty()) {
    out.append("\n  Index ");
    out.append(index_path_);
  }
  out.append("\n  Dateien ");
  out.append(std::to_string(shards_.size()));
  out.append(", Tensoren ");
  out.append(std::to_string(tensors_.size()));
  out.push_back('\n');
  return out;
}

}  // namespace quasar::safetensors

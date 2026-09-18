// quasar — src/model/gguf_value.cpp
//
// Die Zugriffe auf einen einzelnen Metadaten-Wert. Sie stehen getrennt vom
// Leser (`gguf.cpp`), weil das zwei Dinge sind: dort wird die Datei zerlegt,
// hier wird ein bereits zerlegter Wert ausgelesen. Zusammen waeren es ueber
//
// Kein Zugriff kopiert etwas: Zeichenketten und Arrays bleiben in der
// gemappten Datei liegen, herausgereicht werden Sichten darauf. Die Grenzen
// sind beim Parsen geprueft worden; was hier gelesen wird, liegt in der Datei.

#include <algorithm>
#include <bit>
#include <limits>
#include <string>
#include <vector>

#include "gguf_detail.hpp"
#include "quasar/core/error.hpp"
#include "quasar/model/gguf.hpp"

namespace quasar::gguf {
namespace {

using detail::load_le;
using detail::sign_extend;

}  // namespace

// ---------------------------------------------------------------------------
// MetaValue
// ---------------------------------------------------------------------------

void MetaValue::require_array(std::string_view what) const {
  if (raw_.type != ValueType::Array) {
    throw_error("GGUF-Metadaten ", what, ": erwartet wurde ein Array, da steht aber ",
                value_type_name(raw_.type), ".");
  }
}

std::string_view MetaValue::read_string_at(std::uint64_t byte_offset) const {
  // Die Grenzen wurden beim Parsen geprueft; hier wird nur noch gelesen.
  const std::uint64_t len = load_le(raw_.base + byte_offset, 8);
  return std::string_view(reinterpret_cast<const char*>(raw_.base + byte_offset + 8),
                          static_cast<std::size_t>(len));
}

bool MetaValue::as_bool(std::string_view what) const {
  if (raw_.type != ValueType::Bool) {
    throw_error("GGUF-Metadaten ", what, ": erwartet wurde bool, da steht aber ",
                value_type_name(raw_.type), ".");
  }
  return raw_.scalar != 0;
}

std::uint64_t MetaValue::as_u64(std::string_view what) const {
  switch (raw_.type) {
    case ValueType::UInt8:
    case ValueType::UInt16:
    case ValueType::UInt32:
    case ValueType::UInt64: return raw_.scalar;
    case ValueType::Bool: return raw_.scalar != 0 ? 1u : 0u;
    case ValueType::Int8:
    case ValueType::Int16:
    case ValueType::Int32:
    case ValueType::Int64: {
      const std::int64_t v = as_i64(what);
      if (v < 0) {
        throw_error("GGUF-Metadaten ", what, ": erwartet wurde eine nicht negative "
                    "Zahl, da steht aber ", v, ".");
      }
      return static_cast<std::uint64_t>(v);
    }
    default: break;
  }
  throw_error("GGUF-Metadaten ", what, ": erwartet wurde eine ganze Zahl, da steht aber ",
              value_type_name(raw_.type), ".");
}

std::int64_t MetaValue::as_i64(std::string_view what) const {
  switch (raw_.type) {
    case ValueType::Int8: return sign_extend(raw_.scalar, 8);
    case ValueType::Int16: return sign_extend(raw_.scalar, 16);
    case ValueType::Int32: return sign_extend(raw_.scalar, 32);
    case ValueType::Int64: return static_cast<std::int64_t>(raw_.scalar);
    case ValueType::UInt8:
    case ValueType::UInt16:
    case ValueType::UInt32: return static_cast<std::int64_t>(raw_.scalar);
    case ValueType::Bool: return raw_.scalar != 0 ? 1 : 0;
    case ValueType::UInt64:
      if (raw_.scalar > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        throw_error("GGUF-Metadaten ", what, ": u64-Wert ", raw_.scalar,
                    " passt nicht in i64.");
      }
      return static_cast<std::int64_t>(raw_.scalar);
    default: break;
  }
  throw_error("GGUF-Metadaten ", what, ": erwartet wurde eine ganze Zahl, da steht aber ",
              value_type_name(raw_.type), ".");
}

double MetaValue::as_double(std::string_view what) const {
  switch (raw_.type) {
    case ValueType::Float32:
      return static_cast<double>(std::bit_cast<float>(static_cast<std::uint32_t>(raw_.scalar)));
    case ValueType::Float64: return std::bit_cast<double>(raw_.scalar);
    case ValueType::Int8:
    case ValueType::Int16:
    case ValueType::Int32:
    case ValueType::Int64: return static_cast<double>(as_i64(what));
    case ValueType::UInt8:
    case ValueType::UInt16:
    case ValueType::UInt32:
    case ValueType::UInt64: return static_cast<double>(raw_.scalar);
    default: break;
  }
  throw_error("GGUF-Metadaten ", what, ": erwartet wurde eine Zahl, da steht aber ",
              value_type_name(raw_.type), ".");
}

float MetaValue::as_float(std::string_view what) const {
  if (raw_.type == ValueType::Float32) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(raw_.scalar));
  }
  return static_cast<float>(as_double(what));
}

std::string_view MetaValue::as_string(std::string_view what) const {
  if (raw_.type != ValueType::String) {
    throw_error("GGUF-Metadaten ", what, ": erwartet wurde eine Zeichenkette, da steht "
                "aber ", value_type_name(raw_.type), ".");
  }
  return std::string_view(reinterpret_cast<const char*>(raw_.base + raw_.offset),
                          static_cast<std::size_t>(raw_.length));
}

ValueType MetaValue::element_type(std::string_view what) const {
  require_array(what);
  return raw_.element_type;
}

std::uint64_t MetaValue::size(std::string_view what) const {
  require_array(what);
  return raw_.length;
}

std::string_view MetaValue::string_at(std::uint64_t index, std::string_view what) const {
  require_array(what);
  if (raw_.element_type != ValueType::String || !raw_.string_offsets) {
    throw_error("GGUF-Metadaten ", what, ": erwartet wurde ein Zeichenketten-Array, da "
                "steht aber ein Array vom Typ ", value_type_name(raw_.element_type), ".");
  }
  if (index >= raw_.length) {
    throw_error("GGUF-Metadaten ", what, ": Index ", index, " liegt ausserhalb des "
                "Arrays mit ", raw_.length, " Eintraegen.");
  }
  return read_string_at((*raw_.string_offsets)[static_cast<std::size_t>(index)]);
}

void MetaValue::read_strings(std::vector<std::string_view>& out,
                             std::string_view what) const {
  require_array(what);
  if (raw_.element_type != ValueType::String || !raw_.string_offsets) {
    throw_error("GGUF-Metadaten ", what, ": erwartet wurde ein Zeichenketten-Array, da "
                "steht aber ein Array vom Typ ", value_type_name(raw_.element_type), ".");
  }
  out.clear();
  out.reserve(raw_.string_offsets->size());
  for (const std::uint64_t offset : *raw_.string_offsets) {
    out.push_back(read_string_at(offset));
  }
}

template <typename T>
void MetaValue::fill_numbers(std::vector<T>& out, std::string_view what) const {
  require_array(what);
  const ValueType element = raw_.element_type;
  const std::uint64_t element_size = scalar_type_size(element);
  if (element_size == 0 || element == ValueType::Bool) {
    throw_error("GGUF-Metadaten ", what, ": erwartet wurde ein Zahlen-Array, da steht "
                "aber ein Array vom Typ ", value_type_name(element), ".");
  }
  out.clear();
  out.resize(static_cast<std::size_t>(raw_.length));
  const std::byte* p = raw_.base + raw_.offset;
  for (std::uint64_t i = 0; i < raw_.length; ++i, p += element_size) {
    const std::uint64_t bits = load_le(p, static_cast<std::size_t>(element_size));
    double value = 0.0;
    switch (element) {
      case ValueType::UInt8:
      case ValueType::UInt16:
      case ValueType::UInt32:
      case ValueType::UInt64:
        out[static_cast<std::size_t>(i)] = static_cast<T>(bits);
        continue;
      case ValueType::Int8:
        out[static_cast<std::size_t>(i)] = static_cast<T>(sign_extend(bits, 8));
        continue;
      case ValueType::Int16:
        out[static_cast<std::size_t>(i)] = static_cast<T>(sign_extend(bits, 16));
        continue;
      case ValueType::Int32:
        out[static_cast<std::size_t>(i)] = static_cast<T>(sign_extend(bits, 32));
        continue;
      case ValueType::Int64:
        out[static_cast<std::size_t>(i)] = static_cast<T>(static_cast<std::int64_t>(bits));
        continue;
      case ValueType::Float32:
        value = static_cast<double>(std::bit_cast<float>(static_cast<std::uint32_t>(bits)));
        break;
      case ValueType::Float64:
        value = std::bit_cast<double>(bits);
        break;
      default:
        throw_error("GGUF-Metadaten ", what, ": Elementtyp ", value_type_name(element),
                    " kann nicht als Zahl gelesen werden.");
    }
    out[static_cast<std::size_t>(i)] = static_cast<T>(value);
  }
}

void MetaValue::read_numbers(std::vector<float>& out, std::string_view what) const {
  fill_numbers(out, what);
}
void MetaValue::read_numbers(std::vector<double>& out, std::string_view what) const {
  fill_numbers(out, what);
}
void MetaValue::read_numbers(std::vector<std::int32_t>& out, std::string_view what) const {
  fill_numbers(out, what);
}
void MetaValue::read_numbers(std::vector<std::uint32_t>& out, std::string_view what) const {
  fill_numbers(out, what);
}
void MetaValue::read_numbers(std::vector<std::int64_t>& out, std::string_view what) const {
  fill_numbers(out, what);
}
void MetaValue::read_numbers(std::vector<std::uint64_t>& out, std::string_view what) const {
  fill_numbers(out, what);
}

const MetaValue& MetaValue::element_at(std::uint64_t index, std::string_view what) const {
  require_array(what);
  if (raw_.element_type != ValueType::Array || !raw_.children) {
    throw_error("GGUF-Metadaten ", what, ": erwartet wurde ein Array von Arrays, da "
                "steht aber ein Array vom Typ ", value_type_name(raw_.element_type), ".");
  }
  if (index >= raw_.children->size()) {
    throw_error("GGUF-Metadaten ", what, ": Index ", index, " liegt ausserhalb des "
                "Arrays mit ", raw_.children->size(), " Eintraegen.");
  }
  return (*raw_.children)[static_cast<std::size_t>(index)];
}

std::string MetaValue::to_text(std::uint64_t max_elements) const {
  const std::string_view what = "to_text";
  switch (raw_.type) {
    case ValueType::Bool: return raw_.scalar != 0 ? "true" : "false";
    case ValueType::String: {
      const std::string_view s = as_string(what);
      std::string out = "\"";
      out.append(s.size() > 64 ? s.substr(0, 64) : s);
      if (s.size() > 64) out.append("...");
      out.push_back('"');
      return out;
    }
    case ValueType::Float32:
    case ValueType::Float64: return std::to_string(as_double(what));
    case ValueType::Int8:
    case ValueType::Int16:
    case ValueType::Int32:
    case ValueType::Int64: return std::to_string(as_i64(what));
    case ValueType::UInt8:
    case ValueType::UInt16:
    case ValueType::UInt32:
    case ValueType::UInt64: return std::to_string(raw_.scalar);
    case ValueType::Array: break;
  }

  std::string out;
  out.append(value_type_name(raw_.element_type));
  out.push_back('[');
  out.append(std::to_string(raw_.length));
  out.append("]{");
  const std::uint64_t shown = std::min(raw_.length, max_elements);
  for (std::uint64_t i = 0; i < shown; ++i) {
    if (i) out.append(", ");
    if (raw_.element_type == ValueType::String) {
      const std::string_view s = string_at(i, what);
      out.push_back('"');
      out.append(s.size() > 24 ? s.substr(0, 24) : s);
      out.push_back('"');
    } else if (raw_.element_type == ValueType::Array) {
      out.append(element_at(i, what).to_text(4));
    } else {
      MetaValue::Raw scalar;
      scalar.type = raw_.element_type;
      scalar.base = raw_.base;
      const std::uint64_t element_size = scalar_type_size(raw_.element_type);
      scalar.scalar = load_le(raw_.base + raw_.offset + i * element_size,
                              static_cast<std::size_t>(element_size));
      out.append(MetaValue(std::move(scalar)).to_text());
    }
  }
  if (shown < raw_.length) out.append(", ...");
  out.push_back('}');
  return out;
}

}  // namespace quasar::gguf

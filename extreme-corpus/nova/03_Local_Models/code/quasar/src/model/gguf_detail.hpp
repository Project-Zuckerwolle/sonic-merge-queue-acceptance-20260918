#pragma once
//
// quasar — src/model/gguf_detail.hpp  (modulintern)
//
// Zwei Kleinigkeiten, die sowohl der Leser (`gguf.cpp`) als auch die
// Wert-Zugriffe (`gguf_value.cpp`) brauchen. GGUF ist little-endian; gelesen
// wird deshalb Byte fuer Byte und nicht per memcpy auf einen nativen Typ --
// das haelt den Leser auch auf einer big-endian-Maschine richtig.

#include <cstddef>
#include <cstdint>

namespace quasar::gguf::detail {

inline std::uint64_t load_le(const std::byte* p, std::size_t n) noexcept {
  std::uint64_t v = 0;
  for (std::size_t i = 0; i < n; ++i) {
    v |= static_cast<std::uint64_t>(static_cast<unsigned char>(p[i])) << (8 * i);
  }
  return v;
}

inline std::int64_t sign_extend(std::uint64_t bits, unsigned width_bits) noexcept {
  if (width_bits >= 64) return static_cast<std::int64_t>(bits);
  const std::uint64_t mask = (std::uint64_t{1} << width_bits) - 1;
  const std::uint64_t v = bits & mask;
  const std::uint64_t sign = std::uint64_t{1} << (width_bits - 1);
  return static_cast<std::int64_t>((v ^ sign) - sign);
}

}  // namespace quasar::gguf::detail

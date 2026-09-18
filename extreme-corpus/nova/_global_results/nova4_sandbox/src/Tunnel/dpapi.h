// dpapi.h — Windows DPAPI Key-Storage (Aufgabe 6.1).
//
// Private Keys (NOISE-Static, TLS) werden via CryptProtectData an den aktuellen
// Kontext gebunden verschlüsselt — nie Klartext auf Disk (Design §Key Storage).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nova::tunnel {

bool dpapi_protect(const std::vector<uint8_t>& plain, std::vector<uint8_t>& out,
                   std::string* err = nullptr);
bool dpapi_unprotect(const std::vector<uint8_t>& enc, std::vector<uint8_t>& out,
                     std::string* err = nullptr);

}  // namespace nova::tunnel

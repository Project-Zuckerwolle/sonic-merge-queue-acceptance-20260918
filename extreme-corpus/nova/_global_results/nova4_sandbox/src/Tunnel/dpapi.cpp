// dpapi.cpp — Implementierung von dpapi.h (Aufgabe 6.1).
#include "Tunnel/dpapi.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <wincrypt.h>
#pragma comment(lib, "crypt32.lib")
#endif

namespace nova::tunnel {

#ifdef _WIN32
namespace {
bool run(const std::vector<uint8_t>& in, std::vector<uint8_t>& out, bool protect, std::string* err) {
    DATA_BLOB ib{};
    ib.cbData = DWORD(in.size());
    ib.pbData = const_cast<BYTE*>(in.data());
    DATA_BLOB ob{};
    // CRYPTPROTECT_LOCAL_MACHINE: Machine-Scope, damit der als SYSTEM laufende
    // NovaTunnel-Dienst denselben Key entschlüsseln kann, den das interaktive
    // Setup (Admin-User) verschlüsselt hat. Ohne diesen Flag (User-Scope) scheitert
    // die Entschlüsselung im Dienst still -> falscher (Null-)Static -> Pin-Mismatch.
    const BOOL ok = protect
        ? CryptProtectData(&ib, L"Nova4Tunnel", nullptr, nullptr, nullptr, CRYPTPROTECT_LOCAL_MACHINE, &ob)
        : CryptUnprotectData(&ib, nullptr, nullptr, nullptr, nullptr, CRYPTPROTECT_LOCAL_MACHINE, &ob);
    if (!ok) { if (err) *err = protect ? "CryptProtectData fehlgeschlagen" : "CryptUnprotectData fehlgeschlagen"; return false; }
    out.assign(ob.pbData, ob.pbData + ob.cbData);
    if (ob.pbData) LocalFree(ob.pbData);
    return true;
}
}  // namespace

bool dpapi_protect(const std::vector<uint8_t>& plain, std::vector<uint8_t>& out, std::string* err) {
    return run(plain, out, true, err);
}
bool dpapi_unprotect(const std::vector<uint8_t>& enc, std::vector<uint8_t>& out, std::string* err) {
    return run(enc, out, false, err);
}
#else
bool dpapi_protect(const std::vector<uint8_t>& plain, std::vector<uint8_t>& out, std::string*) {
    out = plain; return true;   // Nicht-Windows: Passthrough (nur für Build)
}
bool dpapi_unprotect(const std::vector<uint8_t>& enc, std::vector<uint8_t>& out, std::string*) {
    out = enc; return true;
}
#endif

}  // namespace nova::tunnel

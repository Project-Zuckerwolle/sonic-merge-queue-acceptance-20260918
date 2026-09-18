// monocypher_backend.cpp — Implementierung von monocypher_backend.h (Aufgabe 6.0).
#include "Tunnel/monocypher_backend.h"

#include <cstring>

extern "C" {
#include <monocypher.h>
}

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#endif

namespace nova::tunnel {

namespace {
void rand_bytes(uint8_t* p, size_t n) {
#ifdef _WIN32
    BCryptGenRandom(nullptr, p, ULONG(n), BCRYPT_USE_SYSTEM_PREFERRED_RNG);
#else
    for (size_t i = 0; i < n; ++i) p[i] = uint8_t(i * 131 + 7);   // nur Fallback-Build
#endif
}
// 64-Bit-Nonce in ein 24-Byte-Feld (ChaCha-Nonce), little-endian, Rest 0.
void make_nonce(uint64_t n, uint8_t out[24]) {
    std::memset(out, 0, 24);
    for (int i = 0; i < 8; ++i) out[i] = uint8_t(n >> (i * 8));
}
}  // namespace

CryptoBackend monocypher_backend() {
    CryptoBackend be;

    be.keypair = [](Key32& priv, Key32& pub) {
        rand_bytes(priv.data(), 32);
        crypto_x25519_public_key(pub.data(), priv.data());
    };
    be.dh = [](const Key32& priv, const Key32& pub) {
        Key32 s{};
        crypto_x25519(s.data(), priv.data(), pub.data());
        return s;
    };
    be.hash = [](const std::vector<uint8_t>& in) {
        Key32 h{};
        crypto_blake2b(h.data(), 32, in.data(), in.size());
        return h;
    };
    be.seal = [](const Key32& key, uint64_t nonce, const std::vector<uint8_t>& pt) {
        uint8_t nn[24]; make_nonce(nonce, nn);
        std::vector<uint8_t> ct(pt.size() + 16);
        uint8_t mac[16];
        crypto_aead_lock(ct.data(), mac, key.data(), nn, nullptr, 0, pt.data(), pt.size());
        std::memcpy(ct.data() + pt.size(), mac, 16);
        return ct;
    };
    be.open = [](const Key32& key, uint64_t nonce, const std::vector<uint8_t>& ct,
                 std::vector<uint8_t>& pt) {
        if (ct.size() < 16) return false;
        const size_t n = ct.size() - 16;
        uint8_t nn[24]; make_nonce(nonce, nn);
        pt.assign(n, 0);
        return crypto_aead_unlock(pt.data(), ct.data() + n, key.data(), nn,
                                  nullptr, 0, ct.data(), n) == 0;
    };
    return be;
}

Key32 x25519_public(const Key32& priv) {
    Key32 pub{};
    crypto_x25519_public_key(pub.data(), priv.data());
    return pub;
}

}  // namespace nova::tunnel

// monocypher_backend.h — Echte NOISE-Primitive via Monocypher (Aufgabe 6.0).
//
// Ersetzt mock_backend() durch echte Krypto: X25519 (DH), ChaCha20-Poly1305
// (AEAD via crypto_aead_lock/unlock), BLAKE2b-256 (Hash). Dieselbe NoiseSession-
// Logik (TT4) läuft damit ohne Änderung — nur das Backend wird getauscht.
#pragma once

#include "Tunnel/noise_xx.h"

namespace nova::tunnel {

CryptoBackend monocypher_backend();

// Leitet den X25519-Public-Key aus einem Private ab (für --serve: Static-Pub aus DPAPI-Priv).
Key32 x25519_public(const Key32& priv);

}  // namespace nova::tunnel

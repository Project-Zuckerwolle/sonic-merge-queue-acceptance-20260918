// noise_xx.h — NOISE-XX Handshake + Transport (Aufgabe 6.0, TT4).
//
// Muster: Noise_XX_25519_ChaChaPoly_BLAKE2s (Design §Schicht 3). Die Krypto-
// Primitive sind INJIZIERT (CryptoBackend) — Server = Monocypher (X25519 /
// ChaCha20-Poly1305 / BLAKE2), Tests = mock_backend(). So ist die Handshake-
// Sequenz (e,es / e,ee,se,s,es / s,se), das gerichtete Key-Schedule und der
// Transport (monotone Nonce + Replay-Schutz) ohne echte Krypto beweisbar; die
// Primitive werden getauscht (gleiches Muster wie IInference/GpuInference).
//
// HINWEIS: Das On-Wire-Byteformat wird für die Browser-Interop (libsodium.js)
// beim Deploy exakt an Noise angeglichen; hier steht die verifizierbare LOGIK.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace nova::tunnel {

using Key32 = std::array<uint8_t, 32>;

struct CryptoBackend {
    std::function<void(Key32& priv, Key32& pub)>            keypair;  // erzeugt Schlüsselpaar
    std::function<Key32(const Key32& priv, const Key32& pub)> dh;     // kommutativ (X25519)
    std::function<Key32(const std::vector<uint8_t>&)>       hash;     // 32-Byte (BLAKE2s)
    std::function<std::vector<uint8_t>(const Key32&, uint64_t, const std::vector<uint8_t>&)> seal;
    std::function<bool(const Key32&, uint64_t, const std::vector<uint8_t>&, std::vector<uint8_t>&)> open;
};

CryptoBackend mock_backend();   // deterministische Test-Primitive

class NoiseSession {
public:
    NoiseSession(CryptoBackend be, bool initiator) : be_(std::move(be)), initiator_(initiator) {}

    void set_static(const Key32& priv, const Key32& pub) { s_priv_ = priv; s_pub_ = pub; }
    void set_ephemeral(const Key32& priv, const Key32& pub) { e_priv_ = priv; e_pub_ = pub; has_e_ = true; }

    // 3-Schritt-Handshake (Blobs). Reihenfolge: msg1 init->resp, msg2 resp->init, msg3 init->resp.
    std::vector<uint8_t> write_msg1();               // initiator: e
    void                 read_msg1(const std::vector<uint8_t>& m);   // responder
    std::vector<uint8_t> write_msg2();               // responder: e, s
    void                 read_msg2(const std::vector<uint8_t>& m);   // initiator -> established
    std::vector<uint8_t> write_msg3();               // initiator: s
    void                 read_msg3(const std::vector<uint8_t>& m);   // responder -> established

    bool         established() const { return established_; }
    const Key32& remote_static() const { return rs_pub_; }   // für Geräte-Whitelist

    // Transport (nach Handshake): jede Nachricht mit 8-Byte-Nonce, monoton + Replay-Schutz.
    std::vector<uint8_t> encrypt(const std::vector<uint8_t>& pt);
    bool                 decrypt(const std::vector<uint8_t>& ct, std::vector<uint8_t>& pt);

private:
    void ensure_ephemeral();
    void derive_keys();   // aus allen drei DH-Ergebnissen (ck -> k_send/k_recv)

    CryptoBackend be_;
    bool  initiator_;
    Key32 s_priv_{}, s_pub_{};
    Key32 e_priv_{}, e_pub_{};
    Key32 re_pub_{}, rs_pub_{};   // remote ephemeral/static
    bool  has_e_ = false, established_ = false;
    Key32 k_send_{}, k_recv_{};
    uint64_t send_nonce_ = 0;
    uint64_t last_recv_nonce_ = 0;   // Replay-Fenster (streng monoton)
};

}  // namespace nova::tunnel

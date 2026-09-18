// noise_xx.cpp — Implementierung von noise_xx.h (Aufgabe 6.0).
#include "Tunnel/noise_xx.h"

#include <algorithm>
#include <atomic>
#include <cstring>

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

Key32 sha256_32(const std::vector<uint8_t>& in) {
    Key32 out{};
#ifdef _WIN32
    BCRYPT_ALG_HANDLE alg = nullptr; BCRYPT_HASH_HANDLE h = nullptr;
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) == 0) {
        if (BCryptCreateHash(alg, &h, nullptr, 0, nullptr, 0, 0) == 0) {
            BCryptHashData(h, const_cast<PUCHAR>(reinterpret_cast<const UCHAR*>(in.data())),
                           ULONG(in.size()), 0);
            BCryptFinishHash(h, out.data(), ULONG(out.size()), 0);
            BCryptDestroyHash(h);
        }
        BCryptCloseAlgorithmProvider(alg, 0);
    }
#else
    for (size_t i = 0; i < in.size(); ++i) out[i % 32] ^= uint8_t(in[i] * 131 + 7);
#endif
    return out;
}

void append(std::vector<uint8_t>& v, const Key32& k) { v.insert(v.end(), k.begin(), k.end()); }

}  // namespace

CryptoBackend mock_backend() {
    CryptoBackend be;
    static std::atomic<uint64_t> counter{0x9E3779B9};

    be.keypair = [](Key32& priv, Key32& pub) {
        const uint64_t c = counter.fetch_add(0x1000193ULL);
        std::vector<uint8_t> seed(8);
        for (int i = 0; i < 8; ++i) seed[i] = uint8_t(c >> (i * 8));
        priv = sha256_32(seed);
        pub = priv;   // Mock: Public == Scalar (nur für Logik-Test, unsicher)
    };
    be.dh = [](const Key32& priv, const Key32& pub) {
        // Kommutativ: hash(min||max) — modelliert X25519-Symmetrie.
        const Key32 &a = priv, &b = pub;
        const bool a_first = std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
        std::vector<uint8_t> m;
        append(m, a_first ? a : b);
        append(m, a_first ? b : a);
        return sha256_32(m);
    };
    be.hash = [](const std::vector<uint8_t>& in) { return sha256_32(in); };
    be.seal = [](const Key32& key, uint64_t nonce, const std::vector<uint8_t>& pt) {
        std::vector<uint8_t> ct(pt.size() + 16);
        for (size_t off = 0; off < pt.size(); off += 32) {
            std::vector<uint8_t> blk; append(blk, key);
            for (int i = 0; i < 8; ++i) blk.push_back(uint8_t(nonce >> (i * 8)));
            blk.push_back(uint8_t(off / 32));
            const Key32 ks = sha256_32(blk);
            for (size_t i = 0; i < 32 && off + i < pt.size(); ++i) ct[off + i] = pt[off + i] ^ ks[i];
        }
        std::vector<uint8_t> tin; append(tin, key);
        for (int i = 0; i < 8; ++i) tin.push_back(uint8_t(nonce >> (i * 8)));
        tin.insert(tin.end(), pt.begin(), pt.end());
        const Key32 tag = sha256_32(tin);
        std::copy(tag.begin(), tag.begin() + 16, ct.begin() + pt.size());
        return ct;
    };
    be.open = [](const Key32& key, uint64_t nonce, const std::vector<uint8_t>& ct, std::vector<uint8_t>& pt) {
        if (ct.size() < 16) return false;
        const size_t n = ct.size() - 16;
        pt.assign(n, 0);
        for (size_t off = 0; off < n; off += 32) {
            std::vector<uint8_t> blk; append(blk, key);
            for (int i = 0; i < 8; ++i) blk.push_back(uint8_t(nonce >> (i * 8)));
            blk.push_back(uint8_t(off / 32));
            const Key32 ks = sha256_32(blk);
            for (size_t i = 0; i < 32 && off + i < n; ++i) pt[off + i] = ct[off + i] ^ ks[i];
        }
        std::vector<uint8_t> tin; append(tin, key);
        for (int i = 0; i < 8; ++i) tin.push_back(uint8_t(nonce >> (i * 8)));
        tin.insert(tin.end(), pt.begin(), pt.end());
        const Key32 tag = sha256_32(tin);
        return std::equal(tag.begin(), tag.begin() + 16, ct.begin() + n);
    };
    return be;
}

void NoiseSession::ensure_ephemeral() {
    if (!has_e_) { be_.keypair(e_priv_, e_pub_); has_e_ = true; }
}

void NoiseSession::derive_keys() {
    const Key32 dh_ee = be_.dh(e_priv_, re_pub_);
    Key32 dh_es, dh_se;
    if (initiator_) { dh_es = be_.dh(e_priv_, rs_pub_); dh_se = be_.dh(s_priv_, re_pub_); }
    else            { dh_es = be_.dh(s_priv_, re_pub_); dh_se = be_.dh(e_priv_, rs_pub_); }

    const char* label = "Noise_XX_25519_ChaChaPoly_BLAKE2s";
    std::vector<uint8_t> m(label, label + std::strlen(label));
    append(m, dh_ee); append(m, dh_es); append(m, dh_se);
    const Key32 ck = be_.hash(m);

    std::vector<uint8_t> m1; append(m1, ck); m1.push_back(1);
    std::vector<uint8_t> m2; append(m2, ck); m2.push_back(2);
    const Key32 k1 = be_.hash(m1), k2 = be_.hash(m2);
    k_send_ = initiator_ ? k1 : k2;
    k_recv_ = initiator_ ? k2 : k1;
    established_ = true;
}

std::vector<uint8_t> NoiseSession::write_msg1() {
    ensure_ephemeral();
    std::vector<uint8_t> m; append(m, e_pub_); return m;
}
void NoiseSession::read_msg1(const std::vector<uint8_t>& m) {
    if (m.size() >= 32) std::copy_n(m.begin(), 32, re_pub_.begin());
}
std::vector<uint8_t> NoiseSession::write_msg2() {
    ensure_ephemeral();
    std::vector<uint8_t> m; append(m, e_pub_); append(m, s_pub_); return m;
}
void NoiseSession::read_msg2(const std::vector<uint8_t>& m) {
    if (m.size() < 64) return;
    std::copy_n(m.begin(), 32, re_pub_.begin());
    std::copy_n(m.begin() + 32, 32, rs_pub_.begin());
    derive_keys();
}
std::vector<uint8_t> NoiseSession::write_msg3() {
    std::vector<uint8_t> m; append(m, s_pub_); return m;
}
void NoiseSession::read_msg3(const std::vector<uint8_t>& m) {
    if (m.size() < 32) return;
    std::copy_n(m.begin(), 32, rs_pub_.begin());
    derive_keys();
}

std::vector<uint8_t> NoiseSession::encrypt(const std::vector<uint8_t>& pt) {
    const uint64_t nonce = ++send_nonce_;
    std::vector<uint8_t> out(8);
    for (int i = 0; i < 8; ++i) out[i] = uint8_t(nonce >> ((7 - i) * 8));   // Big-Endian
    const std::vector<uint8_t> body = be_.seal(k_send_, nonce, pt);
    out.insert(out.end(), body.begin(), body.end());
    return out;
}
bool NoiseSession::decrypt(const std::vector<uint8_t>& ct, std::vector<uint8_t>& pt) {
    if (ct.size() < 8) return false;
    uint64_t nonce = 0;
    for (int i = 0; i < 8; ++i) nonce = (nonce << 8) | ct[i];
    if (nonce <= last_recv_nonce_) return false;   // Replay/Reorder -> verwerfen
    const std::vector<uint8_t> body(ct.begin() + 8, ct.end());
    if (!be_.open(k_recv_, nonce, body, pt)) return false;
    last_recv_nonce_ = nonce;
    return true;
}

}  // namespace nova::tunnel

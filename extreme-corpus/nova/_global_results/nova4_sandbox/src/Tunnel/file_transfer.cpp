// file_transfer.cpp — Implementierung von file_transfer.h (Aufgabe 6.3).
#include "Tunnel/file_transfer.h"

#include <algorithm>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#endif

namespace nova::tunnel {

std::string sha256_hex(const uint8_t* data, size_t len) {
    uint8_t digest[32] = {0};
#ifdef _WIN32
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE h = nullptr;
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) == 0) {
        if (BCryptCreateHash(alg, &h, nullptr, 0, nullptr, 0, 0) == 0) {
            BCryptHashData(h, const_cast<PUCHAR>(reinterpret_cast<const UCHAR*>(data)),
                           ULONG(len), 0);
            BCryptFinishHash(h, digest, sizeof(digest), 0);
            BCryptDestroyHash(h);
        }
        BCryptCloseAlgorithmProvider(alg, 0);
    }
#endif
    static const char* hx = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (int i = 0; i < 32; ++i) { out += hx[digest[i] >> 4]; out += hx[digest[i] & 0xF]; }
    return out;
}

uint32_t chunk_count(uint64_t size) {
    return uint32_t((size + kChunkSize - 1) / kChunkSize);
}

FileMeta make_file_meta(const std::string& name, const uint8_t* data, size_t len) {
    FileMeta m;
    m.name = name;
    m.size = len;
    m.sha256_hex = sha256_hex(data, len);
    m.chunk_count = chunk_count(len);
    return m;
}

size_t get_chunk(const uint8_t* data, size_t len, uint32_t index, std::vector<uint8_t>& out) {
    const uint64_t start = uint64_t(index) * kChunkSize;
    if (start >= len) { out.clear(); return 0; }
    const size_t n = size_t(std::min<uint64_t>(kChunkSize, len - start));
    out.assign(data + start, data + start + n);
    return n;
}

FileReassembler::FileReassembler(const FileMeta& meta) : meta_(meta) {
    buf_.assign(size_t(meta_.size), 0);
    got_.assign(meta_.chunk_count, 0);
}

bool FileReassembler::have_chunk(uint32_t index) const {
    return index < got_.size() && got_[index];
}

bool FileReassembler::accept_chunk(uint32_t index, const uint8_t* data, size_t len) {
    if (index >= meta_.chunk_count) return false;
    const uint64_t start = uint64_t(index) * kChunkSize;
    const size_t expect = size_t(std::min<uint64_t>(kChunkSize, meta_.size - start));
    if (len != expect || start + len > buf_.size()) return false;
    std::copy(data, data + len, buf_.begin() + size_t(start));
    if (!got_[index]) { got_[index] = 1; ++got_count_; }   // idempotent -> resumable
    return true;
}

bool FileReassembler::verify(std::string* got_hex) const {
    if (!complete()) return false;
    const std::string h = sha256_hex(buf_.data(), buf_.size());
    if (got_hex) *got_hex = h;
    return h == meta_.sha256_hex;
}

}  // namespace nova::tunnel

// chunk_streamer.cpp — siehe chunk_streamer.h
#include "InferEngine/chunk_streamer.h"
#include "ModelStore/chunk_validator.h"

#include <chrono>
#include <cstring>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#endif

namespace nova::infer {

using modelstore::Chunk;

// ---------------------------------------------------------------------------
// RAM-Modus
// ---------------------------------------------------------------------------
bool ChunkStreamer::preload(const std::string& path, std::string* err) {
    if (mode_ != StreamMode::RAM) return true;  // SSD: nichts vorladen
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { if (err) *err = "preload: kann nicht öffnen: " + path; return false; }
    const std::streamoff sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(size_t(sz < 0 ? 0 : sz));
    if (sz > 0) {
        f.read(reinterpret_cast<char*>(buf.data()), sz);
        if (f.gcount() != sz) { if (err) *err = "preload: Lesefehler: " + path; return false; }
    }
    ram_[path] = std::move(buf);
    return true;
}

void ChunkStreamer::preload_all(const std::vector<std::string>& paths) {
    for (const auto& p : paths) preload(p);
}

// ---------------------------------------------------------------------------
// SSD-Modus: OVERLAPPED (asynchron) ReadFile
// ---------------------------------------------------------------------------
bool ChunkStreamer::read_ssd_overlapped(const std::string& path,
                                        std::vector<uint8_t>& dst, std::string* err) {
#ifdef _WIN32
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING,
                           FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        if (err) *err = "CreateFile fehlgeschlagen (" + std::to_string(GetLastError()) + "): " + path;
        return false;
    }

    LARGE_INTEGER fsize;
    if (!GetFileSizeEx(h, &fsize)) { if (err) *err = "GetFileSizeEx fehlgeschlagen"; CloseHandle(h); return false; }
    dst.resize(size_t(fsize.QuadPart));

    bool ok = true;
    uint64_t total = 0;
    const DWORD CHUNK = 1u << 20;  // 1 MiB Häppchen, jeweils async
    OVERLAPPED ov{};
    ov.hEvent = CreateEventA(nullptr, TRUE, FALSE, nullptr);
    if (!ov.hEvent) { if (err) *err = "CreateEvent fehlgeschlagen"; CloseHandle(h); return false; }

    while (total < uint64_t(fsize.QuadPart)) {
        const DWORD want = DWORD((uint64_t(fsize.QuadPart) - total < CHUNK)
                                 ? (uint64_t(fsize.QuadPart) - total) : CHUNK);
        ov.Offset     = DWORD(total & 0xFFFFFFFFull);
        ov.OffsetHigh = DWORD(total >> 32);
        ResetEvent(ov.hEvent);

        DWORD got = 0;
        BOOL r = ReadFile(h, dst.data() + total, want, nullptr, &ov);
        if (!r) {
            DWORD e = GetLastError();
            if (e == ERROR_IO_PENDING) {
                if (!GetOverlappedResult(h, &ov, &got, TRUE)) {
                    if (GetLastError() == ERROR_HANDLE_EOF) break;
                    if (err) *err = "GetOverlappedResult fehlgeschlagen (" +
                                    std::to_string(GetLastError()) + ")";
                    ok = false; break;
                }
            } else if (e == ERROR_HANDLE_EOF) {
                break;
            } else {
                if (err) *err = "ReadFile fehlgeschlagen (" + std::to_string(e) + ")";
                ok = false; break;
            }
        } else {
            // Synchron abgeschlossen — Ergebnis trotzdem über OVERLAPPED holen.
            if (!GetOverlappedResult(h, &ov, &got, TRUE)) got = want;
        }
        if (got == 0) break;
        total += got;
    }

    CloseHandle(ov.hEvent);
    CloseHandle(h);
    if (ok && total != uint64_t(fsize.QuadPart)) {
        if (err) *err = "unvollständig gelesen: " + path;
        return false;
    }
    return ok;
#else
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { if (err) *err = "kann nicht öffnen: " + path; return false; }
    const std::streamoff sz = f.tellg(); f.seekg(0);
    dst.resize(size_t(sz < 0 ? 0 : sz));
    if (sz > 0) f.read(reinterpret_cast<char*>(dst.data()), sz);
    return bool(f);
#endif
}

// ---------------------------------------------------------------------------
bool ChunkStreamer::read_raw(const std::string& path, std::vector<uint8_t>& dst,
                             std::string* err) {
    if (mode_ == StreamMode::RAM) {
        auto it = ram_.find(path);
        if (it == ram_.end()) {
            // Nicht vorgeladen -> on-demand laden, dann memcpy.
            if (!preload(path, err)) return false;
            it = ram_.find(path);
            if (it == ram_.end()) { if (err) *err = "RAM-Quelle fehlt: " + path; return false; }
        }
        dst.resize(it->second.size());
        std::memcpy(dst.data(), it->second.data(), it->second.size());
        return true;
    }
    return read_ssd_overlapped(path, dst, err);
}

bool ChunkStreamer::read_chunk(const std::string& path, Chunk& out, std::string* err) {
    std::vector<uint8_t> raw;
    if (!read_raw(path, raw, err)) return false;
    return modelstore::parse_chunk(raw.data(), raw.size(), out, err);
}

// ---------------------------------------------------------------------------
StreamStats ChunkStreamer::stream_and_validate(const std::vector<std::string>& paths,
                                               uint64_t* bad_crc_out, bool stop_on_bad) {
    using clock = std::chrono::steady_clock;
    StreamStats st;
    uint64_t bad = 0;
    std::vector<uint8_t> raw;
    Chunk c;

    const auto t0 = clock::now();
    for (const auto& p : paths) {
        if (!read_raw(p, raw, nullptr)) { ++bad; if (stop_on_bad) break; continue; }
        st.bytes += raw.size();
        ++st.chunks;
        std::string err;
        if (!modelstore::parse_chunk(raw.data(), raw.size(), c, &err)) {
            ++bad; if (stop_on_bad) break; continue;
        }
        uint32_t actual = 0;
        if (modelstore::validate_chunk(c, &actual) != modelstore::ChunkStatus::OK) {
            ++bad; if (stop_on_bad) break;
        }
    }
    const auto t1 = clock::now();
    st.seconds = std::chrono::duration<double>(t1 - t0).count();
    if (bad_crc_out) *bad_crc_out = bad;
    return st;
}

}  // namespace nova::infer

// prefetch_stats.cpp — siehe prefetch_stats.h
#include "InferEngine/prefetch_stats.h"

#include <cstring>
#include <fstream>
#include <vector>

namespace nova::infer {

namespace {
constexpr char     PF_MAGIC[4] = {'N', 'V', 'P', 'F'};
constexpr uint16_t PF_VERSION  = 1;

template <class T> void wr(std::ofstream& f, const T& v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <class T> bool rd(std::ifstream& f, T& v) {
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    return f.gcount() == std::streamsize(sizeof(T));
}
}  // namespace

uint32_t PrefetchStats::in_window(const AccessRecord& r, int64_t now_unix) const {
    if (now_unix - r.window_start >= WINDOW_SECONDS) return 0;  // Fenster abgelaufen
    return r.window_count;
}

void PrefetchStats::record_access(const std::string& chunk_path, int64_t now_unix) {
    AccessRecord& r = records_[chunk_path];
    if (r.path.empty()) { r.path = chunk_path; r.window_start = now_unix; }
    r.count++;
    r.last = now_unix;
    if (now_unix - r.window_start >= WINDOW_SECONDS) {
        r.window_start = now_unix;
        r.window_count = 0;
    }
    r.window_count++;
}

uint32_t PrefetchStats::accesses_last_hour(const std::string& chunk_path, int64_t now_unix) const {
    auto it = records_.find(chunk_path);
    return it == records_.end() ? 0 : in_window(it->second, now_unix);
}

bool PrefetchStats::is_hot(const std::string& chunk_path, int64_t now_unix) const {
    return accesses_last_hour(chunk_path, now_unix) > hot_threshold_;
}

std::vector<std::string> PrefetchStats::hot_chunks(int64_t now_unix) const {
    std::vector<std::string> out;
    for (const auto& [path, r] : records_)
        if (in_window(r, now_unix) > hot_threshold_) out.push_back(path);
    return out;
}

size_t PrefetchStats::warm_hot_chunks(int64_t now_unix) const {
    size_t warmed = 0;
    std::vector<char> buf(1 << 16);
    for (const auto& path : hot_chunks(now_unix)) {
        std::ifstream f(path, std::ios::binary);
        if (!f) continue;
        // Sequenziell lesen, um die Datei in den OS-Cache zu ziehen.
        while (f.read(buf.data(), std::streamsize(buf.size())) || f.gcount() > 0) {
            if (f.gcount() == 0) break;
        }
        ++warmed;
    }
    return warmed;
}

bool PrefetchStats::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(PF_MAGIC, 4);
    wr(f, PF_VERSION);
    const uint16_t reserved = 0; wr(f, reserved);
    const uint32_t n = uint32_t(records_.size()); wr(f, n);
    for (const auto& [path_key, r] : records_) {
        const uint16_t plen = uint16_t(r.path.size());
        wr(f, plen);
        f.write(r.path.data(), plen);
        wr(f, r.count);
        wr(f, r.last);
        wr(f, r.window_start);
        wr(f, r.window_count);
    }
    return bool(f);
}

bool PrefetchStats::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[4];
    f.read(magic, 4);
    if (f.gcount() != 4 || std::memcmp(magic, PF_MAGIC, 4) != 0) return false;
    uint16_t ver = 0, reserved = 0;
    if (!rd(f, ver) || !rd(f, reserved)) return false;
    if (ver != PF_VERSION) return false;
    uint32_t n = 0;
    if (!rd(f, n)) return false;

    records_.clear();
    for (uint32_t i = 0; i < n; ++i) {
        uint16_t plen = 0;
        if (!rd(f, plen)) return false;
        std::string p(plen, '\0');
        if (plen) { f.read(p.data(), plen); if (f.gcount() != std::streamsize(plen)) return false; }
        AccessRecord r;
        r.path = p;
        if (!rd(f, r.count) || !rd(f, r.last) || !rd(f, r.window_start) || !rd(f, r.window_count))
            return false;
        records_[p] = std::move(r);
    }
    return true;
}

}  // namespace nova::infer

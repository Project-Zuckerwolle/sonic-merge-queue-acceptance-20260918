// chunk_validator.cpp — siehe chunk_validator.h
#include "ModelStore/chunk_validator.h"

#include <filesystem>

namespace fs = std::filesystem;

namespace nova::modelstore {

ChunkStatus validate_chunk(const Chunk& c, uint32_t* actual_out) {
    uint32_t crc = crc32(c.scales.data(), c.scales.size(), 0);
    crc = crc32(c.payload.data(), c.payload.size(), crc);
    if (actual_out) *actual_out = crc;
    return (crc == c.header.crc32) ? ChunkStatus::OK : ChunkStatus::BAD_CRC;
}

ChunkCheck validate_chunk_file(const std::string& path) {
    ChunkCheck r;
    r.path = path;
    Chunk c;
    std::string err;
    if (!read_chunk(path, c, &err)) {
        r.status = ChunkStatus::READ_ERROR;
        r.detail = err;
        return r;
    }
    if (!c.header.magic_ok()) {
        r.status = ChunkStatus::BAD_MAGIC;
        return r;
    }
    r.expected_crc = c.header.crc32;
    r.status = validate_chunk(c, &r.actual_crc);
    return r;
}

std::vector<ChunkCheck> validate_chunk_dir(const std::string& dir, bool only_failures) {
    std::vector<ChunkCheck> out;
    std::error_code ec;
    if (!fs::exists(dir, ec)) return out;
    for (auto it = fs::recursive_directory_iterator(dir, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) break;
        if (!it->is_regular_file()) continue;
        if (it->path().extension() != ".nv4") continue;
        ChunkCheck r = validate_chunk_file(it->path().string());
        if (!only_failures || r.status != ChunkStatus::OK)
            out.push_back(std::move(r));
    }
    return out;
}

}  // namespace nova::modelstore

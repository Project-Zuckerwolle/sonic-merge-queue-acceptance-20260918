// chunk_validator.h — CRC32-Prüfung aller .nv4-Chunks beim Start (§5.2, §15.1).
#pragma once

#include <string>
#include <vector>

#include "ModelStore/nv4_format.h"

namespace nova::modelstore {

enum class ChunkStatus { OK, BAD_CRC, READ_ERROR, BAD_MAGIC };

struct ChunkCheck {
    std::string path;
    ChunkStatus status = ChunkStatus::OK;
    uint32_t    expected_crc = 0;
    uint32_t    actual_crc   = 0;
    std::string detail;
};

// Prüft die CRC32 eines bereits gelesenen Chunks (Scales + Payload vs Header).
ChunkStatus validate_chunk(const Chunk& c, uint32_t* actual_out = nullptr);

// Liest + prüft eine einzelne .nv4-Datei.
ChunkCheck validate_chunk_file(const std::string& path);

// Prüft alle *.nv4 in einem Verzeichnis (rekursiv). Liefert nur die Defekten
// zurück wenn only_failures=true, sonst alle.
std::vector<ChunkCheck> validate_chunk_dir(const std::string& dir,
                                           bool only_failures = true);

}  // namespace nova::modelstore

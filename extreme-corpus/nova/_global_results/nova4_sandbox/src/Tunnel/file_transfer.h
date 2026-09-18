// file_transfer.h — NovaTunnel Dateitransfer (Aufgabe 6.3, TT6).
//
// Protokoll über denselben verschlüsselten Kanal (Design §Dateitransfer):
//   FILE_INIT{name,size,sha256,chunk_count} -> FILE_READY{transfer_id}
//   pro 1-MB-Chunk: FILE_CHUNK{index,data} -> FILE_ACK{index}
//   FILE_DONE -> FILE_COMPLETE{sha256_verified}
// Max 500 MB, SHA-256 nach vollständigem Empfang verifiziert, resumable via index.
// Diese Datei kapselt die reine Chunk-/Hash-Logik (netzwerkfrei -> testbar).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nova::tunnel {

inline constexpr size_t   kChunkSize   = 1u << 20;      // 1 MB
inline constexpr uint64_t kMaxFileSize = 500ull << 20;  // 500 MB

// SHA-256 (Windows BCrypt) -> Hex-String.
std::string sha256_hex(const uint8_t* data, size_t len);

struct FileMeta {
    std::string name;
    uint64_t    size = 0;
    std::string sha256_hex;
    uint32_t    chunk_count = 0;
};

uint32_t chunk_count(uint64_t size);
FileMeta make_file_meta(const std::string& name, const uint8_t* data, size_t len);
// Kopiert Chunk `index` nach out; liefert dessen Größe (0 bei ungültigem Index).
size_t   get_chunk(const uint8_t* data, size_t len, uint32_t index, std::vector<uint8_t>& out);

// Empfänger: sammelt Chunks (auch out-of-order / resumable), verifiziert am Ende.
class FileReassembler {
public:
    explicit FileReassembler(const FileMeta& meta);

    bool have_chunk(uint32_t index) const;
    bool accept_chunk(uint32_t index, const uint8_t* data, size_t len);  // false bei Bereichsfehler
    bool complete() const { return got_count_ == meta_.chunk_count; }
    // true wenn vollständig UND SHA-256 == meta. got_hex optional gefüllt.
    bool verify(std::string* got_hex = nullptr) const;

    const std::vector<uint8_t>& bytes() const { return buf_; }
    uint32_t received() const { return got_count_; }

private:
    FileMeta             meta_;
    std::vector<uint8_t> buf_;
    std::vector<char>    got_;
    uint32_t             got_count_ = 0;
};

}  // namespace nova::tunnel

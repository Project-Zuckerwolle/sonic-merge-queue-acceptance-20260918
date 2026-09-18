// chunk_streamer.h — Chunk-Streaming-Quelle (Design §6.1–6.3).
//
// Zwei Modi via Config-Flag:
//   SSD  — Gaming-PC: NVMe -> VRAM, Windows OVERLAPPED (async) I/O, ~7 GB/s
//   RAM  — Server-PC: Modelle im RAM vorgeladen -> memcpy, ~16 GB/s
//
// Der Streamer liefert rohe Chunk-Bytes; Parsing/CRC erledigt nv4_format /
// chunk_validator. Triple-Buffering (ring_buffer, TB4) baut hierauf auf.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "ModelStore/nv4_format.h"

namespace nova::infer {

enum class StreamMode { SSD, RAM };

struct StreamStats {
    uint64_t bytes  = 0;
    uint64_t chunks = 0;
    double   seconds = 0.0;
    double   gib_per_s() const {
        return seconds > 0.0 ? (double(bytes) / (1024.0 * 1024.0 * 1024.0)) / seconds : 0.0;
    }
};

class ChunkStreamer {
public:
    explicit ChunkStreamer(StreamMode mode) : mode_(mode) {}

    StreamMode mode() const { return mode_; }

    // RAM-Modus: Datei vorab in RAM laden (simuliert startup_loader/ram_model_store).
    // Im SSD-Modus ein No-Op.
    bool preload(const std::string& path, std::string* err = nullptr);
    void preload_all(const std::vector<std::string>& paths);

    // Liest die rohen Bytes eines Chunk-Files in `dst`.
    //   SSD: OVERLAPPED ReadFile.   RAM: memcpy aus vorgeladenem Puffer.
    bool read_raw(const std::string& path, std::vector<uint8_t>& dst, std::string* err = nullptr);

    // Komfort: rohe Bytes lesen + zu Chunk parsen.
    bool read_chunk(const std::string& path, modelstore::Chunk& out, std::string* err = nullptr);

    // Liest alle Pfade, parst, validiert CRC, akkumuliert Bandbreite.
    // Bei stop_on_bad_crc=true Abbruch beim ersten CRC-Fehler.
    StreamStats stream_and_validate(const std::vector<std::string>& paths,
                                    uint64_t* bad_crc_out,
                                    bool stop_on_bad_crc = false);

private:
    bool read_ssd_overlapped(const std::string& path, std::vector<uint8_t>& dst, std::string* err);

    StreamMode mode_;
    std::unordered_map<std::string, std::vector<uint8_t>> ram_;  // RAM-Modus-Puffer
};

}  // namespace nova::infer

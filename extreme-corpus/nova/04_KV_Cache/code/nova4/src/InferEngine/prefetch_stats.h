// prefetch_stats.h — SSD-Prefetch-Priorisierung (Design §6.4).
//
// Zugriffsfrequenz pro Chunk wird getrackt. Chunks mit > N Zugriffen/Stunde
// gelten als "heiß" und werden für OS-Cache-Priorisierung vorgewärmt. Persistenz
// in cache/chunk_access.bin, Update nach jeder Session.
//
// Zeit wird als Unix-Sekunden hereingereicht (testbar/deterministisch);
// produktive Aufrufer übergeben time(nullptr).
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nova::infer {

struct AccessRecord {
    std::string path;
    uint64_t    count        = 0;   // Lebenszeit-Zugriffe
    int64_t     last         = 0;   // letzter Zugriff (Unix-Sekunden)
    int64_t     window_start = 0;   // Start des aktuellen Stundenfensters
    uint32_t    window_count = 0;   // Zugriffe im aktuellen Fenster
};

class PrefetchStats {
public:
    static constexpr uint32_t DEFAULT_HOT_THRESHOLD = 5;   // > 5 Zugriffe/Stunde
    static constexpr int64_t  WINDOW_SECONDS         = 3600;

    void set_hot_threshold(uint32_t per_hour) { hot_threshold_ = per_hour; }
    uint32_t hot_threshold() const { return hot_threshold_; }

    // Einen Zugriff auf einen Chunk verbuchen.
    void record_access(const std::string& chunk_path, int64_t now_unix);

    // Zugriffe im aktuellen Stundenfenster (0 wenn Fenster abgelaufen).
    uint32_t accesses_last_hour(const std::string& chunk_path, int64_t now_unix) const;

    bool is_hot(const std::string& chunk_path, int64_t now_unix) const;

    // Alle aktuell heißen Chunk-Pfade.
    std::vector<std::string> hot_chunks(int64_t now_unix) const;

    // Heiße Chunks in den OS-Cache lesen (Vorwärmen). Liefert Anzahl gewärmter
    // Dateien. now_unix bestimmt, welche Chunks als heiß gelten.
    size_t warm_hot_chunks(int64_t now_unix) const;

    bool save(const std::string& path) const;
    bool load(const std::string& path);

    size_t size() const { return records_.size(); }
    const std::unordered_map<std::string, AccessRecord>& records() const { return records_; }

private:
    uint32_t in_window(const AccessRecord& r, int64_t now_unix) const;

    std::unordered_map<std::string, AccessRecord> records_;
    uint32_t hot_threshold_ = DEFAULT_HOT_THRESHOLD;
};

}  // namespace nova::infer

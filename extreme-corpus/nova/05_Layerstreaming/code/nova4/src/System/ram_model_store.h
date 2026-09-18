// ram_model_store.h — RAM-Buffer als Streaming-Quelle (Design §1b, §6.3, §18).
//
// [Server-PC] Modelle liegen permanent im RAM (14 GB für Nova) und werden von
// dort nach VRAM gestreamt (~16 GB/s PCIe 3.0). Dieser Store hält die geladenen
// Modell-Dateien als zusammenhängende Byte-Buffer; chunk_streamer (RAM-Modus)
// kann sie als Quelle nutzen. Geladen wird beim Startup (startup_loader).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace nova::system {

class RamModelStore {
public:
    // Lädt eine Datei vollständig in den RAM unter dem logischen Namen.
    bool load_file(const std::string& name, const std::string& path, std::string* err = nullptr);

    bool has(const std::string& name) const { return buffers_.count(name) > 0; }
    const std::vector<uint8_t>* get(const std::string& name) const;
    void free(const std::string& name);   // z.B. Ministral 3B nach Apex

    size_t total_bytes() const;
    size_t model_count() const { return buffers_.size(); }

private:
    std::map<std::string, std::vector<uint8_t>> buffers_;
};

}  // namespace nova::system

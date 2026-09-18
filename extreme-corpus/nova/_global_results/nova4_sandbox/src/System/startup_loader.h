// startup_loader.h — SSD -> RAM Preload beim Start (Design §1b, §14).
//
// [Server-PC] Lädt die permanenten Modelle (Gemma 4 E27B, Ministral 14B,
// Gemma 3 1B) von der SSD in den RAM-Store. Misst die Ladezeit je Modell
// (Design-Erwartung ~18,6 s gesamt bei 0,5 GB/s SATA-SSD).
#pragma once

#include <string>
#include <vector>

#include "System/ram_model_store.h"

namespace nova::system {

struct PreloadEntry { std::string name; std::string path; };

struct PreloadResult {
    std::string name;
    size_t      bytes = 0;
    double      seconds = 0.0;
    bool        ok = false;
    std::string error;
};

// Lädt alle Einträge in den Store; liefert je Modell Größe + Ladezeit.
std::vector<PreloadResult> preload_models(RamModelStore& store,
                                          const std::vector<PreloadEntry>& models);

}  // namespace nova::system

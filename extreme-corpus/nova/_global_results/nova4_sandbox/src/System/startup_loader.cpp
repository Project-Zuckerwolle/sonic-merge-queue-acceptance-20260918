// startup_loader.cpp — Implementierung von startup_loader.h (Design §1b).
#include "System/startup_loader.h"

#include <chrono>

namespace nova::system {

std::vector<PreloadResult> preload_models(RamModelStore& store,
                                          const std::vector<PreloadEntry>& models) {
    std::vector<PreloadResult> out;
    out.reserve(models.size());
    for (const auto& m : models) {
        PreloadResult r; r.name = m.name;
        const auto t0 = std::chrono::high_resolution_clock::now();
        std::string err;
        r.ok = store.load_file(m.name, m.path, &err);
        const auto t1 = std::chrono::high_resolution_clock::now();
        r.seconds = std::chrono::duration<double>(t1 - t0).count();
        if (r.ok) { const auto* b = store.get(m.name); r.bytes = b ? b->size() : 0; }
        else r.error = err;
        out.push_back(std::move(r));
    }
    return out;
}

}  // namespace nova::system

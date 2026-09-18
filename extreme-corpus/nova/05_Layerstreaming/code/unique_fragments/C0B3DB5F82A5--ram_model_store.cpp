// ram_model_store.cpp — Implementierung von ram_model_store.h (Design §1b).
#include "System/ram_model_store.h"

#include <fstream>

namespace nova::system {

bool RamModelStore::load_file(const std::string& name, const std::string& path, std::string* err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { if (err) *err = "RAM-Preload: kann nicht öffnen: " + path; return false; }
    const std::streamoff sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(size_t(sz < 0 ? 0 : sz));
    if (sz > 0) {
        f.read(reinterpret_cast<char*>(buf.data()), sz);
        if (f.gcount() != sz) { if (err) *err = "RAM-Preload: Lesefehler: " + path; return false; }
    }
    buffers_[name] = std::move(buf);
    return true;
}

const std::vector<uint8_t>* RamModelStore::get(const std::string& name) const {
    auto it = buffers_.find(name);
    return it == buffers_.end() ? nullptr : &it->second;
}

void RamModelStore::free(const std::string& name) { buffers_.erase(name); }

size_t RamModelStore::total_bytes() const {
    size_t t = 0;
    for (const auto& [k, v] : buffers_) t += v.size();
    return t;
}

}  // namespace nova::system

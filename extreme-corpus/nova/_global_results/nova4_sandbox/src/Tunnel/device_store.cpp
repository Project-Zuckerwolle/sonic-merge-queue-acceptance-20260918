// device_store.cpp — Implementierung von device_store.h (Aufgabe 6.1).
#include "Tunnel/device_store.h"

#include "Apex/json.h"

#include <fstream>
#include <sstream>

namespace nova::tunnel {

namespace {
std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
    return o;
}
}  // namespace

bool DeviceStore::load(std::string* err) {
    devices_.clear();
    std::ifstream f(path_, std::ios::binary);
    if (!f) return true;  // noch keine Datei = leere Whitelist
    std::ostringstream ss; ss << f.rdbuf();
    apex::JsonValue j;
    if (!apex::json_parse(ss.str(), j, err) || !j.is_array()) {
        if (err && err->empty()) *err = "devices.json ist kein Array";
        return false;
    }
    for (const auto& e : j.arr) {
        if (!e.is_object()) continue;
        Device d;
        if (auto* v = e.find("name")) d.name = v->as_str();
        if (auto* v = e.find("public_key_b64")) d.public_key_b64 = v->as_str();
        if (auto* v = e.find("added_ts")) d.added_ts = int64_t(v->num);
        if (auto* v = e.find("last_seen_ts")) d.last_seen_ts = int64_t(v->num);
        if (!d.public_key_b64.empty()) devices_.push_back(std::move(d));
    }
    return true;
}

bool DeviceStore::save(std::string* err) const {
    std::ofstream f(path_, std::ios::binary | std::ios::trunc);
    if (!f) { if (err) *err = "devices.json nicht schreibbar: " + path_; return false; }
    f << "[\n";
    for (size_t i = 0; i < devices_.size(); ++i) {
        const Device& d = devices_[i];
        f << "  {\"name\": \"" << esc(d.name) << "\", \"public_key_b64\": \""
          << esc(d.public_key_b64) << "\", \"added_ts\": " << d.added_ts
          << ", \"last_seen_ts\": " << d.last_seen_ts << "}"
          << (i + 1 < devices_.size() ? "," : "") << "\n";
    }
    f << "]\n";
    return bool(f);
}

bool DeviceStore::add(const std::string& name, const std::string& public_key_b64,
                      int64_t now_ts, std::string* err) {
    if (public_key_b64.empty()) { if (err) *err = "leerer Public Key"; return false; }
    for (auto& d : devices_)
        if (d.public_key_b64 == public_key_b64) { d.name = name; return true; }  // idempotent
    devices_.push_back({name, public_key_b64, now_ts, 0});
    return true;
}

bool DeviceStore::contains(const std::string& public_key_b64) const {
    for (const auto& d : devices_)
        if (d.public_key_b64 == public_key_b64) return true;
    return false;
}

bool DeviceStore::revoke_by_name(const std::string& name) {
    const size_t before = devices_.size();
    for (auto it = devices_.begin(); it != devices_.end();) {
        if (it->name == name) it = devices_.erase(it);
        else ++it;
    }
    return devices_.size() != before;
}

void DeviceStore::revoke_all() { devices_.clear(); }

void DeviceStore::touch(const std::string& public_key_b64, int64_t now_ts) {
    for (auto& d : devices_)
        if (d.public_key_b64 == public_key_b64) { d.last_seen_ts = now_ts; return; }
}

}  // namespace nova::tunnel

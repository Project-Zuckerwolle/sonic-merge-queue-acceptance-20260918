// device_store.h — NovaTunnel Geräte-Whitelist (Aufgabe 6.1, TT3).
//
// devices.json: [{name, public_key_b64, added_ts, last_seen_ts}]. Nur gelistete
// Public Keys dürfen verbinden — unbekannte Keys werden ohne Fehlermeldung
// getrennt (Design §Schicht 2). Persistenz als Markdown-freies JSON.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nova::tunnel {

struct Device {
    std::string name;
    std::string public_key_b64;
    int64_t     added_ts = 0;
    int64_t     last_seen_ts = 0;
};

class DeviceStore {
public:
    explicit DeviceStore(const std::string& path) : path_(path) {}

    bool load(std::string* err = nullptr);
    bool save(std::string* err = nullptr) const;

    // Registriert ein Gerät (idempotent über public_key_b64).
    bool add(const std::string& name, const std::string& public_key_b64,
             int64_t now_ts, std::string* err = nullptr);

    // Whitelist-Prüfung — Kern von TT3.
    bool contains(const std::string& public_key_b64) const;

    bool revoke_by_name(const std::string& name);   // sofortige Sperre
    void revoke_all();                               // Panic

    void touch(const std::string& public_key_b64, int64_t now_ts);  // last_seen
    const std::vector<Device>& list() const { return devices_; }
    size_t size() const { return devices_.size(); }

private:
    std::string         path_;
    std::vector<Device> devices_;
};

}  // namespace nova::tunnel

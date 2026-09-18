// settings.h — Settings-API / config.json (Design §12.2).
//
// Minimaler Flat-JSON-Store (Objekt aus String/Zahl/Bool-Werten) für die
// konfigurierbaren Parameter aus §12.2. Kein verschachteltes JSON — reicht für
// das Settings-Menü; vollständiges Schema später. R/W von config.json.
#pragma once

#include <map>
#include <string>
#include <vector>

namespace nova::web {

class Settings {
public:
    bool load(const std::string& path, std::string* err = nullptr);
    bool save(const std::string& path, std::string* err = nullptr) const;

    void parse(const std::string& json);
    std::string serialize() const;  // hübsches Flat-JSON

    std::string get(const std::string& key, const std::string& def = "") const;
    bool        get_bool(const std::string& key, bool def = false) const;
    int         get_int(const std::string& key, int def = 0) const;
    void        set(const std::string& key, const std::string& value);

    bool has(const std::string& key) const;
    const std::map<std::string, std::string>& all() const { return kv_; }

    static Settings defaults();  // §12.2-Defaults

private:
    std::map<std::string, std::string> kv_;  // Werte als Roh-String gespeichert
};

}  // namespace nova::web

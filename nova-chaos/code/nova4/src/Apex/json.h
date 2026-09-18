// json.h — Minimaler JSON-Wert (Parser + Serializer) für Apex-Task-State.
//
// Kein externes JSON-Lib im Projekt; dies deckt Objekte/Arrays/Strings/Zahlen/
// Bool/Null vollständig genug für apex_tasks/*.json (§13.7) ab.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace nova::apex {

struct JsonValue {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool                                  b = false;
    double                                num = 0.0;
    std::string                           str;
    std::vector<JsonValue>                arr;
    std::map<std::string, JsonValue>      obj;

    static JsonValue make_object() { JsonValue v; v.type = Object; return v; }
    static JsonValue make_array()  { JsonValue v; v.type = Array;  return v; }
    static JsonValue of(const std::string& s) { JsonValue v; v.type = String; v.str = s; return v; }
    static JsonValue of(int n)     { JsonValue v; v.type = Number; v.num = n; return v; }
    static JsonValue of(bool x)    { JsonValue v; v.type = Bool;   v.b = x; return v; }

    bool is_object() const { return type == Object; }
    bool is_array()  const { return type == Array; }

    const JsonValue* find(const std::string& key) const {
        auto it = obj.find(key); return it == obj.end() ? nullptr : &it->second;
    }
    std::string as_str(const std::string& def = "") const { return type == String ? str : def; }
    int         as_int(int def = 0) const { return type == Number ? int(num) : def; }
    bool        as_bool(bool def = false) const { return type == Bool ? b : def; }

    std::string serialize(int indent = 0) const;
};

// Parst JSON-Text. Liefert false bei Syntaxfehler.
bool json_parse(const std::string& text, JsonValue& out, std::string* err = nullptr);

}  // namespace nova::apex

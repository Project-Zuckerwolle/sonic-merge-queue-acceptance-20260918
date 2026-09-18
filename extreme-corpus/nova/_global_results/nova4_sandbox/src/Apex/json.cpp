// json.cpp — Implementierung von json.h (minimaler JSON-Parser/Serializer).
#include "Apex/json.h"

#include <cmath>
#include <cstdio>
#include <sstream>

namespace nova::apex {

namespace {
void escape(const std::string& s, std::string& out) {
    out += '"';
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': out += "\\r"; break;
            default:   out += c;
        }
    }
    out += '"';
}
void indent_to(std::string& out, int n) { for (int i = 0; i < n; ++i) out += ' '; }
}  // namespace

std::string JsonValue::serialize(int indent) const {
    std::string out;
    switch (type) {
        case Null:   out += "null"; break;
        case Bool:   out += b ? "true" : "false"; break;
        case Number: {
            if (num == std::floor(num) && std::abs(num) < 1e15) {
                char buf[32]; std::snprintf(buf, sizeof(buf), "%lld", (long long)num); out += buf;
            } else { char buf[32]; std::snprintf(buf, sizeof(buf), "%g", num); out += buf; }
            break;
        }
        case String: escape(str, out); break;
        case Array: {
            if (arr.empty()) { out += "[]"; break; }
            out += "[\n";
            for (size_t i = 0; i < arr.size(); ++i) {
                indent_to(out, indent + 2);
                out += arr[i].serialize(indent + 2);
                if (i + 1 < arr.size()) out += ",";
                out += "\n";
            }
            indent_to(out, indent); out += "]";
            break;
        }
        case Object: {
            if (obj.empty()) { out += "{}"; break; }
            out += "{\n";
            size_t i = 0;
            for (const auto& [k, v] : obj) {
                indent_to(out, indent + 2);
                escape(k, out); out += ": ";
                out += v.serialize(indent + 2);
                if (++i < obj.size()) out += ",";
                out += "\n";
            }
            indent_to(out, indent); out += "}";
            break;
        }
    }
    return out;
}

namespace {
struct Parser {
    const char* s; const char* end; std::string* err;
    void skip() { while (s < end && (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r')) ++s; }
    bool fail(const char* m) { if (err) *err = m; return false; }

    bool value(JsonValue& v) {
        skip();
        if (s >= end) return fail("unerwartetes Ende");
        switch (*s) {
            case '{': return object(v);
            case '[': return array(v);
            case '"': { v.type = JsonValue::String; return string(v.str); }
            case 't': case 'f': return boolean(v);
            case 'n': return null(v);
            default:  return number(v);
        }
    }
    bool string(std::string& out) {
        if (*s != '"') return fail("String erwartet");
        ++s; out.clear();
        while (s < end && *s != '"') {
            if (*s == '\\' && s + 1 < end) {
                ++s;
                switch (*s) {
                    case 'n': out += '\n'; break; case 't': out += '\t'; break;
                    case 'r': out += '\r'; break; case '"': out += '"'; break;
                    case '\\': out += '\\'; break; case '/': out += '/'; break;
                    default: out += *s;
                }
            } else out += *s;
            ++s;
        }
        if (s >= end) return fail("String nicht terminiert");
        ++s; return true;
    }
    bool number(JsonValue& v) {
        const char* start = s;
        while (s < end && (*s == '-' || *s == '+' || *s == '.' || *s == 'e' || *s == 'E' ||
                           (*s >= '0' && *s <= '9'))) ++s;
        if (s == start) return fail("Zahl erwartet");
        v.type = JsonValue::Number; v.num = std::strtod(std::string(start, s).c_str(), nullptr);
        return true;
    }
    bool boolean(JsonValue& v) {
        if (end - s >= 4 && std::string(s, s + 4) == "true") { v.type = JsonValue::Bool; v.b = true; s += 4; return true; }
        if (end - s >= 5 && std::string(s, s + 5) == "false") { v.type = JsonValue::Bool; v.b = false; s += 5; return true; }
        return fail("bool erwartet");
    }
    bool null(JsonValue& v) {
        if (end - s >= 4 && std::string(s, s + 4) == "null") { v.type = JsonValue::Null; s += 4; return true; }
        return fail("null erwartet");
    }
    bool array(JsonValue& v) {
        v.type = JsonValue::Array; ++s; skip();
        if (s < end && *s == ']') { ++s; return true; }
        for (;;) {
            JsonValue e;
            if (!value(e)) return false;
            v.arr.push_back(std::move(e));
            skip();
            if (s < end && *s == ',') { ++s; continue; }
            if (s < end && *s == ']') { ++s; return true; }
            return fail("',' oder ']' erwartet");
        }
    }
    bool object(JsonValue& v) {
        v.type = JsonValue::Object; ++s; skip();
        if (s < end && *s == '}') { ++s; return true; }
        for (;;) {
            skip();
            std::string key;
            if (!string(key)) return false;
            skip();
            if (s >= end || *s != ':') return fail("':' erwartet");
            ++s;
            JsonValue val;
            if (!value(val)) return false;
            v.obj[key] = std::move(val);
            skip();
            if (s < end && *s == ',') { ++s; continue; }
            if (s < end && *s == '}') { ++s; return true; }
            return fail("',' oder '}' erwartet");
        }
    }
};
}  // namespace

bool json_parse(const std::string& text, JsonValue& out, std::string* err) {
    Parser p{text.data(), text.data() + text.size(), err};
    if (!p.value(out)) return false;
    return true;
}

}  // namespace nova::apex

// settings.cpp — Implementierung von settings.h (Design §12.2).
#include "Web/settings.h"

#include <fstream>
#include <sstream>

namespace nova::web {

namespace {
std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\t': o += "\\t";  break;
            case '\r': break;
            default:   o += c;
        }
    }
    return o;
}

// Sehr kleiner Parser für ein flaches JSON-Objekt: {"k":"v","n":1,"b":true}.
// Werte werden als Roh-String abgelegt (Strings ohne Quotes, Zahlen/Bools wörtlich).
void parse_flat(const std::string& j, std::map<std::string, std::string>& kv) {
    kv.clear();
    size_t i = 0;
    // UTF-8-BOM überspringen (Windows-Editoren / PowerShell Set-Content -Encoding utf8 schreiben ihn).
    if (j.size() >= 3 && (unsigned char)j[0] == 0xEF && (unsigned char)j[1] == 0xBB && (unsigned char)j[2] == 0xBF) i = 3;
    auto skip_ws = [&] { while (i < j.size() && (j[i]==' '||j[i]=='\t'||j[i]=='\n'||j[i]=='\r')) ++i; };
    auto read_string = [&](std::string& out) -> bool {
        if (i >= j.size() || j[i] != '"') return false;
        ++i; out.clear();
        while (i < j.size() && j[i] != '"') {
            if (j[i] == '\\' && i + 1 < j.size()) {
                ++i;
                char c = j[i];
                out += (c=='n') ? '\n' : (c=='t') ? '\t' : c;
            } else out += j[i];
            ++i;
        }
        if (i < j.size()) ++i;  // schließendes "
        return true;
    };
    skip_ws();
    if (i < j.size() && j[i] == '{') ++i;
    for (;;) {
        skip_ws();
        if (i >= j.size() || j[i] == '}') break;
        std::string key;
        if (!read_string(key)) break;
        skip_ws();
        if (i < j.size() && j[i] == ':') ++i;
        skip_ws();
        std::string val;
        if (i < j.size() && j[i] == '"') { read_string(val); }
        else {  // Zahl / bool / null bis Komma oder }
            size_t s = i;
            while (i < j.size() && j[i] != ',' && j[i] != '}') ++i;
            val = j.substr(s, i - s);
            while (!val.empty() && (val.back()==' '||val.back()=='\t'||val.back()=='\n'||val.back()=='\r')) val.pop_back();
        }
        kv[key] = val;
        skip_ws();
        if (i < j.size() && j[i] == ',') { ++i; continue; }
    }
}
}  // namespace

void Settings::parse(const std::string& json) { parse_flat(json, kv_); }

std::string Settings::serialize() const {
    std::string o = "{\n";
    size_t n = 0;
    for (const auto& [k, v] : kv_) {
        o += "  \"" + json_escape(k) + "\": ";
        // Zahl/Bool unquotiert, sonst String. Achtung: "127.0.0.1" ist KEINE Zahl —
        // daher echte Zahl nur, wenn stod den ganzen String konsumiert.
        bool numeric = (v == "true" || v == "false" || v == "null");
        if (!numeric && !v.empty()) {
            try { size_t pos = 0; (void)std::stod(v, &pos); numeric = (pos == v.size()); }
            catch (...) { numeric = false; }
        }
        if (numeric) o += v;
        else o += "\"" + json_escape(v) + "\"";
        if (++n < kv_.size()) o += ",";
        o += "\n";
    }
    o += "}\n";
    return o;
}

bool Settings::load(const std::string& path, std::string* err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { if (err) *err = "settings: kann nicht öffnen: " + path; return false; }
    std::stringstream ss; ss << f.rdbuf();
    parse(ss.str());
    return true;
}

bool Settings::save(const std::string& path, std::string* err) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) { if (err) *err = "settings: kann nicht schreiben: " + path; return false; }
    const std::string s = serialize();
    f.write(s.data(), std::streamsize(s.size()));
    return bool(f);
}

std::string Settings::get(const std::string& key, const std::string& def) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? def : it->second;
}
bool Settings::get_bool(const std::string& key, bool def) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) return def;
    return it->second == "true" || it->second == "1";
}
int Settings::get_int(const std::string& key, int def) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) return def;
    try { return std::stoi(it->second); } catch (...) { return def; }
}
void Settings::set(const std::string& key, const std::string& value) { kv_[key] = value; }
bool Settings::has(const std::string& key) const { return kv_.find(key) != kv_.end(); }

Settings Settings::defaults() {
    Settings s;
    s.set("bind_address", "127.0.0.1");
    s.set("port_http", "8000");
    s.set("brain_compiler_interval_min", "30");
    s.set("gaming_vram_buffer_gb", "4");
    s.set("idle_timeout_min", "30");
    s.set("daily_briefing_time", "06:00");
    s.set("log_level", "INFO");
    s.set("active_profile", "profile_coding");
    s.set("workspace", "workspace");
    // Nova 4 Server-Erweiterungen (flaches Schema -> flache Keys statt models.chat).
    s.set("unrestricted", "false");     // dedizierter Server setzt beim ersten Start true
    s.set("tier2_confirm", "false");    // Besitzer: Tier-2-Aktionen ohne Rueckfrage
    s.set("vram_floor_mb", "500");      // proaktiver VRAM-Floor (GpuInference)
    s.set("models_chat", "");           // .nv4-Ordner Mistral Small 24B
    s.set("models_draft", "");          // .bin Ministral 3B (== subagent)
    s.set("models_subagent", "");
    s.set("models_brain", "");          // == models_chat (24B)
    // C1-Engine (Fertigstellung): schneller on-GPU-Pfad als Default. engine=real + gguf_path/draft_gguf_path
    // aktivieren die Real-Engine + Spec-Decoding. Die Fast-Flags sind einzeln übersteuerbar.
    s.set("gpu_forward", "true");        // 24B-Forward on-GPU (Aktivierungen+KV resident, Gewichte gestreamt)
    s.set("fmt_block3", "true");         // zweistufige 24B-Skalen (-19% Transfer)
    s.set("spec_single_stream", "true"); // ein 24B-Stream/Verify statt zwei
    s.set("pinned_stream", "true");      // pinned Host-Memory fuer H2D
    s.set("kv_bits", "3");               // residenter KV 3-Bit (near-lossless, ~10x kleiner)
    s.set("tgt_bits", "3");              // 24B-Gewichte-Bits. 3=passt in 16GB-RAM. 4 (bessere Instruct-Qualitaet,
                                         // ~14.7GB) nur bei >=20GB RAM, sonst Swap. Design §4.1: eigtl. Mixed
                                         // (Attention INT4 / FFN INT3) — passt + bessere Qualitaet (TODO).
    s.set("target_kv", "80000");         // 24B-KV-Reserve (Token) resident
    s.set("draft_kv", "2048");           // 3B-Draft-KV-Reserve
    s.set("draft_gguf_path", "");        // 3B-GGUF -> Spec-Decoding (leer = greedy)
    return s;
}

}  // namespace nova::web

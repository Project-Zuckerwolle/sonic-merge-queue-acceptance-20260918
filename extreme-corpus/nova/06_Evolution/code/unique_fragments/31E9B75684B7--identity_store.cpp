// identity_store.cpp — Implementierung von identity_store.h (Design §10.3).
#include "Memory/identity_store.h"

#include <fstream>
#include <sstream>

namespace nova::memory {

namespace {
std::string trim_trailing_newlines(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}
}  // namespace

bool IdentityStore::is_locked(const std::string& heading) {
    // "LOCKED" exakt oder mit Zusatz ("LOCKED ...") — case-sensitiv wie im Design.
    return heading == "LOCKED" || heading.rfind("LOCKED", 0) == 0;
}

void IdentityStore::parse(const std::string& md) {
    blocks_.clear();
    std::istringstream in(md);
    std::string line;
    IdentityBlock cur;          // Preamble-Block (heading "")
    bool have_block = false;    // schon ein Heading gesehen?
    std::string body;

    auto flush = [&]() {
        cur.body = trim_trailing_newlines(body);
        blocks_.push_back(cur);
        body.clear();
    };

    while (std::getline(in, line)) {
        std::string l = line;
        if (!l.empty() && l.back() == '\r') l.pop_back();
        // Heading: "# " am Zeilenanfang (genau ein '#').
        if (l.size() >= 2 && l[0] == '#' && l[1] == ' ') {
            if (have_block || !body.empty()) flush();
            cur = IdentityBlock{};
            cur.heading = l.substr(2);
            have_block = true;
        } else {
            body += l;
            body += '\n';
        }
    }
    if (have_block || !body.empty()) flush();
}

bool IdentityStore::load(const std::string& path, std::string* err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { if (err) *err = "identity: kann nicht öffnen: " + path; return false; }
    std::stringstream ss; ss << f.rdbuf();
    parse(ss.str());
    return true;
}

std::string IdentityStore::serialize() const {
    std::string out;
    for (const auto& b : blocks_) {
        if (!b.heading.empty()) { out += "# "; out += b.heading; out += "\n"; }
        if (!b.body.empty()) { out += b.body; out += "\n"; }
        out += "\n";  // Leerzeile zwischen Blöcken
    }
    return trim_trailing_newlines(out) + "\n";
}

bool IdentityStore::save(const std::string& path, std::string* err) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) { if (err) *err = "identity: kann nicht schreiben: " + path; return false; }
    const std::string s = serialize();
    f.write(s.data(), std::streamsize(s.size()));
    return bool(f);
}

int IdentityStore::index_of(const std::string& heading) const {
    for (int i = 0; i < int(blocks_.size()); ++i)
        if (blocks_[i].heading == heading) return i;
    return -1;
}

const std::string* IdentityStore::find(const std::string& heading) const {
    const int i = index_of(heading);
    return i < 0 ? nullptr : &blocks_[i].body;
}

bool IdentityStore::set_block(const std::string& heading, const std::string& body) {
    if (is_locked(heading)) return false;  // LOCKED nie überschreiben
    const int i = index_of(heading);
    if (i < 0) blocks_.push_back({heading, body});
    else       blocks_[i].body = body;
    return true;
}

bool IdentityStore::write_dreaming(const std::string& new_body) {
    return set_block("Dreaming", new_body);
}

}  // namespace nova::memory

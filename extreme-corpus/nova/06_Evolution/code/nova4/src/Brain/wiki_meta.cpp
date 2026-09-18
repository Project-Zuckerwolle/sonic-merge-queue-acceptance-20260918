// wiki_meta.cpp — Implementierung von wiki_meta.h (Design §11.2).
#include "Brain/wiki_meta.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace nova::brain {

void WikiMeta::add_session(const std::string& id) {
    if (std::find(sessions.begin(), sessions.end(), id) == sessions.end())
        sessions.push_back(id);
}

namespace {
std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

// Findet den Inhalt eines Kommentars <!-- prefix ... --> (erstes Vorkommen).
bool find_comment(const std::string& page, const std::string& prefix, std::string& inner) {
    size_t pos = 0;
    while ((pos = page.find("<!--", pos)) != std::string::npos) {
        const size_t end = page.find("-->", pos);
        if (end == std::string::npos) break;
        std::string body = trim(page.substr(pos + 4, end - pos - 4));
        if (body.rfind(prefix, 0) == 0) { inner = body; return true; }
        pos = end + 3;
    }
    return false;
}
}  // namespace

WikiMeta parse_meta(const std::string& page) {
    WikiMeta m;
    std::string inner;

    // Links
    if (find_comment(page, "Links:", inner)) {
        std::string list = trim(inner.substr(6));
        std::stringstream ss(list);
        std::string item;
        while (std::getline(ss, item, ',')) {
            item = trim(item);
            if (item.empty()) continue;
            const size_t c1 = item.find(':');
            const size_t c2 = item.find(':', c1 == std::string::npos ? c1 : c1 + 1);
            WikiLink l;
            l.target = trim(item.substr(0, c1 == std::string::npos ? item.size() : c1));
            if (c1 != std::string::npos) {
                const std::string sv = item.substr(c1 + 1, (c2 == std::string::npos ? item.size() : c2) - c1 - 1);
                l.strength = std::atof(trim(sv).c_str());
            }
            if (c2 != std::string::npos) l.reason = trim(item.substr(c2 + 1));
            m.links.push_back(l);
        }
    }

    // nova-meta
    if (find_comment(page, "nova-meta", inner)) {
        std::stringstream ss(inner);
        std::string tok;
        while (ss >> tok) {
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) continue;
            const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
            if (k == "last_mentioned") m.last_mentioned = v;
            else if (k == "sessions") {
                std::stringstream sv(v); std::string id;
                while (std::getline(sv, id, ';')) if (!id.empty()) m.sessions.push_back(id);
            }
        }
    }

    m.evergreen = page.find("<!-- evergreen -->") != std::string::npos;
    m.inactive  = page.find("# [inaktiv]") != std::string::npos;
    return m;
}

std::string strip_meta(const std::string& page) {
    std::string out;
    std::stringstream ss(page);
    std::string line;
    while (std::getline(ss, line)) {
        std::string raw = line;
        if (!raw.empty() && raw.back() == '\r') raw.pop_back();
        const std::string t = trim(raw);
        if (t.rfind("<!-- Links:", 0) == 0) continue;
        if (t.rfind("<!-- nova-meta", 0) == 0) continue;
        if (t == "<!-- evergreen -->") continue;
        if (t == "# [inaktiv]") continue;
        out += raw; out += "\n";
    }
    return trim(out);
}

std::string apply_meta(const std::string& body, const WikiMeta& m) {
    std::string out = strip_meta(body);  // idempotent: evtl. vorhandene Meta entfernen
    out += "\n\n";
    if (!m.links.empty()) {
        out += "<!-- Links: ";
        for (size_t i = 0; i < m.links.size(); ++i) {
            if (i) out += ", ";
            char buf[16]; std::snprintf(buf, sizeof(buf), "%.2f", m.links[i].strength);
            out += m.links[i].target; out += ":"; out += buf; out += ":"; out += m.links[i].reason;
        }
        out += " -->\n";
    }
    out += "<!-- nova-meta";
    if (!m.last_mentioned.empty()) out += " last_mentioned=" + m.last_mentioned;
    if (!m.sessions.empty()) {
        out += " sessions=";
        for (size_t i = 0; i < m.sessions.size(); ++i) { if (i) out += ";"; out += m.sessions[i]; }
    }
    out += " -->\n";
    if (m.evergreen) out += "<!-- evergreen -->\n";
    if (m.inactive)  out += "# [inaktiv]\n";
    return out;
}

}  // namespace nova::brain

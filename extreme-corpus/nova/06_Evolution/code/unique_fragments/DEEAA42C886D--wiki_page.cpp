// wiki_page.cpp — Implementierung von wiki_page.h.
#include "Brain/wiki_page.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>

namespace nova::brain {

// ---------------------------------------------------------------------------
// Kategorien
// ---------------------------------------------------------------------------
const char* category_dir(Category c) {
    switch (c) {
        case Category::Entities:  return "entities";
        case Category::Concepts:  return "concepts";
        case Category::Synthesis: return "synthesis";
    }
    return "synthesis";
}

const char* category_label(Category c) {
    switch (c) {
        case Category::Entities:  return "Dinge und Personen";
        case Category::Concepts:  return "Begriffe";
        case Category::Synthesis: return "Zusammenhänge";
    }
    return "Zusammenhänge";
}

Category category_from(const std::string& s) {
    const std::string l = lower_ascii(trim(s));
    if (l.find("entit") != std::string::npos || l.find("ding") != std::string::npos ||
        l.find("person") != std::string::npos)
        return Category::Entities;
    if (l.find("concept") != std::string::npos || l.find("konzept") != std::string::npos ||
        l.find("begriff") != std::string::npos)
        return Category::Concepts;
    return Category::Synthesis;
}

// ---------------------------------------------------------------------------
// WikiPage
// ---------------------------------------------------------------------------
void WikiPage::add_session(const std::string& s) {
    const std::string v = sanitize_field(s);
    if (v.empty()) return;
    if (std::find(sessions.begin(), sessions.end(), v) == sessions.end()) sessions.push_back(v);
}

void WikiPage::add_alias(const std::string& a) {
    const std::string v = sanitize_field(a);
    if (v.empty() || v == title) return;
    if (std::find(aliases.begin(), aliases.end(), v) == aliases.end()) aliases.push_back(v);
}

void WikiPage::touch(const std::string& today, const std::string& session) {
    if (!today.empty()) last_mentioned = today;
    if (created.empty()) created = last_mentioned;
    add_session(session);
    // Kein Decay, nur die Flagge zurücksetzen (Vorgabe: die Entscheidung gegen
    // Decay war richtig — Wissen altert nicht, Aufmerksamkeit schon).
    inactive = false;
}

bool WikiPage::has_link(const std::string& target) const {
    for (const auto& l : links)
        if (l.target == target) return true;
    return false;
}

void WikiPage::merge_link(const std::string& target, double weight) {
    const std::string t = slugify(target);
    if (t.empty() || t == id) return;
    for (auto& l : links) {
        if (l.target == t) {
            l.weight = std::max(l.weight, weight);   // stärkerer Fund gewinnt, keiner geht verloren
            return;
        }
    }
    links.push_back(PageLink{t, weight});
}

int WikiPage::prune_links(const std::vector<std::string>& existing_ids) {
    const size_t before = links.size();
    links.erase(std::remove_if(links.begin(), links.end(),
                               [&](const PageLink& l) {
                                   return std::find(existing_ids.begin(), existing_ids.end(),
                                                    l.target) == existing_ids.end();
                               }),
                links.end());
    return int(before - links.size());
}

// ---------------------------------------------------------------------------
// slugify
// ---------------------------------------------------------------------------
std::string slugify(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    const unsigned char* p = reinterpret_cast<const unsigned char*>(s.c_str());
    const size_t         n = s.size();
    for (size_t i = 0; i < n;) {
        const unsigned char c = p[i];
        // UTF-8-Umlaute ausschreiben — ein Dateiname muss auf NTFS wie auf
        // ext4 gültig und stabil sein.
        if (c == 0xC3 && i + 1 < n) {
            const unsigned char d = p[i + 1];
            const char*         rep = nullptr;
            switch (d) {
                case 0xA4: case 0x84: rep = "ae"; break;
                case 0xB6: case 0x96: rep = "oe"; break;
                case 0xBC: case 0x9C: rep = "ue"; break;
                case 0x9F:            rep = "ss"; break;
                default: break;
            }
            if (rep) { out += rep; i += 2; continue; }
            out.push_back('-');
            i += 2;
            continue;
        }
        if (std::isalnum(c) && c < 0x80) out.push_back(char(std::tolower(c)));
        else if (c < 0x80)               out.push_back('-');
        else                             { ++i; continue; }   // sonstige Mehrbyte: weg
        ++i;
    }
    // Bindestriche zusammenfassen und außen abschneiden.
    std::string cl;
    for (char c : out) {
        if (c == '-' && (cl.empty() || cl.back() == '-')) continue;
        cl.push_back(c);
    }
    while (!cl.empty() && cl.back() == '-') cl.pop_back();
    if (cl.size() > 64) {
        cl.resize(64);
        while (!cl.empty() && cl.back() == '-') cl.pop_back();
    }
    return cl;
}

// ---------------------------------------------------------------------------
// Scanner
// ---------------------------------------------------------------------------
std::vector<std::string> scan_links(const std::string& body) {
    std::vector<std::string> out;
    size_t                   pos = 0;
    while ((pos = body.find("[[", pos)) != std::string::npos) {
        const size_t end = body.find("]]", pos + 2);
        if (end == std::string::npos) break;
        const std::string raw = trim(body.substr(pos + 2, end - pos - 2));
        pos = end + 2;
        if (raw.empty() || raw.size() > 120) continue;
        const std::string t = slugify(raw);
        if (t.empty()) continue;
        if (std::find(out.begin(), out.end(), t) == out.end()) out.push_back(t);
    }
    return out;
}

std::vector<std::string> scan_tags(const std::string& body) {
    std::vector<std::string> out;
    for (size_t i = 0; i < body.size(); ++i) {
        if (body[i] != '#') continue;
        // Kein Tag: Zeilenanfang (Markdown-Überschrift) oder mitten im Wort.
        const bool line_start = (i == 0) || body[i - 1] == '\n';
        if (line_start) continue;
        if (i > 0 && !std::isspace(static_cast<unsigned char>(body[i - 1])) && body[i - 1] != '(')
            continue;
        std::string t;
        size_t      j = i + 1;
        for (; j < body.size(); ++j) {
            const unsigned char c = static_cast<unsigned char>(body[j]);
            if (std::isalnum(c) || c >= 0x80 || c == '_' || c == '-') t.push_back(body[j]);
            else break;
        }
        i = j;
        if (t.size() < 2) continue;
        const std::string s = slugify(t);
        if (!s.empty() && std::find(out.begin(), out.end(), s) == out.end()) out.push_back(s);
    }
    return out;
}

bool append_link_line(std::string& body, const std::string& target, const std::string& reason) {
    const std::string t = slugify(target);
    if (t.empty()) return false;
    if (body.find("[[" + t + "]]") != std::string::npos) return false;

    const std::string line =
        "- [[" + t + "]] - " + (reason.empty() ? std::string("verwandt") : sanitize_field(reason));

    const std::string head = "## Verbindungen";
    const size_t      at = body.find(head);
    if (at == std::string::npos) {
        if (!body.empty() && body.back() != '\n') body.push_back('\n');
        body += "\n" + head + "\n" + line + "\n";
        return true;
    }
    size_t eol = body.find('\n', at);
    if (eol == std::string::npos) {
        body += "\n" + line + "\n";
        return true;
    }
    body.insert(eol + 1, line + "\n");
    return true;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------
bool legacy_format(const std::string& text) {
    if (text.find("<!--nova\n") != std::string::npos) return false;
    return text.find("<!-- nova-meta") != std::string::npos ||
           text.find("<!-- Links:") != std::string::npos ||
           text.find("<!-- evergreen -->") != std::string::npos ||
           text.find("# [inaktiv]") != std::string::npos;
}

namespace {

void parse_header_line(WikiPage& p, const std::string& line) {
    const auto f = split(line, '|');
    if (f.empty()) return;
    const std::string key = lower_ascii(trim(f[0]));
    auto              rest = [&](size_t i) { return i < f.size() ? trim(f[i]) : std::string(); };

    if (key == "category")            p.category = category_from(rest(1));
    else if (key == "title")          p.title = rest(1);
    else if (key == "created")        p.created = rest(1);
    else if (key == "last_mentioned") p.last_mentioned = rest(1);
    else if (key == "evergreen")      p.evergreen = (rest(1) == "1" || rest(1) == "ja");
    else if (key == "status")         p.inactive = (lower_ascii(rest(1)) == "inaktiv");
    else if (key == "aliases") {
        for (size_t i = 1; i < f.size(); ++i) p.add_alias(trim(f[i]));
    } else if (key == "sessions") {
        for (size_t i = 1; i < f.size(); ++i) p.add_session(trim(f[i]));
    } else if (key == "tags") {
        for (size_t i = 1; i < f.size(); ++i) {
            const std::string t = slugify(trim(f[i]));
            if (!t.empty() && std::find(p.tags.begin(), p.tags.end(), t) == p.tags.end())
                p.tags.push_back(t);
        }
    } else if (key == "links") {
        // links|ziel|gewicht|ziel|gewicht|…  — Paare, nie Freitext. Deshalb
        // kann hier nichts zerbrechen, was in wiki_meta.cpp:49-57 zerbrach.
        for (size_t i = 1; i + 1 < f.size(); i += 2) {
            const std::string tgt = slugify(trim(f[i]));
            if (tgt.empty()) continue;
            p.merge_link(tgt, std::atof(trim(f[i + 1]).c_str()));
        }
    }
}

// Alte Fassung: Metadaten als HTML-Kommentare irgendwo in der Datei.
void parse_legacy(WikiPage& p, const std::string& text, std::string& body_out) {
    std::string body;
    for (const auto& raw : split(text, '\n')) {
        const std::string t = trim(raw);
        if (t.rfind("<!-- Links:", 0) == 0) {
            const size_t e = t.find("-->");
            std::string  list = t.substr(11, (e == std::string::npos ? t.size() : e) - 11);
            for (const auto& item : split(list, ',')) {
                const std::string it = trim(item);
                if (it.empty()) continue;
                const auto parts = split(it, ':');
                if (parts.empty()) continue;
                const std::string tgt = slugify(trim(parts[0]));
                if (tgt.empty()) continue;
                const double w = parts.size() > 1 ? std::atof(trim(parts[1]).c_str()) : 0.5;
                p.merge_link(tgt, w > 0.0 ? w : 0.5);
            }
            continue;
        }
        if (t.rfind("<!-- nova-meta", 0) == 0) {
            std::string inner = t.substr(14);
            const size_t e = inner.find("-->");
            if (e != std::string::npos) inner = inner.substr(0, e);
            // Token an Leerzeichen — genau die Schwäche von wiki_meta.cpp:66.
            // Wir lesen sie nur noch, um sie loszuwerden.
            for (const auto& tok : split(inner, ' ')) {
                const size_t eq = tok.find('=');
                if (eq == std::string::npos) continue;
                const std::string k = trim(tok.substr(0, eq));
                const std::string v = trim(tok.substr(eq + 1));
                if (k == "last_mentioned") p.last_mentioned = v;
                else if (k == "sessions")
                    for (const auto& s : split(v, ';')) p.add_session(s);
            }
            continue;
        }
        if (t == "<!-- evergreen -->") { p.evergreen = true; continue; }
        if (t == "# [inaktiv]")        { p.inactive = true; continue; }
        body += raw;
        body.push_back('\n');
    }
    body_out = trim(body);
}

}  // namespace

WikiPage parse_page(const std::string& id, Category c, const std::string& text) {
    WikiPage p;
    p.id = id;
    p.category = c;

    const auto lines = split(text, '\n');

    // Kopfblock suchen — er steht in den ersten Zeilen oder gar nicht.
    size_t hdr_begin = std::string::npos;
    for (size_t i = 0; i < lines.size() && i < 4; ++i) {
        if (trim(lines[i]) == "<!--nova") { hdr_begin = i; break; }
    }

    std::string body;
    if (hdr_begin != std::string::npos) {
        for (size_t i = 0; i < hdr_begin; ++i) {
            const std::string t = trim(lines[i]);
            if (t.rfind("# ", 0) == 0 && p.title.empty()) p.title = trim(t.substr(2));
        }
        size_t i = hdr_begin + 1;
        for (; i < lines.size(); ++i) {
            const std::string t = trim(lines[i]);
            if (t == "-->") { ++i; break; }
            parse_header_line(p, t);
        }
        for (; i < lines.size(); ++i) {
            body += lines[i];
            body.push_back('\n');
        }
        body = trim(body);
    } else if (legacy_format(text)) {
        parse_legacy(p, text, body);
        // Titel aus der ersten H1 der alten Seite.
        for (const auto& raw : split(body, '\n')) {
            const std::string t = trim(raw);
            if (t.rfind("# ", 0) == 0) { p.title = trim(t.substr(2)); break; }
        }
    } else {
        // Handgeschriebene oder fremde Markdown-Datei: alles ist Inhalt.
        body = trim(text);
        for (const auto& raw : lines) {
            const std::string t = trim(raw);
            if (t.rfind("# ", 0) == 0) { p.title = trim(t.substr(2)); break; }
        }
    }

    // Titelzeile nicht doppelt im Inhalt führen.
    if (!p.title.empty()) {
        const std::string h1 = "# " + p.title;
        if (body.rfind(h1, 0) == 0) body = trim(body.substr(h1.size()));
    }
    if (p.title.empty()) p.title = id;

    p.body = body;

    // Der Graph kommt aus dem Fließtext. Der Kopf trägt nur die Gewichte —
    // steht ein Link im Text, aber nicht im Kopf, gilt er trotzdem.
    for (const auto& t : scan_links(p.body)) p.merge_link(t, 0.5);
    for (const auto& t : scan_tags(p.body))
        if (std::find(p.tags.begin(), p.tags.end(), t) == p.tags.end()) p.tags.push_back(t);

    return p;
}

std::string render_page(const WikiPage& p) {
    std::string out;
    out += "# " + sanitize_field(p.title.empty() ? p.id : p.title) + "\n";
    out += "<!--nova\n";
    out += std::string("category|") + category_dir(p.category) + "\n";
    if (!p.aliases.empty()) {
        out += "aliases";
        for (const auto& a : p.aliases) out += "|" + sanitize_field(a);
        out += "\n";
    }
    if (!p.created.empty())        out += "created|" + sanitize_field(p.created) + "\n";
    if (!p.last_mentioned.empty()) out += "last_mentioned|" + sanitize_field(p.last_mentioned) + "\n";
    if (!p.sessions.empty()) {
        out += "sessions";
        for (const auto& s : p.sessions) out += "|" + sanitize_field(s);
        out += "\n";
    }
    if (p.evergreen) out += "evergreen|1\n";
    out += std::string("status|") + (p.inactive ? "inaktiv" : "aktiv") + "\n";
    if (!p.links.empty()) {
        out += "links";
        for (const auto& l : p.links) {
            char buf[16];
            std::snprintf(buf, sizeof buf, "%.2f", l.weight);
            out += "|" + sanitize_field(l.target) + "|" + buf;
        }
        out += "\n";
    }
    if (!p.tags.empty()) {
        out += "tags";
        for (const auto& t : p.tags) out += "|" + sanitize_field(t);
        out += "\n";
    }
    out += "-->\n\n";
    out += trim(p.body);
    out += "\n";
    return out;
}

}  // namespace nova::brain

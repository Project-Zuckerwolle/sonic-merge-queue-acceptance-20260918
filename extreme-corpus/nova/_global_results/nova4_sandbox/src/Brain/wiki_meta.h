// wiki_meta.h — Metadaten in Wiki-Seiten als HTML-Kommentare (Design §11.2).
//
// Brain-Metadaten leben in der Markdown-Seite selbst (direkt lesbar/Git-fähig):
//   <!-- Links: ziel:0.9:grund, ziel2:0.7:grund -->     (Cross-Entry Linking)
//   <!-- nova-meta last_mentioned=YYYY-MM-DD sessions=id1;id2 -->
//   <!-- evergreen -->                                   (>=3 Sessions)
//   # [inaktiv]                                          (>30 Tage nicht erwähnt)
//
// parse_meta() liest sie, strip_meta() liefert den reinen Inhalt, apply_meta()
// hängt sie wieder an. So bleibt der menschliche Inhalt unberührt.
#pragma once

#include <string>
#include <vector>

namespace nova::brain {

struct WikiLink { std::string target; double strength = 0.0; std::string reason; };

struct WikiMeta {
    std::string              last_mentioned;  // "YYYY-MM-DD" oder leer
    std::vector<std::string> sessions;        // distinkte Session-IDs
    bool                     evergreen = false;
    bool                     inactive  = false;
    std::vector<WikiLink>    links;

    void add_session(const std::string& id);  // nur falls noch nicht vorhanden
};

WikiMeta    parse_meta(const std::string& page);
std::string strip_meta(const std::string& page);                 // nur Inhalt
std::string apply_meta(const std::string& body, const WikiMeta& m);  // Inhalt + Footer

}  // namespace nova::brain

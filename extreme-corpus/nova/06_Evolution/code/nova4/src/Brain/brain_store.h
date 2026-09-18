// brain_store.h — Markdown-Wiki R/W + Verzeichnisstruktur (Design §11.1, §18).
//
// brain/
//   raw/                  append-only (Chatlogs, Apex-Transcripts, Notizen)
//   wiki/entities|concepts|synthesis/   *.md
//   wiki/index.md  wiki/hot.md
//   log.md
//
// Reines Dateisystem (std::filesystem). Seiten sind Markdown; Metadaten leben
// als HTML-Kommentare in der Seite (Evergreen, Links, nova-meta) — siehe
// brain_consolidator/brain_evergreen. Optionaler ReadDirectoryChangesW-Watcher
// (Windows) triggert den Brain-Zyklus bei Änderungen.
#pragma once

#include <string>
#include <vector>

namespace nova::brain {

enum class WikiCategory { Entities, Concepts, Synthesis };
const char* category_dir(WikiCategory c);

struct WikiRef { WikiCategory category; std::string name; };  // name ohne ".md"

class BrainStore {
public:
    explicit BrainStore(const std::string& root) : root_(root) {}

    bool init(std::string* err = nullptr);   // legt Verzeichnisbaum an

    const std::string& root() const { return root_; }

    // Wiki-Seiten.
    bool write_wiki(WikiCategory c, const std::string& name, const std::string& content,
                    std::string* err = nullptr);
    bool read_wiki(WikiCategory c, const std::string& name, std::string& out,
                   std::string* err = nullptr) const;
    bool wiki_exists(WikiCategory c, const std::string& name) const;
    std::vector<WikiRef> list_wiki() const;                 // alle Kategorien
    std::vector<std::string> list_wiki(WikiCategory c) const;

    // raw/ append-only.
    bool append_raw(const std::string& name, const std::string& content,
                    std::string* err = nullptr);
    std::vector<std::string> list_raw() const;
    bool read_raw(const std::string& name, std::string& out) const;

    // hot.md / log.md / index.md.
    bool write_hot(const std::string& content, std::string* err = nullptr);
    bool read_hot(std::string& out) const;
    bool append_log(const std::string& line, std::string* err = nullptr);
    bool write_index(const std::string& content, std::string* err = nullptr);

    // Erste n Zeilen einer Wiki-Seite (für BM25-Index + Duplikat-Check).
    std::string first_lines(WikiCategory c, const std::string& name, int n) const;

    std::string wiki_path(WikiCategory c, const std::string& name) const;

private:
    std::string root_;
};

}  // namespace nova::brain

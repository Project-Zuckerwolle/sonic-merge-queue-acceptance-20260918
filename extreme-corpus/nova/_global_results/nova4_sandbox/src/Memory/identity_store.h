// identity_store.h — identity.md mit LOCKED-Block-System (Design §10.3, §12.4).
//
// identity.md besteht aus `# HEADING`-Blöcken. Dreaming (Ministral 14B nach
// jeder Session) schreibt AUSSCHLIESSLICH in den `# Dreaming`-Block; `# LOCKED`-
// Blöcke sind niemals Schreibziel. Der Parser bewahrt Reihenfolge + Inhalt,
// damit manuelle Edits und Dreaming koexistieren.
#pragma once

#include <string>
#include <vector>

namespace nova::memory {

struct IdentityBlock {
    std::string heading;  // ohne führendes "# "
    std::string body;     // Zeilen zwischen diesem und dem nächsten Heading
};

class IdentityStore {
public:
    // Parst Markdown-Text in Blöcke. Text vor dem ersten Heading -> Block "".
    void parse(const std::string& md);
    bool load(const std::string& path, std::string* err = nullptr);
    bool save(const std::string& path, std::string* err = nullptr) const;

    std::string serialize() const;          // zurück nach Markdown
    std::string full_text() const { return serialize(); }  // für Context-Stack

    const std::vector<IdentityBlock>& blocks() const { return blocks_; }
    const std::string* find(const std::string& heading) const;  // body oder nullptr

    // Dreaming: schreibt NUR den "# Dreaming"-Block (legt ihn an wenn nötig).
    // Liefert false und ändert nichts, wenn als heading ein LOCKED-Name kommt.
    bool write_dreaming(const std::string& new_body);

    // Generisches Setzen — verweigert LOCKED-Blöcke (heading == "LOCKED").
    bool set_block(const std::string& heading, const std::string& body);

    static bool is_locked(const std::string& heading);

private:
    int index_of(const std::string& heading) const;
    std::vector<IdentityBlock> blocks_;
};

}  // namespace nova::memory

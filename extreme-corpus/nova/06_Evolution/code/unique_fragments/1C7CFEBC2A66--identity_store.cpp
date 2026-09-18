// identity_store.cpp — Implementierung von identity_store.h.
#include "Memory/identity_store.h"

#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

namespace nova::memory {

namespace {

std::string rtrim_newlines(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

std::string trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    const size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Entfernt eine umschließende ```-Hülle. Das 24B liefert seinen Block regelmäßig
// als Code-Block aus; in testdata/memory/identity.md steht deshalb wörtlich
// "```berlin```" als kompletter Dreaming-Inhalt. Nova 4 hat das übernommen.
std::string strip_fences(std::string s) {
    s = trim(s);
    if (s.rfind("```", 0) == 0) {
        const size_t nl = s.find('\n');
        // ```lang\n…\n```   ODER einzeilig ```text```
        if (nl == std::string::npos) {
            s = s.substr(3);
            if (s.size() >= 3 && s.compare(s.size() - 3, 3, "```") == 0) s.resize(s.size() - 3);
        } else {
            s = s.substr(nl + 1);
            const size_t last = s.rfind("```");
            if (last != std::string::npos) s = s.substr(0, last);
        }
    }
    return trim(s);
}

// Serialisierung ohne Lock — der Aufrufer hält ihn bereits.
std::string serialize_blocks(const std::vector<IdentityBlock>& blocks) {
    std::string out;
    for (const auto& b : blocks) {
        if (b.heading.empty() && b.body.empty()) continue;  // Nova 4 schrieb hier Leerzeilen
        if (!b.heading.empty()) { out += "# "; out += b.heading; out += "\n"; }
        if (!b.body.empty()) { out += b.body; out += "\n"; }
        out += "\n";
    }
    return rtrim_newlines(out) + "\n";
}

}  // namespace

IdentityStore::IdentityStore(std::string path)
    : path_(path.empty() ? Paths::join(Paths::get().memory(), "identity.md") : std::move(path)) {}

void IdentityStore::set_event_sink(EventSink sink) {
    st_.with([&](State& s) { s.sink = std::move(sink); });
}

void IdentityStore::emit(EventKind k, std::string a, std::string b) const {
    // Senke UNTER dem Lock kopieren, aufrufen AUSSERHALB. std::mutex ist nicht
    // rekursiv, und eine UI-Senke darf zurück in den Store greifen dürfen.
    EventSink s = st_.with([](const State& st) { return st.sink; });
    if (s) s(Event{k, std::move(a), std::move(b), 0});
}

bool IdentityStore::is_locked(const std::string& heading) {
    return heading.rfind("LOCKED", 0) == 0;
}

void IdentityStore::parse(const std::string& md) {
    std::vector<IdentityBlock> out;
    std::istringstream in(md);
    std::string line, body;
    IdentityBlock cur;
    bool have_block = false;

    auto flush = [&] {
        cur.body = rtrim_newlines(body);
        out.push_back(cur);
        body.clear();
    };

    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.size() >= 2 && line[0] == '#' && line[1] == ' ') {
            if (have_block || !body.empty()) flush();
            cur = IdentityBlock{};
            cur.heading = trim(line.substr(2));
            have_block = true;
        } else {
            body += line;
            body += '\n';
        }
    }
    if (have_block || !body.empty()) flush();

    st_.with([&](State& s) { s.blocks = std::move(out); });
}

Status IdentityStore::load() {
    if (!file_exists(path_)) {
        // Frische Installation ist ein gültiger Zustand, kein Fehler — Nova 4
        // lieferte hier false und der Aufrufer (context_service.cpp:82) konnte
        // "leer" nicht von "kaputt" unterscheiden.
        st_.with([](State& s) { s.blocks.clear(); });
        return ok();
    }
    auto r = read_file(path_);
    if (!r) {
        emit(EventKind::Error, "memory.identity.read", r.error().message);
        return Status(r.error());
    }
    parse(r.value());
    return ok();
}

std::string IdentityStore::serialize() const {
    return st_.with([](const State& s) { return serialize_blocks(s.blocks); });
}

std::string IdentityStore::locked_text() const {
    return st_.with([](const State& s) {
        std::string out;
        for (const auto& b : s.blocks) {
            if (!is_locked(b.heading)) continue;
            out += "# "; out += b.heading; out += "\n";
            if (!b.body.empty()) { out += b.body; out += "\n"; }
            out += "\n";
        }
        return rtrim_newlines(out);
    });
}

std::string IdentityStore::unlocked_text() const {
    return st_.with([](const State& s) {
        std::string out;
        for (const auto& b : s.blocks) {
            if (is_locked(b.heading)) continue;
            if (b.heading.empty() && b.body.empty()) continue;
            if (!b.heading.empty()) { out += "# "; out += b.heading; out += "\n"; }
            if (!b.body.empty()) { out += b.body; out += "\n"; }
            out += "\n";
        }
        return rtrim_newlines(out);
    });
}

std::vector<IdentityBlock> IdentityStore::blocks() const {
    return st_.with([](const State& s) { return s.blocks; });
}

std::optional<std::string> IdentityStore::find(const std::string& heading) const {
    return st_.with([&](const State& s) -> std::optional<std::string> {
        for (const auto& b : s.blocks)
            if (b.heading == heading) return b.body;
        return std::nullopt;
    });
}

Status IdentityStore::write_file_unlocked(const std::string& text,
                                          std::string* backup_warn) const {
    // Backup der letzten drei Fassungen. Grund: der LOCKED-Schutz greift nur
    // gegen set_block; er hilft nicht gegen ein Dreaming, das den Block mit
    // Unsinn füllt ("```berlin```"). Ohne Historie ist das nicht rückholbar,
    // und identity.md ist die einzige Datei, in die das Modell selbst schreibt.
    std::error_code ec;
    if (file_exists(path_)) {
        for (int i = 3; i >= 2; --i) {
            const std::string from = path_ + ".bak" + std::to_string(i - 1);
            const std::string to   = path_ + ".bak" + std::to_string(i);
            if (file_exists(from)) {
                fs::remove(to, ec);
                fs::rename(from, to, ec);
            }
        }
        ec.clear();
        fs::copy_file(path_, path_ + ".bak1", fs::copy_options::overwrite_existing, ec);
        // Ein fehlgeschlagenes Backup verhindert das Speichern NICHT — es wird
        // nur gemeldet. Sonst blockiert eine gesperrte Backup-Datei die
        // eigentliche Persistenz.
        if (ec && backup_warn) *backup_warn = ec.message();
    }
    // temp + fsync + rename: ein Abbruch hinterlässt entweder die alte oder die
    // neue Datei, nie eine halbe. Nova 4: identity_store.cpp:70-76 (ofstream
    // trunc) — ein Absturz dort löscht LOCKED-Blöcke, die per Definition nie
    // verloren gehen dürfen.
    return write_atomic(path_, text);
}

Status IdentityStore::save() const {
    // Serialisieren UND schreiben unter EINEM Lock. Zwei getrennte Zugriffe
    // erlauben, dass zwischen Serialisierung und Rename ein zweiter Schreiber
    // dieselbe Datei mit einem älteren Stand überholt — genau die Art Race, die
    // in Nova 4 gar nicht erst auffiel, weil es überhaupt keinen Lock gab
    // (chat_service.cpp:331 gegen den Lesepfad in context_service.cpp:82).
    std::string warn;
    Status st = st_.with([&](const State& s) {
        return write_file_unlocked(serialize_blocks(s.blocks), &warn);
    });
    if (!warn.empty()) emit(EventKind::Error, "memory.identity.backup", warn);
    if (!st) {
        emit(EventKind::Error, "memory.identity.write", st.error().message);
        return st;
    }
    return ok();
}

Status IdentityStore::set_block(const std::string& heading, const std::string& body) {
    if (heading.empty())
        return Status(err("memory.identity.no_heading", "Block ohne Überschrift", NOVA_HERE));
    if (is_locked(heading)) {
        emit(EventKind::Error, "memory.identity.locked",
             "Block '" + heading + "' ist gesperrt und wurde nicht geändert.");
        return Status(err("memory.identity.locked",
                          "Gesperrter Block '" + heading + "' wird nie überschrieben.", NOVA_HERE));
    }

    const bool changed = st_.with([&](State& s) {
        for (auto& b : s.blocks) {
            if (b.heading != heading) continue;
            if (b.body == body) return false;
            b.body = body;
            return true;
        }
        s.blocks.push_back(IdentityBlock{heading, body});
        return true;
    });

    if (!changed) return ok();
    Status st = save();
    if (!st) return st;
    emit(EventKind::MemoryWrite, "identity:" + heading, body);
    return ok();
}

Status IdentityStore::write_dreaming(const std::string& new_body) {
    const std::string cleaned = strip_fences(new_body);
    if (cleaned.empty()) {
        // Nova 4 (identity_store.cpp:97) hätte den Block mit Leerstring
        // überschrieben und den gesamten bisherigen Dreaming-Bestand vernichtet,
        // sobald das 24B einmal nichts liefert.
        emit(EventKind::Error, "memory.dreaming.empty",
             "Dreaming lieferte keinen verwertbaren Inhalt — Block bleibt unverändert.");
        return Status(err("memory.dreaming.empty", "Dreaming-Antwort war leer.", NOVA_HERE));
    }
    return set_block(kDreaming, cleaned);
}

}  // namespace nova::memory

// workspace_guard.h — WorkspaceFS Boundary-Check (Design §13.3, §22 Predator v5).
//
// Alle datei_*-Skills und shell_exec operieren ausschließlich im Workspace-
// Verzeichnis. Jeder Pfad wird aufgelöst und auf die Workspace-Grenze geprüft.
// Path-Traversal (../../etc/passwd, absolute Pfade außerhalb) -> Boundary-Fehler,
// Tool-Call bricht ab OHNE Ausführung.
//
// Limits: MAX_READ = 10 MB, MAX_WRITE = 10 MB, MAX_GLOB_RESULTS = 100.
#pragma once

#include <cstddef>
#include <string>

namespace nova::skills {

class WorkspaceGuard {
public:
    static constexpr size_t MAX_READ  = 10 * 1024 * 1024;
    static constexpr size_t MAX_WRITE = 10 * 1024 * 1024;
    static constexpr int    MAX_GLOB  = 100;

    // unrestricted=true: Boundary-Check aus (Nova darf den ganzen PC steuern) —
    // fuer den dedizierten Server. resolve() liefert dann jeden normalisierten
    // Absolutpfad ohne WorkspaceBoundaryError.
    explicit WorkspaceGuard(const std::string& workspace_root, bool unrestricted = false);

    const std::string& root() const { return root_; }
    bool unrestricted() const { return unrestricted_; }

    // Löst path (relativ oder absolut) gegen den Workspace auf. Liefert false +
    // err bei Path-Traversal / Pfad außerhalb des Workspace (WorkspaceBoundaryError).
    bool resolve(const std::string& path, std::string& abs_out, std::string* err) const;

    // Prüft ob ein bereits absoluter, kanonischer Pfad im Workspace liegt.
    bool inside(const std::string& abs_canonical) const;

private:
    std::string root_;  // kanonischer Workspace-Pfad
    bool        unrestricted_ = false;
};

}  // namespace nova::skills

// workspace_guard.cpp — Implementierung von workspace_guard.h (Design §13.3).
#include "Skills/workspace_guard.h"

#include <filesystem>

namespace fs = std::filesystem;

namespace nova::skills {

namespace {
// Normalisiert einen Pfad lexikalisch (löst . und .. auf, ohne Existenzprüfung).
std::string normalize(const fs::path& p) {
    return p.lexically_normal().generic_string();
}
}  // namespace

WorkspaceGuard::WorkspaceGuard(const std::string& workspace_root, bool unrestricted)
    : unrestricted_(unrestricted) {
    std::error_code ec;
    fs::path r = fs::absolute(fs::path(workspace_root), ec);
    root_ = normalize(r);
    // Trailing-Slash entfernen für sauberen Präfix-Vergleich.
    while (root_.size() > 1 && root_.back() == '/') root_.pop_back();
}

bool WorkspaceGuard::inside(const std::string& abs_canonical) const {
    if (abs_canonical == root_) return true;
    // Muss mit root_ + "/" beginnen.
    return abs_canonical.size() > root_.size() &&
           abs_canonical.compare(0, root_.size(), root_) == 0 &&
           abs_canonical[root_.size()] == '/';
}

bool WorkspaceGuard::resolve(const std::string& path, std::string& abs_out, std::string* err) const {
    std::error_code ec;
    fs::path p(path);
    // Relative Pfade an den Workspace hängen; absolute bleiben absolut (und
    // werden dann gegen die Grenze geprüft).
    fs::path combined = p.is_absolute() ? p : (fs::path(root_) / p);
    fs::path abs = fs::absolute(combined, ec);
    const std::string norm = normalize(abs);

    // Vollzugriff-Modus: keine Grenzprüfung (relative Pfade bleiben workspace-relativ,
    // absolute Pfade sind ueberall erlaubt).
    if (!unrestricted_ && !inside(norm)) {
        if (err) *err = "WorkspaceBoundaryError: Pfad außerhalb des Workspace: " + path;
        return false;
    }
    abs_out = norm;
    return true;
}

}  // namespace nova::skills

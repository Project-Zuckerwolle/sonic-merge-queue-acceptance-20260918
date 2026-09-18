// brain_store.cpp — Implementierung von brain_store.h (Design §11.1).
#include "Brain/brain_store.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace nova::brain {

const char* category_dir(WikiCategory c) {
    switch (c) {
        case WikiCategory::Entities:  return "entities";
        case WikiCategory::Concepts:  return "concepts";
        case WikiCategory::Synthesis: return "synthesis";
    }
    return "synthesis";
}

namespace {
bool write_file(const std::string& path, const std::string& content, std::string* err) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) { if (err) *err = "kann nicht schreiben: " + path; return false; }
    f.write(content.data(), std::streamsize(content.size()));
    return bool(f);
}
bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::stringstream ss; ss << f.rdbuf(); out = ss.str();
    return true;
}
}  // namespace

bool BrainStore::init(std::string* err) {
    std::error_code ec;
    fs::create_directories(fs::path(root_) / "raw", ec);
    fs::create_directories(fs::path(root_) / "wiki" / "entities", ec);
    fs::create_directories(fs::path(root_) / "wiki" / "concepts", ec);
    fs::create_directories(fs::path(root_) / "wiki" / "synthesis", ec);
    if (ec) { if (err) *err = "init: " + ec.message(); return false; }
    return true;
}

std::string BrainStore::wiki_path(WikiCategory c, const std::string& name) const {
    return (fs::path(root_) / "wiki" / category_dir(c) / (name + ".md")).string();
}

bool BrainStore::write_wiki(WikiCategory c, const std::string& name,
                            const std::string& content, std::string* err) {
    return write_file(wiki_path(c, name), content, err);
}
bool BrainStore::read_wiki(WikiCategory c, const std::string& name,
                           std::string& out, std::string* err) const {
    if (!read_file(wiki_path(c, name), out)) { if (err) *err = "nicht gefunden: " + name; return false; }
    return true;
}
bool BrainStore::wiki_exists(WikiCategory c, const std::string& name) const {
    return fs::exists(wiki_path(c, name));
}

std::vector<std::string> BrainStore::list_wiki(WikiCategory c) const {
    std::vector<std::string> out;
    const fs::path dir = fs::path(root_) / "wiki" / category_dir(c);
    std::error_code ec;
    if (!fs::exists(dir, ec)) return out;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_regular_file()) continue;
        if (e.path().extension() == ".md") out.push_back(e.path().stem().string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<WikiRef> BrainStore::list_wiki() const {
    std::vector<WikiRef> out;
    for (auto c : {WikiCategory::Entities, WikiCategory::Concepts, WikiCategory::Synthesis})
        for (auto& n : list_wiki(c)) out.push_back({c, n});
    return out;
}

bool BrainStore::append_raw(const std::string& name, const std::string& content, std::string* err) {
    const std::string path = (fs::path(root_) / "raw" / name).string();
    std::ofstream f(path, std::ios::binary | std::ios::app);
    if (!f) { if (err) *err = "raw append fehlgeschlagen: " + path; return false; }
    f.write(content.data(), std::streamsize(content.size()));
    return bool(f);
}
std::vector<std::string> BrainStore::list_raw() const {
    std::vector<std::string> out;
    const fs::path dir = fs::path(root_) / "raw";
    std::error_code ec;
    if (!fs::exists(dir, ec)) return out;
    for (const auto& e : fs::directory_iterator(dir, ec))
        if (e.is_regular_file()) out.push_back(e.path().filename().string());
    std::sort(out.begin(), out.end());
    return out;
}
bool BrainStore::read_raw(const std::string& name, std::string& out) const {
    return read_file((fs::path(root_) / "raw" / name).string(), out);
}

bool BrainStore::write_hot(const std::string& content, std::string* err) {
    return write_file((fs::path(root_) / "wiki" / "hot.md").string(), content, err);
}
bool BrainStore::read_hot(std::string& out) const {
    return read_file((fs::path(root_) / "wiki" / "hot.md").string(), out);
}
bool BrainStore::append_log(const std::string& line, std::string* err) {
    const std::string path = (fs::path(root_) / "log.md").string();
    std::ofstream f(path, std::ios::binary | std::ios::app);
    if (!f) { if (err) *err = "log append fehlgeschlagen"; return false; }
    f << line << "\n";
    return bool(f);
}
bool BrainStore::write_index(const std::string& content, std::string* err) {
    return write_file((fs::path(root_) / "wiki" / "index.md").string(), content, err);
}

std::string BrainStore::first_lines(WikiCategory c, const std::string& name, int n) const {
    std::ifstream f(wiki_path(c, name), std::ios::binary);
    if (!f) return {};
    std::string line, out;
    for (int i = 0; i < n && std::getline(f, line); ++i) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        out += line; out += "\n";
    }
    return out;
}

}  // namespace nova::brain

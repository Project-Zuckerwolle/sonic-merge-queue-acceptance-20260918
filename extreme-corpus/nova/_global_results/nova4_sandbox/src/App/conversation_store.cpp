// conversation_store.cpp — Implementierung von conversation_store.h.
#include "App/conversation_store.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace nova::app {

namespace {

std::string esc(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') { /* verwerfen */ }
        else o += c;
    }
    return o;
}
std::string unesc(const std::string& s) {
    std::string o;
    o.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            const char n = s[++i];
            o += (n == 'n') ? '\n' : n;
        } else o += s[i];
    }
    return o;
}

}  // namespace

ConversationStore::ConversationStore(std::string dir) : dir_(std::move(dir)) {
    std::error_code ec;
    fs::create_directories(dir_, ec);
}

std::string ConversationStore::path_for(const std::string& id) const {
    return (fs::path(dir_) / (id + ".md")).string();
}

void ConversationStore::load_all() {
    std::error_code ec;
    if (!fs::exists(dir_, ec)) return;
    for (auto& de : fs::directory_iterator(dir_, ec)) {
        if (ec) break;
        if (!de.is_regular_file(ec) || de.path().extension() != ".md") continue;
        const std::string id = de.path().stem().string();

        std::ifstream f(de.path(), std::ios::binary);
        if (!f) continue;
        Conversation conv;
        conv.id = id;
        std::string line;
        bool first = true;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (first) {
                first = false;
                if (line.rfind("# ", 0) == 0) { conv.title = line.substr(2); continue; }
            }
            const size_t tab = line.find('\t');
            if (tab == std::string::npos) continue;
            conv.wm.add_turn(line.substr(0, tab), unesc(line.substr(tab + 1)));
        }
        if (conv.title.empty()) conv.title = "Chat " + id;

        // counter_ aus numerischem Suffix "c<N>" hochziehen.
        if (id.size() > 1 && id[0] == 'c') {
            try { counter_ = std::max<uint64_t>(counter_, std::stoull(id.substr(1))); } catch (...) {}
        }
        convs_[id] = std::move(conv);
        order_.push_back(id);
    }
}

std::string ConversationStore::create(const std::string& title) {
    const std::string id = "c" + std::to_string(++counter_);
    Conversation conv;
    conv.id = id;
    conv.title = title;
    convs_[id] = std::move(conv);
    order_.push_back(id);
    persist(id);
    return id;
}

Conversation* ConversationStore::get(const std::string& id) {
    auto it = convs_.find(id);
    return it == convs_.end() ? nullptr : &it->second;
}

bool ConversationStore::remove(const std::string& id) {
    auto it = convs_.find(id);
    if (it == convs_.end()) return false;
    convs_.erase(it);
    order_.erase(std::remove(order_.begin(), order_.end(), id), order_.end());
    std::error_code ec;
    fs::remove(path_for(id), ec);
    return true;
}

bool ConversationStore::rename(const std::string& id, const std::string& title) {
    auto it = convs_.find(id);
    if (it == convs_.end()) return false;
    it->second.title = title;
    persist(id);
    return true;
}

std::vector<std::pair<std::string, std::string>> ConversationStore::list() const {
    std::vector<std::pair<std::string, std::string>> out;
    for (auto rit = order_.rbegin(); rit != order_.rend(); ++rit) {
        auto it = convs_.find(*rit);
        if (it != convs_.end()) out.emplace_back(it->first, it->second.title);
    }
    return out;
}

void ConversationStore::persist(const std::string& id) const {
    auto it = convs_.find(id);
    if (it == convs_.end()) return;
    std::ofstream f(path_for(id), std::ios::binary | std::ios::trunc);
    if (!f) return;
    f << "# " << it->second.title << "\n";
    for (const auto& t : it->second.wm.turns())
        f << t.role << "\t" << esc(t.text) << "\n";
}

}  // namespace nova::app

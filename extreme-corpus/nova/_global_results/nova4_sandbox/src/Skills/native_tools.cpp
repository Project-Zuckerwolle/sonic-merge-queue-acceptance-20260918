// native_tools.cpp — Implementierung der nativen Tools (Design §13.3).
#include "Skills/native_tools.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#include <powrprof.h>
#pragma comment(lib, "PowrProf.lib")
#endif

namespace fs = std::filesystem;

namespace nova::skills {

std::string param(const Params& p, const std::string& key, const std::string& def) {
    for (const auto& [k, v] : p) if (k == key) return v;
    return def;
}
bool has_param(const Params& p, const std::string& key) {
    for (const auto& [k, v] : p) if (k == key) return true;
    return false;
}

namespace tools {

namespace {
// ---- Ausdrucks-Parser (rekursiver Abstieg) --------------------------------
struct ExprParser {
    const char* s; bool ok = true;
    void skip() { while (*s == ' ' || *s == '\t') ++s; }
    double expr() {
        double v = term();
        for (;;) { skip();
            if (*s == '+') { ++s; v += term(); }
            else if (*s == '-') { ++s; v -= term(); }
            else break; }
        return v;
    }
    double term() {
        double v = factor();
        for (;;) { skip();
            if (*s == '*') { ++s; v *= factor(); }
            else if (*s == '/') { ++s; double d = factor(); if (d == 0) { ok = false; return 0; } v /= d; }
            else if (*s == '%') { ++s; double d = factor(); if (d == 0) { ok = false; return 0; } v = std::fmod(v, d); }
            else break; }
        return v;
    }
    double factor() {
        skip();
        if (*s == '(') { ++s; double v = expr(); skip(); if (*s == ')') ++s; else ok = false; return v; }
        if (*s == '-') { ++s; return -factor(); }
        if (*s == '+') { ++s; return factor(); }
        char* end; double v = std::strtod(s, &end);
        if (end == s) { ok = false; return 0; }
        s = end; return v;
    }
};

bool resolve_or_err(ToolContext& ctx, const std::string& path, std::string& abs, ToolResult& err) {
    if (!ctx.guard) { err = tool_err("kein WorkspaceGuard im Kontext"); return false; }
    std::string e;
    if (!ctx.guard->resolve(path, abs, &e)) { err = tool_err(e); return false; }
    return true;
}
}  // namespace

bool eval_expression(const std::string& expr, double& out) {
    ExprParser p{expr.c_str()};
    out = p.expr();
    p.skip();
    return p.ok && *p.s == '\0';
}

// ---- Tier 1 ---------------------------------------------------------------
ToolResult rechner(const Params& p, ToolContext&) {
    const std::string e = param(p, "expr", param(p, "ausdruck", ""));
    if (e.empty()) return tool_err("Parameter 'expr' fehlt");
    double v;
    if (!eval_expression(e, v)) return tool_err("ungültiger Ausdruck: " + e);
    std::ostringstream os; os << e << " = " << v;
    return tool_ok(os.str());
}

ToolResult uhrzeit(const Params&, ToolContext&) {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return tool_ok(std::string("Aktuelle Zeit: ") + buf);
}

ToolResult datei_lesen(const Params& p, ToolContext& ctx) {
    std::string abs; ToolResult err;
    if (!resolve_or_err(ctx, param(p, "path"), abs, err)) return err;
    std::error_code ec;
    if (fs::is_directory(abs, ec)) {
        std::string out = "Verzeichnis " + param(p, "path") + ":\n";
        int n = 0;
        for (const auto& e : fs::directory_iterator(abs, ec)) {
            out += (e.is_directory() ? "[dir] " : "      ") + e.path().filename().string() + "\n";
            if (++n >= WorkspaceGuard::MAX_GLOB) { out += "...\n"; break; }
        }
        return tool_ok(out);
    }
    std::ifstream f(abs, std::ios::binary | std::ios::ate);
    if (!f) return tool_err("Datei nicht lesbar: " + param(p, "path"));
    const std::streamoff sz = f.tellg();
    if (size_t(sz) > WorkspaceGuard::MAX_READ) return tool_err("Datei > MAX_READ (10 MB)");
    f.seekg(0);
    std::stringstream ss; ss << f.rdbuf();
    return tool_ok(ss.str());
}

ToolResult datei_suchen(const Params& p, ToolContext& ctx) {
    std::string abs; ToolResult err;
    if (!resolve_or_err(ctx, param(p, "path", "."), abs, err)) return err;
    const std::string pattern = param(p, "name", param(p, "pattern", ""));
    std::error_code ec;
    std::string out; int n = 0;
    for (const auto& e : fs::recursive_directory_iterator(abs, ec)) {
        if (!e.is_regular_file()) continue;
        const std::string fn = e.path().filename().string();
        if (pattern.empty() || fn.find(pattern) != std::string::npos) {
            out += e.path().lexically_relative(ctx.guard->root()).generic_string() + "\n";
            if (++n >= WorkspaceGuard::MAX_GLOB) { out += "... (Limit 100)\n"; break; }
        }
    }
    return tool_ok(n ? out : "Keine Treffer.");
}

ToolResult codebase_scan(const Params& p, ToolContext& ctx) {
    std::string abs; ToolResult err;
    if (!resolve_or_err(ctx, param(p, "path", "."), abs, err)) return err;
    std::error_code ec;
    std::string out = "Codebase-Scan:\n";
    int files = 0;
    for (const auto& e : fs::recursive_directory_iterator(abs, ec)) {
        if (!e.is_regular_file()) continue;
        out += "## " + e.path().lexically_relative(ctx.guard->root()).generic_string() + "\n";
        std::ifstream f(e.path(), std::ios::binary);
        std::string line;
        for (int i = 0; i < 5 && std::getline(f, line); ++i) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            out += "  " + line + "\n";
        }
        if (++files >= 200) { out += "... (200-Datei-Limit)\n"; break; }
    }
    out = "Dateien: " + std::to_string(files) + "\n" + out;
    return tool_ok(out);
}

ToolResult todo_lesen(const Params&, ToolContext& ctx) {
    if (ctx.todo_path.empty()) return tool_ok("Keine Todos.");
    std::ifstream f(ctx.todo_path, std::ios::binary);
    if (!f) return tool_ok("Keine Todos.");
    std::stringstream ss; ss << f.rdbuf();
    const std::string s = ss.str();
    return tool_ok(s.empty() ? "Keine Todos." : s);
}

// ---- Netzwerk-Tools (http_get injiziert) ----------------------------------
namespace {
ToolResult net_tool(ToolContext& ctx, const std::string& url, const char* label) {
    if (!ctx.http_get) return tool_err(std::string(label) + ": kein HTTP-Client konfiguriert");
    const std::string r = ctx.http_get(url);
    if (r.empty()) return tool_err(std::string(label) + ": leere Antwort");
    return tool_ok(r);
}
// URL-Encode für Query-Parameter (Leerzeichen/Umlaute/Sonderzeichen).
std::string url_encode(const std::string& s) {
    static const char* hex = "0123456789ABCDEF";
    std::string o;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') o += char(c);
        else { o += '%'; o += hex[c >> 4]; o += hex[c & 15]; }
    }
    return o;
}
// Erster "key":<zahl> aus einem JSON-Blob (Geocoding lat/lon).
std::string json_num(const std::string& j, const std::string& key) {
    size_t p = j.find("\"" + key + "\"");
    if (p == std::string::npos) return "";
    p = j.find(':', p); if (p == std::string::npos) return "";
    ++p; while (p < j.size() && j[p] == ' ') ++p;
    size_t s = p;
    while (p < j.size() && (std::isdigit((unsigned char)j[p]) || j[p]=='-'||j[p]=='.'||j[p]=='+'||j[p]=='e'||j[p]=='E')) ++p;
    return j.substr(s, p - s);
}
}  // namespace

ToolResult wetter(const Params& p, ToolContext& ctx) {
    if (!ctx.http_get) return tool_err("wetter: kein HTTP-Client konfiguriert");
    const std::string loc = param(p, "location", "Hamburg");
    // 1) Geocoding Name -> lat/lon (Open-Meteo, kein API-Key). open-meteo hat KEIN location-Param.
    const std::string geo = ctx.http_get(
        "https://geocoding-api.open-meteo.com/v1/search?name=" + url_encode(loc) + "&count=1&language=de&format=json");
    const std::string lat = json_num(geo, "latitude"), lon = json_num(geo, "longitude");
    if (lat.empty() || lon.empty()) return tool_err("wetter: Ort '" + loc + "' nicht gefunden");
    // 2) Forecast + aktuelles Wetter mit lat/lon.
    const std::string r = ctx.http_get(
        "https://api.open-meteo.com/v1/forecast?latitude=" + lat + "&longitude=" + lon +
        "&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&timezone=auto");
    if (r.empty()) return tool_err("wetter: leere Antwort");
    return tool_ok("Ort: " + loc + " (" + lat + ", " + lon + ")\n" + r);
}
ToolResult websearch(const Params& p, ToolContext& ctx) {
    const std::string q = param(p, "query");
    if (q.empty()) return tool_err("Parameter 'query' fehlt");
    // SearXNG (lokal, JSON) wenn per Param gesetzt, sonst DuckDuckGo-HTML (ohne JS, echte Treffer).
    const std::string sx = param(p, "searxng", "");
    if (!sx.empty()) return net_tool(ctx, sx + "/search?format=json&q=" + url_encode(q), "websearch");
    return net_tool(ctx, "https://html.duckduckgo.com/html/?q=" + url_encode(q), "websearch");
}
ToolResult news_rss(const Params& p, ToolContext& ctx) {
    // Default: echter deutscher Nachrichten-Feed (Tagesschau). Per Param 'feed' überschreibbar (§13.3).
    return net_tool(ctx, param(p, "feed", "https://www.tagesschau.de/xml/rss2/"), "news_rss");
}
ToolResult finanzen(const Params& p, ToolContext& ctx) {
    const std::string sym = param(p, "symbol", "AAPL");
    // stooq: öffentlicher CSV-Kurs ohne API-Key (Symbol, Datum, Zeit, OHLCV).
    return net_tool(ctx, "https://stooq.com/q/l/?s=" + url_encode(sym) + "&f=sd2t2ohlcv&h&e=csv", "finanzen");
}

// ---- Tier 2 (WorkspaceFS) -------------------------------------------------
ToolResult datei_schreiben(const Params& p, ToolContext& ctx) {
    std::string abs; ToolResult err;
    if (!resolve_or_err(ctx, param(p, "path"), abs, err)) return err;
    const std::string content = param(p, "content");
    if (content.size() > WorkspaceGuard::MAX_WRITE) return tool_err("Inhalt > MAX_WRITE (10 MB)");
    std::error_code ec; fs::create_directories(fs::path(abs).parent_path(), ec);
    std::ofstream f(abs, std::ios::binary | std::ios::trunc);
    if (!f) return tool_err("kann nicht schreiben: " + param(p, "path"));
    f.write(content.data(), std::streamsize(content.size()));
    return tool_ok("Geschrieben (" + std::to_string(content.size()) + " Bytes): " + param(p, "path"));
}

ToolResult datei_verschieben(const Params& p, ToolContext& ctx) {
    std::string from, to; ToolResult err;
    if (!resolve_or_err(ctx, param(p, "from"), from, err)) return err;
    if (!resolve_or_err(ctx, param(p, "to"), to, err)) return err;
    std::error_code ec;
    fs::create_directories(fs::path(to).parent_path(), ec);
    fs::rename(from, to, ec);
    if (ec) return tool_err("verschieben fehlgeschlagen: " + ec.message());
    return tool_ok("Verschoben: " + param(p, "from") + " -> " + param(p, "to"));
}

ToolResult datei_sortieren(const Params& p, ToolContext& ctx) {
    std::string abs; ToolResult err;
    if (!resolve_or_err(ctx, param(p, "path", "."), abs, err)) return err;
    std::error_code ec; int moved = 0;
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(abs, ec))
        if (e.is_regular_file()) files.push_back(e.path());
    for (const auto& f : files) {
        std::string ext = f.extension().string();
        if (!ext.empty() && ext[0] == '.') ext = ext.substr(1);
        if (ext.empty()) ext = "sonstige";
        const fs::path sub = fs::path(abs) / ext;
        fs::create_directories(sub, ec);
        fs::rename(f, sub / f.filename(), ec);
        if (!ec) ++moved;
    }
    return tool_ok("Sortiert: " + std::to_string(moved) + " Dateien nach Endung");
}

ToolResult ordner_erstellen(const Params& p, ToolContext& ctx) {
    std::string abs; ToolResult err;
    if (!resolve_or_err(ctx, param(p, "path"), abs, err)) return err;
    std::error_code ec;
    if (!fs::create_directories(abs, ec) && ec) return tool_err("mkdir fehlgeschlagen: " + ec.message());
    return tool_ok("Ordner erstellt: " + param(p, "path"));
}

ToolResult datei_loeschen(const Params& p, ToolContext& ctx) {
    std::string abs; ToolResult err;
    if (!resolve_or_err(ctx, param(p, "path"), abs, err)) return err;
    std::error_code ec;
    if (!fs::remove(abs, ec)) return tool_err("löschen fehlgeschlagen: " + param(p, "path"));
    return tool_ok("Gelöscht: " + param(p, "path"));
}

ToolResult todo_schreiben(const Params& p, ToolContext& ctx) {
    if (ctx.todo_path.empty()) return tool_err("kein todo_path im Kontext");
    const std::string text = param(p, "text");
    if (text.empty()) return tool_err("Parameter 'text' fehlt");
    std::ofstream f(ctx.todo_path, std::ios::binary | std::ios::app);
    if (!f) return tool_err("Todo nicht schreibbar");
    const std::string mark = param(p, "done") == "true" ? "[x] " : "[ ] ";
    f << mark << text << "\n";
    return tool_ok("Todo gespeichert: " + text);
}

// ---- Prozess-Helfer (Windows CreateProcess, cwd + Timeout) ----------------
namespace {
#ifdef _WIN32
bool run_process(const std::string& cmdline, const std::string& cwd, int timeout_ms,
                 std::string& out, std::string* err) {
    SECURITY_ATTRIBUTES sa{}; sa.nLength = sizeof(sa); sa.bInheritHandle = TRUE;
    HANDLE rd = nullptr, wr = nullptr;
    if (!::CreatePipe(&rd, &wr, &sa, 0)) { if (err) *err = "CreatePipe"; return false; }
    ::SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{}; si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES; si.hStdOutput = wr; si.hStdError = wr;
    PROCESS_INFORMATION pi{};
    std::string cmd = cmdline;
    BOOL okp = ::CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                                nullptr, cwd.empty() ? nullptr : cwd.c_str(), &si, &pi);
    ::CloseHandle(wr);
    if (!okp) { ::CloseHandle(rd); if (err) *err = "CreateProcess fehlgeschlagen"; return false; }

    // Output lesen bis EOF (Prozess schließt Pipe).
    char buf[4096]; DWORD n = 0;
    while (::ReadFile(rd, buf, sizeof(buf), &n, nullptr) && n > 0) out.append(buf, n);
    const DWORD w = ::WaitForSingleObject(pi.hProcess, DWORD(timeout_ms));
    if (w == WAIT_TIMEOUT) { ::TerminateProcess(pi.hProcess, 1); if (err) *err = "Timeout (60s)"; }
    ::CloseHandle(rd); ::CloseHandle(pi.hProcess); ::CloseHandle(pi.hThread);
    return w != WAIT_TIMEOUT;
}
#endif
}  // namespace

ToolResult git_ops(const Params& p, ToolContext& ctx) {
    const std::string action = param(p, "action", "status");
    std::string cmd = "git " + action;
    if (action == "log") cmd += " --oneline -n 10";
#ifdef _WIN32
    std::string out, err;
    const std::string cwd = ctx.guard ? ctx.guard->root() : "";
    if (!run_process("cmd /c " + cmd, cwd, 30000, out, &err)) return tool_err("git: " + err);
    return tool_ok(out.empty() ? "(git: keine Ausgabe)" : out);
#else
    (void)ctx; return tool_err("git nur unter Windows implementiert");
#endif
}

ToolResult shell_exec(const Params& p, ToolContext& ctx) {
    const std::string cmd = param(p, "cmd");
    if (cmd.empty()) return tool_err("Parameter 'cmd' fehlt");
#ifdef _WIN32
    std::string out, err;
    const std::string cwd = ctx.guard ? ctx.guard->root() : "";
    if (!run_process("cmd /c " + cmd, cwd, 60000, out, &err)) return tool_err("shell_exec: " + err);
    return tool_ok(out.empty() ? "(keine Ausgabe)" : out);
#else
    (void)ctx; return tool_err("shell_exec nur unter Windows");
#endif
}

ToolResult power_sleep(const Params&, ToolContext&) {
#ifdef _WIN32
    ::SetSuspendState(TRUE, FALSE, FALSE);
    return tool_ok("Hibernate (S4) ausgelöst");
#else
    return tool_err("power_sleep nur unter Windows");
#endif
}

}  // namespace tools
}  // namespace nova::skills

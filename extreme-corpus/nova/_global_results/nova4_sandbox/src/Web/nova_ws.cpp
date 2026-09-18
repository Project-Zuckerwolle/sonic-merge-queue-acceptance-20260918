// nova_ws.cpp — Implementierung von nova_ws.h (Design §13.5, §14).
#include "Web/nova_ws.h"

#include <cctype>

namespace nova::web {

std::string ToolCall::param(const std::string& key) const {
    for (const auto& [k, v] : params) if (k == key) return v;
    return {};
}
bool ToolCall::has(const std::string& key) const {
    for (const auto& [k, v] : params) if (k == key) return true;
    return false;
}

namespace {
const std::string kTool = "tool_call";
const std::string kApex = "apex_call";

bool is_prefix_of(const std::string& full, const std::string& part) {
    return part.size() <= full.size() && full.compare(0, part.size(), part) == 0;
}
bool is_name_term(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '>' || c == '/';
}
}  // namespace

bool StreamInterceptor::parse_attrs(const std::string& tag,
                                    std::vector<std::pair<std::string, std::string>>& out) {
    // tag ist z.B.  <tool_call name="wetter" location="Hamburg"/>
    // Extrahiere alle  key="value"  Paare.
    out.clear();
    size_t i = 0;
    while (i < tag.size()) {
        // Suche Schlüsselanfang (Buchstabe/_).
        while (i < tag.size() && !(std::isalpha((unsigned char)tag[i]) || tag[i] == '_')) ++i;
        if (i >= tag.size()) break;
        size_t ks = i;
        while (i < tag.size() && (std::isalnum((unsigned char)tag[i]) || tag[i] == '_')) ++i;
        std::string key = tag.substr(ks, i - ks);
        // Erwarte '=' '"'
        while (i < tag.size() && (tag[i] == ' ' || tag[i] == '\t')) ++i;
        if (i >= tag.size() || tag[i] != '=') continue;
        ++i;
        while (i < tag.size() && (tag[i] == ' ' || tag[i] == '\t')) ++i;
        if (i >= tag.size() || tag[i] != '"') continue;
        ++i;
        size_t vs = i;
        while (i < tag.size() && tag[i] != '"') ++i;
        std::string val = tag.substr(vs, i - vs);
        if (i < tag.size()) ++i;  // schließendes '"'
        out.push_back({key, val});
    }
    return !out.empty();
}

void StreamInterceptor::feed(const std::string& chunk) {
    buf_ += chunk;
    process();
}

void StreamInterceptor::process() {
    size_t scan = 0;
    while (scan < buf_.size()) {
        const size_t lt = buf_.find('<', scan);
        if (lt == std::string::npos) {
            emit_text(buf_.substr(scan));
            scan = buf_.size();
            break;
        }
        // Text vor '<' ausgeben.
        if (lt > scan) emit_text(buf_.substr(scan, lt - scan));

        // Tag-Namen lesen.
        size_t i = lt + 1;
        std::string name;
        while (i < buf_.size() && !is_name_term(buf_[i])) name += buf_[i++];
        const bool terminated = (i < buf_.size());  // Terminator-Zeichen gesehen?

        if (!terminated) {
            // Name evtl. unvollständig. Könnte noch ein bekanntes Tag werden?
            if (is_prefix_of(kTool, name) || is_prefix_of(kApex, name)) {
                buf_.erase(0, lt);  // ab '<' behalten, auf mehr warten
                return;
            }
            // Divergiert -> '<' ist literaler Text.
            emit_text("<");
            scan = lt + 1;
            continue;
        }

        const bool known = (name == kTool || name == kApex);
        if (!known) { emit_text("<"); scan = lt + 1; continue; }

        // Bekanntes Tag: schließendes '>' suchen.
        const size_t gt = buf_.find('>', i);
        if (gt == std::string::npos) { buf_.erase(0, lt); return; }  // Tag unvollständig

        const std::string tag = buf_.substr(lt, gt - lt + 1);
        std::vector<std::pair<std::string, std::string>> attrs;
        parse_attrs(tag, attrs);
        if (name == kTool) {
            ToolCall tc;
            for (auto& [k, v] : attrs) {
                if (k == "name") tc.name = v;
                else tc.params.push_back({k, v});
            }
            if (tool_) tool_(tc);
        } else {
            ApexCall ac;
            for (auto& [k, v] : attrs) {
                if (k == "skill") ac.skill = v;
                else if (k == "task") ac.task = v;
            }
            if (apex_) apex_(ac);
        }
        scan = gt + 1;
    }
    buf_.erase(0, scan);
}

void StreamInterceptor::finish() {
    if (!buf_.empty()) { emit_text(buf_); buf_.clear(); }
}

void ChatHandler::stream(ModelFn model, Sink sink, ToolExec tool) const {
    StreamInterceptor it;
    it.on_text([&](const std::string& s) { sink({ChatEvent::Text, s, {}, {}}); });
    it.on_tool([&](const ToolCall& tc) {
        sink({ChatEvent::Tool, "", tc, {}});
        if (tool) {
            const std::string obs = tool(tc);
            sink({ChatEvent::Text, obs, {}, {}});  // OBS eingebettet
        }
    });
    it.on_apex([&](const ApexCall& ac) { sink({ChatEvent::Apex, "", {}, ac}); });

    for (;;) {
        const std::string tok = model();
        if (tok.empty()) break;
        it.feed(tok);
    }
    it.finish();
    sink({ChatEvent::Done, "", {}, {}});
}

}  // namespace nova::web

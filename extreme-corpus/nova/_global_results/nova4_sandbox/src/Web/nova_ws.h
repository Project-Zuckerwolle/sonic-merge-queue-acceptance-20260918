// nova_ws.h — Chat-Handler + Tool/Apex-Call-Interceptor (Design §13.5, §14).
//
// Nova generiert strukturierte Tags im Output-Stream:
//   <tool_call name="wetter" location="Hamburg"/>   einmaliges Normal-Tool
//   <apex_call skill="coding" task="..."/>           startet ReAct-Loop (Block E)
//
// StreamInterceptor verarbeitet den Token-Stream INKREMENTELL: normaler Text
// fließt sofort zum Sink, sobald sicher ist dass kein Tag-Präfix mehr offen ist;
// vollständige Tags werden geparst und als Event gemeldet. So bleibt Streaming
// flüssig (TB 10), während Tags abgefangen werden (TB 12, Block E).
#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace nova::web {

struct ToolCall {
    std::string name;
    std::vector<std::pair<std::string, std::string>> params;
    std::string param(const std::string& key) const;
    bool has(const std::string& key) const;
};

struct ApexCall {
    std::string skill;
    std::string task;
};

class StreamInterceptor {
public:
    using TextFn = std::function<void(const std::string&)>;
    using ToolFn = std::function<void(const ToolCall&)>;
    using ApexFn = std::function<void(const ApexCall&)>;

    void on_text(TextFn f) { text_ = std::move(f); }
    void on_tool(ToolFn f) { tool_ = std::move(f); }
    void on_apex(ApexFn f) { apex_ = std::move(f); }

    void feed(const std::string& chunk);  // gestreamte Tokens
    void finish();                        // restlichen Text flushen

    // Parser-Helfer (öffentlich für Tests).
    static bool parse_attrs(const std::string& tag,
                            std::vector<std::pair<std::string, std::string>>& out);

private:
    void process();
    void emit_text(const std::string& s) { if (text_ && !s.empty()) text_(s); }

    std::string buf_;
    TextFn text_;
    ToolFn tool_;
    ApexFn apex_;
};

// Tie-together: Modell-Tokenquelle -> Interceptor -> Phrase-Filter -> Sink.
// ModelFn liefert das nächste Token (leer == fertig). Sink bekommt fertigen
// Anzeigetext. ToolExec führt ein Normal-Tool aus und liefert die OBS zurück.
struct ChatEvent {
    enum Type { Text, Tool, Apex, Done } type;
    std::string text;        // bei Text
    ToolCall    tool;        // bei Tool
    ApexCall    apex;        // bei Apex
};

class ChatHandler {
public:
    using ModelFn = std::function<std::string()>;                 // nächstes Token
    using Sink    = std::function<void(const ChatEvent&)>;
    using ToolExec = std::function<std::string(const ToolCall&)>; // -> OBS

    // Streamt die Antwort. Normal-Tools werden ausgeführt und ihre OBS als
    // Text-Event eingebettet; Apex-Calls werden nur gemeldet (Loop = Block E).
    void stream(ModelFn model, Sink sink, ToolExec tool = {}) const;
};

}  // namespace nova::web

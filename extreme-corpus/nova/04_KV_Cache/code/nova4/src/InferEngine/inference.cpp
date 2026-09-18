// inference.cpp — Default-Impl + Adapter für IInference (Aufgabe 0).
#include "InferEngine/inference.h"

namespace nova::infer {

std::string IInference::complete(const GenRequest& req) {
    begin(req);
    std::string out;
    for (;;) {
        const std::string t = next_token();
        if (t.empty()) break;
        out += t;
    }
    return out;
}

std::function<std::string()> as_token_source(IInference& eng) {
    return [&eng]() -> std::string { return eng.next_token(); };
}

std::function<std::string(const std::string&)> as_llm_fn(IInference& eng,
                                                         std::string prefix) {
    return [&eng, prefix = std::move(prefix)](const std::string& prompt) -> std::string {
        GenRequest req;
        req.prefix = prefix;
        req.dynamic = prompt;
        req.instruct = true;  // Hintergrund-LLM (Dreaming/Episode/Brain) braucht die Instruct-Hülle,
                              // sonst rambelt/wiederholt das Instruct-Modell -> Garbage in identity/episodes.
        return eng.complete(req);
    };
}

}  // namespace nova::infer

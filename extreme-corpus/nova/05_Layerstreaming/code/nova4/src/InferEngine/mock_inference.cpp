// mock_inference.cpp — Implementierung der deterministischen CPU-Mock (Aufgabe 0).
#include "InferEngine/mock_inference.h"

#include <cctype>

namespace nova::infer {

namespace {

std::string to_lower_ascii(const std::string& s) {
    std::string o = s;
    for (char& c : o) c = char(std::tolower((unsigned char)c));
    return o;
}

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

// Nimmt das Wort nach " in " (Buchstaben) als Ort; "" wenn keins.
std::string extract_location(const std::string& text) {
    const std::string low = to_lower_ascii(text);
    const size_t p = low.find(" in ");
    if (p == std::string::npos) return {};
    size_t i = p + 4;
    while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
    size_t s = i;
    // Buchstaben (inkl. UTF-8-Fortsetzungsbytes) bis Leerraum/Satzzeichen.
    while (i < text.size()) {
        const unsigned char ch = (unsigned char)text[i];
        if (std::isalpha(ch) || ch >= 0x80) ++i; else break;
    }
    if (i == s) return {};
    return text.substr(s, i - s);
}

}  // namespace

std::string MockInference::extract_expr(const std::string& text) {
    const size_t n = text.size();
    size_t i = 0;
    while (i < n && !std::isdigit((unsigned char)text[i])) ++i;
    if (i >= n) return {};

    std::string acc;
    bool has_op = false;
    for (size_t j = i; j < n; ++j) {
        const char ch = text[j];
        if (std::isdigit((unsigned char)ch) || ch == '.' || ch == '(' || ch == ')') {
            acc += ch;
        } else if (ch == '+' || ch == '-' || ch == '*' || ch == '/') {
            acc += ch;
            has_op = true;
        } else if (ch == ' ' || ch == '\t') {
            continue;  // Leerraum zwischen Termen erlauben ("2 + 3" -> "2+3")
        } else {
            break;     // Buchstabe/Satzzeichen beendet den Ausdruck
        }
    }
    while (!acc.empty()) {
        const char b = acc.back();
        if (b == '+' || b == '-' || b == '*' || b == '/' || b == '.') acc.pop_back();
        else break;
    }
    if (!has_op || acc.empty()) return {};
    return acc;
}

std::vector<std::string> MockInference::build_response(const std::string& user_text) {
    const std::string low = to_lower_ascii(user_text);
    const std::string expr = extract_expr(user_text);

    if (!expr.empty()) {
        // Arithmetik -> rechner-Tool (Tag über mehrere Tokens verteilt).
        return {
            "Ich ", "rechne ", "das ", "kurz ", "aus", ": ",
            "<tool_call name=\"rechner\" ", "expr=\"" + expr + "\"/>",
            " — ", "fertig", ".",
        };
    }
    if (contains(low, "wetter")) {
        std::string loc = extract_location(user_text);
        if (loc.empty()) loc = "Hamburg";
        return {
            "Einen ", "Moment", ", ", "ich ", "prüfe ", "das ", "Wetter ",
            "in ", loc, ": ",
            "<tool_call name=\"wetter\" ", "location=\"" + loc + "\"/>",
            " Gleich ", "da", ".",
        };
    }
    // Generische kohärente Antwort.
    return {
        "Hallo", "! ", "Ich ", "bin ", "Nova 4", ", ", "dein ", "lokaler ",
        "Assistent", ". ", "Wie ", "kann ", "ich ", "helfen", "?",
    };
}

void MockInference::begin(const GenRequest& req) {
    tokens_ = build_response(req.dynamic);
    idx_ = 0;
}

std::string MockInference::next_token() {
    if (idx_ >= tokens_.size()) return {};
    return tokens_[idx_++];
}

}  // namespace nova::infer

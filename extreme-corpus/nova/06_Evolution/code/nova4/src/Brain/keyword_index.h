// keyword_index.h — BM25 Keyword-Index (Design §11.3, Testbed 9).
//
// CPU-seitig. Index über wiki/-Dateinamen + erste 5 Zeilen. Pro User-Message
// liefert query() die Top-15 Treffer. Normalisierte Scores (geteilt durch den
// Top-Score); Schwellwert 0.3 greift erst ab >= 50 Wiki-Seiten (§11.3), darunter
// deaktiviert. Ziel: < 10 ms.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nova::brain {

struct Bm25Doc {
    std::string id;    // z.B. Wiki-Dateiname
    std::string text;  // Dateiname + erste 5 Zeilen
};

struct Bm25Hit {
    std::string id;
    double      score = 0.0;       // roher BM25-Score
    double      norm_score = 0.0;  // score / top_score
};

class KeywordIndex {
public:
    // BM25-Parameter (Standardwerte).
    struct Params { double k1 = 1.5; double b = 0.75;
                    double norm_threshold = 0.3; int threshold_min_docs = 50; };

    explicit KeywordIndex(Params p = {}) : p_(p) {}

    void build(const std::vector<Bm25Doc>& docs);

    // Top-k Treffer für q, absteigend nach Score. Bei >= threshold_min_docs
    // werden Treffer mit norm_score < norm_threshold verworfen.
    std::vector<Bm25Hit> query(const std::string& q, int top_k = 15) const;

    size_t doc_count() const { return docs_.size(); }

    // Tokenizer: ASCII-lowercase, Split an Nicht-Alphanumerischem; UTF-8-
    // Mehrbyte-Sequenzen (Umlaute) bleiben Teil des Tokens.
    static std::vector<std::string> tokenize(const std::string& s);

private:
    Params p_;
    std::vector<std::string> docs_;          // ids
    std::vector<int>         doc_len_;        // Tokenanzahl je Doc
    double                   avgdl_ = 0.0;
    // term -> (doc_index -> term frequency)
    std::unordered_map<std::string, std::vector<std::pair<int, int>>> postings_;
    std::unordered_map<std::string, double> idf_;
};

}  // namespace nova::brain

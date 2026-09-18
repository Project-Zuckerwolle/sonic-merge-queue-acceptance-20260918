// keyword_index.cpp — Implementierung von keyword_index.h (Design §11.3).
#include "Brain/keyword_index.h"

#include <algorithm>
#include <cmath>

namespace nova::brain {

std::vector<std::string> KeywordIndex::tokenize(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (unsigned char c : s) {
        const bool alnum = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
                           (c >= 'A' && c <= 'Z') || (c >= 0x80);  // UTF-8 Mehrbyte
        if (alnum) {
            cur.push_back(char(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c));
        } else if (!cur.empty()) {
            out.push_back(cur); cur.clear();
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

void KeywordIndex::build(const std::vector<Bm25Doc>& docs) {
    docs_.clear(); doc_len_.clear(); postings_.clear(); idf_.clear();
    docs_.reserve(docs.size());
    doc_len_.reserve(docs.size());

    // term -> tf je Doc; gleichzeitig Dokumentlängen.
    long total_len = 0;
    std::vector<std::unordered_map<std::string, int>> tf_per_doc(docs.size());
    for (size_t d = 0; d < docs.size(); ++d) {
        docs_.push_back(docs[d].id);
        const auto toks = tokenize(docs[d].text);
        doc_len_.push_back(int(toks.size()));
        total_len += long(toks.size());
        for (const auto& t : toks) tf_per_doc[d][t]++;
    }
    avgdl_ = docs.empty() ? 0.0 : double(total_len) / docs.size();

    for (size_t d = 0; d < docs.size(); ++d)
        for (const auto& [term, tf] : tf_per_doc[d])
            postings_[term].push_back({int(d), tf});

    const double N = double(docs.size());
    for (const auto& [term, plist] : postings_) {
        const double df = double(plist.size());
        idf_[term] = std::log((N - df + 0.5) / (df + 0.5) + 1.0);  // BM25+ IDF, immer > 0
    }
}

std::vector<Bm25Hit> KeywordIndex::query(const std::string& q, int top_k) const {
    std::vector<double> score(docs_.size(), 0.0);
    const auto qterms = tokenize(q);
    for (const auto& t : qterms) {
        auto pit = postings_.find(t);
        if (pit == postings_.end()) continue;
        const double idf = idf_.at(t);
        for (const auto& [d, tf] : pit->second) {
            const double denom = tf + p_.k1 * (1.0 - p_.b + p_.b * doc_len_[d] / (avgdl_ + 1e-9));
            score[d] += idf * (tf * (p_.k1 + 1.0)) / (denom + 1e-9);
        }
    }

    std::vector<Bm25Hit> hits;
    hits.reserve(docs_.size());
    for (size_t d = 0; d < docs_.size(); ++d)
        if (score[d] > 0.0) hits.push_back({docs_[d], score[d], 0.0});

    std::sort(hits.begin(), hits.end(),
              [](const Bm25Hit& a, const Bm25Hit& b) { return a.score > b.score; });

    const double top = hits.empty() ? 0.0 : hits[0].score;
    for (auto& h : hits) h.norm_score = top > 0.0 ? h.score / top : 0.0;

    if (int(docs_.size()) >= p_.threshold_min_docs) {
        hits.erase(std::remove_if(hits.begin(), hits.end(),
                                  [&](const Bm25Hit& h) { return h.norm_score < p_.norm_threshold; }),
                   hits.end());
    }
    if (int(hits.size()) > top_k) hits.resize(top_k);
    return hits;
}

}  // namespace nova::brain

// spec_decoder.cpp — Implementierung von spec_decoder.h (Design §8).
#include "InferEngine/spec_decoder.h"

#include <algorithm>
#include <cmath>

namespace nova::infer {

int SpeculativeDecoder::sample_from(const ProbDist& dist, double u) {
    double acc = 0.0;
    for (size_t i = 0; i < dist.size(); ++i) {
        acc += dist[i];
        if (u < acc) return int(i);
    }
    // Fallback (Rundungsreste): letzter Index mit Wahrscheinlichkeit > 0.
    for (int i = int(dist.size()) - 1; i >= 0; --i)
        if (dist[i] > 0.0f) return i;
    return 0;
}

SpecStep SpeculativeDecoder::step(std::vector<int>& ctx) const {
    const int K = cfg_.draft_len;
    SpecStep out;
    out.proposed = K;

    std::vector<int> draft;
    std::vector<ProbDist> q;
    draft_(ctx, K, draft, q);

    std::vector<ProbDist> p;
    target_(ctx, draft, p);
    // p muss K+1 Verteilungen liefern; defensiv kürzen/abbrechen.
    if (int(p.size()) < K + 1) {
        // Ohne gültige Target-Ausgabe nichts emittieren.
        return out;
    }

    for (int i = 0; i < K; ++i) {
        const int x = draft[i];
        const double px = (x < int(p[i].size())) ? p[i][x] : 0.0;
        const double qx = (x < int(q[i].size())) ? q[i][x] : 0.0;
        const double ratio = qx > 0.0 ? std::min(1.0, px / qx) : 1.0;
        if (uni_() <= ratio) {
            // Akzeptiert.
            out.tokens.push_back(x);
            ctx.push_back(x);
            ++out.accepted;
        } else {
            // Verworfen: aus der Residualverteilung (p - q)_+ neu sampeln, Stopp.
            ProbDist resid(p[i].size(), 0.0f);
            double sum = 0.0;
            for (size_t v = 0; v < resid.size(); ++v) {
                const double qv = (v < q[i].size()) ? q[i][v] : 0.0;
                const double r = double(p[i][v]) - qv;
                if (r > 0.0) { resid[v] = float(r); sum += r; }
            }
            int tok;
            if (sum > 0.0) {
                for (auto& r : resid) r = float(r / sum);
                tok = sample_from(resid, uni_());
            } else {
                tok = sample_from(p[i], uni_());  // entartet -> direkt aus p
            }
            out.tokens.push_back(tok);
            ctx.push_back(tok);
            return out;  // Schritt endet nach erstem Reject
        }
    }

    // Alle K akzeptiert -> Bonus-Token aus p[K] (kostenlos im selben Pass).
    const int bonus = sample_from(p[K], uni_());
    out.tokens.push_back(bonus);
    ctx.push_back(bonus);
    out.bonus = true;
    return out;
}

SpeculativeDecoder::RunStats SpeculativeDecoder::run(std::vector<int>& ctx,
                                                     int target_len, int eos) const {
    RunStats s;
    while (int(ctx.size()) < target_len) {
        const SpecStep st = step(ctx);
        if (st.tokens.empty()) break;  // Target lieferte nichts -> Abbruch
        ++s.steps;
        s.proposed += st.proposed;
        s.accepted += st.accepted;
        s.emitted  += int(st.tokens.size());
        if (eos >= 0 &&
            std::find(st.tokens.begin(), st.tokens.end(), eos) != st.tokens.end())
            break;
    }
    return s;
}

}  // namespace nova::infer

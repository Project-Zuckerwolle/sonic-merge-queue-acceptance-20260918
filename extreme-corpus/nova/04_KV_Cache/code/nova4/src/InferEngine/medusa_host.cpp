// medusa_host.cpp — Host-Referenz für medusa.h (Design §9).
// Spiegelt die Köpfe-Projektion aus medusa.cu; die Verifikation ist reine Logik.
#include "InferEngine/medusa.h"

#include <algorithm>

namespace nova::infer {

int MedusaDecoder::argmax(const ProbDist& d) {
    int best = 0;
    float bv = d.empty() ? 0.0f : d[0];
    for (int i = 1; i < int(d.size()); ++i)
        if (d[i] > bv) { bv = d[i]; best = i; }
    return best;
}

MedusaStep MedusaDecoder::step(std::vector<int>& ctx) const {
    const int H = cfg_.num_heads;
    MedusaStep out;
    out.proposed = H;

    std::vector<int> heads;
    std::vector<ProbDist> hdist;
    propose_(ctx, heads, hdist);

    std::vector<ProbDist> p;
    target_(ctx, heads, p);
    if (int(p.size()) < H + 1) return out;  // ungültige Target-Ausgabe

    for (int i = 0; i < H; ++i) {
        const int tok = heads[i];
        const int am  = argmax(p[i]);
        bool accept;
        if (cfg_.typical_threshold > 0.0f)
            accept = (tok < int(p[i].size())) && p[i][tok] >= cfg_.typical_threshold;
        else
            accept = (tok == am);
        if (!accept) {
            // Mismatch: emittiere das Target-eigene Token an dieser Position, Stopp.
            out.tokens.push_back(am);
            ctx.push_back(am);
            return out;
        }
        out.tokens.push_back(tok);
        ctx.push_back(tok);
        ++out.accepted;
    }

    // Alle Köpfe akzeptiert -> Bonus-Token aus p[H].
    const int bonus = argmax(p[H]);
    out.tokens.push_back(bonus);
    ctx.push_back(bonus);
    return out;
}

MedusaDecoder::RunStats MedusaDecoder::run(std::vector<int>& ctx,
                                           int target_len, int eos) const {
    RunStats s;
    while (int(ctx.size()) < target_len) {
        const MedusaStep st = step(ctx);
        if (st.tokens.empty()) break;
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

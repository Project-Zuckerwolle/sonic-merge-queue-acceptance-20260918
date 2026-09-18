// mistral3_forward.cpp — Implementierung von mistral3_forward.h (CPU-F32-Referenz).
#include "InferEngine/mistral3_forward.h"

#include <cmath>

namespace nova::infer {

bool Mistral3Reference::load(const std::string& path, const Mistral3Config& cfg,
                             bool rope_interleaved, std::string* err) {
    cfg_ = cfg; interleaved_ = rope_interleaved;
    if (!gg_.open(path, err)) return false;
    for (const auto& t : gg_.tensors()) tmap_[t.name] = &t;
    return true;
}

std::vector<float> Mistral3Reference::W(const std::string& name) {
    auto it = tmap_.find(name);
    if (it == tmap_.end()) return {};
    std::vector<float> out; std::string e;
    gg_.read_tensor_f32(*it->second, out, &e);
    return out;
}

namespace {
// y[o] = sum_i x[i] * Wrow_o[i], W row-major mit ne0=in (Zeile o bei o*in).
void matvec(const float* x, const std::vector<float>& Wm, int in, int out, float* y) {
    for (int o = 0; o < out; ++o) {
        const float* w = Wm.data() + size_t(o) * in;
        double acc = 0; for (int i = 0; i < in; ++i) acc += double(x[i]) * w[i];
        y[o] = float(acc);
    }
}
void rmsnorm(const float* x, const std::vector<float>& g, int H, float eps, float* out) {
    double ss = 0; for (int i = 0; i < H; ++i) ss += double(x[i]) * x[i];
    const float inv = float(1.0 / std::sqrt(ss / H + eps));
    for (int i = 0; i < H; ++i) out[i] = x[i] * inv * g[i];
}
// RoPE fuer ein [S, nHeads, hd]-Layout (flach S*nHeads*hd), Position = s.
void rope(std::vector<float>& t, int S, int nHeads, int hd, float theta, bool interleaved) {
    const int half = hd / 2;
    for (int s = 0; s < S; ++s)
        for (int h = 0; h < nHeads; ++h) {
            float* v = t.data() + (size_t(s) * nHeads + h) * hd;
            for (int i = 0; i < half; ++i) {
                const float freq = std::pow(theta, -2.0f * i / hd);
                const float ang = s * freq, c = std::cos(ang), sn = std::sin(ang);
                int a = interleaved ? 2 * i : i;
                int b = interleaved ? 2 * i + 1 : i + half;
                const float va = v[a], vb = v[b];
                v[a] = va * c - vb * sn;
                v[b] = va * sn + vb * c;
            }
        }
}
float silu(float x) { return x / (1.0f + std::exp(-x)); }
}  // namespace

std::vector<float> Mistral3Reference::forward(const std::vector<int>& tokens) {
    const int H = cfg_.hidden, nH = cfg_.n_heads, nKV = cfg_.n_kv_heads, hd = cfg_.head_dim;
    const int qd = nH * hd, kvd = nKV * hd, F = cfg_.ffn, V = cfg_.vocab;
    const int S = int(tokens.size());
    const int grp = nH / nKV;
    const float eps = cfg_.rms_eps, theta = cfg_.rope_theta;
    const float scale = 1.0f / std::sqrt(float(hd));

    // Embeddings.
    std::vector<float> x(size_t(S) * H);
    { auto emb = W("token_embd.weight");
      for (int s = 0; s < S; ++s)
          for (int h = 0; h < H; ++h) x[size_t(s)*H+h] = emb[size_t(tokens[s])*H + h]; }

    std::vector<float> xn(size_t(S)*H), q(size_t(S)*qd), k(size_t(S)*kvd), v(size_t(S)*kvd);
    std::vector<float> ctx(size_t(S)*qd), tmpH(size_t(S)*H), g(size_t(S)*F), u(size_t(S)*F);

    for (int l = 0; l < cfg_.n_layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        { auto an = W(p+"attn_norm.weight");
          for (int s=0;s<S;++s) rmsnorm(&x[size_t(s)*H], an, H, eps, &xn[size_t(s)*H]); }
        { auto Wq=W(p+"attn_q.weight"); for(int s=0;s<S;++s) matvec(&xn[size_t(s)*H],Wq,H,qd,&q[size_t(s)*qd]); }
        { auto Wk=W(p+"attn_k.weight"); for(int s=0;s<S;++s) matvec(&xn[size_t(s)*H],Wk,H,kvd,&k[size_t(s)*kvd]); }
        { auto Wv=W(p+"attn_v.weight"); for(int s=0;s<S;++s) matvec(&xn[size_t(s)*H],Wv,H,kvd,&v[size_t(s)*kvd]); }
        rope(q, S, nH, hd, theta, interleaved_);
        rope(k, S, nKV, hd, theta, interleaved_);

        // GQA-Attention, kausal.
        for (int s = 0; s < S; ++s)
            for (int h = 0; h < nH; ++h) {
                const int kvh = h / grp;
                const float* qv = &q[(size_t(s)*nH + h)*hd];
                std::vector<float> sc(s+1);
                float mx = -1e30f;
                for (int t = 0; t <= s; ++t) {
                    const float* kv = &k[(size_t(t)*nKV + kvh)*hd];
                    double d=0; for(int i=0;i<hd;++i) d += double(qv[i])*kv[i];
                    sc[t]=float(d)*scale; if(sc[t]>mx) mx=sc[t];
                }
                double sum=0; for(int t=0;t<=s;++t){ sc[t]=std::exp(sc[t]-mx); sum+=sc[t]; }
                float* out = &ctx[(size_t(s)*nH + h)*hd];
                for(int i=0;i<hd;++i) out[i]=0;
                for (int t=0;t<=s;++t){ const float w=float(sc[t]/sum); const float* vv=&v[(size_t(t)*nKV+kvh)*hd];
                    for(int i=0;i<hd;++i) out[i]+=w*vv[i]; }
            }
        { auto Wo=W(p+"attn_output.weight");
          for(int s=0;s<S;++s){ matvec(&ctx[size_t(s)*qd],Wo,qd,H,&tmpH[size_t(s)*H]);
              for(int h=0;h<H;++h) x[size_t(s)*H+h]+=tmpH[size_t(s)*H+h]; } }

        // FFN (SwiGLU).
        { auto fn=W(p+"ffn_norm.weight");
          for(int s=0;s<S;++s) rmsnorm(&x[size_t(s)*H],fn,H,eps,&xn[size_t(s)*H]); }
        { auto Wg=W(p+"ffn_gate.weight"); for(int s=0;s<S;++s) matvec(&xn[size_t(s)*H],Wg,H,F,&g[size_t(s)*F]); }
        { auto Wu=W(p+"ffn_up.weight");   for(int s=0;s<S;++s) matvec(&xn[size_t(s)*H],Wu,H,F,&u[size_t(s)*F]); }
        for(size_t i=0;i<size_t(S)*F;++i) g[i]=silu(g[i])*u[i];
        { auto Wd=W(p+"ffn_down.weight");
          for(int s=0;s<S;++s){ matvec(&g[size_t(s)*F],Wd,F,H,&tmpH[size_t(s)*H]);
              for(int h=0;h<H;++h) x[size_t(s)*H+h]+=tmpH[size_t(s)*H+h]; } }
    }

    // Finaler Norm + Logits (nur letzte Position).
    std::vector<float> xf(H);
    { auto on=W("output_norm.weight"); rmsnorm(&x[size_t(S-1)*H], on, H, eps, xf.data()); }
    std::vector<float> logits(V);
    { auto Wout=W("output.weight"); matvec(xf.data(), Wout, H, V, logits.data()); }
    return logits;
}

}  // namespace nova::infer

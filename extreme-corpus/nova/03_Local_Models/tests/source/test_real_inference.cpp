// test_real_inference.cpp — Korrektheit der INT3-KV-Engine (C1).
//
// Teil A (schnell, kein Modell): Mini-Zufallsmodell. Unabhängiger Voll-Forward
// (alle Positionen neu, KEIN KV) vs. inkrementeller forward_step (MIT KV). Gleiche
// validierte Kernels, aber verschiedene Schleifenstruktur -> deckt Assemblierungs-/
// KV-/Positions-Bugs auf. Muss pro Position übereinstimmen (argmax + max|Δ|).
//
// Teil B (optional, nur wenn NOVA_REAL_GGUF gesetzt): echtes 24B-Modell laden,
// "Die Hauptstadt von Deutschland ist" dekodieren, Kohärenz + tok/s messen.
#include "InferEngine/real_inference.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace nova::infer;

namespace {

uint64_t g_rng = 0;  // in main() gesetzt
uint32_t xr() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return uint32_t(g_rng); }
float frand() { return (int(xr() % 2001) - 1000) / 1000.0f; }  // [-1,1]

Int3Matrix rand_mat(int M, int K) {
    std::vector<float> W(size_t(M) * K);
    for (auto& v : W) v = frand() * 0.3f;
    return pack_int3(W.data(), M, K, 32);
}
std::vector<float> rand_norm(int H) {
    std::vector<float> g(H); for (auto& v : g) v = 0.8f + 0.4f * ((xr() % 1000) / 1000.0f); return g;
}

Int3Weights make_tiny_model(Mistral3Config c) {
    Int3Weights w; w.cfg = c; w.interleaved = true;
    const int H = c.hidden, qd = c.q_dim(), kvd = c.kv_dim(), F = c.ffn, V = c.vocab;
    w.mats["token_embd"] = rand_mat(V, H);
    for (int l = 0; l < c.n_layers; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        w.norms[b + "attn_norm"] = rand_norm(H);
        w.mats[b + "attn_q"] = rand_mat(qd, H);
        w.mats[b + "attn_k"] = rand_mat(kvd, H);
        w.mats[b + "attn_v"] = rand_mat(kvd, H);
        w.mats[b + "attn_output"] = rand_mat(H, qd);
        w.norms[b + "ffn_norm"] = rand_norm(H);
        w.mats[b + "ffn_gate"] = rand_mat(F, H);
        w.mats[b + "ffn_up"] = rand_mat(F, H);
        w.mats[b + "ffn_down"] = rand_mat(H, F);
    }
    w.norms["output_norm"] = rand_norm(H);
    w.mats["output"] = rand_mat(V, H);
    return w;
}

// Unabhängiger Voll-Forward über ALLE Positionen (kein KV) — eigene Schleifenstruktur.
std::vector<float> ref_full(const Int3Weights& w, const std::vector<int>& toks) {
    const Mistral3Config& c = w.cfg;
    const int H = c.hidden, nH = c.n_heads, nKV = c.n_kv_heads, hd = c.head_dim;
    const int qd = c.q_dim(), kvd = c.kv_dim(), F = c.ffn, V = c.vocab, grp = c.kv_group();
    const int S = int(toks.size());
    const float eps = c.rms_eps, theta = c.rope_theta, scale = 1.0f / std::sqrt(float(hd));
    auto gemv = [&](const std::string& n, const float* x, float* y, int od) {
        const Int3Matrix* W = w.mat(n); if (!W) { for (int i=0;i<od;++i) y[i]=0; return; }
        fused_int3_gemv(*W, x, y);
    };
    auto rms = [&](const float* x, const std::vector<float>& gg, float* o) {
        double ss=0; for (int i=0;i<H;++i) ss+=double(x[i])*x[i];
        float inv=float(1.0/std::sqrt(ss/H+eps)); for(int i=0;i<H;++i) o[i]=x[i]*inv*gg[i]; };
    auto rope = [&](std::vector<float>& t, int nh) {
        const int half=hd/2;
        for(int s=0;s<S;++s) for(int h=0;h<nh;++h){ float* v=&t[(size_t(s)*nh+h)*hd];
            for(int i=0;i<half;++i){ float fr=std::pow(theta,-2.0f*i/hd); float a=s*fr,cs=std::cos(a),sn=std::sin(a);
                int p=2*i,q=2*i+1; float va=v[p],vb=v[q]; v[p]=va*cs-vb*sn; v[q]=va*sn+vb*cs; } } };
    std::vector<float> x(size_t(S)*H);
    { const Int3Matrix* emb=w.mat("token_embd");
      for(int s=0;s<S;++s) dequant_row(*emb, toks[s], &x[size_t(s)*H]); }
    std::vector<float> xn(size_t(S)*H), q(size_t(S)*qd), k(size_t(S)*kvd), v(size_t(S)*kvd),
                       ctx(size_t(S)*qd), th(size_t(S)*H), gg(size_t(S)*F), uu(size_t(S)*F);
    for (int l=0;l<c.n_layers;++l){ const std::string b="blk."+std::to_string(l)+".";
        for(int s=0;s<S;++s) rms(&x[size_t(s)*H], *w.norm(b+"attn_norm"), &xn[size_t(s)*H]);
        for(int s=0;s<S;++s) gemv(b+"attn_q",&xn[size_t(s)*H],&q[size_t(s)*qd],qd);
        for(int s=0;s<S;++s) gemv(b+"attn_k",&xn[size_t(s)*H],&k[size_t(s)*kvd],kvd);
        for(int s=0;s<S;++s) gemv(b+"attn_v",&xn[size_t(s)*H],&v[size_t(s)*kvd],kvd);
        rope(q,nH); rope(k,nKV);
        for(int s=0;s<S;++s) for(int h=0;h<nH;++h){ int kvh=h/grp; const float* qv=&q[(size_t(s)*nH+h)*hd];
            std::vector<float> sc(s+1); float mx=-1e30f;
            for(int t=0;t<=s;++t){ const float* kk=&k[(size_t(t)*nKV+kvh)*hd]; double d=0;
                for(int i=0;i<hd;++i) d+=double(qv[i])*kk[i]; sc[t]=float(d)*scale; if(sc[t]>mx)mx=sc[t]; }
            double sm=0; for(int t=0;t<=s;++t){ sc[t]=std::exp(sc[t]-mx); sm+=sc[t]; }
            float* o=&ctx[(size_t(s)*nH+h)*hd]; for(int i=0;i<hd;++i)o[i]=0;
            for(int t=0;t<=s;++t){ float ww=float(sc[t]/sm); const float* vv=&v[(size_t(t)*nKV+kvh)*hd];
                for(int i=0;i<hd;++i)o[i]+=ww*vv[i]; } }
        for(int s=0;s<S;++s){ gemv(b+"attn_output",&ctx[size_t(s)*qd],&th[size_t(s)*H],H);
            for(int i=0;i<H;++i) x[size_t(s)*H+i]+=th[size_t(s)*H+i]; }
        for(int s=0;s<S;++s) rms(&x[size_t(s)*H], *w.norm(b+"ffn_norm"), &xn[size_t(s)*H]);
        for(int s=0;s<S;++s) gemv(b+"ffn_gate",&xn[size_t(s)*H],&gg[size_t(s)*F],F);
        for(int s=0;s<S;++s) gemv(b+"ffn_up",&xn[size_t(s)*H],&uu[size_t(s)*F],F);
        for(size_t i=0;i<size_t(S)*F;++i) gg[i]=(gg[i]/(1.0f+std::exp(-gg[i])))*uu[i];
        for(int s=0;s<S;++s){ gemv(b+"ffn_down",&gg[size_t(s)*F],&th[size_t(s)*H],H);
            for(int i=0;i<H;++i) x[size_t(s)*H+i]+=th[size_t(s)*H+i]; } }
    std::vector<float> xf(H); rms(&x[size_t(S-1)*H], *w.norm("output_norm"), xf.data());
    std::vector<float> lg(V,0.0f); gemv("output", xf.data(), lg.data(), V);
    return lg;
}

int argmax(const std::vector<float>& v){ int b=0; for(int i=1;i<int(v.size());++i) if(v[i]>v[b])b=i; return b; }

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // ungepuffert: Teilergebnisse überleben Abbruch
    g_rng = 0x9E3779B97F4A7C15ULL;
    Mistral3Config c; c.n_layers=3; c.n_heads=4; c.n_kv_heads=2; c.head_dim=16;
    c.hidden=64; c.ffn=128; c.vocab=48; c.rope_theta=10000.0f; c.rms_eps=1e-5f;

    Int3Weights w = make_tiny_model(c);
    std::vector<int> toks = {1, 5, 9, 2, 7, 3, 11, 4};

    // Inkrementell (forward_step + KV) vs. Voll-Forward (ref_full) je Position.
    KvCacheF32 kv; kv.reset(c.n_layers);
    float worst = 0.0f; int mismatches = 0;
    for (int i = 0; i < int(toks.size()); ++i) {
        std::vector<float> inc = forward_step(w, toks[i], i, kv);
        std::vector<int> pre(toks.begin(), toks.begin() + i + 1);
        std::vector<float> ref = ref_full(w, pre);
        float md = 0.0f; for (int j = 0; j < c.vocab; ++j) md = std::max(md, std::fabs(inc[j] - ref[j]));
        if (argmax(inc) != argmax(ref)) ++mismatches;
        worst = std::max(worst, md);
        std::printf("pos %d: max|Δ|=%.2e argmax inc=%d ref=%d\n", i, md, argmax(inc), argmax(ref));
    }
    std::printf("A) worst max|Δ|=%.3e, argmax-mismatches=%d\n", worst, mismatches);
    bool okA = (worst < 1e-2f) && (mismatches == 0);
    std::printf("TESTBED RealInference-A (KV/inkrementell == Voll-Forward): %s\n", okA ? "BESTANDEN" : "FEHLGESCHLAGEN");

    // A2: forward_batch(N) == N × forward_step (Spec-Verify-Korrektheit).
    const int V = c.vocab;
    const int split = 3;                         // 3 prefillen, Rest batchen
    KvCacheF32 kvs; kvs.reset(c.n_layers);
    for (int i = 0; i < split; ++i) forward_step(w, toks[i], i, kvs);
    KvCacheF32 kvb = kvs;                         // gleicher Prefill-Zustand
    const int Nrem = int(toks.size()) - split;
    std::vector<int> rest(toks.begin() + split, toks.end());
    std::vector<float> batch = forward_batch(w, rest.data(), Nrem, split, kvb);
    float worstB = 0.0f; int misB = 0;
    for (int n = 0; n < Nrem; ++n) {
        std::vector<float> seq = forward_step(w, rest[n], split + n, kvs);
        const float* br = &batch[size_t(n) * V];
        int ab = 0, as = 0; float md = 0.0f;
        for (int j = 0; j < V; ++j) { md = std::max(md, std::fabs(br[j] - seq[j]));
            if (br[j] > br[ab]) ab = j; if (seq[j] > seq[as]) as = j; }
        if (ab != as) ++misB; worstB = std::max(worstB, md);
    }
    std::printf("A2) forward_batch vs seq: worst max|Δ|=%.3e, argmax-mismatches=%d\n", worstB, misB);
    bool okA2 = (worstB < 1e-2f) && (misB == 0);
    std::printf("TESTBED RealInference-A2 (batched == sequenziell): %s\n", okA2 ? "BESTANDEN" : "FEHLGESCHLAGEN");
    okA = okA && okA2;

    // D: Spec-Decoding (3B-Draft-Analog) == greedy Target — Token-für-Token identisch.
    {
        Int3Weights tgt = make_tiny_model(c);
        g_rng = 0xD1FFD1FFD1FFULL;               // anderer Seed -> Draft != Target (Rejects)
        Int3Weights drf = make_tiny_model(c);
        GenRequest req; req.dynamic = "";        // ohne Tokenizer: Prompt = [bos]
        RealInference eg; eg.set_weights(tgt);
        eg.begin(req);
        std::vector<int> greedy_ids;
        for (int i = 0; i < 24; ++i) { int t = eg.next_token_id(); if (t < 0) break; greedy_ids.push_back(t); }
        RealInference es; es.set_weights(tgt); es.set_draft_weights(drf); es.set_spec_k(4);
        es.begin(req);
        std::vector<int> spec_ids;
        for (int i = 0; i < 24; ++i) { int t = es.next_token_id(); if (t < 0) break; spec_ids.push_back(t); }
        bool okD = (greedy_ids == spec_ids) && !greedy_ids.empty();
        std::printf("D) greedy=%zu spec=%zu Tokens, accept-rate=%.2f, identisch=%d\n",
                    greedy_ids.size(), spec_ids.size(), es.last_accept_rate(), int(greedy_ids == spec_ids));
        std::printf("TESTBED RealInference-D (Spec == greedy Target): %s\n", okD ? "BESTANDEN" : "FEHLGESCHLAGEN");
        okA = okA && okD;
    }

    // E: on-GPU-Forward (GpuForward) == CPU forward_step, gleiche Token-Sequenz (kein Kaskaden-Drift).
#ifdef NOVA_HAVE_CUDA
    {
        Int3Weights we = make_tiny_model(c);
        GpuForward gf; std::string ge;
        if (!gf.init(we, /*max_seq=*/64, &ge)) { std::printf("E) GpuForward init: %s\n", ge.c_str()); }
        else {
            KvCacheF32 kve; kve.reset(c.n_layers);
            std::vector<float> glog(V);
            float worstE = 0.0f; int misE = 0;
            for (int i = 0; i < int(toks.size()); ++i) {
                std::vector<float> cpu = forward_step(we, toks[i], i, kve);   // CPU (gemv auf GPU)
                gf.step_argmax(toks[i], i, glog.data());                     // komplett on-GPU
                int ac = argmax(cpu), ag = 0; float md = 0.0f;
                for (int j = 0; j < V; ++j) { md = std::max(md, std::fabs(cpu[j] - glog[j])); if (glog[j] > glog[ag]) ag = j; }
                if (ac != ag) ++misE; worstE = std::max(worstE, md);
            }
            std::printf("E) GpuForward vs CPU: worst max|Δ|=%.3e, argmax-mismatches=%d\n", worstE, misE);
            bool okE = (worstE < 1e-2f) && (misE == 0);
            std::printf("TESTBED RealInference-E (on-GPU-Forward == CPU): %s\n", okE ? "BESTANDEN" : "FEHLGESCHLAGEN");
            okA = okA && okE;
        }
    }

    // E2: STREAMED on-GPU-Forward (GpuForward streamed=true, Gewichte NICHT resident) == CPU forward_step.
    //     Beweist die gestreamte Device-Pipeline (24B-Pfad, nimmt CPU aus dem Loop) token-identisch.
    {
        Int3Weights we = make_tiny_model(c);
        GpuForward gf; std::string ge;
        if (!gf.init(we, /*max_seq=*/64, &ge, /*streamed=*/true)) { std::printf("E2) init: %s\n", ge.c_str()); }
        else {
            KvCacheF32 kve; kve.reset(c.n_layers);
            std::vector<float> glog(V);
            float worstE = 0.0f; int misE = 0;
            for (int i = 0; i < int(toks.size()); ++i) {
                std::vector<float> cpu = forward_step(we, toks[i], i, kve);
                gf.step_logits(toks[i], i, glog.data());                     // gestreamt, komplett on-GPU
                int ac = argmax(cpu), ag = 0; float md = 0.0f;
                for (int j = 0; j < V; ++j) { md = std::max(md, std::fabs(cpu[j] - glog[j])); if (glog[j] > glog[ag]) ag = j; }
                if (ac != ag) ++misE; worstE = std::max(worstE, md);
            }
            std::printf("E2) Streamed-GPU vs CPU: worst max|Δ|=%.3e, argmax-mismatches=%d\n", worstE, misE);
            bool okE2 = (worstE < 1e-2f) && (misE == 0);
            std::printf("TESTBED RealInference-E2 (streamed-on-GPU-Forward == CPU): %s\n", okE2 ? "BESTANDEN" : "FEHLGESCHLAGEN");
            okA = okA && okE2;
        }
    }

    // E3: batched STREAMED on-GPU forward_batch_gpu == CPU forward_batch (N Spalten in EINEM Weight-Read).
    //     Der Spec-Verify-Pfad des gestreamten 24B. Zeilenweise (Position start_pos+n) token-identisch.
    {
        Int3Weights we = make_tiny_model(c);
        GpuForward gf; std::string ge;
        const int N = std::min<int>(int(toks.size()), GpuForward::MAX_BATCH);
        if (!gf.init(we, /*max_seq=*/64, &ge, /*streamed=*/true)) { std::printf("E3) init: %s\n", ge.c_str()); }
        else {
            KvCacheF32 kvb; kvb.reset(c.n_layers);
            std::vector<float> cpu = forward_batch(we, toks.data(), N, 0, kvb);   // [N][V]
            std::vector<float> gpu(size_t(N) * V);
            gf.forward_batch_gpu(toks.data(), N, 0, gpu.data());                  // [N][V]
            float worst = 0.0f; int mis = 0;
            for (int n = 0; n < N; ++n) {
                int ac = 0, ag = 0;
                for (int j = 0; j < V; ++j) {
                    worst = std::max(worst, std::fabs(cpu[size_t(n) * V + j] - gpu[size_t(n) * V + j]));
                    if (cpu[size_t(n) * V + j] > cpu[size_t(n) * V + ac]) ac = j;
                    if (gpu[size_t(n) * V + j] > gpu[size_t(n) * V + ag]) ag = j;
                }
                if (ac != ag) ++mis;
            }
            std::printf("E3) Streamed-GPU-batch vs CPU: N=%d worst max|Δ|=%.3e, argmax-mismatches=%d\n", N, worst, mis);
            bool okE3 = (worst < 1e-2f) && (mis == 0);
            std::printf("TESTBED RealInference-E3 (streamed-GPU forward_batch == CPU): %s\n", okE3 ? "BESTANDEN" : "FEHLGESCHLAGEN");
            okA = okA && okE3;
        }
    }

    // E3b (Phase 4): N>32 aktiviert die gemm-N-Kachelung (grid.y>1). Streamed forward_batch_gpu == CPU.
    {
        Int3Weights we = make_tiny_model(c);
        GpuForward gf; std::string ge;
        const int N = 40;                              // > 32 -> zwei 32-Spalten-Kacheln
        std::vector<int> t40(N);
        for (int i = 0; i < N; ++i) t40[i] = (i * 7 + 3) % c.vocab;
        if (!gf.init(we, /*max_seq=*/64, &ge, /*streamed=*/true)) { std::printf("E3b) init: %s\n", ge.c_str()); }
        else {
            KvCacheF32 kvb; kvb.reset(c.n_layers);
            std::vector<float> cpu = forward_batch(we, t40.data(), N, 0, kvb);
            std::vector<float> gpu(size_t(N) * V);
            gf.forward_batch_gpu(t40.data(), N, 0, gpu.data());
            float worst = 0.0f; int mis = 0;
            for (int n = 0; n < N; ++n) {
                int ac = 0, ag = 0; float bc = cpu[size_t(n) * V], bg = gpu[size_t(n) * V];
                for (int v = 0; v < V; ++v) {
                    float d = std::fabs(cpu[size_t(n) * V + v] - gpu[size_t(n) * V + v]); if (d > worst) worst = d;
                    if (cpu[size_t(n) * V + v] > bc) { bc = cpu[size_t(n) * V + v]; ac = v; }
                    if (gpu[size_t(n) * V + v] > bg) { bg = gpu[size_t(n) * V + v]; ag = v; }
                }
                if (ac != ag) ++mis;
            }
            std::printf("E3b) N=%d (>32, gekachelt) worst max|Δ|=%.3e, argmax-mismatches=%d\n", N, worst, mis);
            bool okE3b = (worst < 1e-2f) && (mis == 0);
            std::printf("TESTBED RealInference-E3b (gemm N-Kachelung N>32 == CPU): %s\n", okE3b ? "BESTANDEN" : "FEHLGESCHLAGEN");
            okA = okA && okE3b;
        }
    }

    // E5 (Phase 6A Tree-Verify): Ahnen-Maske + per-Knoten-Positionen.
    //   (a) Ketten-Baum (parent=i-1, pos=i, Maske unteres Dreieck) == linearer forward_batch.
    //   (b) Geschwister-Isolation: zwei Kinder desselben Elternknotens sehen sich NICHT.
    {
        const uint64_t rng_save_e5 = g_rng;   // RNG um E5 sichern (E4 danach unverändert)
        Int3Weights we = make_tiny_model(c);
        GpuForward gf; std::string ge;
        if (!gf.init(we, 64, &ge, /*streamed=*/true)) { std::printf("E5) init: %s\n", ge.c_str()); }
        else {
            const int N = 5;
            std::vector<int> toks(N); for (int i = 0; i < N; ++i) toks[i] = (i * 13 + 2) % c.vocab;
            std::vector<float> lin((size_t)N * V), tree((size_t)N * V);
            gf.forward_batch_gpu(toks.data(), N, 0, lin.data());                               // linear
            std::vector<int> pos(N); for (int i = 0; i < N; ++i) pos[i] = i;
            std::vector<unsigned char> mask((size_t)N * N, 0);
            for (int n = 0; n < N; ++n) for (int j = 0; j <= n; ++j) mask[n * N + j] = 1;       // Kette: n sieht 0..n
            gf.forward_batch_gpu(toks.data(), N, 0, tree.data(), pos.data(), mask.data());
            float wa = 0.0f; for (size_t i = 0; i < lin.size(); ++i) wa = std::max(wa, std::fabs(lin[i] - tree[i]));
            // (b) node0 (pos0), node1 (Kind, pos1), node2 (Geschwister, pos1). node1 sieht {0,1}, node2 {0,2}.
            std::vector<int> tb = {toks[0], toks[1], toks[2]}, tpos = {0, 1, 1};
            std::vector<unsigned char> tmask(9, 0);
            tmask[0] = 1;                              // node0 -> self
            tmask[3] = 1; tmask[4] = 1;                // node1 -> node0,self
            tmask[6] = 1; tmask[8] = 1;                // node2 -> node0,self (NICHT node1)
            std::vector<float> tr3((size_t)3 * V);
            gf.forward_batch_gpu(tb.data(), 3, 0, tr3.data(), tpos.data(), tmask.data());
            std::vector<int> pA = {toks[0], toks[1]}, pB = {toks[0], toks[2]};                  // Referenz-Paare
            std::vector<float> rA((size_t)2 * V), rB((size_t)2 * V);
            gf.forward_batch_gpu(pA.data(), 2, 0, rA.data());
            gf.forward_batch_gpu(pB.data(), 2, 0, rB.data());
            float wsib = 0.0f;
            for (int v = 0; v < V; ++v) { wsib = std::max(wsib, std::fabs(tr3[(size_t)1 * V + v] - rA[(size_t)1 * V + v]));
                                          wsib = std::max(wsib, std::fabs(tr3[(size_t)2 * V + v] - rB[(size_t)1 * V + v])); }
            std::printf("E5) Ketten-Baum==linear Δ=%.3e | Geschwister-Isolation Δ=%.3e\n", wa, wsib);
            bool okE5 = (wa < 1e-2f) && (wsib < 1e-2f);
            std::printf("TESTBED RealInference-E5 (Tree-Attention-Maske korrekt): %s\n", okE5 ? "BESTANDEN" : "FEHLGESCHLAGEN");
            okA = okA && okE5;
        }
        g_rng = rng_save_e5;   // RNG wiederherstellen -> E4 unverändert
    }

    // E4: 2-Bit-KV (kv_bits=2) Roundtrip-Sanity — VERLUSTBEHAFTET, nicht token-identisch. Gate: Δ
    //     BESCHRÄNKT (kein NaN/Garbage aus Packing-Bug); mismatches informativ (2-Bit ist verlustig).
    {
        for (int kb : {2, 3, 4}) {
            Int3Weights we = make_tiny_model(c);
            GpuForward g0, gq; std::string e0s, eqs;
            g0.init(we, 64, &e0s, /*streamed=*/true, /*kv_bits=*/0);
            gq.init(we, 64, &eqs, /*streamed=*/true, /*kv_bits=*/kb);
            std::vector<float> l0(V), lq(V); float worst = 0.0f; int mis = 0; bool finite = true;
            for (int i = 0; i < int(toks.size()); ++i) {
                g0.step_logits(toks[i], i, l0.data());
                gq.step_logits(toks[i], i, lq.data());
                int a0 = 0, aq = 0;
                for (int j = 0; j < V; ++j) {
                    worst = std::max(worst, std::fabs(l0[j] - lq[j]));
                    if (!std::isfinite(lq[j])) finite = false;
                    if (l0[j] > l0[a0]) a0 = j; if (lq[j] > lq[aq]) aq = j;
                }
                if (a0 != aq) ++mis;
            }
            std::printf("E4) %d-Bit-KV vs FP32: worst max|Δ|=%.3e, argmax-mismatches=%d/%d, finite=%d\n",
                        kb, worst, mis, int(toks.size()), int(finite));
        }
        {   // Gate: 4-Bit-KV muss NAH an FP32 sein (Rotation aktiv -> Integration korrekt).
            Int3Weights we = make_tiny_model(c);
            GpuForward g0, g4; std::string e0s, e4s;
            g0.init(we, 64, &e0s, true, 0); g4.init(we, 64, &e4s, true, 4);
            std::vector<float> l0(V), l4(V); float worst = 0.0f;
            for (int i = 0; i < int(toks.size()); ++i) {
                g0.step_logits(toks[i], i, l0.data()); g4.step_logits(toks[i], i, l4.data());
                for (int j = 0; j < V; ++j) worst = std::max(worst, std::fabs(l0[j] - l4[j]));
            }
            bool okE4 = (worst < 1.0f);   // 4-Bit+Rotation ~lossless -> kleiner Δ; groß => Integrationsbug
            std::printf("TESTBED RealInference-E4 (4-Bit-KV ~ FP32, Δ=%.3e): %s\n", worst, okE4 ? "BESTANDEN" : "FEHLGESCHLAGEN");
            okA = okA && okE4;
        }
    }
#endif

    // Teil B: echtes Modell (optional).
    if (const char* gg = std::getenv("NOVA_REAL_GGUF")) {
        const char* tk = std::getenv("NOVA_TEKKEN");
        std::printf("B) Lade echtes 24B-Modell (%s) ...\n", gg);
        RealInference eng; std::string err;
        if (!eng.load(gg, Mistral3Config::m24b(), tk ? tk : "", {}, &err)) {
            std::printf("B) Laden fehlgeschlagen: %s\n", err.c_str()); return okA ? 0 : 1;
        }
        if (const char* dg = std::getenv("NOVA_DRAFT_GGUF")) {
            std::string de;
            if (eng.load_draft(dg, Mistral3Config::m3b(), &de)) std::printf("B) 3B-Draft geladen -> Spec-Decoding AKTIV\n");
            else std::printf("B) Draft laden fehlgeschlagen (%s) -> plain greedy\n", de.c_str());
        }
        GenRequest req; req.prefix = ""; req.dynamic = "Die Hauptstadt von Deutschland ist";
        using clk = std::chrono::steady_clock;
        const int NGEN = 32;
        real_inference_use_gpu(true);   // nur GPU — keine CPU-Benchmarks
        auto decode = [&](std::string& out) {           // NGEN Tokens greedy dekodieren, Text sammeln
            out.clear(); int n = 0;
            for (int i = 0; i < NGEN; ++i) { std::string t = eng.next_token(); if (t.empty()) break; out += t; ++n; }
            return n;
        };
        if (eng.has_draft()) {
            // K-Sweep: mehr Draft-Tokens pro Pass. Greedy-Spec == Greedy-Dekodierung UNABHÄNGIG von K,
            // also MUSS der Output über alle K identisch bleiben (identisch=1) — kein Qualitäts-Tradeoff.
            // Gesucht ist der tok/s-Peak. Laden nur 1× (3B bereits VRAM-resident, bleibt über begin()).
            const int Ks[] = {4, 8, 12, 16, 20, 24};
            std::string ref;
            for (int ki = 0; ki < 6; ++ki) {
                const int K = Ks[ki];
                eng.set_spec_k(K);
                auto t0 = clk::now(); eng.begin(req); auto t1 = clk::now();
                std::string out; int ntok = decode(out); auto t2 = clk::now();
                const double dec = std::chrono::duration<double>(t2 - t1).count();
                if (ki == 0) { ref = out; std::printf("B) Ausgabe: \"%s\"\n", out.c_str()); }
                std::printf("B[K=%2d]) %d Tok in %.2fs = %.2f tok/s | accept=%.2f | prefill=%.2fs | identisch=%d\n",
                            K, ntok, dec, ntok > 0 && dec > 0 ? ntok / dec : 0.0, eng.last_accept_rate(),
                            std::chrono::duration<double>(t1 - t0).count(), int(out == ref));
            }
        } else {
            auto t0 = clk::now(); eng.begin(req); auto t1 = clk::now();
            std::string out; int ntok = decode(out); auto t2 = clk::now();
            const double dec = std::chrono::duration<double>(t2 - t1).count();
            std::printf("B) Ausgabe: \"%s\"\n", out.c_str());
            std::printf("B) greedy (kein Draft): %d Tok in %.2fs = %.2f tok/s | prefill=%.2fs\n",
                        ntok, dec, ntok > 0 && dec > 0 ? ntok / dec : 0.0,
                        std::chrono::duration<double>(t1 - t0).count());
        }
    }
    return okA ? 0 : 1;
}

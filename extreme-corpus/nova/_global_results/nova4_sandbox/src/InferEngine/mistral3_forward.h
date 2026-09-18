// mistral3_forward.h — CPU-Referenz-Forward fuer mistral3 (Engine-Gate 2, Orakel).
//
// F32-Referenz: liest Gewichte pro Tensor aus dem GGUF (gguf_reader dequantisiert
// Q4_K/Q6_K -> F32), rechnet einen vollstaendigen Decoder-Forward (RMSNorm, GQA-
// Attention mit RoPE, SwiGLU-FFN) und liefert die Logits der letzten Position.
// Korrektheit ist das Ziel, nicht Geschwindigkeit (kein KV-Cache; ~Minuten/Token
// beim 24B). Dient als Orakel fuer den spaeteren GPU/INT3-Pfad und macht Nova
// (langsam) real sprechfaehig hinter IInference.
#pragma once

#include "InferEngine/mistral3_config.h"
#include "ModelStore/gguf_reader.h"

#include <map>
#include <string>
#include <vector>

namespace nova::infer {

class Mistral3Reference {
public:
    // rope_interleaved: true = ggml-NORM (Paare 2i,2i+1, GGUF-Default fuer LLaMA/Mistral),
    // false = rotate_half (HF-Konvention). Bei falschem Output umschalten.
    bool load(const std::string& gguf_path, const Mistral3Config& cfg,
              bool rope_interleaved = true, std::string* err = nullptr);

    // Volle Sequenz -> Logits[vocab] der letzten Position.
    std::vector<float> forward(const std::vector<int>& tokens);

    const Mistral3Config& config() const { return cfg_; }

private:
    std::vector<float> W(const std::string& name);   // dequantisierter Tensor (F32)

    Mistral3Config              cfg_;
    bool                        interleaved_ = true;
    modelstore::GgufReader      gg_;
    std::map<std::string, const modelstore::GgufTensorInfo*> tmap_;
};

}  // namespace nova::infer

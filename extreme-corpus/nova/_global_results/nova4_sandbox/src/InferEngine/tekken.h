// tekken.h — Tekken/llama-bpe Byte-Level-BPE-Tokenizer (mistral3, vocab 131072).
//
// Laedt vocab.txt (Zeile i = Byte-Level-Token fuer id i), merges.txt ("a b" je
// Rang) und ttypes.txt (1 normal, 3 control/special, 6 byte) aus einem Verzeichnis
// (Export via scratchpad/export_tok.py aus dem GGUF). GPT-2-Byte-Level-Encoding +
// vereinfachter llama-bpe-Pre-Tokenizer (Buchstaben/Ziffern/Space/Sonst-Laeufe).
// Ziel: gueltige, kohaerente Tokenisierung (nicht zwingend byte-exakt vs Ollama).
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nova::infer {

class Tekken {
public:
    bool load(const std::string& dir, std::string* err = nullptr);

    std::vector<int> encode(const std::string& text, bool add_bos = true) const;
    std::string      decode(const std::vector<int>& ids) const;

    int  bos() const { return 1; }
    int  eos() const { return 2; }
    int  vocab_size() const { return int(id2tok_.size()); }
    bool ready() const { return !id2tok_.empty(); }

private:
    std::vector<std::string>            id2tok_;      // id -> Byte-Level-Token
    std::unordered_map<std::string,int> tok2id_;
    std::unordered_map<std::string,int> merge_rank_;  // "a b" -> Rang
    std::vector<std::string>            specials_;     // control-Tokens (literal matchen)
    std::unordered_map<std::string,int> special_id_;

    std::string byte2u_[256];                          // Byte -> Byte-Level-Unicode (UTF-8)
    std::unordered_map<std::string,int> u2byte_;       // umgekehrt

    std::vector<int> bpe_piece(const std::string& piece) const;  // ein Pre-Token -> ids
};

}  // namespace nova::infer

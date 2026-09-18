// tekken.cpp — Implementierung von tekken.h (GPT-2-Byte-Level-BPE, mistral3).
#include "InferEngine/tekken.h"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace nova::infer {

namespace {
std::string utf8(unsigned cp) {
    std::string s;
    if (cp < 0x80) s += char(cp);
    else if (cp < 0x800) { s += char(0xC0 | (cp >> 6)); s += char(0x80 | (cp & 0x3F)); }
    else { s += char(0xE0 | (cp >> 12)); s += char(0x80 | ((cp >> 6) & 0x3F)); s += char(0x80 | (cp & 0x3F)); }
    return s;
}
// UTF-8 -> Codepoints (mit Byte-Spannen). Ungueltiges wird als Einzelbytes behandelt.
struct CP { unsigned cp; size_t off, len; };
std::vector<CP> decode_utf8(const std::string& s) {
    std::vector<CP> out;
    for (size_t i = 0; i < s.size();) {
        unsigned c = (unsigned char)s[i]; size_t n = 1; unsigned cp = c;
        if (c >= 0xF0 && i + 3 < s.size()) { cp = ((c&7)<<18)|((s[i+1]&0x3F)<<12)|((s[i+2]&0x3F)<<6)|(s[i+3]&0x3F); n=4; }
        else if (c >= 0xE0 && i + 2 < s.size()) { cp = ((c&0xF)<<12)|((s[i+1]&0x3F)<<6)|(s[i+2]&0x3F); n=3; }
        else if (c >= 0xC0 && i + 1 < s.size()) { cp = ((c&0x1F)<<6)|(s[i+1]&0x3F); n=2; }
        out.push_back({cp, i, n}); i += n;
    }
    return out;
}
bool is_space(unsigned c){ return c==' '||c=='\t'||c=='\n'||c=='\r'||c==0x0B||c==0x0C||c==0xA0; }
bool is_digit(unsigned c){ return c>='0'&&c<='9'; }
bool is_letter(unsigned c){
    if((c>='A'&&c<='Z')||(c>='a'&&c<='z')) return true;
    if(c>=0xC0 && c<=0x24F) return true;      // Latin-1 Supplement + Extended-A/B (Umlaute etc.)
    if(c>=0x370 && c<=0x1FFF) return true;     // Griechisch/Kyrillisch grob
    return false;
}
int cls(unsigned c){ return is_letter(c)?1 : is_digit(c)?2 : 3; }  // 1=L 2=N 3=other
}  // namespace

bool Tekken::load(const std::string& dir, std::string* err) {
    // GPT-2 Byte<->Unicode-Tabelle.
    std::vector<int> bs, cs;
    for (int b='!'; b<='~'; ++b) bs.push_back(b);
    for (int b=0xA1; b<=0xAC; ++b) bs.push_back(b);
    for (int b=0xAE; b<=0xFF; ++b) bs.push_back(b);
    cs = bs; int n = 0;
    for (int b=0; b<256; ++b) if (std::find(bs.begin(),bs.end(),b)==bs.end()) { bs.push_back(b); cs.push_back(256+n++); }
    for (size_t i=0;i<bs.size();++i){ byte2u_[bs[i]] = utf8(unsigned(cs[i])); u2byte_[byte2u_[bs[i]]] = bs[i]; }

    std::ifstream fv(dir + "\\vocab.txt", std::ios::binary);
    if (!fv) { if (err) *err = "vocab.txt fehlt in " + dir; return false; }
    std::string line;
    std::ifstream ft(dir + "\\ttypes.txt");
    std::string tl;
    while (std::getline(fv, line)) {
        if (!line.empty() && line.back()=='\r') line.pop_back();
        const int id = int(id2tok_.size());
        id2tok_.push_back(line); tok2id_[line] = id;
        int ty = 1; if (ft && std::getline(ft, tl)) ty = std::atoi(tl.c_str());
        if (ty == 3) { specials_.push_back(line); special_id_[line] = id; }   // control
    }
    std::ifstream fm(dir + "\\merges.txt", std::ios::binary);
    int rank = 0;
    while (std::getline(fm, line)) {
        if (!line.empty() && line.back()=='\r') line.pop_back();
        if (!line.empty()) merge_rank_[line] = rank++;
    }
    // Specials nach Laenge absteigend (longest-match).
    std::sort(specials_.begin(), specials_.end(),
              [](const std::string&a,const std::string&b){return a.size()>b.size();});
    return !id2tok_.empty();
}

std::vector<int> Tekken::bpe_piece(const std::string& piece) const {
    // Bytes -> Byte-Level-Symbole.
    std::vector<std::string> sym;
    for (unsigned char b : piece) sym.push_back(byte2u_[b]);
    if (sym.empty()) return {};
    for (;;) {
        int best = -1, bi = -1;
        for (size_t i=0;i+1<sym.size();++i) {
            auto it = merge_rank_.find(sym[i] + " " + sym[i+1]);
            if (it != merge_rank_.end() && (best<0 || it->second<best)) { best=it->second; bi=int(i); }
        }
        if (bi < 0) break;
        sym[bi] += sym[bi+1]; sym.erase(sym.begin()+bi+1);
    }
    std::vector<int> ids;
    for (auto& s : sym) { auto it = tok2id_.find(s); if (it!=tok2id_.end()) ids.push_back(it->second); }
    return ids;
}

std::vector<int> Tekken::encode(const std::string& text, bool add_bos) const {
    std::vector<int> ids;
    if (add_bos) ids.push_back(bos());
    std::string norm;
    auto flush_norm = [&](){
        if (norm.empty()) return;
        const auto cps = decode_utf8(norm);
        size_t j = 0;
        while (j < cps.size()) {
            size_t start = j;
            if (is_space(cps[j].cp)) {
                size_t k=j; while(k<cps.size() && is_space(cps[k].cp)) ++k;
                if (k-j==1 && k<cps.size()) {          // ein Space fuehrt das naechste Wort an
                    int c=cls(cps[k].cp); size_t e=k+1;
                    while(e<cps.size() && !is_space(cps[e].cp) && cls(cps[e].cp)==c) ++e;
                    size_t bo=cps[start].off, bend=(e<cps.size()?cps[e].off:norm.size());
                    for (auto id : bpe_piece(norm.substr(bo,bend-bo))) ids.push_back(id); j=e;
                } else {
                    size_t bo=cps[start].off, bend=(k<cps.size()?cps[k].off:norm.size());
                    for (auto id : bpe_piece(norm.substr(bo,bend-bo))) ids.push_back(id); j=k;
                }
            } else {
                int c=cls(cps[j].cp); size_t e=j+1;
                while(e<cps.size() && !is_space(cps[e].cp) && cls(cps[e].cp)==c) ++e;
                size_t bo=cps[start].off, bend=(e<cps.size()?cps[e].off:norm.size());
                for (auto id : bpe_piece(norm.substr(bo,bend-bo))) ids.push_back(id); j=e;
            }
        }
        norm.clear();
    };
    size_t i = 0;
    while (i < text.size()) {
        bool sp = false;
        for (const auto& s : specials_) {          // longest-match special
            if (!s.empty() && text.compare(i, s.size(), s) == 0) {
                flush_norm(); ids.push_back(special_id_.at(s)); i += s.size(); sp = true; break;
            }
        }
        if (!sp) { norm += text[i]; ++i; }
    }
    flush_norm();
    return ids;
}

std::string Tekken::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= int(id2tok_.size())) continue;
        const std::string& t = id2tok_[id];
        if (special_id_.count(t)) continue;        // <s>/</s>/[INST]… nicht ausgeben
        // Byte-Level-Token -> Original-Bytes.
        for (const auto& c : decode_utf8(t)) {
            auto it = u2byte_.find(t.substr(c.off, c.len));
            if (it != u2byte_.end()) out += char(it->second);
        }
    }
    return out;
}

}  // namespace nova::infer

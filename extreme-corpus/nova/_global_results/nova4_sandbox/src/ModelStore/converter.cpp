// converter.cpp — Nova 4 Modell-Konverter (Design §5).
//
//   converter.exe <modell.gguf> [--out DIR] [--quant int2|int3|int4] [--no-compress]
//       GGUF -> .nv4 Chunk-Streaming-Format (Auto: Gemma4=INT2, sonst INT4).
//
//   converter.exe --format flat-int4 <modell.gguf> [--out DIR]
//       GGUF -> .bin Flat-INT4 (Ministral 3B, §5.3).
//
//   converter.exe --gen-test-model DIR [--layers N --rows R --cols C
//                                       --quant int2|int4 --no-compress]
//       Erzeugt synthetische .nv4-Chunks (für Testbed 2/3 ohne reale GGUF).
//
//   converter.exe --validate DIR        CRC32-Prüfung aller .nv4 in DIR.
//   converter.exe --selftest            Round-Trip-Tests des Formats.
//
//   converter.exe --medusa-heads <f> --base <dir>   Medusa-Köpfe zu <dir> hinzufügen.
#include "Apex/json.h"
#include "ModelStore/chunk_validator.h"
#include "ModelStore/gguf_reader.h"
#include "ModelStore/nv4_format.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace nova::modelstore;

namespace {

struct Args {
    std::string positional;
    std::string out;
    std::string format;       // "flat-int4" erzwingt .bin
    std::string quant;        // "int2" | "int4"
    std::string medusa_heads;
    std::string base;
    bool compress = true;
    int  layers = 4, rows = 256, cols = 512;
    bool gen_test = false, validate = false, selftest = false;
    std::string gen_dir, validate_dir;
};

[[noreturn]] void usage_and_exit(int code) {
    std::cout <<
        "Nova 4 converter\n"
        "  converter <modell.gguf> [--out DIR] [--quant int2|int3|int4] [--no-compress]\n"
        "  converter --format flat-int4 <modell.gguf> [--out DIR]\n"
        "  converter --gen-test-model DIR [--layers N --rows R --cols C --quant int2|int4 --no-compress]\n"
        "  converter --validate DIR\n"
        "  converter --selftest\n";
    std::exit(code);
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    return s;
}

const char* quant_name(QuantType qt) {
    switch (qt) {
        case QuantType::INT2: return "int2";
        case QuantType::INT3: return "int3";
        case QuantType::INT4: return "int4";
    }
    return "int4";
}

QuantType pick_quant(const Args& a, const std::string& model_hint) {
    if (a.quant == "int2") return QuantType::INT2;
    if (a.quant == "int3") return QuantType::INT3;
    if (a.quant == "int4") return QuantType::INT4;
    // Auto: Gemma 4 / E27B -> INT2 (MoE, §4.1). Mistral (dense 24B) -> INT3
    // (bessere Qualität als INT2, ~8 GB, passt in 14 GB RAM). Sonst INT4.
    const std::string h = lower(model_hint);
    if (h.find("gemma4") != std::string::npos || h.find("e27b") != std::string::npos ||
        h.find("gemma-4") != std::string::npos)
        return QuantType::INT2;
    if (h.find("mistral") != std::string::npos || h.find("24b") != std::string::npos)
        return QuantType::INT3;
    return QuantType::INT4;
}

// Schreibt meta.json (von chunk_index.cpp später gelesen).
struct MetaEntry { std::string file; uint32_t layer, chunk, shape0, shape1, comp, uncomp; };

bool write_meta(const fs::path& dir, const std::string& model, const std::string& fmt,
                QuantType qt, const std::vector<MetaEntry>& chunks) {
    std::ofstream m(dir / "meta.json", std::ios::trunc);
    if (!m) return false;
    m << "{\n";
    m << "  \"model\": \"" << model << "\",\n";
    m << "  \"format\": \"" << fmt << "\",\n";
    m << "  \"quant\": \"" << quant_name(qt) << "\",\n";
    m << "  \"group_size\": " << GROUP_SIZE << ",\n";
    m << "  \"chunks\": [\n";
    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto& c = chunks[i];
        m << "    {\"file\": \"" << c.file << "\", \"layer\": " << c.layer
          << ", \"chunk\": " << c.chunk
          << ", \"shape\": [" << c.shape0 << ", " << c.shape1 << "]"
          << ", \"comp\": " << c.comp << ", \"uncomp\": " << c.uncomp << "}"
          << (i + 1 < chunks.size() ? "," : "") << "\n";
    }
    m << "  ]\n}\n";
    return bool(m);
}

// Layer-Index aus GGUF-Tensornamen (z.B. "blk.7.attn_q.weight") ableiten.
uint16_t infer_layer(const std::string& name, uint16_t fallback) {
    size_t p = name.find("blk.");
    if (p == std::string::npos) p = name.find("layers.");
    if (p == std::string::npos) return fallback;
    p = name.find('.', p);
    if (p == std::string::npos) return fallback;
    int v = std::atoi(name.c_str() + p + 1);
    return (v >= 0 && v < 65535) ? uint16_t(v) : fallback;
}

// --------------------------------------------------------------------------
int cmd_convert_nv4(const Args& a) {
    GgufReader g;
    std::string err;
    if (!g.open(a.positional, &err)) { std::cerr << "Fehler: " << err << "\n"; return 1; }

    const QuantType qt = pick_quant(a, a.positional);
    fs::path outdir = a.out.empty()
        ? fs::path(a.positional).stem().concat("_nv4")
        : fs::path(a.out);
    fs::create_directories(outdir / "chunks");

    std::cout << "GGUF v" << g.version() << ", " << g.tensors().size()
              << " Tensoren -> .nv4 (" << quant_name(qt)
              << (a.compress ? "+Zstd" : "") << ")\n";

    std::vector<MetaEntry> meta;
    std::vector<float> data;
    uint16_t seq = 0, skipped = 0;
    for (const auto& t : g.tensors()) {
        if (!g.read_tensor_f32(t, data, &err)) {
            std::cerr << "  [skip] " << t.name << ": " << err << "\n";
            ++skipped; ++seq;
            continue;
        }
        const uint64_t n = t.num_elements();
        QuantResult q = quantize_tensor(data.data(), n, qt);

        const uint16_t layer = infer_layer(t.name, seq);
        char fname[64];
        std::snprintf(fname, sizeof(fname), "chunks/L%05u_C%05u.nv4", layer, seq);
        const uint32_t s0 = t.dims.empty() ? uint32_t(n) : uint32_t(t.dims[0]);
        const uint32_t s1 = t.dims.size() > 1 ? uint32_t(t.dims[1]) : 1u;

        if (!write_chunk((outdir / fname).string(), layer, seq, qt, s0, s1,
                         q.scales, q.packed, a.compress)) {
            std::cerr << "Fehler beim Schreiben von " << fname << "\n"; return 1;
        }
        Chunk rb; read_chunk((outdir / fname).string(), rb);
        meta.push_back({fname, layer, seq, s0, s1, rb.header.compressed_size,
                        rb.header.uncompressed_size});
        ++seq;
    }
    write_meta(outdir, fs::path(a.positional).stem().string(), "nv4", qt, meta);
    std::cout << "Fertig: " << meta.size() << " Chunks, " << skipped
              << " übersprungen -> " << outdir.string() << "\n";
    return skipped && meta.empty() ? 1 : 0;
}

// --------------------------------------------------------------------------
int cmd_convert_flat(const Args& a) {
    GgufReader g;
    std::string err;
    if (!g.open(a.positional, &err)) { std::cerr << "Fehler: " << err << "\n"; return 1; }
    fs::path outdir = a.out.empty() ? fs::path(".") : fs::path(a.out);
    fs::create_directories(outdir);
    const fs::path outfile = outdir / (fs::path(a.positional).stem().string() + ".bin");

    // Alle Tensoren in einen kontinuierlichen INT4-Block (§5.3).
    std::vector<uint8_t> block;
    std::vector<float> data;
    uint32_t layers = 0; uint16_t hidden = 0;
    for (const auto& t : g.tensors()) {
        if (!g.read_tensor_f32(t, data, &err)) {
            std::cerr << "  [skip] " << t.name << ": " << err << "\n"; continue;
        }
        QuantResult q = quantize_tensor(data.data(), t.num_elements(), QuantType::INT4);
        // Flat: nur gepackte Codes (Scales hier weggelassen — vereinfachtes 3B-Format).
        block.insert(block.end(), q.packed.begin(), q.packed.end());
        if (!t.dims.empty()) hidden = uint16_t(t.dims[0]);
        ++layers;
    }
    Nv4bHeader h{};
    std::memcpy(h.magic, NV4B_MAGIC, 4);
    h.version = NV4_VERSION; h.layers = layers; h.hidden_dim = hidden;
    h.checksum = crc32(block.data(), block.size(), 0);
    uint8_t hdr[NV4B_HEADER_SIZE]; h.serialize(hdr);

    std::ofstream f(outfile, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(hdr), NV4B_HEADER_SIZE);
    f.write(reinterpret_cast<const char*>(block.data()), std::streamsize(block.size()));
    std::cout << "Flat-INT4 geschrieben: " << outfile.string() << " ("
              << block.size() << " Byte INT4)\n";
    return 0;
}

// --------------------------------------------------------------------------
int cmd_gen_test(const Args& a) {
    const QuantType qt = (a.quant == "int2") ? QuantType::INT2
                       : (a.quant == "int3") ? QuantType::INT3 : QuantType::INT4;
    fs::path outdir(a.gen_dir);
    fs::create_directories(outdir / "chunks");

    std::mt19937 rng(0xC0FFEE);  // deterministisch
    std::normal_distribution<float> dist(0.0f, 0.7f);

    std::cout << "gen-test-model: " << a.layers << " Layer x " << a.rows << "x" << a.cols
              << " (" << quant_name(qt)
              << (a.compress ? "+Zstd" : "") << ") -> " << outdir.string() << "\n";

    std::vector<MetaEntry> meta;
    const uint64_t n = uint64_t(a.rows) * a.cols;
    std::vector<float> data(static_cast<size_t>(n));
    for (int l = 0; l < a.layers; ++l) {
        for (auto& v : data) v = dist(rng);
        QuantResult q = quantize_tensor(data.data(), n, qt);
        char fname[64];
        std::snprintf(fname, sizeof(fname), "chunks/L%05u_C%05u.nv4", l, l);
        const std::string path = (outdir / fname).string();
        if (!write_chunk(path, uint16_t(l), uint16_t(l), qt,
                         uint32_t(a.rows), uint32_t(a.cols), q.scales, q.packed, a.compress)) {
            std::cerr << "Schreibfehler: " << path << "\n"; return 1;
        }
        Chunk rb; read_chunk(path, rb);
        meta.push_back({fname, uint32_t(l), uint32_t(l), uint32_t(a.rows), uint32_t(a.cols),
                        rb.header.compressed_size, rb.header.uncompressed_size});
    }
    write_meta(outdir, "test-model", "nv4", qt, meta);
    std::cout << "Fertig: " << meta.size() << " Chunks geschrieben.\n";
    return 0;
}

// --------------------------------------------------------------------------
int cmd_validate(const Args& a) {
    auto fails = validate_chunk_dir(a.validate_dir, /*only_failures=*/false);
    int bad = 0;
    for (const auto& c : fails) {
        const bool ok = c.status == ChunkStatus::OK;
        if (!ok) ++bad;
        std::cout << (ok ? "[OK]  " : "[BAD] ") << c.path;
        if (!ok) std::cout << "  (" << c.detail << " exp=" << std::hex << c.expected_crc
                           << " act=" << c.actual_crc << std::dec << ")";
        std::cout << "\n";
    }
    std::cout << fails.size() << " Chunks, " << bad << " defekt.\n";
    return bad ? 1 : 0;
}

// --------------------------------------------------------------------------
// Medusa-Köpfe: safetensors (F16/F32) -> INT4 .nv4 in <base>/medusa/ (§5.5).
int cmd_medusa_heads(const Args& a) {
    std::ifstream f(a.medusa_heads, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "safetensors nicht gefunden: " << a.medusa_heads << "\n"; return 1; }
    const std::streamoff sz = f.tellg();
    f.seekg(0);
    if (sz < 8) { std::cerr << "safetensors zu klein\n"; return 1; }

    uint64_t hlen = 0;
    f.read(reinterpret_cast<char*>(&hlen), 8);           // 8-Byte LE Header-Länge
    if (8 + hlen > uint64_t(sz)) { std::cerr << "safetensors Header-Länge ungültig\n"; return 1; }
    std::string header(size_t(hlen), '\0');
    f.read(header.data(), std::streamsize(hlen));
    std::vector<uint8_t> blob(size_t(uint64_t(sz) - 8 - hlen));
    f.read(reinterpret_cast<char*>(blob.data()), std::streamsize(blob.size()));

    nova::apex::JsonValue j;
    std::string jerr;
    if (!json_parse(header, j, &jerr) || !j.is_object()) {
        std::cerr << "safetensors Header-JSON: " << jerr << "\n"; return 1;
    }

    const fs::path outdir = fs::path(a.base) / "medusa";
    fs::create_directories(outdir / "chunks");

    std::vector<MetaEntry> meta;
    uint16_t seq = 0;
    for (const auto& [name, tv] : j.obj) {
        if (name == "__metadata__" || !tv.is_object()) continue;
        const auto* dt = tv.find("dtype");
        const auto* shp = tv.find("shape");
        const auto* offs = tv.find("data_offsets");
        if (!dt || !shp || !offs || !offs->is_array() || offs->arr.size() != 2) continue;

        const std::string dtype = dt->as_str();
        const uint64_t o0 = uint64_t(offs->arr[0].num), o1 = uint64_t(offs->arr[1].num);
        if (o1 > blob.size() || o1 < o0) continue;

        uint64_t nel = 1;
        std::vector<uint32_t> dims;
        for (const auto& d : shp->arr) { const uint64_t dv = uint64_t(d.num); nel *= dv ? dv : 1; dims.push_back(uint32_t(dv)); }

        std::vector<float> data(static_cast<size_t>(nel), 0.0f);
        const uint8_t* p = blob.data() + o0;
        if (dtype == "F32") {
            std::memcpy(data.data(), p, size_t(nel) * 4);
        } else if (dtype == "F16") {
            for (uint64_t i = 0; i < nel; ++i) { uint16_t h; std::memcpy(&h, p + i * 2, 2); data[size_t(i)] = half_to_float(h); }
        } else {
            std::cerr << "  [skip] " << name << ": dtype " << dtype << "\n"; continue;
        }

        QuantResult q = quantize_tensor(data.data(), nel, QuantType::INT4);
        char fname[80];
        std::snprintf(fname, sizeof(fname), "chunks/H%05u.nv4", seq);
        const uint32_t s0 = dims.empty() ? uint32_t(nel) : dims[0];
        const uint32_t s1 = dims.size() > 1 ? dims[1] : 1u;
        if (!write_chunk((outdir / fname).string(), seq, seq, QuantType::INT4, s0, s1,
                         q.scales, q.packed, true)) {
            std::cerr << "Schreibfehler: " << fname << "\n"; return 1;
        }
        Chunk rb; read_chunk((outdir / fname).string(), rb);
        meta.push_back({fname, seq, seq, s0, s1, rb.header.compressed_size, rb.header.uncompressed_size});
        ++seq;
    }
    write_meta(outdir, "medusa-heads", "nv4", QuantType::INT4, meta);
    std::cout << "Medusa-Köpfe konvertiert: " << meta.size() << " Tensoren -> " << outdir.string() << "\n";
    return meta.empty() ? 1 : 0;
}

// --------------------------------------------------------------------------
int cmd_selftest() {
    int fails = 0;
    auto check = [&](bool cond, const char* what) {
        std::cout << (cond ? "  [OK] " : "  [FAIL] ") << what << "\n";
        if (!cond) ++fails;
    };

    // 1) CRC32-Referenzwert: crc32("123456789") == 0xCBF43926
    check(crc32("123456789", 9, 0) == 0xCBF43926u, "CRC32 Referenzvektor");

    // 2) FP16 Round-Trip
    bool fp_ok = true;
    for (float v : {0.0f, 1.0f, -1.0f, 0.5f, 3.14159f, -2.7f, 65504.0f, 1e-4f}) {
        float r = half_to_float(float_to_half(v));
        if (std::fabs(r - v) > std::fabs(v) * 0.01f + 1e-3f) fp_ok = false;
    }
    check(fp_ok, "FP16 <-> FP32 Round-Trip");

    // 3) Quant/Dequant Round-Trip (INT4): mittlerer relativer Fehler klein
    {
        const uint64_t n = 4096;
        std::mt19937 rng(42);
        std::normal_distribution<float> d(0.f, 1.f);
        std::vector<float> x(static_cast<size_t>(n));
        for (auto& v : x) v = d(rng);
        auto q = quantize_tensor(x.data(), n, QuantType::INT4);
        auto y = dequantize_tensor(q.scales, q.packed, n, QuantType::INT4);
        double num = 0, da = 0, db = 0;
        for (size_t i = 0; i < n; ++i) { num += x[i]*y[i]; da += x[i]*x[i]; db += y[i]*y[i]; }
        double cos = num / (std::sqrt(da) * std::sqrt(db) + 1e-12);
        std::cout << "       INT4 cosine-sim = " << cos << "\n";
        check(cos > 0.98, "INT4 Quant Round-Trip (cos > 0.98)");
    }

    // 3b) INT3 Round-Trip (Mistral 24B): byte-straddelnde 3-Bit-Packung.
    {
        const uint64_t n = 4096;
        std::mt19937 rng(43);
        std::normal_distribution<float> d(0.f, 1.f);
        std::vector<float> x(static_cast<size_t>(n));
        for (auto& v : x) v = d(rng);
        auto q = quantize_tensor(x.data(), n, QuantType::INT3);
        auto y = dequantize_tensor(q.scales, q.packed, n, QuantType::INT3);
        double num = 0, da = 0, db = 0;
        for (size_t i = 0; i < n; ++i) { num += x[i]*y[i]; da += x[i]*x[i]; db += y[i]*y[i]; }
        double cos = num / (std::sqrt(da) * std::sqrt(db) + 1e-12);
        std::cout << "       INT3 cosine-sim = " << cos << " ("
                  << packed_bytes(n, QuantType::INT3) << " B vs INT4 "
                  << packed_bytes(n, QuantType::INT4) << " B) — Größe/Qualität-Tradeoff\n";
        check(cos > 0.97, "INT3 Quant Round-Trip (cos > 0.97)");
    }

    // 4) Chunk write -> read -> CRC OK (mit Zstd)
    {
        const uint64_t n = 2048;
        std::vector<float> x(static_cast<size_t>(n));
        std::mt19937 rng(7);
        std::normal_distribution<float> d(0.f, 0.5f);
        for (auto& v : x) v = d(rng);
        auto q = quantize_tensor(x.data(), n, QuantType::INT2);
        const std::string p = (fs::temp_directory_path() / "nova4_selftest.nv4").string();
        bool w = write_chunk(p, 3, 1, QuantType::INT2, 32, 64, q.scales, q.packed, true);
        ChunkCheck c = validate_chunk_file(p);
        check(w && c.status == ChunkStatus::OK, "Chunk write/read + CRC (INT2+Zstd)");

        // 5) Payload-Dekomprimierung stimmt mit Original-Packed überein
        Chunk rb; read_chunk(p, rb);
        std::vector<uint8_t> dec;
        bool dok = decompress_payload(rb, dec) && dec == q.packed;
        check(dok, "Zstd-Dekomprimierung == Original");
        std::error_code ec; fs::remove(p, ec);
    }

    // 6) Medusa-Heads: synthetisches safetensors -> INT4 .nv4 + CRC.
    {
        const std::string hdr =
            "{\"w0\":{\"dtype\":\"F32\",\"shape\":[4,8],\"data_offsets\":[0,128]},"
            "\"w1\":{\"dtype\":\"F32\",\"shape\":[8],\"data_offsets\":[128,160]}}";
        std::vector<float> w(40);
        std::mt19937 rng(9); std::normal_distribution<float> d(0.f, 1.f);
        for (auto& v : w) v = d(rng);
        const std::string stp = (fs::temp_directory_path() / "nova4_heads.safetensors").string();
        {
            std::ofstream o(stp, std::ios::binary | std::ios::trunc);
            uint64_t hl = hdr.size();
            o.write(reinterpret_cast<const char*>(&hl), 8);
            o.write(hdr.data(), std::streamsize(hdr.size()));
            o.write(reinterpret_cast<const char*>(w.data()), 160);   // 40 floats
        }
        Args ma;
        ma.medusa_heads = stp;
        ma.base = (fs::temp_directory_path() / "nova4_medusa_base").string();
        const int rc = cmd_medusa_heads(ma);
        auto mf = validate_chunk_dir((fs::path(ma.base) / "medusa").string(), false);
        bool ok = (rc == 0) && (mf.size() == 2);
        for (const auto& c : mf) if (c.status != ChunkStatus::OK) ok = false;
        check(ok, "Medusa-Heads safetensors -> .nv4 + CRC");
        std::error_code ec; fs::remove(stp, ec); fs::remove_all(fs::path(ma.base), ec);
    }

    std::cout << (fails ? "SELFTEST FEHLGESCHLAGEN" : "SELFTEST OK") << " (" << fails << " Fehler)\n";
    return fails ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::cerr << name << " braucht ein Argument\n"; std::exit(2); }
            return argv[++i];
        };
        if (s == "--out") a.out = next("--out");
        else if (s == "--format") a.format = next("--format");
        else if (s == "--quant") a.quant = next("--quant");
        else if (s == "--no-compress") a.compress = false;
        else if (s == "--layers") a.layers = std::atoi(next("--layers").c_str());
        else if (s == "--rows") a.rows = std::atoi(next("--rows").c_str());
        else if (s == "--cols") a.cols = std::atoi(next("--cols").c_str());
        else if (s == "--medusa-heads") a.medusa_heads = next("--medusa-heads");
        else if (s == "--base") a.base = next("--base");
        else if (s == "--gen-test-model") { a.gen_test = true; a.gen_dir = next("--gen-test-model"); }
        else if (s == "--validate") { a.validate = true; a.validate_dir = next("--validate"); }
        else if (s == "--selftest") a.selftest = true;
        else if (s == "-h" || s == "--help") usage_and_exit(0);
        else if (!s.empty() && s[0] == '-') { std::cerr << "unbekannte Option: " << s << "\n"; usage_and_exit(2); }
        else a.positional = s;
    }

    if (a.selftest)  return cmd_selftest();
    if (a.gen_test)  return cmd_gen_test(a);
    if (a.validate)  return cmd_validate(a);
    if (!a.medusa_heads.empty()) {
        if (a.base.empty()) { std::cerr << "--medusa-heads braucht --base <draft-dir>\n"; return 2; }
        return cmd_medusa_heads(a);
    }
    if (a.positional.empty()) usage_and_exit(2);
    if (a.format == "flat-int4") return cmd_convert_flat(a);
    return cmd_convert_nv4(a);
}

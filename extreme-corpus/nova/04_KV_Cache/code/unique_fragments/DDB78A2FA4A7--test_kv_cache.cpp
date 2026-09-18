// quasar — tests/test_kv_cache.cpp
//
// Stufe 6, erster Teil: der INT4-KV-Cache (SPEC Abschnitt 8).
//
// Geprueft wird die **Leseseite** (SPEC Abschnitt 10): nicht "wurde
// angehaengt", sondern "kommt beim naechsten Lesen genau das heraus, was
// angehaengt wurde" — und zwar sowohl aus den rohen Bytes als auch aus dem
// Attention-Kernel, der sie tatsaechlich benutzt.
//
// Und es wird geprueft, was der Cache **nicht** tut: er wirft nichts von selbst
// weg, wenn er voll ist (kein Evicting), und `truncate` verschiebt nichts.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "quasar/gpu/buffer.hpp"
#include "quasar/gpu/device.hpp"
#include "quasar/gpu/stream.hpp"
#include "quasar/kernels/kernels.hpp"
#include "quasar/kv/cache.hpp"

#include "harness/golden.hpp"
#include "reference/cpu_reference.hpp"

namespace {

constexpr const char* kName = "kv.cache";
namespace ref = quasar::test::reference;
namespace gpu = quasar::gpu;
namespace kv = quasar::kv;
namespace kern = quasar::kernels;

using quasar::test::Report;

template <typename T>
gpu::DeviceBuffer upload_vector(const std::vector<T>& v) {
    gpu::DeviceBuffer b(v.size() * sizeof(T));
    b.upload(v.data(), v.size() * sizeof(T));
    return b;
}

// Groesster Abstand zweier fp16-Felder in ULP, plus Zahl der bitgleichen
// Werte. Dieselbe Rechnung wie in `test_kernels.cpp`: fp16 als
// Betrag-Vorzeichen in eine monotone Ordnung bringen, dann die Differenz.
struct HalfMatch {
    std::size_t equal = 0;
    int worst_ulp = 0;
};

HalfMatch compare_halves(const std::vector<std::uint16_t>& actual,
                         const std::vector<std::uint16_t>& expected) {
    HalfMatch m;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] == expected[i]) {
            ++m.equal;
            continue;
        }
        auto ordered = [](std::uint16_t h) -> int {
            return (h & 0x8000u) ? -static_cast<int>(h & 0x7FFFu) : static_cast<int>(h);
        };
        const int diff = std::abs(ordered(actual[i]) - ordered(expected[i]));
        if (diff > m.worst_ulp) m.worst_ulp = diff;
    }
    return m;
}

}  // namespace

// Der Cache gibt nur `const`-Geraetezeiger heraus. Um den Inhalt zu pruefen,
// laesst der Test ihn durch den Attention-Kernel laufen — das ist ohnehin der
// einzige Verbraucher, und damit prueft der Test die Leseseite des echten
// Verbrauchers statt einer Hilfsansicht.

int main() {
    using namespace quasar;

    try {
        if (gpu::device_count() == 0) {
            return test::skip(kName, "kein GPU-Geraet sichtbar");
        }
        gpu::set_device(0);
        Report report(kName);
        gpu::Stream stream;

        // Befund B1 (Pruefrunde 3): der Cache lief hier mit Kapazitaet 16 und
        // 8 belegten Token -- also durchweg **unterhalb** jeder Kachel- und
        // Stapelgrenze des Attention-Kernels (8 KV-Token je Schritt) und weit
        // unterhalb des Betriebs (`--context 4096`). Er faehrt jetzt eine
        // Kapazitaet, die viele Kacheln umspannt, und die Leseprobe unten geht
        // ueber 115 belegte Token statt ueber 8.
        //
        // `head_dim = 128` mit `kv_group = 32` ist zugleich die
        // Betriebskonfiguration (`engine.cpp: group_size = min(32, head_dim)`)
        // und schickt die Leseprobe ueber den **schnellen** Attention-Weg --
        // vorher lief sie mit head_dim = 64 ausschliesslich generisch.
        constexpr int kLayers = 3;
        constexpr int kKvHeads = 2;
        constexpr int kHeadDim = 128;
        constexpr int kCapacity = 160;
        constexpr int kGroup = 32;
        constexpr int kHeads = 4;  // GQA 4/2

        // Beide Schuebe liegen ueber der Kachelbreite; ihre Summe (115) liegt
        // ueber 64 und ist kein Vielfaches von 8, damit die Restkachel wirklich
        // vorkommt.
        constexpr int kFirstPush = 70;
        constexpr int kSecondPush = 45;
        constexpr int kBothPushes = kFirstPush + kSecondPush;

        kv::Shape shape{kLayers, kKvHeads, kHeadDim, kCapacity, kGroup};
        kv::Cache cache = kv::Cache::create(shape);
        report.check(cache.size() == 0 && cache.empty(), "frischer Cache ist leer");
        report.note(shape.describe());

        const std::size_t per_token = static_cast<std::size_t>(kKvHeads) * kHeadDim;

        // --- anhaengen, in zwei Schueben ----------------------------------
        // Erst 70 Token, dann 45. Der zweite Schub darf den ersten nicht
        // anfassen und muss genau dahinter landen.
        std::vector<float> all_k, all_v;
        std::uint64_t seed = 0x4242'0001ULL;
        auto push = [&](int n) {
            const std::vector<float> k = test::make_uniform(per_token * static_cast<std::size_t>(n),
                                                            seed++, -1.5f, 1.5f);
            const std::vector<float> v = test::make_uniform(per_token * static_cast<std::size_t>(n),
                                                            seed++, -1.5f, 1.5f);
            for (int l = 0; l < kLayers; ++l) {
                // Jeder Layer bekommt bewusst andere Zahlen, damit eine
                // vertauschte Layer-Adresse auffaellt.
                std::vector<float> kl = k, vl = v;
                for (std::size_t i = 0; i < kl.size(); ++i) {
                    kl[i] += static_cast<float>(l);
                    vl[i] -= static_cast<float>(l);
                }
                gpu::DeviceBuffer dkl = upload_vector(kl);
                gpu::DeviceBuffer dvl = upload_vector(vl);
                cache.append(l, static_cast<const float*>(dkl.data()),
                             static_cast<const float*>(dvl.data()), n, stream);
                stream.synchronize();
            }
            cache.advance(n);
            all_k.insert(all_k.end(), k.begin(), k.end());
            all_v.insert(all_v.end(), v.begin(), v.end());
        };
        push(kFirstPush);
        {
            char buf[160];
            std::snprintf(buf, sizeof buf, "nach dem ersten Schub stehen %d Token im Cache",
                          kFirstPush);
            report.check(cache.size() == kFirstPush, buf);
        }
        push(kSecondPush);
        {
            char buf[200];
            std::snprintf(buf, sizeof buf,
                          "nach dem zweiten Schub stehen %d Token im Cache (der erste Schub "
                          "bleibt unangetastet)",
                          kBothPushes);
            report.check(cache.size() == kBothPushes, buf);
        }

        // --- Leseseite: durch den Attention-Kernel -------------------------
        // Ein Q-Vektor, der genau eine Dimension anspricht, macht aus der
        // Attention eine Ablesung: mit einem sehr grossen Wert auf Dimension d
        // und Nullen sonst gewinnt der Schluessel mit dem groessten K[.][d],
        // und die Ausgabe ist praktisch dessen V-Vektor. Statt dieses Kniffs
        // wird hier direkt gegen die CPU-Referenz derselben Attention
        // gerechnet — sie liest denselben Cacheinhalt und ist unabhaengig.
        auto check_layer_matches = [&](int layer) {
            // Erwarteter Cacheinhalt dieses Layers als Bytes.
            std::vector<float> kl(all_k), vl(all_v);
            for (std::size_t i = 0; i < kl.size(); ++i) {
                kl[i] += static_cast<float>(layer);
                vl[i] -= static_cast<float>(layer);
            }
            kl.resize(static_cast<std::size_t>(kCapacity) * per_token, 0.0f);
            vl.resize(static_cast<std::size_t>(kCapacity) * per_token, 0.0f);
            const ref::Int4Blob k_expected = ref::quantize_int4(kl, kGroup);
            const ref::Int4Blob v_expected = ref::quantize_int4(vl, kGroup);

            const int n_kv = cache.size();
            const std::size_t q_count = static_cast<std::size_t>(kHeads) * kHeadDim;
            const std::vector<float> q =
                test::make_uniform(q_count, 0x9999'0000ULL + static_cast<std::uint64_t>(layer),
                                   -1.0f, 1.0f);
            const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

            const std::vector<std::uint16_t> expected = ref::attention_int4_kv(
                q, k_expected.quants, k_expected.scales, v_expected.quants, v_expected.scales, 1,
                n_kv, kHeads, kKvHeads, kHeadDim, kCapacity, kGroup, n_kv - 1, scale);

            gpu::DeviceBuffer dq = upload_vector(q);
            gpu::DeviceBuffer dout(q_count * sizeof(std::uint16_t));

            // Derselbe Kernel, dieselbe Form, dieselben Zahlen -- einmal ueber
            // die Puffer **des Caches** und einmal ueber unabhaengig
            // hochgeladene Puffer mit dem erwarteten Inhalt.
            auto run = [&](const std::uint8_t* kq, const std::uint16_t* ks, const std::uint8_t* vq,
                           const std::uint16_t* vs) {
                dout.fill_zero();
                kern::attention_int4_kv(static_cast<kern::half_t*>(dout.data()),
                                        static_cast<const float*>(dq.data()), kq, ks, vq, vs, 1,
                                        n_kv, kHeads, kKvHeads, kHeadDim, kCapacity, kGroup,
                                        n_kv - 1, scale, stream);
                stream.synchronize();
                std::vector<std::uint16_t> got(q_count);
                dout.download(got.data(), got.size() * sizeof(std::uint16_t));
                return got;
            };

            const std::vector<std::uint16_t> actual =
                run(cache.k_quants(layer), cache.k_scales(layer), cache.v_quants(layer),
                    cache.v_scales(layer));

            // --- 1. Der Cacheinhalt selbst: bitgleich ------------------------
            //
            // Befund B1 (Pruefrunde 3): frueher stand hier bitgleich **gegen
            // die CPU-Referenz**. Das ging nur durch, weil der Cache 8 Token
            // fuehrte; ueber 115 Token weichen `__expf` in float und die
            // Referenz in double regulaer um 1 fp16-ULP ab -- eine
            // Kernel-Eigenschaft, kein Cachefehler. Die Bitgleichheit bleibt
            // deshalb, aber an der Stelle, an der sie etwas ueber den **Cache**
            // aussagt: gegen unabhaengig hochgeladene Puffer mit dem erwarteten
            // Inhalt, durch denselben Kernel. Jeder Byteunterschied im Cache --
            // ein verschobener Layer, ein ueberschriebener erster Schub, eine
            // falsche Skalenadresse -- faellt hier bitgenau auf. Die Toleranz
            // ist damit nicht gelockert, sondern die Zusicherung geschaerft:
            // sie trennt jetzt Cachefehler von Kernel-Rundung, statt beides zu
            // vermengen.
            gpu::DeviceBuffer ekq = upload_vector(k_expected.quants);
            gpu::DeviceBuffer eks = upload_vector(k_expected.scales);
            gpu::DeviceBuffer evq = upload_vector(v_expected.quants);
            gpu::DeviceBuffer evs = upload_vector(v_expected.scales);
            const std::vector<std::uint16_t> from_upload =
                run(static_cast<const std::uint8_t*>(ekq.data()),
                    static_cast<const std::uint16_t*>(eks.data()),
                    static_cast<const std::uint8_t*>(evq.data()),
                    static_cast<const std::uint16_t*>(evs.data()));

            std::size_t equal = 0;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (actual[i] == from_upload[i]) ++equal;
            }
            char buf[280];
            std::snprintf(buf, sizeof buf,
                          "Layer %d: der Cacheinhalt kommt beim Attention-Kernel an — ueber %d "
                          "belegte Token %zu von %zu Werten bitgleich mit demselben Kernel auf "
                          "unabhaengig hochgeladenen Puffern",
                          layer, n_kv, equal, actual.size());
            report.check(equal == actual.size(), buf);

            // --- 2. Und das Ergebnis ist auch richtig ------------------------
            // Gegen die unabhaengige CPU-Referenz, mit derselben Schranke, die
            // `kernels.attention` fuer diesen Kernel fuehrt.
            const HalfMatch m = compare_halves(actual, expected);
            std::snprintf(buf, sizeof buf,
                          "Layer %d: gegen die CPU-Referenz hoechstens 2 fp16-ULP Abstand "
                          "(gemessen %d, %zu von %zu bitgleich)",
                          layer, m.worst_ulp, m.equal, actual.size());
            report.check(m.worst_ulp <= 2, buf);
        };
        for (int l = 0; l < kLayers; ++l) check_layer_matches(l);

        // --- ab Position N abschneiden -------------------------------------
        cache.truncate(kFirstPush);
        {
            char buf[160];
            std::snprintf(buf, sizeof buf, "truncate(%d) laesst genau %d Token stehen", kFirstPush,
                          kFirstPush);
            report.check(cache.size() == kFirstPush, buf);
        }
        cache.truncate(kCapacity + 39);
        report.check(cache.size() == kFirstPush, "truncate hinter dem Fuellstand aendert nichts");

        // Nach dem Abschneiden dieselben Token noch einmal anhaengen: der
        // Cache muss danach wieder genau das enthalten wie vorher.
        all_k.resize(static_cast<std::size_t>(kFirstPush) * per_token);
        all_v.resize(static_cast<std::size_t>(kFirstPush) * per_token);
        seed = 0x4242'0003ULL;  // derselbe Keim wie beim zweiten Schub
        push(kSecondPush);
        {
            char buf[200];
            std::snprintf(buf, sizeof buf,
                          "nach truncate(%d) und erneutem Anhaengen wieder %d Token", kFirstPush,
                          kBothPushes);
            report.check(cache.size() == kBothPushes, buf);
        }
        for (int l = 0; l < kLayers; ++l) check_layer_matches(l);

        // --- verwerfen ------------------------------------------------------
        cache.clear();
        report.check(cache.size() == 0 && cache.empty(), "clear() verwirft alles");

        // --- was der Cache NICHT tut ---------------------------------------
        {
            // Voll laufen lassen und dann eins zu viel: er wirft nichts weg,
            // er bricht ab (SPEC Abschnitt 8: kein Evicting).
            const std::vector<float> chunk =
                test::make_uniform(per_token * kCapacity, 0x7777'0001ULL, -1.0f, 1.0f);
            gpu::DeviceBuffer d = upload_vector(chunk);
            for (int l = 0; l < kLayers; ++l) {
                cache.append(l, static_cast<const float*>(d.data()),
                             static_cast<const float*>(d.data()), kCapacity, stream);
            }
            stream.synchronize();
            cache.advance(kCapacity);
            report.check(cache.size() == kCapacity, "Cache laeuft bis genau zur Kapazitaet voll");

            bool threw = false;
            try {
                cache.append(0, static_cast<const float*>(d.data()),
                             static_cast<const float*>(d.data()), 1, stream);
            } catch (const std::exception&) {
                threw = true;
            }
            report.check(threw,
                         "ein Token ueber die Kapazitaet hinaus bricht ab — der Cache wirft "
                         "nichts von selbst weg (kein Evicting, SPEC Abschnitt 8)");
        }

        // --- unbrauchbare Form bricht ab ------------------------------------
        {
            bool threw = false;
            try {
                kv::Cache::create(kv::Shape{2, 2, 64, 8, 48});  // 48 teilt 64 nicht
            } catch (const std::exception&) {
                threw = true;
            }
            report.check(threw,
                         "group_size, die head_dim nicht teilt, bricht ab — sonst laege eine "
                         "Skalengrenze mitten in einem Kopfvektor");

            threw = false;
            try {
                (void)cache.k_quants(kLayers);
            } catch (const std::exception&) {
                threw = true;
            }
            report.check(threw, "Layer ausserhalb des Bereichs bricht ab");
        }

        return report.finish();
    } catch (const std::exception& error) {
        std::printf("[FEHLSCHLAG] %s: Ausnahme: %s\n", kName, error.what());
        return 1;
    }
}

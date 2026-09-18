// quasar — src/model/hf_config.cpp

#include "quasar/model/hf_config.hpp"

#include <array>
#include <filesystem>
#include <limits>

#include "quasar/core/error.hpp"
#include "quasar/core/json.hpp"
#include "quasar/core/utf8_path.hpp"
#include "quasar/model/gguf.hpp"

namespace quasar {
namespace {

namespace fs = std::filesystem;
using quasar::core::path_to_utf8;

// Feldnamen unterscheiden sich je Architektur. Die Liste ist absichtlich kurz:
// sie deckt die Schreibweisen ab, die in HuggingFace-Konfigurationen
// tatsaechlich vorkommen, und nicht mehr. Der erste Treffer gewinnt.
struct AliasList {
  std::string_view purpose;                 // fuer die Fehlermeldung
  std::array<std::string_view, 5> names;    // leere Eintraege werden uebersprungen
};

constexpr AliasList kLayers{"Layerzahl",
                            {"num_hidden_layers", "n_layer", "n_layers", "num_layers", ""}};
constexpr AliasList kEmbd{"Hidden-Groesse",
                          {"hidden_size", "n_embd", "d_model", "dim", ""}};
constexpr AliasList kHeads{"Attention-Heads",
                           {"num_attention_heads", "n_head", "n_heads", "num_heads", ""}};
constexpr AliasList kHeadsKv{"KV-Heads",
                             {"num_key_value_heads", "n_head_kv", "n_kv_heads",
                              "num_kv_heads", ""}};
constexpr AliasList kHeadDim{"Head-Dim", {"head_dim", "head_size", "", "", ""}};
constexpr AliasList kVocab{"Vokabulargroesse", {"vocab_size", "n_vocab", "", "", ""}};
constexpr AliasList kFf{"FFN-Groesse",
                        {"intermediate_size", "ffn_dim", "n_inner", "d_ff", ""}};
constexpr AliasList kContext{"Kontextlaenge",
                             {"max_position_embeddings", "n_ctx", "n_positions",
                              "max_sequence_length", ""}};
constexpr AliasList kTheta{"RoPE-Theta",
                           {"rope_theta", "rotary_emb_base", "rope_base", "", ""}};
constexpr AliasList kEps{"Norm-Epsilon",
                         {"rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon",
                          "norm_eps", "layer_norm_rms_epsilon"}};

// Die Objekte, in denen gesucht wird -- Wurzel zuerst, danach die
// legen die Sprachparameter unter "text_config" ab; ohne diesen Schritt
// findet man dort kein einziges Pflichtfeld.
std::vector<const json::Object*> search_scopes(const json::Object& root) {
  std::vector<const json::Object*> scopes{&root};
  for (const std::string_view nested : {"text_config", "llm_config", "language_config"}) {
    if (const json::Value* value = root.find(nested)) {
      if (value->is_object()) scopes.push_back(&value->as_object(nested));
    }
  }
  return scopes;
}

const json::Value* lookup(const std::vector<const json::Object*>& scopes,
                          const AliasList& aliases, std::string_view* found_name) {
  for (const json::Object* scope : scopes) {
    for (const std::string_view name : aliases.names) {
      if (name.empty()) continue;
      if (const json::Value* value = scope->find(name)) {
        if (found_name) *found_name = name;
        return value;
      }
    }
  }
  return nullptr;
}

std::string alias_text(const AliasList& aliases) {
  std::string out;
  for (const std::string_view name : aliases.names) {
    if (name.empty()) continue;
    if (!out.empty()) out.append(" / ");
    out.append(name);
  }
  return out;
}

[[noreturn]] void throw_missing(const std::string& source, const AliasList& aliases) {
  throw_error("HuggingFace-Konfiguration ", source, ": das Pflichtfeld fuer ",
              aliases.purpose, " fehlt. Gesucht wurde nach ", alias_text(aliases),
              " -- in der Wurzel und in text_config. quasar setzt hier keinen "
              "Ersatzwert ein.");
}

std::uint32_t to_u32(const json::Value& value, std::string_view what,
                     const std::string& source) {
  const std::int64_t v = value.as_int(what);
  if (v <= 0 || v > std::numeric_limits<std::uint32_t>::max()) {
    throw_error("HuggingFace-Konfiguration ", source, ": ", what, " ist ", v,
                ", erwartet wird eine positive Zahl.");
  }
  return static_cast<std::uint32_t>(v);
}

std::uint32_t require_u32(const std::vector<const json::Object*>& scopes,
                          const AliasList& aliases, const std::string& source) {
  std::string_view name;
  const json::Value* value = lookup(scopes, aliases, &name);
  if (!value) throw_missing(source, aliases);
  return to_u32(*value, name, source);
}

float require_f32(const std::vector<const json::Object*>& scopes,
                  const AliasList& aliases, const std::string& source) {
  std::string_view name;
  const json::Value* value = lookup(scopes, aliases, &name);
  if (!value) throw_missing(source, aliases);
  return static_cast<float>(value->as_double(name));
}

std::string architecture_of(const json::Object& root, const std::string& source) {
  if (const json::Value* value = root.find("model_type")) {
    return value->as_string("model_type");
  }
  if (const json::Value* value = root.find("architectures")) {
    const json::Array& list = value->as_array("architectures");
    if (!list.empty()) return list[0].as_string("architectures[0]");
  }
  throw_error("HuggingFace-Konfiguration ", source,
              ": weder 'model_type' noch 'architectures' vorhanden -- die "
              "Architektur ist damit unbestimmt.");
}

void read_rope_scaling(const std::vector<const json::Object*>& scopes,
                       ModelParams& params) {
  const json::Value* value = nullptr;
  for (const json::Object* scope : scopes) {
    value = scope->find("rope_scaling");
    if (!value) value = scope->find("rope_parameters");
    if (value && !value->is_null()) break;
    value = nullptr;
  }
  if (!value) return;

  const json::Object& object = value->as_object("rope_scaling");
  if (const json::Value* type = object.find("rope_type")) {
    params.rope_scaling.type = type->as_string("rope_scaling.rope_type");
  } else if (const json::Value* legacy = object.find("type")) {
    params.rope_scaling.type = legacy->as_string("rope_scaling.type");
  }
  if (const json::Value* factor = object.find("factor")) {
    params.rope_scaling.factor = factor->as_double("rope_scaling.factor");
  }
  if (const json::Value* original = object.find("original_max_position_embeddings")) {
    const std::int64_t v = original->as_int("rope_scaling.original_max_position_embeddings");
    if (v > 0 && v <= std::numeric_limits<std::uint32_t>::max()) {
      params.rope_scaling.original_context_length = static_cast<std::uint32_t>(v);
    }
  }
  const auto optional_double = [&](const char* key, double& out) {
    if (const json::Value* field = object.find(key)) out = field->as_double(key);
  };
  optional_double("low_freq_factor", params.rope_scaling.low_frequency_factor);
  optional_double("high_freq_factor", params.rope_scaling.high_frequency_factor);
  optional_double("attention_factor", params.rope_scaling.attention_factor);
  optional_double("beta_fast", params.rope_scaling.beta_fast);
  optional_double("beta_slow", params.rope_scaling.beta_slow);
  // Ohne Typangabe ist die Skalierung nicht auswertbar -- dann gilt sie als
  // nicht vorhanden, statt einen Typ zu erfinden.
  if (params.rope_scaling.type.empty()) params.rope_scaling.factor = 1.0;
}

void finish(ModelParams& params) {
  if (params.head_dim == 0) {
    if (params.n_embd % params.n_heads != 0) {
      throw_error(params.source, ": head_dim fehlt und laesst sich nicht ableiten -- ",
                  params.n_embd, " (hidden) ist nicht durch ", params.n_heads,
                  " (Heads) teilbar.");
    }
    params.head_dim = params.n_embd / params.n_heads;
    params.head_dim_derived = true;
  }
  params.validate();
}

}  // namespace

void ModelParams::validate() const {
  const auto fail = [&](std::string_view what) {
    throw_error(source, ": ", what);
  };
  if (architecture.empty()) fail("die Architektur-Kennung ist leer.");
  if (n_layers == 0) fail("die Layerzahl ist 0.");
  if (n_embd == 0) fail("die Hidden-Groesse ist 0.");
  if (n_heads == 0) fail("die Zahl der Attention-Heads ist 0.");
  if (n_heads_kv == 0) fail("die Zahl der KV-Heads ist 0.");
  if (head_dim == 0) fail("head_dim ist 0.");
  if (n_vocab == 0) fail("die Vokabulargroesse ist 0.");
  if (n_ff == 0) fail("die FFN-Groesse ist 0.");
  if (context_length == 0) fail("die Kontextlaenge ist 0.");
  if (!(rope_theta > 0.0f)) fail("rope_theta ist nicht groesser als 0.");
  if (!(norm_epsilon > 0.0f)) fail("das Norm-Epsilon ist nicht groesser als 0.");
  if (n_heads_kv > n_heads) {
    throw_error(source, ": es gibt ", n_heads_kv, " KV-Heads, aber nur ", n_heads,
                " Attention-Heads.");
  }
  if (n_heads % n_heads_kv != 0) {
    throw_error(source, ": ", n_heads, " Attention-Heads sind nicht durch ", n_heads_kv,
                " KV-Heads teilbar -- so ist keine Gruppierung moeglich.");
  }
}

std::string ModelParams::describe() const {
  std::string out = "Modell ";
  out.append(architecture);
  out.append("\n  Layer ");
  out.append(std::to_string(n_layers));
  out.append(", Hidden ");
  out.append(std::to_string(n_embd));
  out.append(", FFN ");
  out.append(std::to_string(n_ff));
  out.append("\n  Heads ");
  out.append(std::to_string(n_heads));
  out.append(", KV-Heads ");
  out.append(std::to_string(n_heads_kv));
  if (n_heads_kv_derived) out.append(" (abgeleitet)");
  out.append(", Head-Dim ");
  out.append(std::to_string(head_dim));
  if (head_dim_derived) out.append(" (abgeleitet)");
  out.append("\n  Vokabular ");
  out.append(std::to_string(n_vocab));
  out.append(", Kontext ");
  out.append(std::to_string(context_length));
  out.append("\n  RoPE-Theta ");
  out.append(std::to_string(rope_theta));
  if (rope_scaling.present()) {
    out.append(", Skalierung ");
    out.append(rope_scaling.type);
    out.append(" x");
    out.append(std::to_string(rope_scaling.factor));
  }
  out.append(", Norm-Epsilon ");
  out.append(std::to_string(norm_epsilon));
  out.append("\n  Quelle ");
  out.append(source);
  out.push_back('\n');
  return out;
}

ModelParams read_hf_config(const std::string& path) {
  std::string file = path;
  const fs::path p = quasar::core::utf8_path(path);
  std::error_code ec;
  if (fs::is_directory(p, ec)) file = path_to_utf8(p / "config.json");
  if (!fs::is_regular_file(quasar::core::utf8_path(file), ec)) {
    throw_error("HuggingFace-Konfiguration nicht gefunden: ", file);
  }

  const json::Value root_value = json::parse_file(file);
  const json::Object& root = root_value.as_object("config.json");
  const std::vector<const json::Object*> scopes = search_scopes(root);

  ModelParams params;
  params.source = file;
  params.architecture = architecture_of(root, file);
  params.n_layers = require_u32(scopes, kLayers, file);
  params.n_embd = require_u32(scopes, kEmbd, file);
  params.n_heads = require_u32(scopes, kHeads, file);
  params.n_vocab = require_u32(scopes, kVocab, file);
  params.n_ff = require_u32(scopes, kFf, file);
  params.context_length = require_u32(scopes, kContext, file);
  params.rope_theta = require_f32(scopes, kTheta, file);
  params.norm_epsilon = require_f32(scopes, kEps, file);

  std::string_view name;
  if (const json::Value* value = lookup(scopes, kHeadsKv, &name)) {
    params.n_heads_kv = to_u32(*value, name, file);
  } else {
    // HuggingFace-Vorgabe: ohne num_key_value_heads ist es klassische MHA.
    params.n_heads_kv = params.n_heads;
    params.n_heads_kv_derived = true;
  }
  if (const json::Value* value = lookup(scopes, kHeadDim, &name)) {
    params.head_dim = to_u32(*value, name, file);
  }

  read_rope_scaling(scopes, params);
  finish(params);
  return params;
}

ModelParams params_from_gguf(const gguf::GgufFile& file) {
  ModelParams params;
  params.source = "GGUF " + file.path();

  const std::string arch(file.architecture());
  params.architecture = arch;

  const auto key = [&arch](std::string_view suffix) { return arch + std::string(suffix); };

  params.n_layers = file.require_u32(key(".block_count"));
  params.n_embd = file.require_u32(key(".embedding_length"));
  params.n_heads = file.require_u32(key(".attention.head_count"));
  params.n_ff = file.require_u32(key(".feed_forward_length"));
  params.context_length = file.require_u32(key(".context_length"));
  params.rope_theta = file.require_f32(key(".rope.freq_base"));
  params.norm_epsilon = file.require_f32(key(".attention.layer_norm_rms_epsilon"));

  if (const std::optional<std::uint32_t> kv = file.get_u32(key(".attention.head_count_kv"))) {
    params.n_heads_kv = *kv;
  } else {
    params.n_heads_kv = params.n_heads;
    params.n_heads_kv_derived = true;
  }
  if (const std::optional<std::uint32_t> dim = file.get_u32(key(".attention.key_length"))) {
    params.head_dim = *dim;
  }

  // Vokabulargroesse: erst das ausdrueckliche Feld, sonst die Laenge der
  // Tokenliste. Beides steht in der Datei -- geraten wird nichts.
  if (const std::optional<std::uint32_t> vocab = file.get_u32(key(".vocab_size"))) {
    params.n_vocab = *vocab;
  } else if (const gguf::MetaValue* tokens = file.metadata().find("tokenizer.ggml.tokens")) {
    const std::uint64_t n = tokens->size("tokenizer.ggml.tokens");
    if (n == 0 || n > std::numeric_limits<std::uint32_t>::max()) {
      throw_error(params.source, ": tokenizer.ggml.tokens hat ", n, " Eintraege.");
    }
    params.n_vocab = static_cast<std::uint32_t>(n);
  } else {
    throw_error(params.source, ": weder '", key(".vocab_size"),
                "' noch 'tokenizer.ggml.tokens' vorhanden -- die Vokabulargroesse "
                "ist unbestimmt.");
  }

  if (const std::optional<float> factor = file.get_f32(key(".rope.scaling.factor"))) {
    // Befund M13: hier stand `get_string_or(..., "linear")`. Steht in der Datei
    // ein Skalierungsfaktor, aber kein Typ, erfand quasar "linear" -- obwohl der
    // Unterschied zwischen linear, yarn und llama3 die Positionskodierung
    // veraendert. Der Schwesterpfad `read_hf_config` lehnt genau das ab ("statt
    // einen Typ zu erfinden"); zwei Befueller derselben Struktur mit
    // entgegengesetzter Politik.
    //
    // Jetzt gilt in beiden Pfaden dasselbe: ohne Typangabe ist die Skalierung
    // nicht auswertbar und gilt als nicht vorhanden -- geraten wird nichts.
    const std::string_view type = file.get_string_or(key(".rope.scaling.type"), "");
    if (type.empty()) {
      params.rope_scaling.factor = 1.0;
      params.rope_scaling.type.clear();
    } else {
      params.rope_scaling.factor = static_cast<double>(*factor);
      params.rope_scaling.type = std::string(type);
      params.rope_scaling.original_context_length =
          file.get_u32_or(key(".rope.scaling.original_context_length"), 0);
      params.rope_scaling.low_frequency_factor =
          file.get_f32(key(".rope.scaling.low_freq_factor")).value_or(0.0f);
      params.rope_scaling.high_frequency_factor =
          file.get_f32(key(".rope.scaling.high_freq_factor")).value_or(0.0f);
      params.rope_scaling.attention_factor =
          file.get_f32(key(".rope.scaling.attn_factor")).value_or(0.0f);
      params.rope_scaling.beta_fast =
          file.get_f32(key(".rope.scaling.yarn_beta_fast")).value_or(0.0f);
      params.rope_scaling.beta_slow =
          file.get_f32(key(".rope.scaling.yarn_beta_slow")).value_or(0.0f);
    }
  }

  finish(params);
  return params;
}

}  // namespace quasar

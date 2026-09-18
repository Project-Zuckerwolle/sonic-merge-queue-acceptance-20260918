# quasar — src/model/CMakeLists.txt
#
# Metadaten, Tensor-Verzeichnis". Nur lesen und beschreiben; konvertiert und
# quantisiert wird in `format/`, `quant/` und `convert/`.
#
# Reines C++, kein HIP. Alles Datei- und JSON-Handwerk kommt aus quasar_core.

add_library(quasar_model STATIC
  gguf.cpp
  gguf_value.cpp
  gguf_types.cpp
  safetensors.cpp
  hf_config.cpp)

target_link_libraries(quasar_model PUBLIC quasar_core)
quasar_public_includes(quasar_model)
target_compile_features(quasar_model PUBLIC cxx_std_20)
set_target_properties(quasar_model PROPERTIES EXPORT_NAME model)

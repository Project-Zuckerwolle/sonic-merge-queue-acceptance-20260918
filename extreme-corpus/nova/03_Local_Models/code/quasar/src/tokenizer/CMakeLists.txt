# quasar — src/tokenizer/CMakeLists.txt
#
# Vorzerleger-Ketten. Haengt an `quasar_core` (Unicode, JSON, mmap) und an
# `quasar_model` (GGUF-Vokabular).

add_library(quasar_tokenizer STATIC
  tokenizer.cpp
  normalizers.cpp
  pre_tokenizers.cpp
  models.cpp
  decoders.cpp
  precompiled.cpp
  gguf_vocab.cpp)

target_include_directories(quasar_tokenizer
  PRIVATE "${PROJECT_SOURCE_DIR}/src")
quasar_public_includes(quasar_tokenizer)

target_link_libraries(quasar_tokenizer PUBLIC quasar_core quasar_model)
target_compile_features(quasar_tokenizer PUBLIC cxx_std_20)
set_target_properties(quasar_tokenizer PROPERTIES EXPORT_NAME tokenizer)

// apex_batch.h — Batched Subagent-Inferenz (Design §4.4, §13, §18).
//
// Ministral 3B verarbeitet 2 Sequenzen gleichzeitig in EINEM GEMM-Pass
// (interleaved mit dem 14B in den GPU-Idle-Fenstern). Der eigentliche
// Token-Generator ist ein injizierter Callback (SubagentFn) — ohne Modell
// testbar. Der CUDA-Pfad (apex_batch.cu) spiegelt den GEMM-Batch.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace nova::apex {

using SubagentFn = std::function<std::string(const std::string& prompt)>;  // 3B

// Host-Pfad: verarbeitet prompts in 2er-Batches (GEMM-Batch-Größe 2).
// weights muss geladen (nicht leer) sein. Liefert je Prompt eine Ausgabe.
std::vector<std::string> apex_batch_infer_host(const std::vector<uint8_t>& weights,
                                               const std::vector<std::string>& prompts,
                                               const SubagentFn& fn);

#ifdef NOVA_HAVE_CUDA
// GPU-Pfad (apex_batch.cu): batched GEMM für 2 Sequenzen. logits_out:
// 2 * vocab. Skelett — Verdrahtung an reale 3B-Gewichte bei TB 13 auf dem Server.
bool apex_batch_gemm_cuda(const float* weights, int rows, int cols,
                          const float* seq0, const float* seq1,
                          float* logits_out, std::string* err = nullptr);
#endif

}  // namespace nova::apex

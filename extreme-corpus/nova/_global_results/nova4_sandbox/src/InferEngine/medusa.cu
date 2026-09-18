// medusa.cu — CUDA-Pfad für Medusa-Kopf-Auswertung (Design §9, TB 8).
//
// Wird NUR auf dem Server-PC kompiliert (NOVA_HAVE_CUDA) und NUR genutzt, wenn
// trainierte Köpfe vorliegen (§5.5). Ein Block je Kopf reduziert die Logits
// (vocab) auf den Argmax-Token — der GPU-relevante Schritt der Kopf-Auswertung.
// Die Akzeptanzlogik selbst ist günstig und läuft host-seitig (medusa_host.cpp).
//
// Gerüst-Status: Argmax-Reduktion + Launcher vollständig; Verdrahtung an reale
// Köpfe erfolgt bei TB 8, sobald Heads konvertiert sind (converter --medusa-heads).
#include "InferEngine/medusa.h"

#include <cuda_runtime.h>
#include <float.h>

#include <string>

namespace nova::infer {

namespace {
constexpr int kThreads = 256;

#define NV_CUDA_CHECK(call, errptr)                                          \
    if (cudaError_t _e = (call); _e != cudaSuccess) {                       \
        if (errptr) *(errptr) = std::string("CUDA: ") +                     \
                    cudaGetErrorString(_e) + " (" #call ")";                \
        ok = false;                                                         \
        break;                                                              \
    } else (void)0

// Ein Block pro Kopf: Block-weite Reduktion über vocab -> Argmax-Index.
__global__ void argmax_heads_kernel(const float* __restrict__ logits,
                                    int vocab, int* __restrict__ out_tokens) {
    const int head = blockIdx.x;
    const float* row = logits + size_t(head) * vocab;

    __shared__ float s_val[kThreads];
    __shared__ int   s_idx[kThreads];

    float best = -FLT_MAX; int bi = 0;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float v = row[i];
        if (v > best) { best = v; bi = i; }
    }
    s_val[threadIdx.x] = best;
    s_idx[threadIdx.x] = bi;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (s_val[threadIdx.x + stride] > s_val[threadIdx.x]) {
                s_val[threadIdx.x] = s_val[threadIdx.x + stride];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out_tokens[head] = s_idx[0];
}
}  // namespace

bool medusa_argmax_heads_cuda(const float* logits, int num_heads, int vocab,
                              int* out_tokens, std::string* err) {
    if (num_heads <= 0 || vocab <= 0) return true;
    const size_t n = size_t(num_heads) * vocab;

    float* d_logits = nullptr; int* d_out = nullptr;
    bool ok = true;
    do {
        NV_CUDA_CHECK(cudaMalloc(&d_logits, n * sizeof(float)), err);
        NV_CUDA_CHECK(cudaMalloc(&d_out, size_t(num_heads) * sizeof(int)), err);
        NV_CUDA_CHECK(cudaMemcpy(d_logits, logits, n * sizeof(float),
                                 cudaMemcpyHostToDevice), err);
        argmax_heads_kernel<<<num_heads, kThreads>>>(d_logits, vocab, d_out);
        NV_CUDA_CHECK(cudaGetLastError(), err);
        NV_CUDA_CHECK(cudaDeviceSynchronize(), err);
        NV_CUDA_CHECK(cudaMemcpy(out_tokens, d_out, size_t(num_heads) * sizeof(int),
                                 cudaMemcpyDeviceToHost), err);
    } while (false);

    if (d_logits) cudaFree(d_logits);
    if (d_out)    cudaFree(d_out);
    return ok;
}

}  // namespace nova::infer

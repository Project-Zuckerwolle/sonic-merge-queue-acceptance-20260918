// turboquant.cu — CUDA-Pfad für TurboQuant-Rotation (Design §7.1, TB 6).
//
// Wird NUR auf dem Server-PC kompiliert (NOVA_HAVE_CUDA, NVIDIA RTX 3080).
// PolarQuant-Rotation r = Q*v ist der dominante Kostenanteil (head_dim^2 je
// Vektor, batched über den ganzen KV-Cache). Outlier-Codebuch-Suche, 2-Bit-
// Quantisierung und QJL-Bit-Packing sind günstig und laufen bit-identisch zur
// Host-Referenz (turboquant_host.cpp) CPU-seitig.
//
// Gerüst-Status: Rotations-Kernel + Launcher vollständig; Verdrahtung in den
// kv_manager-Compute-Pfad erfolgt bei TB 6 auf dem Server.
#include "InferEngine/turboquant.h"

#include <cuda_runtime.h>

#include <string>

namespace nova::infer {

namespace {
#define NV_CUDA_CHECK(call, errptr)                                          \
    if (cudaError_t _e = (call); _e != cudaSuccess) {                       \
        if (errptr) *(errptr) = std::string("CUDA: ") +                     \
                    cudaGetErrorString(_e) + " (" #call ")";                \
        ok = false;                                                         \
        break;                                                              \
    } else (void)0

// Ein Block pro Vektor, ein Thread pro Ausgabe-Kanal i.
// transpose=false: out[i] = sum_k Q[i*d+k] * in[k]   (r = Q v)
// transpose=true:  out[k] = sum_i Q[i*d+k] * in[i]   (v = Q^T r)
__global__ void rotate_kernel(const float* __restrict__ Q,
                              const float* __restrict__ in,
                              float* __restrict__ out,
                              int d, int transpose) {
    const int vec = blockIdx.x;
    const int i   = threadIdx.x;
    if (i >= d) return;
    const float* x = in + size_t(vec) * d;
    float acc = 0.0f;
    if (!transpose) {
        const float* row = Q + size_t(i) * d;
        for (int k = 0; k < d; ++k) acc += row[k] * x[k];
    } else {
        for (int j = 0; j < d; ++j) acc += Q[size_t(j) * d + i] * x[j];
    }
    out[size_t(vec) * d + i] = acc;
}
}  // namespace

bool turboquant_rotate_batch_cuda(const float* Q, const float* in, float* out,
                                  int head_dim, int num_vectors, bool transpose,
                                  std::string* err) {
    if (head_dim <= 0 || num_vectors <= 0) return true;
    const size_t qn = size_t(head_dim) * head_dim;
    const size_t vn = size_t(num_vectors) * head_dim;

    float* d_Q = nullptr; float* d_in = nullptr; float* d_out = nullptr;
    bool ok = true;
    do {
        NV_CUDA_CHECK(cudaMalloc(&d_Q, qn * sizeof(float)), err);
        NV_CUDA_CHECK(cudaMalloc(&d_in, vn * sizeof(float)), err);
        NV_CUDA_CHECK(cudaMalloc(&d_out, vn * sizeof(float)), err);
        NV_CUDA_CHECK(cudaMemcpy(d_Q, Q, qn * sizeof(float), cudaMemcpyHostToDevice), err);
        NV_CUDA_CHECK(cudaMemcpy(d_in, in, vn * sizeof(float), cudaMemcpyHostToDevice), err);

        // Ein Thread pro Ausgabe-Kanal; Blockgröße auf Vielfaches von 32 aufgerundet.
        const int threads = ((head_dim + 31) / 32) * 32;
        rotate_kernel<<<num_vectors, threads>>>(d_Q, d_in, d_out, head_dim,
                                                transpose ? 1 : 0);
        NV_CUDA_CHECK(cudaGetLastError(), err);
        NV_CUDA_CHECK(cudaDeviceSynchronize(), err);
        NV_CUDA_CHECK(cudaMemcpy(out, d_out, vn * sizeof(float), cudaMemcpyDeviceToHost), err);
    } while (false);

    if (d_Q)   cudaFree(d_Q);
    if (d_in)  cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    return ok;
}

}  // namespace nova::infer

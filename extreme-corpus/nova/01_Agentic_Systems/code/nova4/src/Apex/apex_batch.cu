// apex_batch.cu — CUDA batched Subagent-Inferenz (Design §4.4, §13, TB 13).
//
// Wird NUR auf dem Server-PC kompiliert (NOVA_HAVE_CUDA). Ein GEMM-Pass
// verarbeitet 2 Subagent-Sequenzen gleichzeitig (Batch-Dim 2): logits[b] =
// W * seq[b]. Das füllt die GPU-Idle-Fenster zwischen den 14B-Chunk-Loads
// (interleaved, §6.2). Token-Sampling/Decoding läuft host-seitig.
//
// Gerüst-Status: Batch-GEMM + Launcher vollständig; Verdrahtung an reale 3B-
// Gewichte (ministral-3b.bin) erfolgt bei TB 13 auf dem Server.
#include "Apex/apex_batch.h"

#include <cuda_runtime.h>

#include <string>

namespace nova::apex {

namespace {
constexpr int kThreads = 128;

#define NV_CUDA_CHECK(call, errptr)                                          \
    if (cudaError_t _e = (call); _e != cudaSuccess) {                       \
        if (errptr) *(errptr) = std::string("CUDA: ") +                     \
                    cudaGetErrorString(_e) + " (" #call ")";                \
        ok = false;                                                         \
        break;                                                              \
    } else (void)0

// Batch-GEMM: out[b*rows + r] = sum_c W[r*cols + c] * seq_b[c], b in {0,1}.
// grid.x = rows, grid.y = 2 (Batch).
__global__ void batch_gemv_kernel(const float* __restrict__ W,
                                  const float* __restrict__ s0,
                                  const float* __restrict__ s1,
                                  int rows, int cols, float* __restrict__ out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* seq = (blockIdx.y == 0) ? s0 : s1;
    const float* row = W + size_t(r) * cols;
    float acc = 0.0f;
    for (int c = 0; c < cols; ++c) acc += row[c] * seq[c];
    out[size_t(blockIdx.y) * rows + r] = acc;
}
}  // namespace

bool apex_batch_gemm_cuda(const float* weights, int rows, int cols,
                          const float* seq0, const float* seq1,
                          float* logits_out, std::string* err) {
    if (rows <= 0 || cols <= 0) return true;
    float* d_w = nullptr; float* d_s0 = nullptr; float* d_s1 = nullptr; float* d_o = nullptr;
    bool ok = true;
    do {
        NV_CUDA_CHECK(cudaMalloc(&d_w, size_t(rows) * cols * sizeof(float)), err);
        NV_CUDA_CHECK(cudaMalloc(&d_s0, size_t(cols) * sizeof(float)), err);
        NV_CUDA_CHECK(cudaMalloc(&d_s1, size_t(cols) * sizeof(float)), err);
        NV_CUDA_CHECK(cudaMalloc(&d_o, size_t(2) * rows * sizeof(float)), err);
        NV_CUDA_CHECK(cudaMemcpy(d_w, weights, size_t(rows) * cols * sizeof(float), cudaMemcpyHostToDevice), err);
        NV_CUDA_CHECK(cudaMemcpy(d_s0, seq0, size_t(cols) * sizeof(float), cudaMemcpyHostToDevice), err);
        NV_CUDA_CHECK(cudaMemcpy(d_s1, seq1, size_t(cols) * sizeof(float), cudaMemcpyHostToDevice), err);

        dim3 grid((rows + kThreads - 1) / kThreads, 2);
        batch_gemv_kernel<<<grid, kThreads>>>(d_w, d_s0, d_s1, rows, cols, d_o);
        NV_CUDA_CHECK(cudaGetLastError(), err);
        NV_CUDA_CHECK(cudaDeviceSynchronize(), err);
        NV_CUDA_CHECK(cudaMemcpy(logits_out, d_o, size_t(2) * rows * sizeof(float), cudaMemcpyDeviceToHost), err);
    } while (false);
    if (d_w) cudaFree(d_w); if (d_s0) cudaFree(d_s0);
    if (d_s1) cudaFree(d_s1); if (d_o) cudaFree(d_o);
    return ok;
}

}  // namespace nova::apex

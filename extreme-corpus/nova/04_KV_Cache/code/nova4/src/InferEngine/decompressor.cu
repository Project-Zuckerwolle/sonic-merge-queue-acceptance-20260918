// decompressor.cu — CUDA-Dequantisierung INT2/INT4 -> FP16 (Design §6.2, TB 3).
//
// Wird NUR auf dem Server-PC kompiliert (NOVA_HAVE_CUDA, NVIDIA RTX 3080).
// Zstd-Dekomprimierung läuft CPU-seitig (decompress_payload); dieser Kernel
// übernimmt die INT->FP16-Dequantisierung auf der GPU. Mathematik identisch
// zu dequantize_to_fp16_host (decompressor_host.cpp) -> bit-kompatibel.
//
// Gerüst-Status: Kernels + Launcher vollständig; Integration in die
// ring_buffer-Compute-Stufe (Stream B) erfolgt bei TB 4/5 auf dem Server.
#include "InferEngine/decompressor.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <string>

namespace nova::infer {

namespace {
constexpr int kGroupSize = 32;       // == modelstore::GROUP_SIZE
constexpr int kThreads   = 256;

// Setzt ok=false und verlässt das umgebende do/while, damit cudaFree läuft.
// if-Form (kein do/while-Wrapper), sonst bräche break nur aus dem Makro aus.
#define NV_CUDA_CHECK(call, errptr)                                          \
    if (cudaError_t _e = (call); _e != cudaSuccess) {                       \
        if (errptr) *(errptr) = std::string("CUDA: ") +                     \
                    cudaGetErrorString(_e) + " (" #call ")";                \
        ok = false;                                                         \
        break;                                                              \
    } else (void)0

__global__ void dequant_int4_kernel(const uint8_t* __restrict__ packed,
                                    const __half* __restrict__ scales,
                                    unsigned long long n,
                                    __half* __restrict__ out) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float scale = __half2float(scales[i / kGroupSize]);
    const int   byte  = packed[i >> 1];
    const int   code  = (byte >> ((i & 1) * 4)) & 0xF;
    out[i] = __float2half(float(code - 8) * scale);
}

__global__ void dequant_int2_kernel(const uint8_t* __restrict__ packed,
                                    const __half* __restrict__ scales,
                                    unsigned long long n,
                                    __half* __restrict__ out) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float scale = __half2float(scales[i / kGroupSize]);
    const int   byte  = packed[i >> 2];
    const int   code  = (byte >> ((i & 3) * 2)) & 0x3;
    out[i] = __float2half(float(code - 2) * scale);
}
}  // namespace

bool cuda_device_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

bool decompress_chunk_cuda(const modelstore::Chunk& c,
                           std::vector<uint16_t>& fp16_out, std::string* err) {
    std::vector<uint8_t> packed;
    if (!modelstore::decompress_payload(c, packed, err)) return false;

    const unsigned long long n = c.header.num_elements();
    const auto qt = modelstore::QuantType(c.header.quant_type);
    const size_t need = modelstore::packed_bytes(n, qt);
    if (packed.size() < need) { if (err) *err = "Payload kürzer als erwartet"; return false; }

    uint8_t* d_packed = nullptr;
    __half*  d_scales = nullptr;
    __half*  d_out    = nullptr;
    fp16_out.resize(static_cast<size_t>(n));

    bool ok = true;
    do {
        NV_CUDA_CHECK(cudaMalloc(&d_packed, packed.size()), err);
        NV_CUDA_CHECK(cudaMalloc(&d_scales, c.scales.size()), err);
        NV_CUDA_CHECK(cudaMalloc(&d_out, size_t(n) * sizeof(__half)), err);
        NV_CUDA_CHECK(cudaMemcpy(d_packed, packed.data(), packed.size(),
                                 cudaMemcpyHostToDevice), err);
        NV_CUDA_CHECK(cudaMemcpy(d_scales, c.scales.data(), c.scales.size(),
                                 cudaMemcpyHostToDevice), err);

        const unsigned long long blocks = (n + kThreads - 1) / kThreads;
        if (qt == modelstore::QuantType::INT4)
            dequant_int4_kernel<<<(unsigned)blocks, kThreads>>>(d_packed, d_scales, n, d_out);
        else
            dequant_int2_kernel<<<(unsigned)blocks, kThreads>>>(d_packed, d_scales, n, d_out);

        NV_CUDA_CHECK(cudaGetLastError(), err);
        NV_CUDA_CHECK(cudaDeviceSynchronize(), err);
        NV_CUDA_CHECK(cudaMemcpy(fp16_out.data(), d_out, size_t(n) * sizeof(__half),
                                 cudaMemcpyDeviceToHost), err);
    } while (false);

    if (d_packed) cudaFree(d_packed);
    if (d_scales) cudaFree(d_scales);
    if (d_out)    cudaFree(d_out);
    return ok;
}

}  // namespace nova::infer

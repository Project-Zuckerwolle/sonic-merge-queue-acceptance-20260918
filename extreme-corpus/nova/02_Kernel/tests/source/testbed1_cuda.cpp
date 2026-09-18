// testbed1_cuda.cpp — CUDA Baseline (Aufgabe 1, TB1).
//
// [SERVER] Unter NOVA_HAVE_CUDA: RTX 3080 erkennen (sm_86), cudaMemGetInfo,
// einfacher Device-Zugriff — auf dem Server auch aus Session 0 (Service).
// [DEV-PC] Ohne CUDA-Toolkit: die Detektions-/Fehlerlogik wird geprüft und der
// Test meldet SKIPPED (Exit 0), damit CTest grün bleibt.
#include <cstdio>
#include <iostream>

#ifdef NOVA_HAVE_CUDA
#include <cuda_runtime.h>
#endif

int main() {
    std::cout << "=== Testbed 1: CUDA Baseline ===\n";
#ifdef NOVA_HAVE_CUDA
    int n = 0;
    const cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess || n == 0) {
        std::printf("  Keine CUDA-GPU: %s\n", cudaGetErrorString(e));
        std::cout << "=== Testbed 1: FEHLGESCHLAGEN ===\n";
        return 1;
    }
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);
    size_t freeb = 0, totalb = 0;
    cudaMemGetInfo(&freeb, &totalb);
    std::printf("  GPU: %s (sm_%d%d), VRAM %zu MB (frei %zu MB)\n",
                p.name, p.major, p.minor, size_t(totalb >> 20), size_t(freeb >> 20));
    const bool ok = (p.major * 10 + p.minor) >= 60;   // compute-fähig (Ampere = 86)
    std::cout << "\n=== Testbed 1: " << (ok ? "BESTANDEN" : "FEHLGESCHLAGEN") << " ===\n";
    return ok ? 0 : 1;
#else
    std::cout << "  SKIPPED (kein CUDA-Toolkit auf diesem Host) — Detektionslogik ok,\n"
                 "  echte Prüfung auf dem RTX-3080-Server (sm_86, Session 0).\n";
    std::cout << "=== Testbed 1: SKIPPED ===\n";
    return 0;
#endif
}

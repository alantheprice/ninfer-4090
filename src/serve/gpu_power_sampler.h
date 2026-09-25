#pragma once

// NVML dynamic loading for GPU power sampling behind the /usage energy report.
// Loaded once; sampling is a no-op returning 0 when the library, device, or a
// required symbol is unavailable (energy accounting then reports unavailable
// rather than wrong numbers).
//
// Unlike the mainline implementation, the NVML device is resolved from this
// process's CUDA device ordinal (ServeOptions::device) via the PCI bus ID, so
// a server pinned to GPU 1 measures GPU 1 instead of whichever device first
// answers NVML's index order. That matters on multi-GPU hosts where two
// ninfer-serve processes run side by side: each must bill its own board.

#include <cuda_runtime_api.h>

#include <cstdio>
#include <dlfcn.h>

namespace ninfer::serve {

struct GpuPowerSampler {
    // nvmlDeviceGetPowerUsage(nvmlDevice_t, unsigned int*) — milliwatts.
    using PowerFn = int (*)(void*, unsigned int*);
    // nvmlDeviceGetHandleByPciBusId_v2(const char*, nvmlDevice_t*).
    using HandleByPciFn = int (*)(const char*, void*);
    // nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t*).
    using HandleByIdxFn = int (*)(unsigned int, void*);
    PowerFn query_power = nullptr;
    HandleByPciFn handle_by_pci_fn = nullptr;
    HandleByIdxFn handle_by_idx_fn = nullptr;
    void* nvml_handle = nullptr;
    void* device = nullptr;
    bool initialized = false;

    explicit GpuPowerSampler(int cuda_device) {
        nvml_handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
        if (nvml_handle == nullptr) {
            std::fprintf(stderr, "[nvml] dlopen failed\n");
            return;
        }
        auto* init_fn = (int (*)())dlsym(nvml_handle, "nvmlInit_v2");
        handle_by_pci_fn =
            (HandleByPciFn)dlsym(nvml_handle, "nvmlDeviceGetHandleByPciBusId_v2");
        handle_by_idx_fn =
            (HandleByIdxFn)dlsym(nvml_handle, "nvmlDeviceGetHandleByIndex_v2");
        query_power = (PowerFn)dlsym(nvml_handle, "nvmlDeviceGetPowerUsage");
        if (init_fn == nullptr || query_power == nullptr ||
            (handle_by_pci_fn == nullptr && handle_by_idx_fn == nullptr)) {
            std::fprintf(stderr, "[nvml] symbol lookup failed\n");
            return;
        }
        if (init_fn() != 0) {
            std::fprintf(stderr, "[nvml] init failed\n");
            return;
        }

        // Resolve this server's board from the CUDA device the engine runs on.
        int ordinal = cuda_device >= 0 ? cuda_device : 0;
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, ordinal) != cudaSuccess) {
            std::fprintf(stderr, "[nvml] cudaGetDeviceProperties(%d) failed\n", ordinal);
            return;
        }
        char pci_bus_id[32];
        if (cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), ordinal) != cudaSuccess) {
            std::fprintf(stderr, "[nvml] cudaDeviceGetPCIBusId(%d) failed\n", ordinal);
            return;
        }
        int rc = 1;
        if (handle_by_pci_fn != nullptr) {
            rc = handle_by_pci_fn(pci_bus_id, &device);
        } else {
            // No PCI lookup symbol: fall back to assuming NVML index == CUDA
            // ordinal under CUDA_DEVICE_ORDER=PCI_BUS_ID (the deploy convention).
            rc = handle_by_idx_fn(static_cast<unsigned int>(ordinal), &device);
        }
        if (rc != 0) {
            std::fprintf(stderr, "[nvml] no device handle for PCI %s (rc=%d)\n", pci_bus_id,
                         rc);
            return;
        }
        initialized = true;
    }
    ~GpuPowerSampler() = default;
    GpuPowerSampler(const GpuPowerSampler&) = delete;
    GpuPowerSampler& operator=(const GpuPowerSampler&) = delete;

    void sample() {
        if (!initialized) { return; }
        unsigned int mw = 0;
        if (query_power(device, &mw) == 0) { last_milliwatts = mw; }
    }
    unsigned int last_milliwatts = 0;
};

} // namespace ninfer::serve

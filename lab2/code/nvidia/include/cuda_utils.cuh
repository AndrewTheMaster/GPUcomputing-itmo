#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace lab {
inline void cuda_check(cudaError_t status, const char* expr, const char* file, int line) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + " " + expr + ": " + cudaGetErrorString(status));
}
#define CUDA_CHECK(expr) ::lab::cuda_check((expr), #expr, __FILE__, __LINE__)

template<class T> class DeviceBuffer {
    T* ptr_ = nullptr;
    std::size_t n_ = 0;
public:
    explicit DeviceBuffer(std::size_t n) : n_(n) {
        if (n) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr_) (void)cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return n_; }
    void upload(const T* p) {
        if (n_) CUDA_CHECK(cudaMemcpy(ptr_, p, n_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* p) const {
        if (n_) CUDA_CHECK(cudaMemcpy(p, ptr_, n_ * sizeof(T), cudaMemcpyDeviceToHost));
    }
};
class Event {
    cudaEvent_t event_{};
public:
    Event() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~Event() { (void)cudaEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t get() const { return event_; }
};
inline cudaDeviceProp device_properties() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    return prop;
}
inline void check_block(int b) {
    const auto p = device_properties();
    if (b != 32 && b != 64 && b != 128 && b != 256 && b != 512)
        throw std::invalid_argument("threads must be 32, 64, 128, 256 or 512");
    if (b > p.maxThreadsPerBlock) throw std::invalid_argument("Block exceeds GPU limit");
}
} // namespace lab

#include <cuda_runtime.h>
#include <iostream>
int main() {
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        std::cerr << "CUDA GPU not available: " << cudaGetErrorString(status) << '\n'; return 1;
    }
    if (count == 0) { std::cerr << "No CUDA GPU found\n"; return 1; }
    for (int i=0; i<count; ++i) {
        cudaDeviceProp p{};
        const auto s = cudaGetDeviceProperties(&p, i);
        if (s != cudaSuccess) { std::cerr << cudaGetErrorString(s) << '\n'; return 1; }
        std::cout << "GPU " << i << ": " << p.name << '\n'
                  << "compute capability: " << p.major << '.' << p.minor << '\n'
                  << "CMAKE_CUDA_ARCHITECTURES=" << p.major << p.minor << '\n'
                  << "memory MiB: " << p.totalGlobalMem / (1024*1024) << '\n'
                  << "max threads/block: " << p.maxThreadsPerBlock << '\n';
    }
}

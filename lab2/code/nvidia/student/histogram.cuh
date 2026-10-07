#pragma once
#include "cuda_utils.cuh"
#include <algorithm>
#include <cstddef>

namespace student {

// Bin index of one value. Must match the CPU reference exactly:
//     bin = abs(static_cast<int>(10.0f * x)) % bins
// The truncation toward zero of static_cast<int> is important for negative x.
__device__ __forceinline__ int histogram_bin(float x, int bins) {
    const int t = static_cast<int>(10.0f * x);
    return (t < 0 ? -t : t) % bins;
}

// Optimized variant: every block builds a private histogram in shared memory,
// then merges it into the global histogram with atomicAdd. Shared-memory
// atomics stay inside one block, so only `bins` global atomics per block occur
// instead of one global atomic per input element.
__global__ void histogram_shared_kernel(const float* __restrict__ input,
                                        int* __restrict__ output,
                                        std::size_t n, int bins) {
    extern __shared__ int block_hist[];
    // Cooperative zeroing; the strided loop supports bins > blockDim.x.
    for (int b = threadIdx.x; b < bins; b += blockDim.x) block_hist[b] = 0;
    __syncthreads();

    // Grid-stride loop: correct for any n, including the partial last block.
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        atomicAdd(&block_hist[histogram_bin(input[i], bins)], 1);
    }
    __syncthreads();

    // Merge the block-local counters into the global result.
    for (int b = threadIdx.x; b < bins; b += blockDim.x)
        atomicAdd(&output[b], block_hist[b]);
}

// Base variant: one atomicAdd into the global histogram per element. Used when
// the whole histogram does not fit into the shared memory of a block.
__global__ void histogram_global_kernel(const float* __restrict__ input,
                                        int* __restrict__ output,
                                        std::size_t n, int bins) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        atomicAdd(&output[histogram_bin(input[i], bins)], 1);
    }
}

// output: bins integer counters. Bin = abs(int(10 * x)) % bins. Initialize all counters.
// All pointers refer to GPU memory. Inputs are read-only; output is NOT initialized.
inline void histogram(const float* input, int* output, std::size_t n, int bins, int threads) {
    if (bins <= 0) return;

    // The framework does not initialize the output: clear every counter first.
    CUDA_CHECK(cudaMemsetAsync(output, 0, static_cast<std::size_t>(bins) * sizeof(int), 0));
    if (n == 0) return;  // nothing to count; the cleared output is the answer

    const int block = threads;
    const std::size_t total_blocks = (n + static_cast<std::size_t>(block) - 1) / block;
    const std::size_t shmem = static_cast<std::size_t>(bins) * sizeof(int);
    const cudaDeviceProp prop = lab::device_properties();

    if (shmem <= prop.sharedMemPerBlock) {
        // Saturating grid: enough resident blocks to fill the device, but no
        // more. This keeps the number of global atomics small.
        int active = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, histogram_shared_kernel, block, shmem));
        if (active < 1) active = 1;
        std::size_t grid = static_cast<std::size_t>(prop.multiProcessorCount) * active;
        grid = std::min(grid, total_blocks);
        if (grid == 0) grid = 1;
        histogram_shared_kernel<<<static_cast<unsigned int>(grid), block, shmem, 0>>>(
            input, output, n, bins);
    } else {
        // Large histogram: fall back to the direct global-atomic variant.
        histogram_global_kernel<<<static_cast<unsigned int>(total_blocks), block, 0, 0>>>(
            input, output, n, bins);
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace student

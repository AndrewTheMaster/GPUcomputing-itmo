#pragma once
#include "cpu_reference.hpp"
#include "verify.hpp"
#include "cuda_utils.cuh"
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/merge.h>
#include <thrust/transform.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

#if LAB_ALGORITHM_ID == 0
#include "reduce.cuh"
#elif LAB_ALGORITHM_ID == 1
#include "prefix_sum.cuh"
#elif LAB_ALGORITHM_ID == 2
#include "segmented_scan.cuh"
#elif LAB_ALGORITHM_ID == 3
#include "group_by.cuh"
#elif LAB_ALGORITHM_ID == 4
#include "sort.cuh"
#elif LAB_ALGORITHM_ID == 5
#include "histogram.cuh"
#else
#error Invalid LAB_ALGORITHM_ID
#endif

namespace lab {
struct BinIndex {
    int bins;
    __host__ __device__ int operator()(float v) const {
        const int t = static_cast<int>(10.0f * v);
        return (t < 0 ? -t : t) % bins;
    }
};
class GpuRunner {
    const Input& x_;
    DeviceBuffer<float> a_, b_, out_;
    DeviceBuffer<int> ka_, kb_, keys_, counts_;
    float reference_scalar_ = 0;
public:
    explicit GpuRunner(const Input& x) : x_(x), a_(x.a.size()), b_(x.b.size()),
        out_(algorithm == Algorithm::histogram ? 0 : algorithm == Algorithm::reduce ? 1 : elements(x)),
        ka_(x.ka.size()), kb_(x.kb.size()),
        keys_(algorithm == Algorithm::sort || algorithm == Algorithm::group_by ? elements(x) : 0),
        counts_(algorithm == Algorithm::histogram ? x.bins : 0) {}
    void upload() {
        a_.upload(x_.a.data()); b_.upload(x_.b.data());
        ka_.upload(x_.ka.data()); kb_.upload(x_.kb.data());
        // Poison outputs. Student code must overwrite every result, not accumulate across calls.
        if (out_.size()) CUDA_CHECK(cudaMemset(out_.data(), 0xff, out_.size() * sizeof(float)));
        if (keys_.size()) CUDA_CHECK(cudaMemset(keys_.data(), 0xff, keys_.size() * sizeof(int)));
        if (counts_.size()) CUDA_CHECK(cudaMemset(counts_.data(), 0xff, counts_.size() * sizeof(int)));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    void run_student(int threads) {
#if LAB_ALGORITHM_ID == 0
        student::reduce(a_.data(), out_.data(), x_.a.size(), threads);
#elif LAB_ALGORITHM_ID == 1
        student::exclusive_scan(a_.data(), out_.data(), x_.a.size(), threads);
#elif LAB_ALGORITHM_ID == 2
        student::segmented_scan(ka_.data(), a_.data(), out_.data(), x_.a.size(), threads);
#elif LAB_ALGORITHM_ID == 3
        student::merge_by_key(ka_.data(), a_.data(), x_.a.size(), kb_.data(), b_.data(), x_.b.size(),
                             keys_.data(), out_.data(), threads);
#elif LAB_ALGORITHM_ID == 4
        student::sort_by_key(ka_.data(), a_.data(), keys_.data(), out_.data(), x_.a.size(), threads);
#elif LAB_ALGORITHM_ID == 5
        student::histogram(a_.data(), counts_.data(), x_.a.size(), x_.bins, threads);
#endif
        CUDA_CHECK(cudaGetLastError());
    }
    void run_thrust() {
        auto a = thrust::device_pointer_cast(a_.data());
        auto b = thrust::device_pointer_cast(b_.data());
        auto ka = thrust::device_pointer_cast(ka_.data());
        auto kb = thrust::device_pointer_cast(kb_.data());
        auto out = thrust::device_pointer_cast(out_.data());
        auto keys = thrust::device_pointer_cast(keys_.data());
        auto counts = thrust::device_pointer_cast(counts_.data());
        const auto n = x_.a.size(), nb = x_.b.size();
        // Avoid pointer arithmetic on null buffers in zero-length inputs.
        if (algorithm == Algorithm::histogram) {
            thrust::fill(thrust::device, counts, counts + x_.bins, 0);
            if (n) {
                thrust::device_vector<int> indices(n), cumulative(x_.bins);
                thrust::transform(thrust::device, a, a+n, indices.begin(), BinIndex{x_.bins});
                thrust::sort(thrust::device, indices.begin(), indices.end());
                auto query = thrust::make_counting_iterator<int>(0);
                thrust::upper_bound(thrust::device, indices.begin(), indices.end(), query,
                                    query + x_.bins, cumulative.begin());
                thrust::adjacent_difference(thrust::device, cumulative.begin(), cumulative.end(), counts);
            }
        } else if (algorithm == Algorithm::reduce) {
            reference_scalar_ = n ? thrust::reduce(thrust::device, a, a+n, 0.0f) : 0.0f;
        } else if (algorithm == Algorithm::group_by) {
            if (!n && nb) {
                thrust::copy(thrust::device, kb, kb+nb, keys);
                thrust::copy(thrust::device, b, b+nb, out);
            } else if (n && !nb) {
                thrust::copy(thrust::device, ka, ka+n, keys);
                thrust::copy(thrust::device, a, a+n, out);
            } else if (n && nb) {
                thrust::merge_by_key(thrust::device, ka, ka+n, kb, kb+nb, a, b, keys, out);
            }
        } else if (n) {
            if (algorithm == Algorithm::prefix_sum)
                thrust::exclusive_scan(thrust::device, a, a+n, out, 0.0f);
            else if (algorithm == Algorithm::segmented_scan)
                thrust::exclusive_scan_by_key(thrust::device, ka, ka+n, a, out, 0.0f);
            else if (algorithm == Algorithm::sort) {
                thrust::copy(thrust::device, ka, ka+n, keys);
                thrust::copy(thrust::device, a, a+n, out);
                thrust::sort_by_key(thrust::device, keys, keys+n, out);
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }
    void download(Output& y, bool is_student) const {
        if (algorithm == Algorithm::reduce && !is_student) y.values[0] = reference_scalar_;
        else out_.download(y.values.data());
        keys_.download(y.keys.data()); counts_.download(y.counts.data());
    }
    void require_inputs_unchanged() const {
        Input copy = x_;
        a_.download(copy.a.data()); b_.download(copy.b.data());
        ka_.download(copy.ka.data()); kb_.download(copy.kb.data());
        if (copy.a != x_.a || copy.b != x_.b || copy.ka != x_.ka || copy.kb != x_.kb)
            throw std::runtime_error("Input modified: student interfaces require read-only inputs");
    }
};
} // namespace lab

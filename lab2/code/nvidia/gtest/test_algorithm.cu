#include "gpu_runner.cuh"
#include <gtest/gtest.h>
#include <string>
using namespace lab;

namespace {
Input example() {
    Input x;
    switch (algorithm) {
    case Algorithm::reduce: case Algorithm::prefix_sum: x.a = {3,1,4,2}; break;
    case Algorithm::segmented_scan: x.ka = {0,0,0,1,1,2,2}; x.a = {2,4,3,10,20,5,7}; break;
    case Algorithm::group_by: x.ka = {1,3,5}; x.a = {0,1,2}; x.kb = {2,3,6}; x.b = {3,4,5}; break;
    case Algorithm::sort: x.ka = {2,0,3,1}; x.a = {0,1,2,3}; break;
    case Algorithm::histogram: x.a = {0,0.2f,0.1f,0.2f}; x.bins = 4; break;
    }
    return x;
}
Output expected_example() {
    Output y;
    switch (algorithm) {
    case Algorithm::reduce: y.values = {10}; break;
    case Algorithm::prefix_sum: y.values = {0,3,4,8}; break;
    case Algorithm::segmented_scan: y.values = {0,2,6,0,10,0,5}; break;
    case Algorithm::group_by: y.keys = {1,2,3,3,5,6}; y.values = {0,3,1,4,2,5}; break;
    case Algorithm::sort: y.keys = {0,1,2,3}; y.values = {1,3,0,2}; break;
    case Algorithm::histogram: y.counts = {1,1,2,0}; break;
    }
    return y;
}
std::string check_gpu(const Input& x, int threads, bool student) {
    try {
        check_block(threads);
        auto ref = allocate_output(algorithm, x), y = allocate_output(algorithm, x);
        run_cpu(algorithm, x, ref);
        GpuRunner run(x);
        // Repeat with re-used buffers: reject dependence on a previous output.
        for (int repeat = 0; repeat < 2; ++repeat) {
            run.upload();
            if (student) run.run_student(threads); else run.run_thrust();
            CUDA_CHECK(cudaDeviceSynchronize());
            run.download(y, student);
            const auto c = verify(algorithm, x, ref, y);
            if (!c.ok) return c.message;
            run.require_inputs_unchanged();
        }
        return {};
    } catch (const std::exception& e) { return e.what(); }
}
std::vector<std::string> patterns() {
    if (algorithm == Algorithm::histogram) return {"one-bin"};
    if (algorithm == Algorithm::sort) return {"sorted", "reverse"};
    if (algorithm == Algorithm::group_by) return {"ties"};
    if (algorithm == Algorithm::segmented_scan) return {"one-segment", "singletons", "zeros", "ones"};
    return {"zeros", "ones"};
}
}

TEST(CPU, KnownExample) {
    const auto x = example();
    auto y = allocate_output(algorithm, x); run_cpu(algorithm, x, y);
    const auto expected = expected_example();
    EXPECT_EQ(y.keys, expected.keys); EXPECT_EQ(y.values, expected.values); EXPECT_EQ(y.counts, expected.counts);
}
TEST(CPU, RejectsCorruptedResult) {
    const auto x = example(); auto y = expected_example();
    if (!y.counts.empty()) ++y.counts[0];
    else if (!y.keys.empty()) y.keys[0] += 100;
    else y.values[0] += 100;
    EXPECT_FALSE(verify(algorithm, x, expected_example(), y).ok);
}
TEST(Reference, MatchesCPU) {
    ASSERT_EQ(check_gpu(example(), 128, false), "");
    for (std::size_t n : {0u, 1u, 31u, 32u, 33u, 255u, 256u, 257u, 1003u, 65537u}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        ASSERT_EQ(check_gpu(make_input(algorithm, n), 128, false), "");
    }
    for (const auto& p : patterns()) {
        SCOPED_TRACE(p);
        ASSERT_EQ(check_gpu(make_input(algorithm, 1003, 42, 256, p), 128, false), "");
    }
    if (algorithm == Algorithm::group_by) {
        ASSERT_EQ(check_gpu(make_input(algorithm, 0, 42, 256, "random", 37), 128, false), "");
        ASSERT_EQ(check_gpu(make_input(algorithm, 37, 42, 256, "random", 0), 128, false), "");
        ASSERT_EQ(check_gpu(make_input(algorithm, 17, 42, 256, "random", 253), 128, false), "");
    }
    if (algorithm == Algorithm::segmented_scan) {
        Input x; x.ka={0,0,1,0,0}; x.a={2,4,3,10,20};
        ASSERT_EQ(check_gpu(x, 128, false), "");
    }
    if (algorithm == Algorithm::histogram)
        for (int bins : {1, 16, 257}) ASSERT_EQ(check_gpu(make_input(algorithm, 1003, 42, bins), 128, false), "");
}
TEST(Student, KnownExample) {
    ASSERT_EQ(check_gpu(example(), 128, true), "");
}
TEST(Student, BoundariesAndBlocks) {
    for (int b : {32, 64, 128, 256, 512}) {
        for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t(b-1), std::size_t(b), std::size_t(b+1), std::size_t{1003}}) {
            SCOPED_TRACE("n=" + std::to_string(n) + ", B=" + std::to_string(b));
            ASSERT_EQ(check_gpu(make_input(algorithm, n), b, true), "");
        }
    }
}
TEST(Student, PatternsAndUnequalInputs) {
    for (const auto& p : patterns()) {
        SCOPED_TRACE(p);
        ASSERT_EQ(check_gpu(make_input(algorithm, 1003, 42, 256, p), 128, true), "");
    }
    if (algorithm == Algorithm::group_by) {
        ASSERT_EQ(check_gpu(make_input(algorithm, 0, 42, 256, "random", 37), 128, true), "");
        ASSERT_EQ(check_gpu(make_input(algorithm, 37, 42, 256, "random", 0), 128, true), "");
        ASSERT_EQ(check_gpu(make_input(algorithm, 17, 42, 256, "random", 253), 128, true), "");
    }
    if (algorithm == Algorithm::segmented_scan) {
        Input x; x.ka={0,0,1,0,0}; x.a={2,4,3,10,20};
        ASSERT_EQ(check_gpu(x, 128, true), "");
    }
    if (algorithm == Algorithm::histogram)
        for (int bins : {1, 16, 257}) ASSERT_EQ(check_gpu(make_input(algorithm, 1003, 42, bins), 128, true), "");
}
TEST(Student, LargeInput) {
    for (int b : {32, 64, 128, 256, 512}) {
        SCOPED_TRACE("B=" + std::to_string(b));
        ASSERT_EQ(check_gpu(make_input(algorithm, 65537, 77), b, true), "");
    }
    ASSERT_EQ(check_gpu(make_input(algorithm, 1u << 20, 123), 256, true), "");
    if (algorithm == Algorithm::segmented_scan)
        ASSERT_EQ(check_gpu(make_input(algorithm, 65537, 123, 256, "one-segment"), 256, true), "");
}

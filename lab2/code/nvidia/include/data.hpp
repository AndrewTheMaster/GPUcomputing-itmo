#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace lab {
enum class Algorithm { reduce, prefix_sum, segmented_scan, group_by, sort, histogram };
#ifndef LAB_ALGORITHM_ID
#define LAB_ALGORITHM_ID 0
#endif
constexpr Algorithm algorithm = static_cast<Algorithm>(LAB_ALGORITHM_ID);
inline const char* name(Algorithm a) {
    switch (a) {
    case Algorithm::reduce: return "reduce";
    case Algorithm::prefix_sum: return "prefix_sum";
    case Algorithm::segmented_scan: return "segmented_scan";
    case Algorithm::group_by: return "group_by";
    case Algorithm::sort: return "sort";
    case Algorithm::histogram: return "histogram";
    }
    throw std::invalid_argument("Unknown algorithm");
}
struct Input {
    std::vector<float> a, b;
    std::vector<int> ka, kb;
    int bins = 256;
};
struct Output {
    std::vector<float> values;
    std::vector<int> keys, counts;
};
inline std::size_t elements(const Input& x) { return x.a.size() + x.b.size(); }
inline Output allocate_output(Algorithm a, const Input& x) {
    Output y;
    if (a == Algorithm::histogram) y.counts.resize(x.bins);
    else {
        const auto n = a == Algorithm::reduce ? 1 : elements(x);
        y.values.resize(n);
        if (a == Algorithm::group_by || a == Algorithm::sort) y.keys.resize(n);
    }
    return y;
}
// All generated values are finite, bounded, and exactly representable as float.
inline Input make_input(Algorithm alg, std::size_t n, std::uint32_t seed = 42,
                        int bins = 256, const std::string& pattern = "random",
                        std::size_t second_size = static_cast<std::size_t>(-1)) {
    const bool pairs = alg == Algorithm::group_by;
    const std::size_t nb = pairs ? (second_size == static_cast<std::size_t>(-1) ? n : second_size) : 0;
    constexpr std::size_t limit = 1u << 24;
    if (n > limit || nb > limit - n || bins < 1 || bins > 65536)
        throw std::invalid_argument("Require total elements <= 2^24 and 1 <= bins <= 65536");
    const bool numeric = alg == Algorithm::reduce || alg == Algorithm::prefix_sum || alg == Algorithm::segmented_scan;
    const bool ok = pattern == "random" ||
        (numeric && (pattern == "zeros" || pattern == "ones")) ||
        (alg == Algorithm::segmented_scan && (pattern == "one-segment" || pattern == "singletons")) ||
        (alg == Algorithm::sort && (pattern == "sorted" || pattern == "reverse")) ||
        (alg == Algorithm::group_by && pattern == "ties") ||
        (alg == Algorithm::histogram && pattern == "one-bin");
    if (!ok) throw std::invalid_argument("Unsupported pattern for this algorithm");
    Input x; x.bins = bins; x.a.resize(n); x.b.resize(nb);
    std::mt19937 rng(seed);
    auto sample = [&]() -> float {
        // Direct use of mt19937 avoids implementation-dependent distributions.
        const int r = static_cast<int>(rng() % 17) - 8;
        return r * 0.125f;
    };
    for (auto& v : x.a) v = pattern == "ones" ? 1.0f : pattern == "zeros" ? 0.0f : sample();
    for (auto& v : x.b) v = sample();
    if (alg == Algorithm::segmented_scan) {
        x.ka.resize(n); int key = 0; std::size_t remaining = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (pattern == "one-segment") x.ka[i] = 0;
            else if (pattern == "singletons") x.ka[i] = static_cast<int>(i);
            else {
                if (remaining == 0) { ++key; remaining = 1 + rng() % 701; }
                x.ka[i] = key; --remaining;
            }
        }
    }
    if (alg == Algorithm::sort || pairs) {
        x.ka.resize(n); x.kb.resize(nb);
        for (auto& k : x.ka) k = pattern == "ties" ? 7 : static_cast<int>(rng() % 4096) - 2048;
        for (auto& k : x.kb) k = pattern == "ties" ? 7 : static_cast<int>(rng() % 4096) - 2048;
        if (pairs || pattern == "sorted" || pattern == "reverse") std::sort(x.ka.begin(), x.ka.end());
        if (pattern == "reverse") std::reverse(x.ka.begin(), x.ka.end());
        if (pairs) std::sort(x.kb.begin(), x.kb.end());
        // Payloads identify original positions, so pair preservation can be checked.
        for (std::size_t i = 0; i < n; ++i) x.a[i] = static_cast<float>(i);
        for (std::size_t i = 0; i < nb; ++i) x.b[i] = static_cast<float>(n + i);
    }
    if (alg == Algorithm::histogram)
        for (auto& v : x.a)
            v = pattern == "one-bin" ? 0.0f : (static_cast<int>(rng() % 8193) - 4096) * 0.125f;
    return x;
}
} // namespace lab

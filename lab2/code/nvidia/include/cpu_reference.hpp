#pragma once
#include "data.hpp"
#include <utility>

namespace lab {
// Sequential reference; output storage is allocated before this function.
inline void run_cpu(Algorithm alg, const Input& x, Output& y) {
    switch (alg) {
    case Algorithm::reduce:
        y.values[0] = std::accumulate(x.a.begin(), x.a.end(), 0.0f); break;
    case Algorithm::prefix_sum:
        std::exclusive_scan(x.a.begin(), x.a.end(), y.values.begin(), 0.0f); break;
    case Algorithm::segmented_scan: {
        float acc = 0.0f;
        for (std::size_t i = 0; i < x.a.size(); ++i) {
            if (i == 0 || x.ka[i] != x.ka[i-1]) acc = 0.0f;
            y.values[i] = acc; acc += x.a[i];
        }
        break;
    }
    case Algorithm::group_by: {
        std::size_t a = 0, b = 0, o = 0;
        while (a < x.a.size() || b < x.b.size()) {
            // Stable merge: take A first for equal keys.
            if (b == x.b.size() || (a < x.a.size() && x.ka[a] <= x.kb[b])) {
                y.keys[o] = x.ka[a]; y.values[o++] = x.a[a++];
            } else { y.keys[o] = x.kb[b]; y.values[o++] = x.b[b++]; }
        }
        break;
    }
    case Algorithm::sort: {
        std::vector<std::pair<int, float>> pairs(x.a.size());
        for (std::size_t i = 0; i < pairs.size(); ++i) pairs[i] = {x.ka[i], x.a[i]};
        std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (std::size_t i = 0; i < pairs.size(); ++i) {
            y.keys[i] = pairs[i].first; y.values[i] = pairs[i].second;
        }
        break;
    }
    case Algorithm::histogram:
        std::fill(y.counts.begin(), y.counts.end(), 0);
        for (float v : x.a) ++y.counts[std::abs(static_cast<int>(10.0f * v)) % x.bins];
        break;
    }
}
} // namespace lab

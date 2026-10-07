#pragma once
#include "data.hpp"
#include <sstream>
#include <limits>

namespace lab {
struct Check { bool ok = true; double max_abs_error = 0; std::string message; };
inline Check verify(Algorithm alg, const Input& x, const Output& ref, const Output& got) {
    Check c;
    auto fail = [&](const std::string& msg) { c.ok = false; c.message = msg; return c; };
    if (ref.values.size() != got.values.size() || ref.keys.size() != got.keys.size() || ref.counts.size() != got.counts.size())
        return fail("Output size mismatch");
    if (alg == Algorithm::histogram) {
        if (ref.counts != got.counts) return fail("Histogram counters differ");
        if (std::accumulate(got.counts.begin(), got.counts.end(), std::int64_t{0}) != static_cast<std::int64_t>(x.a.size()))
            return fail("Histogram total differs from input size");
        return c;
    }
    if (alg == Algorithm::sort) {
        if (!std::is_sorted(got.keys.begin(), got.keys.end())) return fail("Keys are not sorted");
        // Equal keys may be permuted: require an exact permutation of input pairs.
        std::vector<unsigned char> seen(x.a.size(), 0);
        for (std::size_t i = 0; i < got.values.size(); ++i) {
            const float v = got.values[i];
            if (!std::isfinite(v) || v < 0 || static_cast<double>(v) >= x.a.size() || std::floor(v) != v)
                return fail("Invalid sort payload");
            const auto id = static_cast<std::size_t>(v);
            if (seen[id] || x.ka[id] != got.keys[i]) return fail("Lost, repeated, or mismatched key/value pair");
            seen[id] = 1;
        }
        return c;
    }
    if (got.keys != ref.keys) return fail("Keys differ from reference");
    for (std::size_t i = 0; i < got.values.size(); ++i) {
        const double r = ref.values[i], v = got.values[i];
        if (!std::isfinite(v)) return fail("Non-finite output at index " + std::to_string(i));
        const double e = std::abs(v - r);
        c.max_abs_error = std::max(c.max_abs_error, e);
        const double limit = alg == Algorithm::group_by ? 0.0 : 1e-4 + 1e-5 * std::abs(r);
        if (e > limit) return fail("Value differs at index " + std::to_string(i));
    }
    return c;
}
} // namespace lab

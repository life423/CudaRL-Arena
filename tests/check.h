#ifndef CUDARL_TESTS_CHECK_H
#define CUDARL_TESTS_CHECK_H

// Minimal self-registering test harness. TEST_CASE(name) defines a test and
// GPU_TEST_CASE(name) one that needs a CUDA device (skipped without one);
// CHECK(...) records a failure and keeps going so one run reports everything.

#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace cudarl::test {

struct TestCase {
    const char*           name;
    std::function<void()> fn;
    bool                  needs_gpu;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

inline int& failures() {
    static int count = 0;
    return count;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn, bool needs_gpu) {
        registry().push_back({name, std::move(fn), needs_gpu});
    }
};

inline void fail(const char* file, int line, const std::string& msg) {
    ++failures();
    std::printf("    FAIL %s:%d: %s\n", file, line, msg.c_str());
}

// Largest |a - b| divided by the largest |b|: robust to individual values near
// zero, which is where per-element relative error is meaningless. Invalid
// comparisons return infinity so CHECK_CLOSE cannot silently accept them.
template <typename A, typename B>
double normalized_max_error(const A& a, const B& b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double max_diff = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < b.size(); ++i) {
        const double actual = static_cast<double>(a[i]), expected = static_cast<double>(b[i]);
        if (!std::isfinite(actual) || !std::isfinite(expected))
            return std::numeric_limits<double>::infinity();
        max_diff = std::fmax(max_diff, std::fabs(actual - expected));
        max_ref  = std::fmax(max_ref, std::fabs(expected));
    }
    return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}

} // namespace cudarl::test

#define CUDARL_CONCAT_INNER(a, b) a##b
#define CUDARL_CONCAT(a, b) CUDARL_CONCAT_INNER(a, b)

#define CUDARL_REGISTER_TEST(name, needs_gpu)                                        \
    static void CUDARL_CONCAT(test_fn_, __LINE__)();                                 \
    static ::cudarl::test::Registrar CUDARL_CONCAT(test_reg_, __LINE__)(             \
        name, CUDARL_CONCAT(test_fn_, __LINE__), needs_gpu);                         \
    static void CUDARL_CONCAT(test_fn_, __LINE__)()

#define TEST_CASE(name) CUDARL_REGISTER_TEST(name, false)
#define GPU_TEST_CASE(name) CUDARL_REGISTER_TEST(name, true)

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) ::cudarl::test::fail(__FILE__, __LINE__, #cond);                \
    } while (0)

#define CHECK_MSG(cond, msg)                                                         \
    do {                                                                             \
        if (!(cond)) ::cudarl::test::fail(__FILE__, __LINE__, (msg));                \
    } while (0)

// Normalized max error between two equal-length sequences must be <= tol.
#define CHECK_CLOSE(actual, expected, tol)                                           \
    do {                                                                             \
        const double cudarl_err_ = ::cudarl::test::normalized_max_error((actual), (expected)); \
        if (!(cudarl_err_ <= (tol))) {                                               \
            char cudarl_buf_[256];                                                   \
            std::snprintf(cudarl_buf_, sizeof cudarl_buf_, "%s vs %s: error %.3g > %.3g", \
                          #actual, #expected, cudarl_err_, static_cast<double>(tol)); \
            ::cudarl::test::fail(__FILE__, __LINE__, cudarl_buf_);                   \
        }                                                                            \
    } while (0)

#endif // CUDARL_TESTS_CHECK_H

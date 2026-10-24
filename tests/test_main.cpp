#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

#include <cuda_runtime.h>

#include "check.h"

namespace {

bool env_flag_set(const char* name) {
#ifdef _MSC_VER
    char*  value = nullptr;
    size_t len   = 0;
    const bool set = _dupenv_s(&value, &len, name) == 0 && value && std::strcmp(value, "1") == 0;
    std::free(value);
    return set;
#else
    const char* value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
#endif
}

bool cuda_device_present() {
    int count = 0;
    const bool ok = cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    cudaGetLastError();   // clear the error a missing driver leaves behind
    return ok;
}

} // namespace

// Runs every registered test, or only those whose name contains argv[1].
// GPU tests are skipped when no CUDA device is present, unless
// CUDARL_REQUIRE_GPU=1, which turns a missing GPU into a failure.
int main(int argc, char** argv) {
    using cudarl::test::failures;
    using cudarl::test::registry;

    const char* filter      = argc > 1 ? argv[1] : nullptr;
    const bool  require_gpu = env_flag_set("CUDARL_REQUIRE_GPU");
    const bool  have_gpu    = cuda_device_present();
    std::printf("CUDA device: %s%s\n\n", have_gpu ? "present" : "none",
                have_gpu || require_gpu ? "" : " (GPU tests will be skipped)");

    int ran = 0, failed_tests = 0, skipped = 0;
    for (const auto& t : registry()) {
        if (filter && !std::strstr(t.name, filter)) continue;
        if (t.needs_gpu && !have_gpu) {
            if (require_gpu) {
                ++ran;
                ++failed_tests;
                std::printf("FAIL  %-56s (no CUDA device and CUDARL_REQUIRE_GPU=1)\n", t.name);
            } else {
                ++skipped;
                std::printf("SKIP  %-56s (no CUDA device)\n", t.name);
            }
            continue;
        }
        ++ran;
        const int before = failures();
        const auto start = std::chrono::steady_clock::now();
        try {
            t.fn();
        } catch (const std::exception& e) {
            cudarl::test::fail(__FILE__, __LINE__, std::string("uncaught exception: ") + e.what());
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const bool ok = failures() == before;
        if (!ok) ++failed_tests;
        std::printf("%s  %-56s %8.1f ms\n", ok ? "PASS" : "FAIL", t.name, ms);
    }
    std::printf("\n%d/%d tests passed, %d skipped\n", ran - failed_tests, ran, skipped);
    return failed_tests == 0 && ran > 0 ? 0 : 1;
}

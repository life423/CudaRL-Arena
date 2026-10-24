#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>

#include "check.h"

// Runs every registered test, or only those whose name contains argv[1].
int main(int argc, char** argv) {
    using cudarl::test::failures;
    using cudarl::test::registry;

    const char* filter = argc > 1 ? argv[1] : nullptr;
    int ran = 0, failed_tests = 0;
    for (const auto& t : registry()) {
        if (filter && !std::strstr(t.name, filter)) continue;
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
    std::printf("\n%d/%d tests passed\n", ran - failed_tests, ran);
    return failed_tests == 0 && ran > 0 ? 0 : 1;
}

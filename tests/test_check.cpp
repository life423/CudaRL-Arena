#include <limits>
#include <vector>

#include "check.h"

using cudarl::test::normalized_max_error;

TEST_CASE("check: normalized error preserves finite and empty comparisons") {
    CHECK(normalized_max_error(std::vector<float>{}, std::vector<double>{}) == 0);
    CHECK(normalized_max_error(std::vector<float>{1, -2}, std::vector<double>{1, -2}) == 0);
    CHECK(normalized_max_error(std::vector<float>{2, -4}, std::vector<double>{1, -2}) == 1);
    CHECK(normalized_max_error(std::vector<float>{0, 2}, std::vector<double>{0, 0}) == 2);
}

TEST_CASE("check: normalized error rejects nonfinite actual and reference values") {
    const std::vector<double> finite{1, -2};
    for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()}) {
        const std::vector<double> damaged{1, invalid};
        CHECK(normalized_max_error(damaged, finite) > 1);
        CHECK(normalized_max_error(finite, damaged) > 1);
        CHECK(normalized_max_error(damaged, damaged) > 1);
    }
}

TEST_CASE("check: normalized error rejects either unequal length before indexing") {
    const std::vector<float> empty, shorter{1}, longer{1, 2};
    CHECK(normalized_max_error(shorter, longer) > 1);
    CHECK(normalized_max_error(longer, shorter) > 1);
    CHECK(normalized_max_error(empty, shorter) > 1);
    CHECK(normalized_max_error(shorter, empty) > 1);
}

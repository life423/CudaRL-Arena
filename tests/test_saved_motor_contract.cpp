#include "check.h"
#include "core/checkpoint.h"
#include "motor/saved_motor_contract.h"
#include "rl/learned_amplitude_math.h"
#include <cmath>
#include <filesystem>
#include <string>

TEST_CASE("saved_motor: immutable provenance matches original amplitude package without simulation") {
    namespace cp=cudarl::checkpoint;
    const auto root=std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto bytes=cp::read((root/"live/research_data/learned-amplitude-v1.bin").string());
    cp::Reader reader{bytes};
    CHECK(reader.value<uint32_t>()==0x31564c48);
    CHECK(reader.value<uint32_t>()==cudarl::motor::saved_motor_package_version);
    for(uint32_t expected:{6u,6u,3u,4u,41u,4u,336u,1u,7u,1u})CHECK(reader.value<uint32_t>()==expected);
    CHECK(reader.value<uint64_t>()!=0);
    const auto parent=reader.array<uint8_t>(64);
    CHECK(std::string(parent.begin(),parent.end())=="810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0");
    const auto& hashes=cudarl::motor::saved_motor_source_contract_hashes();
    CHECK(reader.value<uint32_t>()==hashes.size());
    for(auto expected:hashes){const auto stored=reader.array<uint8_t>(64);CHECK(std::string(stored.begin(),stored.end())==expected);}
    const auto weights=reader.array<float>(2532);
    CHECK(weights.size()==2532);
    for(float weight:weights)CHECK(std::isfinite(weight));
    const auto amplitude=reader.array<float>(2);
    CHECK(amplitude.size()==2);
    const auto gain=reader.value<float>(),sigma=reader.value<float>();
    CHECK(std::abs(gain-cudarl::learned_amplitude::sigmoid(amplitude[0]))<=2e-7f);
    CHECK(std::abs(sigma-cudarl::learned_amplitude::standard_deviation(amplitude[1]))<=2e-6f);
    reader.finish();
    static_assert(cudarl::motor::saved_motor_provenance_version==1,"Saved provenance decoder changed");
}

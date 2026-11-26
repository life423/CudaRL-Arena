#ifndef CUDARL_SAVED_MOTOR_CONTRACT_H
#define CUDARL_SAVED_MOTOR_CONTRACT_H
#include <array>
#include <cstdint>
#include <string_view>

namespace cudarl::motor {
// Package provenance, not a claim that the current runtime is the old runtime.
// These immutable hashes were read from the preserved, qualified amplitude
// package and checked against its original source snapshot before retirement.
inline constexpr uint32_t saved_motor_provenance_version=1;
inline constexpr uint32_t saved_motor_package_version=2;
inline constexpr std::string_view saved_motor_reference_runtime="model6-appended41-amplitude-v2";
const std::array<std::string_view,12>& saved_motor_source_contract_hashes();
}
#endif

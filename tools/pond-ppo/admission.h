#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// Host-only file/receipt validation. No CUDA context, world integration, neural
// inference or optimizer is constructed here. Keep this interface stable so a
// receipt/source binding correction does not rebuild the physical CUDA kernel.
namespace cudarl::rl::pond_ppo_host {
std::vector<uint8_t> whole_file(const std::filesystem::path& path);
std::string text_file(const std::filesystem::path& path);
std::string sha256(const std::vector<uint8_t>& bytes);
std::string scalar(const std::string& text,const std::string& wanted);
const char* source_sha256();
const char* physics_source_sha256();
uint64_t source_identity();
void validate_gate_documents(const std::string& cpu,const std::string& gpu,
                             uint64_t parent_checksum,uint64_t cpu_gate_checksum);
void prerequisites(const std::filesystem::path& protocol,
                   const std::filesystem::path& cpu_gate,
                   const std::filesystem::path& cuda_gate,
                   uint64_t evaluator_source,uint64_t parent_checksum,uint32_t turn_curriculum=0);
}

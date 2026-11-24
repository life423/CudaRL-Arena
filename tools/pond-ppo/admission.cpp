#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include "admission.h"
#include "core/checkpoint.h"
#include "pond_ppo_identity.h"
#include <windows.h>
#include <bcrypt.h>
#include <array>
#include <cctype>
#include <climits>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cudarl::rl::pond_ppo_host {
namespace cp=cudarl::checkpoint;
namespace {
void require(bool yes,const std::string& reason){if(!yes)throw std::runtime_error(reason);}
}
std::vector<uint8_t> whole_file(const std::filesystem::path& path) {
    std::ifstream file(path,std::ios::binary);require(bool(file),"Missing input: "+path.string());
    return {(std::istreambuf_iterator<char>(file)),{}};
}
std::string text_file(const std::filesystem::path& path){const auto bytes=whole_file(path);return {bytes.begin(),bytes.end()};}
std::string sha256(const std::vector<uint8_t>& bytes) {
    require(bytes.size()<=ULONG_MAX,"SHA256 input too large");BCRYPT_ALG_HANDLE algorithm=nullptr;
    require(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0,"SHA256 provider failed");
    std::array<unsigned char,32> digest{};
    const auto status=BCryptHash(algorithm,nullptr,0,const_cast<PUCHAR>(bytes.data()),static_cast<ULONG>(bytes.size()),digest.data(),32);
    BCryptCloseAlgorithmProvider(algorithm,0);require(status>=0,"SHA256 failed");
    std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto byte:digest)out<<std::setw(2)<<unsigned(byte);return out.str();
}
const char* source_sha256(){return pond_ppo_source_sha256;}
const char* physics_source_sha256(){return pond_ppo_physics_sha256;}
uint64_t source_identity(){const std::string value=source_sha256();return cp::checksum({value.begin(),value.end()});}

// Only root scalars are admissible. A nested diagnostic's 'passed' does not
// override the actual root result. Compound values are skipped intact.
std::string scalar(const std::string& text,const std::string& wanted) {
    size_t i=0;const auto ws=[&](){while(i<text.size()&&std::isspace(static_cast<unsigned char>(text[i])))++i;};
    const auto quoted=[&](){require(i<text.size()&&text[i++]=='"',"Expected JSON string");std::string result;bool escape=false;
        while(i<text.size()){const char c=text[i++];if(escape){result+=c;escape=false;}else if(c=='\\')escape=true;else if(c=='"')return result;else result+=c;}
        throw std::runtime_error("Unclosed JSON string");};
    ws();require(i<text.size()&&text[i++]=='{',"Expected root JSON object");
    while(i<text.size()) {
        ws();if(i==text.size()||text[i]=='}')break;const auto key=quoted();ws();require(i<text.size()&&text[i++]==':',"Expected JSON colon");ws();
        std::string value;
        if(i<text.size()&&text[i]=='"')value=quoted();
        else {const size_t start=i;int depth=0;bool in_string=false,escape=false;
            while(i<text.size()){const char c=text[i];if(in_string){if(escape)escape=false;else if(c=='\\')escape=true;else if(c=='"')in_string=false;}
                else if(c=='"')in_string=true;else if(c=='{'||c=='[')++depth;else if(c=='}'||c==']'){if(depth==0)break;--depth;}
                else if(c==','&&depth==0)break;++i;}
            value=text.substr(start,i-start);while(!value.empty()&&std::isspace(static_cast<unsigned char>(value.back())))value.pop_back();}
        if(key==wanted)return value;ws();if(i<text.size()&&text[i]==','){++i;continue;}break;
    }
    throw std::runtime_error("Missing root JSON binding: "+wanted);
}
void validate_gate_documents(const std::string& cpu,const std::string& gpu,
                             uint64_t parent_checksum,uint64_t cpu_gate_checksum) {
    for(const auto* result:{&cpu,&gpu}) {
        require(scalar(*result,"passed")=="true"&&scalar(*result,"optimizer_steps")=="0","Physics prerequisite did not pass without optimization");
        require(scalar(*result,"schema")=="1"&&scalar(*result,"preflight_only")=="false","Completed v4 dynamics required; static preflight is not admission");
        require(scalar(*result,"runtime_version")=="4"&&scalar(*result,"query_arithmetic_version")=="2","Corrected v4 runtime/query arithmetic required");
    }
    require(scalar(cpu,"protocol_id")=="shoreline-product-v4-2026-11-23"&&scalar(gpu,"protocol_id")=="shoreline-product-v4-cuda-2026-11-23","Require corrected CPUv4 and CUDAv4 qualification; closed v3 diagnostic is not admission");
    require(scalar(cpu,"reason")=="product_physics_and_retention_passed_navigation_unqualified"&&scalar(cpu,"completed_cases")=="36"&&scalar(cpu,"failed_cases")=="0"&&scalar(cpu,"skipped_cases")=="0","CPU qualification cases incomplete");
    require(scalar(cpu,"retention_old_accepted_controls")=="30400"&&scalar(cpu,"retention_new_accepted_controls")=="30400"&&scalar(cpu,"bank_accepted_controls")=="1600"&&scalar(cpu,"half_step_accepted_controls")=="800","CPU qualification accounting incomplete");
    require(scalar(cpu,"package_checksum_fnv1a64")==std::to_string(parent_checksum),"CPU qualification tested another saved motor");
    require(scalar(gpu,"arithmetic_profile")=="cpu_compatible_no_implicit_fma_v1"&&scalar(gpu,"canonical_radius_version")=="1","CUDA physics profile differs");
    require(scalar(gpu,"reason")=="matched_fixed_input_dynamics_passed"&&scalar(gpu,"controls_completed_per_case")=="128"&&scalar(gpu,"CUDA_substep_readback_complete")=="true","CUDA qualification incomplete");
    require(scalar(gpu,"CPU_attempted_controls")=="1024"&&scalar(gpu,"CUDA_attempted_controls")=="1024"&&scalar(gpu,"CPU_attempted_substeps")=="10240"&&scalar(gpu,"CUDA_attempted_substeps")=="10240","CUDA prerequisite physical accounting differs");
    require(scalar(gpu,"saved_payload_checksum_fnv1a64")==std::to_string(parent_checksum)&&scalar(gpu,"cpu_gate_checksum_fnv1a64")==std::to_string(cpu_gate_checksum),"CUDA gate is not bound to this CPU gate and saved motor");
}
void prerequisites(const std::filesystem::path& protocol_path,
                   const std::filesystem::path& cpu_gate,
                   const std::filesystem::path& cuda_gate,
                   uint64_t evaluator_source,uint64_t parent_checksum) {
    require(!protocol_path.empty()&&!cpu_gate.empty()&&!cuda_gate.empty(),"Sealed protocol and CPU/CUDA physics results required");
    const auto protocol=text_file(protocol_path);
    require(scalar(protocol,"protocol_id")=="pond-motor-ppo-v2-2026-11-23","Wrong PPO protocol");
    require(scalar(protocol,"source_identity_fnv1a64")==std::to_string(source_identity()),"Trainer source identity changed");
    require(scalar(protocol,"qualified_physics_source_sha256")==physics_source_sha256(),"Qualified physics source binding differs from compiled model");
    require(scalar(protocol,"max_updates")=="32"&&scalar(protocol,"worlds")=="512"&&scalar(protocol,"rollout_controls")=="64","Protocol work allocation changed");
    require(scalar(protocol,"max_wall_seconds")=="1200"&&scalar(protocol,"max_evidence_bytes")=="4294967296","Protocol aggregate caps changed");
    require(evaluator_source!=0&&scalar(protocol,"retention_evaluator_source_fnv1a64")==std::to_string(evaluator_source),"Retention evaluator source differs from sealed protocol");
    for(const auto& entry:{std::make_pair(cpu_gate,"cpu_gate_sha256"),std::make_pair(cuda_gate,"cuda_gate_sha256")})
        require(sha256(whole_file(entry.first))==scalar(protocol,entry.second),"Physics gate binding differs");
    validate_gate_documents(text_file(cpu_gate),text_file(cuda_gate),parent_checksum,cp::checksum(whole_file(cpu_gate)));
}
}

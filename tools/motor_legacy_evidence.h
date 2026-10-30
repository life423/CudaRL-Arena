#ifndef CUDARL_TOOLS_MOTOR_LEGACY_EVIDENCE_H
#define CUDARL_TOOLS_MOTOR_LEGACY_EVIDENCE_H
#include "envs/swimmer/motor_types.h"
#include <array>

// Read-only POD layouts for protected historical evidence. No legacy motor,
// integrator, sampler, training loop, or model implementation lives here.
namespace cudarl::motor_legacy_evidence {
using State=motor::TaskState;
struct PairedState {
    State task{};float held_target[4]{},actual_torques[40]{},unclamped_torques[40]{};
    float mean_squared_torque=0,substep_max_target_error=0,substep_max_joint_rate=0;
    int first_unsafe_substep=-1,unsafe_substeps=0;State first_unsafe_task{};
};
enum class Kind:int {Move,Brake,Turn,Hold,Recover};
struct Fixture {
    State initial{};Kind kind=Kind::Move;uint32_t seed=0x4D072600u;
    int id=0,warmup_ticks=0,evaluation_ticks=600,pulse_start=-1,pulse_ticks=0;
    deep::V pulse_velocity{};float requested_speed=0,actual_entry_speed=0;bool needs_warmup=false,qualified=true;
};
inline deep::BodyState first_unsafe_body(const std::vector<uint8_t>& bytes) {
    checkpoint::Reader in{bytes};if(in.value<uint32_t>()!=0x504D4953u||in.value<uint32_t>()!=1) throw std::runtime_error("Expected protected historical unsafe-state record");
    const auto state=in.value<PairedState>();in.finish();if(state.first_unsafe_substep<0||state.first_unsafe_substep>10||!motor::finite(state.first_unsafe_task)) throw std::runtime_error("Invalid historical unsafe state");return state.first_unsafe_task.body;
}
struct RegulationGate {
    motor::Gains gains{};uint64_t source_fingerprint=0;bool passed=false;
    void require_passed(uint64_t source) const {if(!passed||!source||source_fingerprint!=source||!motor::same_gains(gains,motor::validated_gains())) throw std::runtime_error("Historical regulation source/gain proof differs");}
    static RegulationGate read(const std::vector<uint8_t>& bytes) {
        checkpoint::Reader in{bytes};if(in.value<uint32_t>()!=0x504D5247u||in.value<uint32_t>()!=1) throw std::runtime_error("Unsupported historical regulation gate");
        const auto protocol=in.array<uint8_t>(1000000);checkpoint::Reader p{protocol};if(p.value<uint32_t>()!=0x504D5250u||p.value<uint32_t>()!=1||p.value<uint32_t>()!=sizeof(PairedState)) throw std::runtime_error("Historical regulation POD layout differs");
        const auto source=p.value<uint64_t>(),fixtures=p.value<uint64_t>();checkpoint::Writer catalog;
        for(int i=0;i<8;++i) {const auto state=p.value<PairedState>();catalog.value(state);for(int j=0;j<4;++j) catalog.value(p.value<float>());catalog.value(p.value<int>());catalog.value(p.value<bool>());if(!motor::finite(state.task)) throw std::runtime_error("Nonfinite historical regulation state");}
        const auto protocol_gains=p.value<motor::Gains>();p.finish();RegulationGate out;out.gains=in.value<motor::Gains>();out.source_fingerprint=in.value<uint64_t>();const auto catalog_hash=in.value<uint64_t>(),protocol_hash=in.value<uint64_t>();
        const int cpu_steps=in.value<int>(),gpu_steps=in.value<int>(),cpu_cases=in.value<int>(),gpu_cases=in.value<int>(),cpu_passed=in.value<int>(),gpu_passed=in.value<int>();const bool completed=in.value<bool>();out.passed=in.value<bool>();const bool parity=in.value<bool>();in.finish();
        if(!source||!fixtures||out.source_fingerprint!=source||catalog_hash!=motor::fingerprint(catalog.data)||protocol_hash!=motor::fingerprint(protocol)||!motor::same_gains(protocol_gains,out.gains)||
            !motor::same_gains(out.gains,motor::validated_gains())||!completed||!out.passed||!parity||cpu_steps!=1600||gpu_steps!=1600||cpu_cases!=8||gpu_cases!=8||cpu_passed!=8||gpu_passed!=8) throw std::runtime_error("Historical regulation evidence is not the completed validated proof");return out;
    }
};
}
#endif

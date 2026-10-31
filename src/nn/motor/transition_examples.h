#ifndef CUDARL_TRANSITION_EXAMPLES_H
#define CUDARL_TRANSITION_EXAMPLES_H
#include "motor/demonstration_data.h"
#include "nn/mlp_spec.h"
#include <algorithm>
namespace cudarl::demonstration {
constexpr uint32_t kSignedCadenceMapping=2,kTransitionDataVersion=1;
constexpr int kTransitionRows=480,kMaximumTransitionRows=1440;
constexpr uint32_t kAcceptedTransitionMask=15;
struct SignedReference {task::Action action{};float amplitude=0,cadence=0;};
inline bool signed_action(const float* latent,SignedReference& out){
    SignedReference next;if(!task::action_from_latent(latent,next.action))return false;
    next.amplitude=.35f*(next.action.command[0]+1);next.cadence=1.5f*next.action.command[1];out=next;return true;
}
inline bool signed_teacher(const float* observation,SignedReference& out){
    BrakingLabel teacher;if(!feedback_brake(observation,teacher))return false;
    const float latent[2]{atanhf(2*teacher.amplitude/.7f-1),atanhf(teacher.cadence/1.5f)};
    return signed_action(latent,out);
}
inline bool step_signed(task::World& world,const SignedReference& reference){
    return stroke_motor::step_state(world,reference.action.command,stroke_motor::ActionMapping::SignedCadence);
}
struct TransitionExamples {
    Dataset data;
    std::vector<float> executed_commands;
    std::vector<uint32_t> source_action_mapping;
    int reused_rows=0,new_rows=0;
    uint32_t accepted_segments=0;
};
inline void append_example(TransitionExamples& out,const float* observation,const SignedReference& executed,uint32_t source_mapping=kSignedCadenceMapping,const float* source_commands=nullptr){
    SignedReference expected;if(!signed_teacher(observation,expected))throw std::runtime_error("Invalid complete transition observation");
    for(int j=0;j<2;++j)if(fabsf(executed.action.latent[j]-expected.action.latent[j])>2e-5f||fabsf(executed.action.command[j]-expected.action.command[j])>2e-6f)
        throw std::runtime_error("Executed action differs from the recorded current-state teacher");
    out.data.observation.insert(out.data.observation.end(),observation,observation+42);
    out.source_action_mapping.push_back(source_mapping);
    for(int j=0;j<2;++j){out.data.target.push_back(executed.action.latent[j]);out.executed_commands.push_back(source_commands?source_commands[j]:executed.action.command[j]);}
}
inline TransitionExamples reuse_compatible(const Dataset& accepted){
    TransitionExamples out;out.data.source_dataset_checksum=accepted.source_dataset_checksum;out.data.source_gate_checksum=accepted.source_gate_checksum;
    for(int row=0;row<accepted.count();++row){const float* obs=accepted.observation.data()+row*42;BrakingLabel braking;Label forward;
        if(!feedback_brake(obs,braking)||!label(obs,forward))throw std::runtime_error("Invalid previously accepted example");
        if(braking.brake_weight!=0||fabsf(braking.amplitude-forward.amplitude)>1e-6f)continue;
        SignedReference converted;if(!signed_teacher(obs,converted))throw std::runtime_error("Cannot encode accepted forward example");
        task::Action original;if(!task::action_from_latent(accepted.target.data()+row*2,original))throw std::runtime_error("Invalid source action");
        append_example(out,obs,converted,1,original.command);++out.reused_rows;
    }
    return out;
}
inline std::vector<uint8_t> encode_transitions(const TransitionExamples& data,uint64_t protocol){
    checkpoint::Writer out;out.value(uint32_t{0x32584454});out.value(kTransitionDataVersion);out.value(kSignedCadenceMapping);
    out.value(protocol);out.value(data.data.source_dataset_checksum);out.value(data.data.source_gate_checksum);out.value(data.accepted_segments);
    out.value(data.reused_rows);out.value(data.new_rows);out.array(data.data.observation);out.array(data.data.target);out.array(data.executed_commands);out.array(data.source_action_mapping);return out.data;
}
inline TransitionExamples decode_transitions(const std::vector<uint8_t>& bytes,uint64_t protocol){
    checkpoint::Reader in{bytes};if(in.value<uint32_t>()!=0x32584454||in.value<uint32_t>()!=kTransitionDataVersion||in.value<uint32_t>()!=kSignedCadenceMapping||in.value<uint64_t>()!=protocol)
        throw std::runtime_error("Wrong transition data schema/action mapping/protocol");
    TransitionExamples out;out.data.source_dataset_checksum=in.value<uint64_t>();out.data.source_gate_checksum=in.value<uint64_t>();out.accepted_segments=in.value<uint32_t>();
    out.reused_rows=in.value<int>();out.new_rows=in.value<int>();out.data.observation=in.array<float>(kMaximumTransitionRows*42);out.data.target=in.array<float>(kMaximumTransitionRows*2);out.executed_commands=in.array<float>(kMaximumTransitionRows*2);out.source_action_mapping=in.array<uint32_t>(kMaximumTransitionRows);in.finish();
    const int count=out.data.count();if(!out.data.source_dataset_checksum||!out.data.source_gate_checksum||out.accepted_segments!=kAcceptedTransitionMask||out.new_rows!=kTransitionRows||out.reused_rows<1||out.reused_rows>960||count!=out.reused_rows+out.new_rows||out.data.target.size()!=static_cast<size_t>(count)*2||out.data.observation.size()!=static_cast<size_t>(count)*42||out.executed_commands.size()!=static_cast<size_t>(count)*2||out.source_action_mapping.size()!=static_cast<size_t>(count))
        throw std::runtime_error("Missing accepted segment evidence or incomplete observation/action rows");
    for(int row=0;row<count;++row){SignedReference expected;if(!signed_teacher(out.data.observation.data()+row*42,expected))throw std::runtime_error("Invalid saved teacher observation");
        const auto mapping=out.source_action_mapping[row];if(mapping!=(row<out.reused_rows?1u:kSignedCadenceMapping))throw std::runtime_error("Wrong source action mapping");
        for(int j=0;j<2;++j)if(!std::isfinite(out.data.target[row*2+j])||!std::isfinite(out.executed_commands[row*2+j])||fabsf(out.executed_commands[row*2+j])>1||fabsf(out.data.target[row*2+j]-expected.action.latent[j])>2e-5f)
            throw std::runtime_error("Saved transition action or target was changed");
        const float amplitude=.35f*(out.executed_commands[row*2]+1),cadence=mapping==1?.25f+.625f*(out.executed_commands[row*2+1]+1):1.5f*out.executed_commands[row*2+1];
        if(fabsf(amplitude-expected.amplitude)>2e-6f||fabsf(cadence-expected.cadence)>2e-6f)throw std::runtime_error("Saved source action has a different physical reference");}
    return out;
}
inline std::vector<float> initialize_signed(std::vector<float> params,const nn::MlpSpec& spec){
    if(spec.input_size()!=42||spec.output_size()!=2||params.size()!=static_cast<size_t>(spec.num_params()))throw std::runtime_error("Wrong signed actor shape");
    const int last=spec.num_layers()-1;
    for(int j=0;j<spec.in(last);++j)params[spec.weight_offset(last)+spec.in(last)+j]=0;
    params[spec.bias_offset(last)+1]=atanhf(1.f/1.5f);return params;
}
// Equal representation of accepted constants, negative-cadence examples and
// other transition examples. These are explicit resampled rows, not new evidence.
inline Dataset balanced_training(const TransitionExamples& accepted){
    std::vector<int> group[3];
    for(int row=0;row<accepted.data.count();++row)group[row<accepted.reused_rows?0:(accepted.data.target[row*2+1]<0?1:2)].push_back(row);
    size_t count=0;for(const auto& rows:group){if(rows.empty())throw std::runtime_error("Missing signed training regime");count=std::max(count,rows.size());}
    Dataset out;
    for(size_t i=0;i<count;++i)for(const auto& rows:group){const int row=rows[i%rows.size()];
        out.observation.insert(out.observation.end(),accepted.data.observation.begin()+row*42,accepted.data.observation.begin()+(row+1)*42);
        out.target.insert(out.target.end(),accepted.data.target.begin()+row*2,accepted.data.target.begin()+(row+1)*2);}
    return out;
}
}
#endif

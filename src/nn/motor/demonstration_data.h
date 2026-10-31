#ifndef CUDARL_MOTOR_DEMONSTRATION_DATA_H
#define CUDARL_MOTOR_DEMONSTRATION_DATA_H
#include "motor/demonstration.h"
#include "core/checkpoint.h"
namespace cudarl::demonstration {
constexpr int kAcceptedRows=960;
struct Dataset {
    std::vector<float> observation,target;
    uint64_t source_dataset_checksum=0,source_gate_checksum=0;
    int count()const{return static_cast<int>(target.size()/2);}
};
// The launcher separately verifies the source's six accepted gates and row
// provenance. This checks the frozen slice's schema, goal order and every label.
inline Dataset accepted_examples(const std::vector<uint8_t>& payload){
    checkpoint::Reader in{payload};
    if(in.value<uint32_t>()!=0x4344454D||in.value<uint32_t>()!=1||in.value<uint32_t>()!=kVersion||in.value<uint32_t>()!=kAcceptedRows)
        throw std::runtime_error("Wrong accepted-demonstration schema/count");
    Dataset data;data.source_dataset_checksum=in.value<uint64_t>();data.source_gate_checksum=in.value<uint64_t>();
    if(!data.source_dataset_checksum||!data.source_gate_checksum)throw std::runtime_error("Missing demonstration lineage");
    data.observation=in.array<float>(kAcceptedRows*42);data.target=in.array<float>(kAcceptedRows*2);in.finish();
    if(data.observation.size()!=kAcceptedRows*42||data.target.size()!=kAcceptedRows*2)throw std::runtime_error("Wrong example dimensions");
    for(int row=0;row<kAcceptedRows;++row){Label expected;const float* obs=data.observation.data()+row*42;
        if(!label(obs,expected)||fabsf(expected.requested_speed-(.004f+.002f*float((row%6)/2)))>1e-7f)
            throw std::runtime_error("Invalid accepted example or goal order");
        for(int j=0;j<2;++j)if(!std::isfinite(data.target[row*2+j])||fabsf(data.target[row*2+j]-expected.action.latent[j])>2e-5f)
            throw std::runtime_error("Recorded action differs from validated source teacher");
    }
    return data;
}
}
#endif

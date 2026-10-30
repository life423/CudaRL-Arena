#include "check.h"
#include "simulation/world.h"
#include <cuda_runtime.h>
#include <array>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace {
namespace s=cudarl::simulation;
std::vector<float> fields(const s::WorldState& w) {
    const auto& b=w.body;const auto& t=w.telemetry;const auto& i=t.inspection;
    std::vector<float> out{b.position.x,b.position.y,b.position.z,b.orientation.x,b.orientation.y,b.orientation.z,b.orientation.w,b.work,b.travel,
        w.current.x,w.current.y,w.current.z,w.fluid.density,w.fluid.viscosity,w.fluid.gravity,w.mean_squared_torque};
    for(float v:b.angle) out.push_back(v);for(float v:b.velocity) out.push_back(v);
    for(float v:w.gains.kp) out.push_back(v);for(float v:w.gains.kd) out.push_back(v);
    for(float v:w.held_targets) out.push_back(v);for(float v:w.raw_action) out.push_back(v);
    for(float v:w.last_actual_torques) out.push_back(v);
    for(float v:w.actual_torques) out.push_back(v);for(float v:w.requested_torques) out.push_back(v);
    for(float v:{i.floor_gap,i.min_other_gap,i.min_gap,i.max_floor_gap,t.minimum_floor_gap,t.minimum_other_gap,t.floor_spring_force_max,t.total_spring_force_max}) out.push_back(v);
    return out;
}
void discrete(const s::WorldState& a,const s::WorldState& b) {
    CHECK(a.control_steps==b.control_steps);CHECK(a.substeps==b.substeps);CHECK(a.time_seconds==b.time_seconds);
    CHECK(a.paused==b.paused&&a.unsafe==b.unsafe&&a.invalid==b.invalid&&a.actuation==b.actuation);
    const auto& x=a.telemetry;const auto& y=b.telemetry;
    CHECK(x.executed_substeps==y.executed_substeps&&x.contact_samples==y.contact_samples&&x.first_unsafe_substep==y.first_unsafe_substep);
    const auto& p=x.inspection;const auto& q=y.inspection;
    CHECK(p.contact==q.contact&&p.floor_contact==q.floor_contact&&p.excessive==q.excessive&&p.escape==q.escape&&p.invalid==q.invalid);
}
}
GPU_TEST_CASE("physics CUDA: invalid shared settings reject before integration (zero integration)") {
    std::vector<s::WorldState> states(8);for(auto& state:states) state.paused=false;
    states[0].current.y=.001f;states[1].current.x=.008f;
    states[2].gains.kp[0]*=2;states[3].gains.kd[3]*=2;
    states[4].fluid.density=-1;states[5].fluid.viscosity=-.001f;
    states[6].fluid.gravity=-1;states[7].actuation=static_cast<s::Actuation>(9);
    const auto before=states;std::vector<std::array<float,4>> input(states.size());
    cudaStream_t stream=nullptr;
    if(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)!=cudaSuccess) throw std::runtime_error("Cannot create settings-check stream");
    try {s::cuda_step(states,input,stream);} catch(...) {cudaStreamDestroy(stream);throw;}
    cudaStreamDestroy(stream);
    for(size_t row=0;row<states.size();++row) {
        const auto& rejected=states[row];CHECK(rejected.invalid&&!s::valid_settings(rejected));
        CHECK(rejected.substeps==0&&rejected.control_steps==0&&rejected.time_seconds==0&&rejected.telemetry.executed_substeps==0);
        const auto old_fields=fields(before[row]),new_fields=fields(rejected);
        CHECK(old_fields==new_fields); // no physical/body or command changes
    }
}
GPU_TEST_CASE("physics CUDA: same-stream CPU agreement and exact GPU replay (4CPU+8GPU controls)") {
    std::vector<s::WorldState> initial(4);
    std::vector<std::array<float,4>> input{{{.1f,-.1f,.05f,-.05f}},{{.07f,.03f,-.08f,.02f}},{{.1f,-.1f,.05f,-.05f}},{{.07f,.03f,-.08f,.02f}}};
    for(int n=0;n<4;++n) {
        auto& w=initial[n];w.paused=false;w.actuation=n<2?s::Actuation::Torque:s::Actuation::JointTarget;
        if(n%2) w.current={.005f,0,0};
        w.body.angle[0]=.03f;w.body.angle[2]=-.02f;w.body.velocity[0]=.001f;
    }
    auto cpu=initial,gpu=initial,replay=initial;
    for(int n=0;n<4;++n) CHECK(s::step_state(cpu[n],input[n].data()));
    cudaStream_t stream=nullptr;
    if(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)!=cudaSuccess) throw std::runtime_error("Cannot create foundation nonblocking stream");
    try {s::cuda_step(gpu,input,stream);s::cuda_step(replay,input,stream);}
    catch(...) {cudaStreamDestroy(stream);throw;}
    cudaStreamDestroy(stream);
    double maximum=0;
    for(int n=0;n<4;++n) {
        discrete(cpu[n],gpu[n]);discrete(gpu[n],replay[n]);
        const auto a=fields(cpu[n]),g=fields(gpu[n]),r=fields(replay[n]);
        CHECK(a.size()==g.size()&&g.size()==r.size());
        for(size_t j=0;j<a.size();++j) {
            CHECK(std::isfinite(g[j]));
            const double error=std::fabs(double(g[j])-a[j]);maximum=std::fmax(maximum,error);
            CHECK(error<=2e-4+2e-3*std::fabs(double(a[j])));
            CHECK(g[j]==r[j]); // defined fields only, never struct padding
        }
        const auto& a_gap=cpu[n].telemetry;const auto& g_gap=gpu[n].telemetry;
        CHECK(std::fabs(a_gap.minimum_floor_gap-g_gap.minimum_floor_gap)<=2e-6f+2e-3f*std::fabs(a_gap.minimum_floor_gap));
        CHECK(std::fabs(a_gap.minimum_other_gap-g_gap.minimum_other_gap)<=2e-6f+2e-3f*std::fabs(a_gap.minimum_other_gap));
        for(float torque:gpu[n].actual_torques) CHECK(std::fabs(torque)<=1);
        CHECK(gpu[n].substeps==10&&!gpu[n].invalid&&!gpu[n].unsafe);
    }
    std::printf("    shared-physics CPU/CUDA max defined-field difference %.9g; GPU repeat exact\n",maximum);
}

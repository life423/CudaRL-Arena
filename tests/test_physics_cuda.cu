#include "check.h"
#include "simulation/world.h"
#include "cuda/cuda_memory.h"
#include "motor/straight_task.h"
#include <cuda_runtime.h>
#include <array>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace {
namespace s=cudarl::simulation;
// Synthetic initialization/scoring only. No physics or policy update occurs.
CUDARL_HD void recovery_score_fixture(int i, float *out) {
    namespace m = cudarl::straight_motor;
    namespace p = cudarl::physics;
    auto w = m::initial_world(0);
    m::ForwardTask t;
    const m::ForwardConfig c;
    bool ok = m::begin_forward(w, c, t);
    w.physical.body.orientation = w.physical.body.orientation *
        p::axis_rotation(i % 2 ? p::V{1,0,0} : p::V{0,0,1}, 5 * cudarl::movement::pi / 180);
    w.physical.body.position += t.start_com_model - cudarl::motor::motion(w.physical.body).com_model;
    ok = ok && m::offset_episode_start(w, t, c, ((i / 2) % 3 - 1) * .0005f);
    w.physical.body.velocity[1] = ((i / 6) % 3 - 1) * .00025f;
    float obs[40]{};
    ok = ok && m::observe(t, w, c, obs);
    const int prior = i >= 18 ? 40 : 0;
    t.controls = prior; t.last_world_control = prior; w.physical.control_steps = prior + 1;
    const auto result = m::record_control(t, w, c);
    const auto &r = result.costs;
    out[0] = ok && result.accepted && w.attempted_substeps == 0 ? 1.f : 0.f;
    out[1] = t.start_com_model.y;
    out[2] = -.072f * (obs[0]*obs[9]+obs[1]*obs[10]+obs[2]*obs[11]);
    out[3] = .02f * (obs[12]*obs[9]+obs[13]*obs[10]+obs[14]*obs[11]);
    out[4] = result.metrics.roll_deg; out[5] = result.metrics.pitch_deg;
    out[6] = result.metrics.tilt_deg; out[7] = result.reward;
    out[8] = r.speed; out[9] = r.lane; out[10] = r.depth; out[11] = r.posture; out[12] = r.effort;
    out[13] = r.horizontal; out[14] = r.recovery; out[15] = r.roll; out[16] = r.pitch;
    out[17] = r.total();
}
__global__ void recovery_score_kernel(float *out) {
    const int i=threadIdx.x;
    if(i<36) recovery_score_fixture(i,out+i*18);
}
__global__ void recovery_reset_targets(float *out) {
    namespace m=cudarl::straight_motor;
    const int row=threadIdx.x;
    if(row>=96)return;
    m::TrainingEpisode episode;episode.controls=640;episode.fixed_depth_offsets=true;episode.offset_phase=row/32;
    const int i=row%32;m::World w;m::ForwardTask t;
    const bool ok=m::begin_training_episode(i,false,episode.task_config(),w,t,episode.arena(),episode.initial_vertical_offset(i));
    out[row*6]=ok?1.f:0.f;out[row*6+1]=t.start_com_model.x;out[row*6+2]=t.start_com_model.y;out[row*6+3]=t.start_com_model.z;
    out[row*6+4]=t.request.position.y;
    out[row*6+5]=(cudarl::motor::motion(w.physical.body).com_model.y-t.start_com_model.y)*cudarl::physics::scale;
}
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
GPU_TEST_CASE("straight_reward5 CUDA: fixed reference and decomposed reward match CPU (zero integration)") {
    device_ptr<float> output(36*18);
    recovery_score_kernel<<<1,64>>>(output.get());
    CUDA_CHECK_THROW(cudaGetLastError());
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    std::vector<float> actual(36*18);
    output.copy_to_host(actual.data(),actual.size());
    double maximum=0;
    for(int i=0;i<36;++i) {
        float expected[18]{};
        recovery_score_fixture(i,expected);
        CHECK(expected[0]==1 && actual[i*18]==1 && expected[1]==-2 && actual[i*18+1]==-2);
        for(int j=0;j<18;++j) {
            const double error=std::fabs(double(actual[i*18+j])-expected[j]);
            maximum=std::fmax(maximum,error);
            CHECK(error<=2e-6+2e-5*std::fabs(double(expected[j])));
        }
    }
    std::printf("    36 CPU/CUDA reward/offset cases, maximum defined-field difference %.9g; zero physics\n",maximum);
}
GPU_TEST_CASE("straight_depth_mix CUDA: actual reset target coordinates across all phases (zero integration)") {
    device_ptr<float> buffer(96*6);recovery_reset_targets<<<1,128>>>(buffer.get());
    CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaDeviceSynchronize());
    std::vector<float> actual(96*6);buffer.copy_to_host(actual.data(),actual.size());
    double maximum=0;int exact_mismatches=0;
    for(int row=0;row<96;++row){
        const int i=row%32;const auto base=cudarl::straight_motor::initial_world(i%2?3:0);
        const auto target=cudarl::motor::motion(base.physical.body).com_model;
        const float expected[3]={target.x,target.y,target.z};
        CHECK(actual[row*6]==1 && actual[row*6+2]==-2 && actual[row*6+4]==-2);
        int centered=i%2;while((centered/2+row/32)%3!=0)centered+=2;
        for(int j=0;j<3;++j)CHECK(actual[row*6+1+j]==actual[(row/32*32+centered)*6+1+j]);
        for(int j=0;j<3;++j){const double difference=std::fabs(double(actual[row*6+1+j])-expected[j]);maximum=std::fmax(maximum,difference);if(difference!=0)++exact_mismatches;CHECK(difference<1e-6);}
        if(row==0)std::printf("    first reset CPUtargetXYZ %.12g %.12g %.12g; GPU %.12g %.12g %.12g\n",target.x,target.y,target.z,actual[1],actual[2],actual[3]);
    }
    std::printf("    96 zero-step reset targets: max CPU/CUDA coordinate difference %.12g modelunits, %d non-bitwise coordinates; ALL targetY exactly-2\n",maximum,exact_mismatches);
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

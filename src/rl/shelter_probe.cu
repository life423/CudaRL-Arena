#include "shelter_probe.h"
#include "envs/swimmer/predator.h"
#include "envs/swimmer/shelter_motor.h"
#include "swimmer_ppo.h"

namespace cudarl::rl {
namespace {
__global__ void inputs(const arch::State* states,const ShelterProbeCase* cases,float* out,int n,int dimensions) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    if(dimensions==arch::kObs) shelter::motor_observe(states[i],cases[i].goal,out+i*dimensions,cases[i].basin);
    else {
        shelter_motor::State world;world.state=states[i];world.request.cover=true;world.request.goal=cases[i].goal;
        world.request.eye=cases[i].predator_eye;
        const auto location=predator::cover(cases[i].goal.cover,cases[i].predator_eye,states[i].habitat,cases[i].basin);
        world.request.heading=deep::axis_rotation({0,1,0},location.yaw).rotate({1,0,0});
        shelter_motor::observe(world,out+i*dimensions,cases[i].basin);
    }
}
__global__ void tick(arch::State* states,const ShelterProbeCase* cases,const float* torques,
                    ShelterProbeResult* reports,int* streak,int n,int step,int approach,int total) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    auto& state=states[i];auto& report=reports[i];const auto& item=cases[i];
    (void)arch::advance(state,torques+i*4,item.current,item.basin);
    arch::replenish(state,item.basin);
    const float distance=deep::norm(state.pond.body.position-item.goal.position);
    const float exposure=predator::exposure(state.pond.body,item.predator_eye,state.habitat,item.basin);
    const float clearance=predator::body_clearance(state.pond.body,state.habitat,item.basin);
    if(!isfinite(distance) || !isfinite(exposure) || !isfinite(clearance)) report.finite=false;
    report.minimum_goal_distance=fminf(report.minimum_goal_distance,distance);
    report.minimum_clearance=fminf(report.minimum_clearance,clearance);
    report.mean_exposure+=exposure/static_cast<float>(total);report.last_exposure=exposure;
    const bool safe=exposure<=.05f && clearance>=.08f;
    if(report.arrival_step<0 && distance<=.45f && safe) report.arrival_step=step+1;
    if(step>=approach) {
        if(safe) {++report.safe_dwell_ticks;++streak[i];}
        else streak[i]=0;
        report.longest_safe_dwell_ticks=max(report.longest_safe_dwell_ticks,streak[i]);
    }
    report.final_goal_distance=distance;
    if(step+1==total) {report.final_state=state;report.contacts=state.contacts-item.state.contacts;}
}
}
std::vector<ShelterProbeResult> probe_shelter_motor(const std::vector<float>& weights,
    const std::vector<ShelterProbeCase>& cases,int approach_ticks,int dwell_ticks,shelter_motor::ActuatorScale scales) {
    if(cases.empty() || cases.size()>2048 || approach_ticks<1 || dwell_ticks<1 || approach_ticks+dwell_ticks>2400)
        throw std::invalid_argument("Invalid frozen shelter motor probe");
    const int n=static_cast<int>(cases.size()),total=approach_ticks+dwell_ticks;
    const int dimensions=weights.size()==static_cast<size_t>(swimmer_specification(4,shelter_motor::kObs).num_params())?shelter_motor::kObs:arch::kObs;
    const auto spec=swimmer_specification(4,dimensions);
    if(weights.size()!=static_cast<size_t>(spec.num_params())) throw std::invalid_argument("Shelter probe requires the inherited 55-input motor");
    for(float x:weights) if(!std::isfinite(x)) throw std::invalid_argument("Non-finite shelter probe motor");
    for(float value:scales.value) if(!std::isfinite(value) || value<1 || value>1048576)
        throw std::invalid_argument("Invalid shelter probe calibration");
    std::vector<arch::State> starts;starts.reserve(cases.size());
    for(const auto& item:cases) {
        const auto& body=item.state.pond.body;const auto& h=item.state.habitat;
        const float qnorm=body.orientation.x*body.orientation.x+body.orientation.y*body.orientation.y+
            body.orientation.z*body.orientation.z+body.orientation.w*body.orientation.w;
        bool finite=std::isfinite(qnorm) && fabsf(qnorm-1.f)<.001f &&
            std::isfinite(deep::norm(body.position));
        for(float value:body.angle) finite=finite && std::isfinite(value);
        for(float value:body.velocity) finite=finite && std::isfinite(value);
        for(float value:item.state.pond.last_action) finite=finite && std::isfinite(value);
        for(const auto& food:item.state.pond.pellets) if(food.active)
            finite=finite && std::isfinite(deep::norm(food.position)) && std::isfinite(deep::norm(food.velocity));
        if(!item.goal.valid || item.goal.food!=-1 || item.goal.cover<0 || item.goal.cover>=predator::kCoverCount ||
           !std::isfinite(deep::norm(item.goal.position)) || !std::isfinite(deep::norm(item.goal.velocity)) ||
           !std::isfinite(deep::norm(item.predator_eye)) || !std::isfinite(deep::norm(item.current)) ||
           deep::norm(item.current)>.0075001f || fabsf(item.current.y)>1e-8f || item.basin.layout!=2 ||
           !h.arch || !std::isfinite(h.scale) || !(h.scale>0) || !std::isfinite(h.major) ||
           !std::isfinite(h.minor) || !(h.major>h.minor && h.minor>0) || !std::isfinite(h.yaw) ||
           !std::isfinite(deep::norm(h.center)) || !finite || item.state.pond.invalid ||
           !arch::water(item.goal.position,h,.08f,item.basin) ||
           !(predator::body_clearance(body,h,item.basin)>=.08f))
            throw std::invalid_argument("Invalid frozen shelter probe case");
        auto start=item.state;
        // Goal choice metadata must agree with the real cover execution path:
        // both real foods remain secondary observations, rather than excluding
        // the inherited food target from the copied source frame.
        start.pond.target=-1;starts.push_back(start);
    }
    SwimmerStream stream;nn::GpuMlp motor(spec,n,7);motor.set_params(weights);
    device_ptr<arch::State> worlds(n);worlds.copy_from_host(starts.data(),starts.size());
    device_ptr<ShelterProbeCase> goals(n);goals.copy_from_host(cases.data(),cases.size());
    std::vector<ShelterProbeResult> results(cases.size());
    device_ptr<ShelterProbeResult> reports(n);reports.copy_from_host(results.data(),results.size());
    device_ptr<int> streak(n);CUDA_CHECK_THROW(cudaMemset(streak.get(),0,streak.size_bytes()));
    device_ptr<float> obs(static_cast<size_t>(n)*dimensions);
    device_ptr<float> means(static_cast<size_t>(n)*4);
    for(int step=0;step<total;++step) {
        inputs<<<(n+63)/64,64,0,stream.value>>>(worlds.get(),goals.get(),obs.get(),n,dimensions);CUDA_CHECK_THROW(cudaGetLastError());
        motor.forward(obs.get(),n,stream.value);
        const float* torques=motor.output();
        if(dimensions==shelter_motor::kObs) {
            shelter_motor::policy_means(torques,obs.get(),means.get(),n,scales,stream.value);torques=means.get();
        }
        tick<<<(n+63)/64,64,0,stream.value>>>(worlds.get(),goals.get(),torques,reports.get(),streak.get(),n,step,approach_ticks,total);
        CUDA_CHECK_THROW(cudaGetLastError());
    }
    CUDA_CHECK_THROW(cudaMemcpyAsync(results.data(),reports.get(),reports.size_bytes(),cudaMemcpyDeviceToHost,stream.value));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));return results;
}
}

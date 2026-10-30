#include "envs/swimmer/command_motor.h"
#include "cuda/device.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

// Focused ordering regression, not a new motor/controller experiment. The
// production path already specifies its consumer stream correctly. The local
// deliberately wrong control below proves why that explicit stream matters.
namespace {
using namespace cudarl;
namespace cm=command_motor;
constexpr int population=73; // includes a partial second64-thread block
constexpr uint32_t seed=0x57EA2026u;
void require(bool valid,const char* message) {if(!valid) throw std::runtime_error(message);}
struct Stream {
    cudaStream_t value=nullptr;
    Stream() {CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
    ~Stream() {if(value) {cudaStreamSynchronize(value);cudaStreamDestroy(value);}}
};
struct RegisteredStates {
    void* data;
    explicit RegisteredStates(std::vector<cm::State>& states):data(states.data()) {
        CUDA_CHECK_THROW(cudaHostRegister(data,states.size()*sizeof(cm::State),cudaHostRegisterDefault));
    }
    ~RegisteredStates() {cudaHostUnregister(data);}
};
template<class T> std::vector<T> download(const T* pointer,size_t count,cudaStream_t stream) {
    std::vector<T> result(count);
    CUDA_CHECK_THROW(cudaMemcpyAsync(result.data(),pointer,count*sizeof(T),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));return result;
}
template<class T> void upload(T* pointer,const std::vector<T>& values,cudaStream_t stream) {
    CUDA_CHECK_THROW(cudaMemcpyAsync(pointer,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice,stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}
template<class T> bool identical(const std::vector<T>& a,const std::vector<T>& b) {
    return a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0;
}
__global__ void observe_local(const cm::State* states,float* observations,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) cm::observe(states[i],observations+i*cm::kObs);
}
__global__ void queued_delay(unsigned long long cycles,uint32_t* marker) {
    const auto begin=clock64();while(clock64()-begin<cycles) {}*marker=1;
}
__global__ void reference_step(const cm::State* starts,const float* action,cm::State* advanced,float* observations,
        float* reward,float* done,float* truncated,float* final_observations,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n) return;
    auto state=starts[i];const auto result=cm::advance(state,action+i*cm::kAct);advanced[i]=state;
    cm::observe(state,observations+i*cm::kObs);reward[i]=result.reward;done[i]=result.timeout||result.invalid?1.f:0.f;
    truncated[i]=0; //600 ticks is a true finite-horizon terminal, not a truncation
    if(result.timeout||result.invalid) cm::observe(state,final_observations+i*cm::kObs);
}
std::vector<cm::State> make_states(int round) {
    std::vector<cm::State> states(population);
    for(int i=0;i<population;++i) {
        auto& state=states[static_cast<size_t>(i)];state=cm::initial(seed,static_cast<uint32_t>(i),static_cast<uint32_t>(round+1));
        state.steps=31+i%29;state.request.position+=deep::V{.13f*(round+1),-.07f,.04f};
        for(int j=0;j<cm::kAct;++j) state.previous_action[j]=.03f*(j+1)*(round%2?1.f:-1.f);
    }
    states.back().steps=cm::kEpisodeSteps-1;return states;
}
struct Outputs {
    device_ptr<float> reward{population},done{population},truncated{population},final_obs{static_cast<size_t>(population)*cm::kObs};
    swimmer::Outputs view() {return {reward.get(),done.get(),truncated.get(),final_obs.get()};}
    void clear(cudaStream_t stream) {
        for(auto* pointer:{&reward,&done,&truncated,&final_obs}) CUDA_CHECK_THROW(cudaMemsetAsync(pointer->get(),0,pointer->size_bytes(),stream));
    }
};
struct StepCapture {std::vector<cm::State> states;std::vector<float> obs,reward,done,truncated,final_obs;};
StepCapture capture(cm::Batch& batch,const Outputs& out,cudaStream_t stream) {
    return {batch.host_states(stream),download(batch.obs(),static_cast<size_t>(population)*cm::kObs,stream),
        download(out.reward.get(),population,stream),download(out.done.get(),population,stream),download(out.truncated.get(),population,stream),
        download(out.final_obs.get(),static_cast<size_t>(population)*cm::kObs,stream)};
}
std::array<float,36> defined_floats(const cm::State& s) {
    std::array<float,36> out{};int at=0;
    for(float value:{s.body.position.x,s.body.position.y,s.body.position.z,s.body.orientation.x,s.body.orientation.y,s.body.orientation.z,s.body.orientation.w}) out[at++]=value;
    for(float value:s.body.angle) out[at++]=value;for(float value:s.body.velocity) out[at++]=value;
    for(float value:{s.body.work,s.body.travel,s.request.position.x,s.request.position.y,s.request.position.z,s.request.velocity.x,s.request.velocity.y,s.request.velocity.z,
                    s.request.heading.x,s.request.heading.y,s.request.heading.z}) out[at++]=value;
    for(float value:s.previous_action) out[at++]=value;return out;
}
void require_independent_parity(const StepCapture& actual,const StepCapture& reference) {
    // These are separately compiled kernels: the small oracle specializes
    // default Physics, whereas production selects runtime per-world Physics.
    // Use the already registered physical parity tolerance for that comparison,
    // not bit equality across different compiler specializations. This does
    // not change any readiness/safety threshold or the exact replay below.
    auto compare=[](float a,float b,double& maximum) {
        require(std::isfinite(a)&&std::isfinite(b),"Independent physical-kernel parity contains nonfinite data");
        const double error=fabs(static_cast<double>(a)-b);maximum=std::max(maximum,error);
        require(error<=2e-4+2e-3*fabs(static_cast<double>(b)),"Independent physical-kernel parity exceeded its preregistered tolerance");
    };
    double state_error=0,observation_error=0,reward_error=0,final_observation_error=0;
    require(actual.states.size()==reference.states.size(),"Independent physical-kernel state population differs");
    for(size_t row=0;row<actual.states.size();++row) {
        const auto& a=actual.states[row];const auto& b=reference.states[row];
        require(a.seed==b.seed&&a.environment==b.environment&&a.episode==b.episode&&a.steps==b.steps,"Independent physical-kernel counters differ");
        const auto av=defined_floats(a),bv=defined_floats(b);
        for(size_t field=0;field<av.size();++field) compare(av[field],bv[field],state_error);
    }
    require(actual.obs.size()==reference.obs.size()&&actual.reward.size()==reference.reward.size()&&actual.final_obs.size()==reference.final_obs.size(),
        "Independent physical-kernel output dimensions differ");
    for(size_t i=0;i<actual.obs.size();++i) compare(actual.obs[i],reference.obs[i],observation_error);
    for(size_t i=0;i<actual.reward.size();++i) compare(actual.reward[i],reference.reward[i],reward_error);
    for(size_t i=0;i<actual.final_obs.size();++i) compare(actual.final_obs[i],reference.final_obs[i],final_observation_error);
    require(identical(actual.done,reference.done)&&identical(actual.truncated,reference.truncated),"Independent physical-kernel terminal flags differ");
    std::cout<<std::setprecision(17)<<"COMMAND_MOTOR_STREAM_INDEPENDENT_KERNEL_PARITY state_max_abs="<<state_error<<" observation_max_abs="<<observation_error
        <<" reward_max_abs="<<reward_error<<" final_observation_max_abs="<<final_observation_error<<" fixed_abs_tolerance=0.0002 fixed_relative_tolerance=0.002 counters_and_terminal_flags_exact=1\n";
}
void require_identical(const StepCapture& a,const StepCapture& b,const char* phase) {
    if(!identical(a.states,b.states)) {
        static constexpr const char* names[36]={"body.position.x","body.position.y","body.position.z","body.orientation.x","body.orientation.y","body.orientation.z","body.orientation.w",
            "body.angle0","body.angle1","body.angle2","body.angle3","body.velocity0","body.velocity1","body.velocity2","body.velocity3","body.velocity4","body.velocity5","body.velocity6","body.velocity7","body.velocity8","body.velocity9",
            "body.work","body.travel","request.position.x","request.position.y","request.position.z","request.velocity.x","request.velocity.y","request.velocity.z","request.heading.x","request.heading.y","request.heading.z",
            "previous_action0","previous_action1","previous_action2","previous_action3"};
        double maximum=0;int differing=0,counter_differences=0;
        for(size_t row=0;row<a.states.size();++row) {
            const auto av=defined_floats(a.states[row]),bv=defined_floats(b.states[row]);
            for(size_t field=0;field<av.size();++field) {
                uint32_t abits=0,bbits=0;std::memcpy(&abits,&av[field],sizeof(float));std::memcpy(&bbits,&bv[field],sizeof(float));
                if(abits!=bbits) {
                    const double error=fabs(static_cast<double>(av[field])-bv[field]);maximum=std::max(maximum,error);
                    if(differing<12) std::cerr<<std::setprecision(17)<<"COMMAND_MOTOR_STREAM_STATE_DIFFERENCE phase="<<phase<<" row="<<row<<" field="<<names[field]
                        <<" actual="<<av[field]<<" reference="<<bv[field]<<" abs="<<error<<" actual_bits="<<abits<<" reference_bits="<<bbits<<'\n';
                    ++differing;
                }
            }
            const auto& as=a.states[row];const auto& bs=b.states[row];
            if(as.seed!=bs.seed||as.environment!=bs.environment||as.episode!=bs.episode||as.steps!=bs.steps) ++counter_differences;
        }
        std::ostringstream message;message<<std::setprecision(17)<<"One-step state comparison phase="<<phase<<" differing_float_fields="<<differing
            <<" counter_rows="<<counter_differences<<" max_abs="<<maximum<<" sizeof_state="<<sizeof(cm::State)<<" padding_only="<<(differing==0&&counter_differences==0);
        throw std::runtime_error(message.str());
    }
    if(!identical(a.obs,b.obs)) throw std::runtime_error(std::string("One-step observations differ phase=")+phase);
    require(identical(a.reward,b.reward)&&identical(a.done,b.done)&&identical(a.truncated,b.truncated)&&identical(a.final_obs,b.final_obs),
        "One-step replay changed reward/terminal/final-observation outputs");
}
}
int main() {
    try {
        const auto device=probe_device();require(device.available,"A working CUDA device is required; this focused test must not skip GPU execution");
        Stream stream;unsigned int flags=0;CUDA_CHECK_THROW(cudaStreamGetFlags(stream.value,&flags));
        require((flags&cudaStreamNonBlocking)!=0,"Focused test did not create a nonblocking stream");
        cm::Batch batch(population,seed);auto incoming=make_states(0);RegisteredStates pin(incoming);
        device_ptr<cm::State> reference_states(population);device_ptr<float> expected_obs(static_cast<size_t>(population)*cm::kObs);
        device_ptr<uint32_t> delay_marker(1);float maximum_cpu_observation_error=0;
        for(int round=0;round<3;++round) {
            const auto next=make_states(round);std::copy(next.begin(),next.end(),incoming.begin());
            CUDA_CHECK_THROW(cudaMemsetAsync(delay_marker.get(),0,sizeof(uint32_t),stream.value));
            queued_delay<<<1,1,0,stream.value>>>(50000000ull,delay_marker.get());CUDA_CHECK_THROW(cudaGetLastError());
            require(cudaStreamQuery(stream.value)==cudaErrorNotReady,"Queued producer delay did not remain in flight before set_states");
            batch.set_states(incoming,stream.value); // actual production call under test
            require(identical(batch.host_states(stream.value),incoming),"set_states did not preserve uploaded complete states");
            require(download(delay_marker.get(),1,stream.value)[0]==1,"Same-stream producer delay did not finish");
            upload(reference_states.get(),incoming,stream.value);
            observe_local<<<(population+63)/64,64,0,stream.value>>>(reference_states.get(),expected_obs.get(),population);CUDA_CHECK_THROW(cudaGetLastError());
            const auto expected=download(expected_obs.get(),expected_obs.size(),stream.value);
            const auto actual=download(batch.obs(),expected.size(),stream.value);
            require(identical(actual,expected),"Production observations did not consume the uploaded states on their supplied stream");
            for(int i=0;i<population;++i) {
                float cpu[cm::kObs];cm::observe(incoming[static_cast<size_t>(i)],cpu);
                for(int j=0;j<cm::kObs;++j) {
                    const float error=fabsf(actual[static_cast<size_t>(i)*cm::kObs+j]-cpu[j]);maximum_cpu_observation_error=fmaxf(maximum_cpu_observation_error,error);
                    require(error<=2e-5f+2e-5f*fabsf(cpu[j]),"CPU observation check disagrees beyond its fixed float tolerance");
                }
            }
        }
        // Deterministic sensitivity control: deliberately observe OLD data on
        // the default stream BEFORE submitting the producer's NEW upload.
        // WDDM may serialize kernels even on concurrent-capable hardware, so
        // this control does not require or claim concurrent GPU execution.
        // It proves that the observation oracle rejects stale-but-finite data;
        // the delayed actual production call above separately tests ordering.
        const auto old=make_states(7);upload(reference_states.get(),old,stream.value);
        observe_local<<<(population+63)/64,64,0,stream.value>>>(reference_states.get(),expected_obs.get(),population);CUDA_CHECK_THROW(cudaGetLastError());
        const auto old_obs=download(expected_obs.get(),expected_obs.size(),stream.value);
        device_ptr<float> wrong_obs(expected_obs.size());
        observe_local<<<(population+63)/64,64>>>(reference_states.get(),wrong_obs.get(),population);CUDA_CHECK_THROW(cudaGetLastError());
        CUDA_CHECK_THROW(cudaStreamSynchronize(nullptr));
        CUDA_CHECK_THROW(cudaMemcpyAsync(reference_states.get(),incoming.data(),incoming.size()*sizeof(cm::State),cudaMemcpyHostToDevice,stream.value));
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));
        const auto stale=download(wrong_obs.get(),wrong_obs.size(),stream.value);
        require(identical(stale,old_obs),"Default-stream negative control did not capture the old state observations");
        require(!identical(stale,download(batch.obs(),wrong_obs.size(),stream.value)),"Default-stream control failed to expose stale observations");
        require(identical(download(reference_states.get(),incoming.size(),stream.value),incoming),"Negative control's queued upload did not eventually finish");

        checkpoint::Writer before;batch.save(before);
        std::vector<float> actions(static_cast<size_t>(population)*cm::kAct);
        for(int i=0;i<population;++i) for(int j=0;j<cm::kAct;++j) actions[static_cast<size_t>(i)*cm::kAct+j]=.035f*(j%2?-.5f:1.f);
        device_ptr<float> d_actions(actions.size());upload(d_actions.get(),actions,stream.value);
        Outputs actual_output;actual_output.clear(stream.value);batch.step(d_actions.get(),actual_output.view(),stream.value,false);const auto first=capture(batch,actual_output,stream.value);
        require(batch.invalid_count(stream.value)==0,"Focused one-step trial produced nonfinite dynamics");
        upload(reference_states.get(),incoming,stream.value);device_ptr<cm::State> expected_advanced(population);Outputs reference_output;reference_output.clear(stream.value);
        reference_step<<<(population+63)/64,64,0,stream.value>>>(reference_states.get(),d_actions.get(),expected_advanced.get(),expected_obs.get(),reference_output.reward.get(),
            reference_output.done.get(),reference_output.truncated.get(),reference_output.final_obs.get(),population);CUDA_CHECK_THROW(cudaGetLastError());
        StepCapture direct{download(expected_advanced.get(),population,stream.value),download(expected_obs.get(),expected_obs.size(),stream.value),
            download(reference_output.reward.get(),population,stream.value),download(reference_output.done.get(),population,stream.value),
            download(reference_output.truncated.get(),population,stream.value),download(reference_output.final_obs.get(),reference_output.final_obs.size(),stream.value)};
        require_independent_parity(first,direct);
        require(first.done.back()==1.f&&first.truncated.back()==0.f,"Finite-horizon step600 must be terminal without truncation/bootstrap");
        require(first.states.back().steps==cm::kEpisodeSteps&&first.states.back().episode==incoming.back().episode,"Frozen terminal step unexpectedly reset its world");
        checkpoint::Writer after_first;batch.save(after_first);
        checkpoint::Reader reader{before.data};batch.load(reader);reader.finish();checkpoint::Writer restored;batch.save(restored);
        require(restored.data==before.data,"Serialized checkpoint restore changed state/observations/counters");
        actual_output.clear(stream.value);batch.step(d_actions.get(),actual_output.view(),stream.value,false);const auto replay=capture(batch,actual_output,stream.value);
        require_identical(first,replay,"production_snapshot_restore_replay");checkpoint::Writer after_replay;batch.save(after_replay);
        require(after_first.data==after_replay.data,"Serialized one-step checkpoint continuation was not byte-identical");
        std::cout<<std::setprecision(9)<<"COMMAND_MOTOR_STREAM_PASS production_order_preexisting_correct=1 nonblocking=1 registered_host_input=1 delayed_upload_rounds=3 population="<<population
            <<" gpu_observations_bit_identical=1 cpu_observation_max_abs="<<maximum_cpu_observation_error<<" negative_control_observe_before_upload_stale=1 concurrent_gpu_scheduling_required=0 independent_kernel_physical_parity=1 production_replay_outputs_bit_identical=1 terminal600_truncated0=1 serialized_replay_bit_identical=1 training_updates=0\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<"COMMAND_MOTOR_STREAM_FAIL "<<error.what()<<'\n';return 1;}
}

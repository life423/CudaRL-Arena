#include "holding_fit_cases.h"
#include "rl/shelter_motor_run.h"
#include "cuda/device.h"
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

// Fixed supervised diagnosis only. No production PPO, rewards, physics,
// parameterization, source saves or deployed controllers are changed.
namespace {
using namespace cudarl;
namespace fs=std::filesystem;
namespace sm=shelter_motor;
constexpr int input_count=59,action_count=4,batch_size=256,fit_steps=1000;
constexpr uint32_t food_train_seed=0xF17A2026u,food_test_seed=0xF17B2026u,index_seed=0x51F12026u;
constexpr uint32_t food_old_seed=0x571EF001u,food_fresh_seed=0x5F152629u;
void need(bool valid,const std::string& message) {if(!valid) throw std::runtime_error(message);}
uint64_t hash_data(const void* data,size_t bytes) {
    uint64_t hash=14695981039346656037ull;const auto* p=static_cast<const uint8_t*>(data);
    for(size_t i=0;i<bytes;++i) {hash^=p[i];hash*=1099511628211ull;}return hash;
}

uint64_t hash_bytes(const std::vector<uint8_t>& data) {return hash_data(data.data(),data.size());}
template<class T> std::vector<T> download(const T* pointer,size_t count,cudaStream_t stream) {
    std::vector<T> result(count);CUDA_CHECK_THROW(cudaMemcpyAsync(result.data(),pointer,count*sizeof(T),cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));return result;
}
template<class T> void upload(device_ptr<T>& pointer,const std::vector<T>& values,cudaStream_t stream) {
    need(pointer.size()==values.size(),"Diagnostic upload dimension mismatch");
    CUDA_CHECK_THROW(cudaMemcpyAsync(pointer.get(),values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice,stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}
void finite(const std::vector<float>& values,const char* name) {for(float value:values) need(std::isfinite(value),std::string("Non-finite ")+name);}
nn::AdamConfig fit_optimizer() {nn::AdamConfig c;c.lr=.0003f;c.beta1=.9f;c.beta2=.999f;c.eps=.00001f;c.max_grad_norm=.5f;return c;}
std::vector<uint8_t> optimizer_bytes(const nn::GpuAdam& optimizer) {checkpoint::Writer out;optimizer.save(out);return out.data;}
struct RowInfo {uint32_t seed=0;int trajectory=0,tick=0,goal=-1;bool training=false,cover=false;float current=0;uint64_t condition=0;};
struct Dataset {
    std::vector<float> observations,targets;std::vector<RowInfo> rows;std::vector<sm::State> starts;
    void add(const float* obs,const float* target,RowInfo row) {
        observations.insert(observations.end(),obs,obs+input_count);targets.insert(targets.end(),target,target+action_count);rows.push_back(row);
    }
    void append(const Dataset& other) {observations.insert(observations.end(),other.observations.begin(),other.observations.end());targets.insert(targets.end(),other.targets.begin(),other.targets.end());rows.insert(rows.end(),other.rows.begin(),other.rows.end());starts.insert(starts.end(),other.starts.begin(),other.starts.end());}
    std::vector<uint8_t> bytes() const {
        checkpoint::Writer out;out.value(uint32_t{0x48464431u});out.array(observations);out.array(targets);out.value(static_cast<uint64_t>(rows.size()));
        for(const auto& row:rows) {out.value(row.seed);out.value(row.trajectory);out.value(row.tick);out.value(row.goal);out.value(row.training);out.value(row.cover);out.value(row.current);out.value(row.condition);}out.array(starts);return out.data;
    }
};
uint64_t condition_hash(const sm::State& world,float current) {
    checkpoint::Writer out;const auto& p=world.state.pond;const auto& b=p.body;
    out.value(b.position);out.value(b.orientation);for(float value:b.angle) out.value(value);for(float value:b.velocity) out.value(value);
    const auto& h=world.state.habitat;out.value(h.scale);out.value(h.arch);out.value(h.center);out.value(h.major);out.value(h.minor);out.value(h.yaw);out.value(current);
    out.value(p.target);out.value(p.steps);out.value(p.since_food);for(float value:p.last_action) out.value(value);
    for(const auto& food:p.pellets) {out.value(food.active);if(food.active) {out.value(food.position);out.value(food.velocity);}}
    out.value(world.request.cover);out.value(world.request.goal.position);out.value(world.request.goal.cover);out.value(world.request.heading);return hash_bytes(out.data);
}
struct Geometry {float exposure=1,head=1,proximal=1,distal=1,bed=1e10f,solid=1e10f,surface=1e10f;int bad=0;};
__device__ __noinline__ Geometry geometry(const sm::State& state) {
    Geometry out;const auto& body=state.state.pond.body;const auto& h=state.state.habitat;
    const auto visible=predator::exposure_detail(body,state.request.eye,h,{2});out.exposure=visible.visible;out.head=visible.head;out.proximal=visible.proximal_tail;out.distal=visible.distal_tail;
    out.solid=deep::obstacle_body_clearance(body,h,{2});const auto shape=deep::shape(body);
    for(int link=0;link<3;++link) for(int axial=0;axial<9;++axial) {
        const auto& part=shape.link[link];const float f=(axial-4)*.25f;
        const auto point=(part.center+part.q.rotate({1,0,0})*(f*part.length*.5f))/deep::scale;
        const auto gradient=deep::habitat_gradient(point.x,point.z,h,{2});const float length=sqrtf(1+gradient.x*gradient.x+gradient.z*gradient.z);
        const auto normal=deep::V{-gradient.x,1,-gradient.z}/length;
        out.bed=fminf(out.bed,(point.y-deep::habitat_floor(point.x,point.z,h,{2}))/length-deep::arch_slice_support(part,link,f,normal)/deep::scale);
        out.surface=fminf(out.surface,-point.y-deep::arch_slice_support(part,link,f,{0,1,0})/deep::scale);
    }
    if(out.exposure>.05f) out.bad|=1;if(out.bed<.08f) out.bad|=2;if(out.solid<.08f) out.bad|=4;if(out.surface<.08f) out.bad|=8;
    if(!isfinite(deep::norm(body.position)) || !isfinite(out.exposure) || !isfinite(out.bed) || !isfinite(out.solid) || !isfinite(out.surface) || state.state.pond.invalid) out.bad|=16;
    for(float value:body.angle) if(!isfinite(value)) out.bad|=16;for(float value:body.velocity) if(!isfinite(value)) out.bad|=16;
    const auto q=body.orientation;if(!isfinite(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)) out.bad|=16;return out;
}
struct PassiveProof {float max_exposure=0,min_clearance=1e10f,max_displacement=0;int rejected=0;};
__global__ void passive_proof(const sm::State* states,const deep::V* initial,int count,PassiveProof* proof) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=count) return;const auto g=geometry(states[row]);auto& out=proof[row];
    out.rejected|=g.bad;out.max_exposure=fmaxf(out.max_exposure,g.exposure);out.min_clearance=fminf(out.min_clearance,fminf(g.bed,fminf(g.solid,g.surface)));
    out.max_displacement=fmaxf(out.max_displacement,deep::norm(states[row].state.pond.body.position-initial[row]));
}
struct CoverData {Dataset data;std::vector<sm::State> selected;};
CoverData cover_data(const rl::ShelterMotorRun& source,bool training,std::ofstream& admission) {
    const auto seed=training?diagnostics::holding_fit_train_seed:diagnostics::holding_fit_test_seed;
    const auto candidates=diagnostics::holding_fit_candidates(source.source.frame,seed,32,training);
    std::vector<diagnostics::HoldingFitCase> valid;std::vector<sm::State> states;std::vector<deep::V> positions;
    for(const auto& item:candidates) {
        if(!diagnostics::holding_fit_initial_safe(item)) {admission<<training<<','<<seed<<','<<item.trajectory<<','<<item.cover<<",0,initial_geometry,0,-1,-1,-1,0\n";continue;}
        for(float value:item.world.state.pond.body.angle) need(value==0,"Moving/curved cover label candidate");
        for(float value:item.world.state.pond.body.velocity) need(value==0,"Nonstationary cover label candidate");
        need(item.world.state.pond.steps==0,"Cover qualification must start at clock0");
        valid.push_back(item);states.push_back(item.world);positions.push_back(item.world.state.pond.body.position);
    }
    need(!states.empty(),"No initially safe cover candidates");const int count=static_cast<int>(states.size());rl::SwimmerStream stream;
    sm::Batch batch(count,seed);auto context=source.source.context;context.current={};batch.configure(source.source.ancestry.policy.actor,context,stream.value,false);CUDA_CHECK_THROW(cudaDeviceSynchronize());batch.set_states(states,stream.value);
    const auto initial_obs=download(batch.obs(),static_cast<size_t>(count)*input_count,stream.value);
    device_ptr<deep::V> initial(count);upload(initial,positions,stream.value);device_ptr<PassiveProof> proof(count);std::vector<PassiveProof> fresh(count);upload(proof,fresh,stream.value);
    device_ptr<float> zero(static_cast<size_t>(count)*4);std::vector<float> zeros(zero.size(),0);upload(zero,zeros,stream.value);
    for(int tick=0;tick<=200;++tick) {
        passive_proof<<<(count+31)/32,32,0,stream.value>>>(batch.states(),initial.get(),count,proof.get());CUDA_CHECK_THROW(cudaGetLastError());
        if(tick<200) batch.step(zero.get(),{},stream.value,false,false);
    }
    const auto reports=download(proof.get(),count,stream.value);const auto final_states=batch.host_states(stream.value);
    for(const auto& state:final_states) need(state.state.pond.steps==200,"Cover qualification did not finish at clock200");
    int selected[4]{};const int requested=training?8:4;CoverData out;const float target[4]{};
    for(int row=0;row<count;++row) {
        const auto& item=valid[row];const auto& report=reports[row];const bool admitted=report.rejected==0 && selected[item.cover]<requested;
        admission<<training<<','<<seed<<','<<item.trajectory<<','<<item.cover<<','<<admitted<<','<<(report.rejected?"passive_proof_failed":(admitted?"first_qualifying":"quota_already_filled"))<<','<<report.max_displacement<<','<<report.max_exposure<<','<<report.min_clearance<<','<<report.rejected<<",200\n";
        if(!admitted) continue;++selected[item.cover];out.selected.push_back(item.world);out.data.starts.push_back(item.world);
        out.data.add(initial_obs.data()+static_cast<size_t>(row)*input_count,target,{seed,item.trajectory,0,item.cover,training,true,0,condition_hash(item.world,0)});
    }
    for(int goal=0;goal<4;++goal) need(selected[goal]==requested,"Bounded cover pool did not supply the preregistered count; no extension authorized");
    need(out.data.rows.size()==static_cast<size_t>(training?32:16),"Cover dataset size differs from preregistration");return out;
}
__global__ void first55(const float* source,float* destination,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count*55) destination[i]=source[(i/55)*59+i%55];
}
__global__ void finite_food_worlds(const sm::State* states,const float* actions,int count,int* failed);
Dataset food_data(const rl::ShelterMotorRun& source,bool training,double& equivalence_maximum) {
    Dataset result;const uint32_t seed=training?food_train_seed:food_test_seed;const int count=training?32:16;
    const auto teacher59=rl::normalized_shelter_actor(rl::transfer_shelter_motor(source.source.motor),source.scales);
    for(int condition=0;condition<2;++condition) {
        rl::SwimmerStream stream;sm::Batch batch(count,seed);auto context=source.source.context;context.current={condition?.0075f:0,0,0};
        batch.configure(source.source.ancestry.policy.actor,context,stream.value,false);
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        std::vector<sm::State> starts(count);std::vector<uint64_t> hashes(count);
        for(int i=0;i<count;++i) {starts[i].state=arch::initial(seed,static_cast<uint32_t>(condition*count+i),0,false,{2});hashes[i]=condition_hash(starts[i],context.current.x);}
        batch.set_states(starts,stream.value);
        const auto actual_starts=batch.host_states(stream.value);result.starts.insert(result.starts.end(),actual_starts.begin(),actual_starts.end());
        for(int i=0;i<count;++i) hashes[i]=condition_hash(actual_starts[i],context.current.x);
        nn::GpuMlp teacher(rl::swimmer_specification(4,55),count,7),expanded(rl::swimmer_specification(4,59),count,7);teacher.set_params(source.source.motor);expanded.set_params(teacher59);CUDA_CHECK_THROW(cudaDeviceSynchronize());
        device_ptr<float> observation55(static_cast<size_t>(count)*55),expanded_mean(static_cast<size_t>(count)*4);device_ptr<int> failed(count);std::vector<int> fresh_flags(count,0);upload(failed,fresh_flags,stream.value);
        for(int tick=0;tick<600;++tick) {
            first55<<<(count*55+255)/256,256,0,stream.value>>>(batch.obs(),observation55.get(),count);CUDA_CHECK_THROW(cudaGetLastError());teacher.forward(observation55.get(),count,stream.value);
            if(tick%10==0) {
                expanded.forward(batch.obs(),count,stream.value);sm::policy_means(expanded.output(),batch.obs(),expanded_mean.get(),count,source.scales,stream.value);
                const auto obs=download(batch.obs(),static_cast<size_t>(count)*59,stream.value),targets=download(teacher.output(),static_cast<size_t>(count)*4,stream.value),comparison=download(expanded_mean.get(),static_cast<size_t>(count)*4,stream.value);
                finite(obs,"food observations");finite(targets,"original55 teacher means");
                for(size_t i=0;i<targets.size();++i) {const double error=std::abs(static_cast<double>(comparison[i])-targets[i]);equivalence_maximum=std::max(equivalence_maximum,error);need(error<=1e-5*(1+std::abs(targets[i])),"Original55 vs normalized59 physical teacher mismatch");}
                for(int i=0;i<count;++i) {for(int j=55;j<59;++j) need(obs[i*59+j]==0,"Food dataset contains a cover mode input");result.add(obs.data()+i*59,targets.data()+i*4,{seed,condition*count+i,tick,-1,training,false,context.current.x,hashes[i]});}
            }
            batch.step(teacher.output(),{},stream.value,false,false);
            finite_food_worlds<<<(count+31)/32,32,0,stream.value>>>(batch.states(),teacher.output(),count,failed.get());CUDA_CHECK_THROW(cudaGetLastError());
        }
        const auto final_states=batch.host_states(stream.value);const auto invalid=download(failed.get(),count,stream.value);
        for(int i=0;i<count;++i) need(final_states[i].state.pond.steps==600 && invalid[i]==0,"Invalid/nonfinite or reset food data trajectory");
        need(teacher.get_params()==source.source.motor && expanded.get_params()==teacher59,"Frozen original teacher changed while collecting data");
    }
    need(result.rows.size()==static_cast<size_t>(training?3840:1920),"Food dataset size differs from preregistration");return result;
}
struct LossScale {float food[4]{1,1,1,1};};
LossScale loss_scale(const Dataset& food) {
    LossScale result;
    for(int joint=0;joint<4;++joint) {double squared=0;for(size_t row=0;row<food.rows.size();++row) squared+=double(food.targets[row*4+joint])*food.targets[row*4+joint];result.food[joint]=static_cast<float>(std::max(1.,std::sqrt(squared/food.rows.size())));}return result;
}
enum class FitArm:int {CoverOnly=0,FoodOnly=1,Joint=2};
const char* arm_name(FitArm arm) {return arm==FitArm::CoverOnly?"cover_only":(arm==FitArm::FoodOnly?"food_only":"joint");}
__global__ void gather_fit(const float* observations,const float* targets,const int* indices,float* batch_obs,float* batch_targets) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<batch_size*59) batch_obs[i]=observations[static_cast<size_t>(indices[i/59])*59+i%59];
    if(i<batch_size*4) batch_targets[i]=targets[static_cast<size_t>(indices[i/4])*4+i%4];
}
__global__ void mse_gradient(const float* means,const float* targets,const float* observations,int count,int arm,LossScale scale,float* loss,float* gradient) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count*4) return;const bool cover=observations[static_cast<size_t>(i/4)*59+55]>=.5f;
    const bool active=arm==2 || (arm==0 && cover) || (arm==1 && !cover);const float unit=cover?1.f:scale.food[i%4],error=means[i]-targets[i];
    const float denominator=static_cast<float>(count*4);loss[i]=active?(error/unit)*(error/unit)/denominator:0;
    gradient[i]=active?2.f*error/(unit*unit*denominator):0;
}
void finite_difference_check(sm::ActuatorScale gains,LossScale units,std::ofstream& file) {
    rl::SwimmerStream stream;std::vector<float> observations(2*59,0),raw(8),targets(8,-.125f);observations[59+55]=1;
    for(int row=0;row<2;++row) for(int joint=0;joint<4;++joint) raw[row*4+joint]=.375f/(row?1.f:gains.value[joint]);
    device_ptr<float> obs(observations.size()),native_raw(raw.size()),label(8),means(8),loss(8),gradient(8);
    upload(obs,observations,stream.value);upload(native_raw,raw,stream.value);upload(label,targets,stream.value);
    for(int arm=0;arm<3;++arm) {
        sm::policy_means(native_raw.get(),obs.get(),means.get(),2,gains,stream.value);
        mse_gradient<<<1,32,0,stream.value>>>(means.get(),label.get(),obs.get(),2,arm,units,loss.get(),gradient.get());CUDA_CHECK_THROW(cudaGetLastError());
        sm::policy_mean_grad(gradient.get(),obs.get(),2,gains,stream.value);const auto actual=download(gradient.get(),8,stream.value);
        for(int row=0;row<2;++row) for(int joint=0;joint<4;++joint) {
            const bool active=arm==2 || (arm==0 && row==1) || (arm==1 && row==0);const double gain=row?1:gains.value[joint],unit=row?1:units.food[joint];
            const double center=raw[row*4+joint],epsilon=1e-6/gain;
            const auto objective=[&](double value) {const double error=(value*gain+.125)/unit;return active?error*error/8:0;};
            const double derivative=(objective(center+epsilon)-objective(center-epsilon))/(2*epsilon);
            need(std::abs(actual[row*4+joint]-derivative)<=2e-5*(1+std::abs(derivative)),"Physical MSE/gain chain finite difference failed");
            file<<arm<<','<<row<<','<<joint<<','<<gain<<','<<unit<<','<<actual[row*4+joint]<<','<<derivative<<'\n';
        }
    }
}
void dataset_guards(const Dataset& train,const Dataset& test,std::ofstream& checks) {
    std::unordered_set<uint64_t> train_conditions,test_conditions;
    for(const auto& row:train.rows) train_conditions.insert(row.condition);for(const auto& row:test.rows) test_conditions.insert(row.condition);
    for(uint64_t hash:train_conditions) need(test_conditions.count(hash)==0,"Train/test starting condition hash overlaps");
    struct Seen {const Dataset* dataset;size_t row;};std::unordered_map<uint64_t,std::vector<Seen>> seen;int exact_duplicates=0,cross_split_duplicates=0;
    for(const auto* data:{&train,&test}) for(size_t row=0;row<data->rows.size();++row) {
        const float* input=data->observations.data()+row*59;const auto hash=hash_data(input,59*sizeof(float));
        for(const auto& old:seen[hash]) {
            const float* previous=old.dataset->observations.data()+old.row*59;if(std::memcmp(input,previous,59*sizeof(float))!=0) continue;
            ++exact_duplicates;cross_split_duplicates+=data!=old.dataset;
            for(int joint=0;joint<4;++joint) {const float a=data->targets[row*4+joint],b=old.dataset->targets[old.row*4+joint];need(std::abs(a-b)<=1e-6f*(1+std::abs(a)),"Contradictory target for the same exact network input");}
        }
        seen[hash].push_back({data,row});
    }
    checks<<"{\"train_conditions\":"<<train_conditions.size()<<",\"test_conditions\":"<<test_conditions.size()<<",\"starting_condition_overlap\":0,\"exact_input_duplicates\":"<<exact_duplicates<<",\"cross_split_exact_input_duplicates\":"<<cross_split_duplicates<<",\"target_contradictions\":0}\n";
}
struct FitResult {std::string name;std::vector<float> parameters;};
FitResult fit_arm(FitArm arm,const rl::ShelterMotorRun& source,const Dataset& train,const std::vector<int>& indices,LossScale units,
                  const fs::path& output,std::vector<uint8_t>& common_initial,std::ofstream& progress) {
    rl::SwimmerStream stream;nn::GpuMlp model(rl::swimmer_specification(4,59),256,7);model.set_params(source.actor);nn::GpuAdam optimizer(model.num_params(),fit_optimizer());CUDA_CHECK_THROW(cudaDeviceSynchronize());
    checkpoint::Writer initial;initial.array(model.get_params());optimizer.save(initial);
    if(common_initial.empty()) common_initial=initial.data;else need(initial.data==common_initial,"Student/fresh Adam initial states differ across arms");need(optimizer.steps()==0,"Supervised Adam did not start fresh");
    device_ptr<float> all_obs(train.observations.size()),all_labels(train.targets.size()),obs(256*59),labels(256*4),means(256*4),loss(256*4),gradient(256*4);device_ptr<int> fixed(indices.size());
    upload(all_obs,train.observations,stream.value);upload(all_labels,train.targets,stream.value);upload(fixed,indices,stream.value);
    for(int step=0;step<fit_steps;++step) {
        gather_fit<<<(256*59+255)/256,256,0,stream.value>>>(all_obs.get(),all_labels.get(),fixed.get()+static_cast<size_t>(step)*256,obs.get(),labels.get());CUDA_CHECK_THROW(cudaGetLastError());
        model.forward(obs.get(),256,stream.value);sm::policy_means(model.output(),obs.get(),means.get(),256,source.scales,stream.value);
        mse_gradient<<<4,256,0,stream.value>>>(means.get(),labels.get(),obs.get(),256,static_cast<int>(arm),units,loss.get(),gradient.get());CUDA_CHECK_THROW(cudaGetLastError());
        sm::policy_mean_grad(gradient.get(),obs.get(),256,source.scales,stream.value);model.zero_grad(stream.value);model.backward(gradient.get(),256,nullptr,stream.value);optimizer.step(model.params(),model.grads(),stream.value);
        const float magnitude=optimizer.last_grad_norm();need(std::isfinite(magnitude),"Non-finite supervised gradient norm");
        if(step==0 || (step+1)%100==0) {const auto components=download(loss.get(),256*4,stream.value);double total=0;for(float value:components) total+=value;
            progress<<arm_name(arm)<<','<<step+1<<','<<total<<','<<magnitude<<','<<fminf(1.f,.5f/(magnitude+1e-6f))<<'\n';progress.flush();std::cout<<"HOLDING_FIT arm="<<arm_name(arm)<<" step="<<step+1<<"/1000 batch_loss_before_step="<<total<<'\n'<<std::flush;}
    }
    const auto parameters=model.get_params();finite(parameters,"fitted actor");need(optimizer.steps()==1000,"Unexpected supervised optimization budget");
    checkpoint::Writer saved;saved.value(uint32_t{0x48534631u});saved.value(uint32_t{1});saved.value(static_cast<int>(arm));saved.value(source.update);saved.value(source.source.fingerprint);saved.value(source.scales);saved.value(units);
    saved.array(parameters);std::vector<float> original_std(4);checkpoint::Reader source_reader{source.training};
    // Keep fixed original std as metadata; it is never fitted or used as a loss.
    (void)source_reader.value<uint32_t>();(void)source_reader.value<uint32_t>();(void)source_reader.value<uint32_t>();(void)source_reader.value<int>();(void)source_reader.value<uint32_t>();(void)source_reader.value<int>();(void)source_reader.value<sm::ActuatorScale>();
    (void)source_reader.array<float>(model.num_params());(void)source_reader.array<float>();original_std=source_reader.array<float>(4);saved.array(original_std);optimizer.save(saved);
    const auto path=output/(std::string(arm_name(arm))+"-supervised-state.bin");checkpoint::write_atomic(path.u8string(),saved.data);const auto bytes=checkpoint::read(path.u8string());need(bytes==saved.data,"Supervised state file readback differs");
    checkpoint::Reader input{bytes};need(input.value<uint32_t>()==0x48534631u && input.value<uint32_t>()==1,"Invalid supervised diagnostic format");
    (void)input.value<int>();(void)input.value<int>();(void)input.value<uint64_t>();(void)input.value<sm::ActuatorScale>();(void)input.value<LossScale>();const auto restored_parameters=input.array<float>(model.num_params());(void)input.array<float>(4);
    nn::GpuMlp restored(model.spec(),256,7);restored.set_params(restored_parameters);nn::GpuAdam restored_optimizer(model.num_params(),fit_optimizer());restored_optimizer.load(input);input.finish();CUDA_CHECK_THROW(cudaDeviceSynchronize());
    need(restored.get_params()==parameters && optimizer_bytes(restored_optimizer)==optimizer_bytes(optimizer),"Fresh supervised actor/Adam restore differs");return {arm_name(arm),parameters};
}
struct FitMetrics {bool food=true,cover=true;};
FitMetrics evaluate_data(const std::string& name,const char* split,const std::vector<float>& parameters,const Dataset& data,sm::ActuatorScale gains,LossScale units,std::ofstream& file) {
    struct Joint {double squared=0,normalized=0,clipped=0,max_error=0,max_clipped_error=0,max_physical=0,max_teacher=0,max_internal=0;};
    Joint sums[2][4];int count[2]{};rl::SwimmerStream stream;nn::GpuMlp model(rl::swimmer_specification(4,59),256,7);model.set_params(parameters);CUDA_CHECK_THROW(cudaDeviceSynchronize());
    for(size_t first=0;first<data.rows.size();first+=256) {
        const int rows=static_cast<int>(std::min<size_t>(256,data.rows.size()-first));device_ptr<float> input(static_cast<size_t>(rows)*59),means(static_cast<size_t>(rows)*4);
        std::vector<float> observations(data.observations.begin()+first*59,data.observations.begin()+(first+rows)*59);upload(input,observations,stream.value);
        model.forward(input.get(),rows,stream.value);sm::policy_means(model.output(),input.get(),means.get(),rows,gains,stream.value);
        const auto physical=download(means.get(),static_cast<size_t>(rows)*4,stream.value),internal=download(model.output(),static_cast<size_t>(rows)*4,stream.value);finite(physical,"evaluated physical means");
        for(int row=0;row<rows;++row) {
            const bool cover=data.rows[first+row].cover;need((observations[row*59+55]>=.5f)==cover,"Dataset mode metadata mismatch");++count[cover?1:0];
            for(int joint=0;joint<4;++joint) {auto& s=sums[cover?1:0][joint];const float predicted=physical[row*4+joint],target=data.targets[(first+row)*4+joint];
                const double error=static_cast<double>(predicted)-target,clipped=deep::clamp(predicted,-1,1)-deep::clamp(target,-1,1),unit=cover?1:units.food[joint];
                s.squared+=error*error;s.normalized+=(error/unit)*(error/unit);s.clipped+=clipped*clipped;s.max_error=std::max(s.max_error,std::abs(error));s.max_clipped_error=std::max(s.max_clipped_error,std::abs(clipped));
                s.max_physical=std::max(s.max_physical,double(std::abs(predicted)));s.max_teacher=std::max(s.max_teacher,double(std::abs(target)));s.max_internal=std::max(s.max_internal,double(std::abs(internal[row*4+joint])));
            }
        }
    }
    FitMetrics result;
    for(int mode=0;mode<2;++mode) {need(count[mode]>0,"Missing data mode at final evaluation");for(int joint=0;joint<4;++joint) {
        const auto& s=sums[mode][joint];const double raw=std::sqrt(s.squared/count[mode]),normalized=std::sqrt(s.normalized/count[mode]),clipped=std::sqrt(s.clipped/count[mode]);
        const bool pass=mode?(raw<=.01 && s.max_error<=.025):(raw<=.05 && normalized<=.01 && clipped<=.02);if(mode) result.cover=result.cover&&pass;else result.food=result.food&&pass;
        file<<name<<','<<split<<','<<(mode?"cover":"food")<<','<<joint<<','<<count[mode]<<','<<raw<<','<<normalized<<','<<clipped<<','<<s.max_error<<','<<s.max_clipped_error<<','<<s.max_internal<<','<<s.max_physical<<','<<s.max_teacher<<','<<pass<<'\n';
    }}return result;
}
struct EvalFrame {deep::BodyState body;Geometry geometry;float raw[4]{},physical[4]{},reward=0,cached_exposure=1;int clock=0,contacts=0;};
__global__ void record_eval(const sm::State* states,const float* raw,const float* means,const float* rewards,int count,EvalFrame* frames) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=count) return;auto& frame=frames[row];frame.body=states[row].state.pond.body;frame.geometry=geometry(states[row]);
    for(int joint=0;joint<4;++joint) {frame.raw[joint]=raw[row*4+joint];frame.physical[joint]=means[row*4+joint];if(!isfinite(frame.raw[joint]) || !isfinite(frame.physical[joint])) frame.geometry.bad|=16;}
    frame.reward=rewards?rewards[row]:0;frame.cached_exposure=states[row].exposure;frame.clock=states[row].state.pond.steps;frame.contacts=states[row].state.contacts;
    if(!isfinite(frame.reward)) frame.geometry.bad|=16;
}
struct SafetyResult {int heldout5=0,heldout10=0,nominal5=0,nominal10=0,invalid=0;bool pass() const {return heldout10==16 && nominal10==4 && invalid==0;}};
SafetyResult evaluate_safety(const std::string& name,const std::vector<float>& parameters,const rl::ShelterMotorRun& source,const std::vector<sm::State>& starts,std::ofstream& trace,std::ofstream& cases) {
    need(starts.size()==20,"Final closed-loop test needs16 heldout plus4 nominal starts");const int count=20;rl::SwimmerStream stream;sm::Batch batch(count,0x51FE2026u);
    auto context=source.source.context;context.current={};batch.configure(source.source.ancestry.policy.actor,context,stream.value,false);CUDA_CHECK_THROW(cudaDeviceSynchronize());batch.set_states(starts,stream.value);
    nn::GpuMlp model(rl::swimmer_specification(4,59),count,7);model.set_params(parameters);CUDA_CHECK_THROW(cudaDeviceSynchronize());
    device_ptr<float> means(count*4),rewards(count);device_ptr<EvalFrame> frames(201*count);
    model.forward(batch.obs(),count,stream.value);sm::policy_means(model.output(),batch.obs(),means.get(),count,source.scales,stream.value);
    record_eval<<<1,32,0,stream.value>>>(batch.states(),model.output(),means.get(),nullptr,count,frames.get());CUDA_CHECK_THROW(cudaGetLastError());
    for(int tick=0;tick<200;++tick) {
        model.forward(batch.obs(),count,stream.value);sm::policy_means(model.output(),batch.obs(),means.get(),count,source.scales,stream.value);
        batch.step(means.get(),{rewards.get(),nullptr,nullptr,nullptr},stream.value,false,false);
        record_eval<<<1,32,0,stream.value>>>(batch.states(),model.output(),means.get(),rewards.get(),count,frames.get()+static_cast<size_t>(tick+1)*count);CUDA_CHECK_THROW(cudaGetLastError());
    }
    const auto output=download(frames.get(),201*count,stream.value);need(model.get_params()==parameters,"Closed-loop frozen actor changed");SafetyResult result;
    for(int row=0;row<count;++row) {
        int first_failure=-1,first_mask=0,streak=0,longest=0;bool all_safe=true,valid=true;float minimum=1e10f,maximum_exposure=0;double total_reward=0,discounted_reward=0;
        for(int tick=0;tick<=200;++tick) {
            const auto& f=output[static_cast<size_t>(tick)*count+row];need(f.clock==tick,"Closed-loop test reset or changed its200-step clock");const auto& g=f.geometry;
            if(g.bad) {all_safe=false;if(first_failure<0) {first_failure=tick;first_mask=g.bad;}}valid=valid && !(g.bad&16);
            minimum=fminf(minimum,fminf(g.bed,fminf(g.solid,g.surface)));maximum_exposure=fmaxf(maximum_exposure,g.exposure);
            if(tick>0) {streak=g.bad==0?streak+1:0;longest=std::max(longest,streak);total_reward+=f.reward;discounted_reward+=std::pow(.99,tick-1)*f.reward;}
            trace<<name<<','<<row<<','<<(row<16?"heldout":"nominal")<<','<<tick<<','<<tick*deep::dt;
            for(float value:f.raw) trace<<','<<value;for(float value:f.physical) trace<<','<<value;for(float value:f.physical) trace<<','<<deep::clamp(value,-1,1);
            trace<<','<<f.body.position.x<<','<<f.body.position.y<<','<<f.body.position.z<<','<<f.body.orientation.x<<','<<f.body.orientation.y<<','<<f.body.orientation.z<<','<<f.body.orientation.w;
            for(float value:f.body.angle) trace<<','<<value;for(float value:f.body.velocity) trace<<','<<value;
            trace<<','<<g.exposure<<','<<g.head<<','<<g.proximal<<','<<g.distal<<','<<g.bed<<','<<g.solid<<','<<g.surface<<','<<g.bad<<','<<f.cached_exposure<<','<<f.reward<<','<<f.contacts<<'\n';
        }
        const bool five=longest>=100,ten=all_safe&&valid;result.invalid+=!valid;
        if(row<16) {result.heldout5+=five;result.heldout10+=ten;}else {result.nominal5+=five;result.nominal10+=ten;}
        cases<<name<<','<<row<<','<<(row<16?"heldout":"nominal")<<','<<first_failure<<','<<first_mask<<','<<longest*deep::dt<<','<<five<<','<<ten<<','<<minimum<<','<<maximum_exposure<<','<<total_reward<<','<<discounted_reward<<','<<valid<<'\n';
    }
    return result;
}
__global__ void finite_food_worlds(const sm::State* states,const float* actions,int count,int* failed) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=count) return;const auto& pond=states[row].state.pond;const auto& body=pond.body;bool valid=!pond.invalid && isfinite(deep::norm(body.position));
    const auto q=body.orientation;valid=valid&&isfinite(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);
    for(float value:body.angle) valid=valid&&isfinite(value);for(float value:body.velocity) valid=valid&&isfinite(value);
    for(int joint=0;joint<4;++joint) valid=valid&&isfinite(actions[row*4+joint]);
    for(const auto& food:pond.pellets) if(food.active) valid=valid&&isfinite(deep::norm(food.position))&&isfinite(deep::norm(food.velocity));
    if(!valid) failed[row]=1;
}
struct FoodResult {double rate=0;bool finite=true;};
FoodResult food_rate(const std::vector<float>& parameters,bool original55,const rl::ShelterMotorRun& source,uint32_t seed,float current) {
    constexpr int count=16;rl::SwimmerStream stream;sm::Batch batch(count,seed);auto context=source.source.context;context.current={current,0,0};
    batch.configure(source.source.ancestry.policy.actor,context,stream.value,false);CUDA_CHECK_THROW(cudaDeviceSynchronize());
    std::vector<sm::State> starts(count);for(int i=0;i<count;++i) starts[i].state=arch::initial(seed,static_cast<uint32_t>(i),0,false,{2});batch.set_states(starts,stream.value);
    nn::GpuMlp model(rl::swimmer_specification(4,original55?55:59),count,7);model.set_params(parameters);CUDA_CHECK_THROW(cudaDeviceSynchronize());
    device_ptr<float> obs55(count*55),means(count*4);device_ptr<int> failed(count);std::vector<int> zeros(count,0);upload(failed,zeros,stream.value);
    for(int tick=0;tick<600;++tick) {
        const float* input=batch.obs();if(original55) {first55<<<4,256,0,stream.value>>>(input,obs55.get(),count);CUDA_CHECK_THROW(cudaGetLastError());input=obs55.get();}
        model.forward(input,count,stream.value);const float* action=model.output();
        if(!original55) {sm::policy_means(action,batch.obs(),means.get(),count,source.scales,stream.value);action=means.get();}
        batch.step(action,{},stream.value,false,false);finite_food_worlds<<<1,32,0,stream.value>>>(batch.states(),action,count,failed.get());CUDA_CHECK_THROW(cudaGetLastError());
    }
    const auto states=batch.host_states(stream.value);const auto invalid=download(failed.get(),count,stream.value);FoodResult result;int foods=0;
    for(int i=0;i<count;++i) {foods+=states[i].state.pond.eaten;result.finite=result.finite && invalid[i]==0;need(states[i].state.pond.steps==600,"Food retention test reset or advanced an unexpected clock");}
    need(model.get_params()==parameters,"Frozen food evaluator modified its actor");result.rate=foods*60./(count*600*deep::dt);return result;
}
}

int main(int argc,char** argv) {
    using namespace cudarl;
    try {
        need(argc==4,"Usage: holding_supervised_fit <v5-update300-full-run> <original-boulder-full-save> <NEW-build-output>");
        const auto output=fs::weakly_canonical(fs::absolute(argv[3])),build=fs::weakly_canonical(fs::current_path()/"build");const auto relative=output.lexically_relative(build);
        need(!relative.empty() && relative!=fs::path(".") && *relative.begin()!=fs::path("..") && !fs::exists(output),"Use a NEW ignored output directory under repository build/");
        const auto original_path=fs::weakly_canonical(fs::absolute(argv[2]));
        need(original_path==fs::weakly_canonical(fs::current_path()/"build/predator-shelter/source-boulder.bin"),"Read the actual protected original build/predator-shelter/source-boulder.bin");
        const auto source_bytes=checkpoint::read(argv[1]),original_bytes=checkpoint::read(argv[2]);const auto source=rl::read_shelter_motor_run(source_bytes);const auto original=rl::read_boulder_source(original_bytes);
        need(rl::shelter_motor_protocol==5 && source.update==300,"Supervised diagnostic is preregistered for the selected protocol5/update300 student");
        need(original.parent==source.source.parent && original.motor==source.source.motor && original.fingerprint==source.source.fingerprint,"Independent original boulder teacher does not match the embedded protected source");
        const auto device=probe_device();need(device.available,"CUDA device/kernel probe failed: "+device.error);fs::create_directories(output);
        {
            std::ofstream plan(output/"protocol.json");plan<<"{\"student_protocol\":5,\"student_update\":300,\"arms\":[\"cover_only\",\"food_only\",\"joint\"],\"gradient_steps_per_arm\":1000,\"batch\":256,\"food_rows\":128,\"cover_rows\":128,\"index_seed\":"<<index_seed
                <<",\"loss\":\"sum active ((physical_mean-target)/task_scale)^2/(256*4); excluded rows zero; required food gain Jacobian unchanged\",\"food_task_scale\":\"max(1,TRAIN original55 teacher physical RMS per joint), frozen\",\"cover_task_scale\":1,\"fresh_adam\":{\"lr\":0.0003,\"beta1\":0.9,\"beta2\":0.999,\"eps\":0.00001,\"global_clip\":0.5},\"food_train_seed\":"<<food_train_seed<<",\"food_test_seed\":"<<food_test_seed
                <<",\"food_train_trajectories\":64,\"food_test_trajectories\":32,\"food_ticks\":600,\"food_harvest\":\"0,10,...590; trajectories split by independent seed, frames within trajectory correlated\",\"cover_train_seed\":"<<diagnostics::holding_fit_train_seed<<",\"cover_test_seed\":"<<diagnostics::holding_fit_test_seed
                <<",\"cover_candidate_pool_per_goal_and_split\":32,\"cover_train_starts\":32,\"cover_test_starts\":16,\"cover_label\":\"ONLY initial observation, physical zero, admitted after initial+all200 fresh native Batch zero-torque ticks pass whole-body safety\",\"heldout_student_evaluation\":\"final only, after all three fixed1000-step fits\",\"closed_loop\":\"16 heldout+4 familiar training nominal states, deterministic own physical means,200 actual Batch steps,noreset,nopractice\",\"safety_gate\":\"all20 safe continuously10s;5s separately reported\",\"failure_mask\":{\"1\":\"exposure>.05\",\"2\":\"bed margin<.08\",\"4\":\"solid margin<.08\",\"8\":\"surface margin<.08\",\"16\":\"nonfinite/invalid\"},\"trace_actions\":\"tick0 initial preview; tick>=1 raw/physical/clamped command that drove the preceding step, paired with resulting state and reward\",\"source_payload_fingerprint\":"<<hash_bytes(source_bytes)<<",\"original_payload_fingerprint\":"<<hash_bytes(original_bytes)<<",\"ppo_updates\":0,\"critic_learning\":false,\"std_learning\":false,\"adopted\":false}\n";need(bool(plan),"Could not write fixed supervised protocol");
        }
        std::ofstream admission(output/"cover-admission.csv");admission<<std::setprecision(12)<<"training,seed,trajectory,goal,admitted,reason,max_displacement,max_exposure,min_full_clearance,rejection_mask,proof_ticks\n";
        const auto train_cover=cover_data(source,true,admission),test_cover=cover_data(source,false,admission);
        double equivalence_maximum=0;const auto train_food=food_data(source,true,equivalence_maximum),test_food=food_data(source,false,equivalence_maximum);
        Dataset train=train_food;train.append(train_cover.data);Dataset test=test_food;test.append(test_cover.data);
        finite(train.observations,"training inputs");finite(train.targets,"training targets");finite(test.observations,"heldout inputs");finite(test.targets,"heldout targets");
        const auto units=loss_scale(train_food);const auto train_payload=train.bytes(),test_payload=test.bytes();
        checkpoint::write_atomic((output/"train-dataset.bin").u8string(),train_payload);checkpoint::write_atomic((output/"heldout-dataset.bin").u8string(),test_payload);
        need(checkpoint::read((output/"train-dataset.bin").u8string())==train_payload && checkpoint::read((output/"heldout-dataset.bin").u8string())==test_payload,"Checksummed dataset readback differs");
        std::ofstream checks(output/"dataset-checks.json");dataset_guards(train,test,checks);
        std::ofstream row_file(output/"dataset-rows.csv");row_file<<"split,row,mode,seed,trajectory,tick,goal,current_m_s,starting_condition_hash\n";
        for(const auto* data:{&train,&test}) for(size_t i=0;i<data->rows.size();++i) {const auto& r=data->rows[i];row_file<<(r.training?"train":"heldout")<<','<<i<<','<<(r.cover?"cover":"food")<<','<<r.seed<<','<<r.trajectory<<','<<r.tick<<','<<r.goal<<','<<r.current<<','<<r.condition<<'\n';}
        std::vector<int> indices(static_cast<size_t>(1000)*256);
        for(uint32_t step=0;step<1000;++step) for(uint32_t row=0;row<256;++row) indices[step*256+row]=row<128?static_cast<int>(rng::bits(index_seed,step,row)%train_food.rows.size()):
            static_cast<int>(train_food.rows.size()+rng::bits(index_seed,step,row)%train_cover.data.rows.size());
        checkpoint::Writer index_payload;index_payload.array(indices);checkpoint::write_atomic((output/"fixed-training-indices.bin").u8string(),index_payload.data);
        std::ofstream derivatives(output/"finite-difference.csv");derivatives<<std::setprecision(12)<<"arm,cover,joint,food_gain,task_scale,gpu_parameter_mean_gradient,central_finite_difference\n";finite_difference_check(source.scales,units,derivatives);
        std::ofstream progress(output/"training-progress.csv");progress<<std::setprecision(12)<<"arm,completed_steps,batch_loss_before_step,preclip_global_gradient_norm,clip_scale\n";
        const auto teacher59=rl::normalized_shelter_actor(rl::transfer_shelter_motor(original.motor),source.scales);
        std::vector<FitResult> policies{{"original_teacher_normalized59",teacher59},{"v5_update300",source.actor}};std::vector<uint8_t> common_initial;
        for(int arm=0;arm<3;++arm) policies.push_back(fit_arm(static_cast<FitArm>(arm),source,train,indices,units,output,common_initial,progress));
        checkpoint::write_atomic((output/"common-initial-student-and-fresh-adam.bin").u8string(),common_initial);
        // No fitted heldout prediction or closed-loop outcome was inspected
        // before all three fixed optimization budgets completed above.
        std::ofstream prediction(output/"prediction-errors.csv");prediction<<std::setprecision(12)<<"policy,split,mode,joint,rows,physical_unclipped_rmse,normalized_rmse,clipped_rmse,max_unclipped_error,max_clipped_error,max_internal_actor_output,max_physical_mean,max_teacher_physical_mean,fit_threshold_pass\n";
        std::vector<FitMetrics> heldout_metrics;
        for(const auto& policy:policies) {evaluate_data(policy.name,"train",policy.parameters,train,source.scales,units,prediction);heldout_metrics.push_back(evaluate_data(policy.name,"heldout",policy.parameters,test,source.scales,units,prediction));}
        std::vector<sm::State> safety_starts=test_cover.selected;
        for(int goal=0;goal<4;++goal) {
            const size_t row=static_cast<size_t>(goal)*8;need(train_cover.data.rows[row].trajectory==goal*32 && train_cover.selected[row].request.goal.cover==goal,"The familiar nominal start was not the qualified training candidate0");
            safety_starts.push_back(train_cover.selected[row]);
        }
        checkpoint::Writer start_payload;start_payload.value(uint32_t{sizeof(sm::State)});start_payload.array(safety_starts);checkpoint::write_atomic((output/"final-closed-loop-starts.bin").u8string(),start_payload.data);
        std::ofstream trajectory(output/"closed-loop-trace.csv"),case_file(output/"closed-loop-cases.csv");trajectory<<std::setprecision(12);case_file<<std::setprecision(12);
        trajectory<<"policy,case,suite,tick,time_s";for(const char* kind:{"raw","physical","applied_clamped"}) for(int j=0;j<4;++j) trajectory<<','<<kind<<j;
        trajectory<<",x,y,z,qx,qy,qz,qw";for(int j=0;j<4;++j) trajectory<<",joint"<<j;for(int j=0;j<10;++j) trajectory<<",velocity"<<j;
        trajectory<<",exposure,head_exposure,proximal_exposure,distal_exposure,bed_margin,solid_margin,surface_margin,failure_mask,cached_exposure,actual_previous_step_reward,obstacle_contacts\n";
        case_file<<"policy,case,suite,first_failure_step,first_failure_mask,longest_safe_seconds,five_second_hold,full_ten_seconds,min_full_clearance,max_exposure,return_sum,discounted_return,finite\n";
        std::vector<SafetyResult> safety;
        for(const auto& policy:policies) {std::cout<<"HOLDING_FIT_CLOSED_LOOP policy="<<policy.name<<" fitted_updates=0\n"<<std::flush;safety.push_back(evaluate_safety(policy.name,policy.parameters,source,safety_starts,trajectory,case_file));}
        std::ofstream food_file(output/"food-retention.csv");food_file<<std::setprecision(12)<<"policy,seed,current_m_s,food_per_minute,original_teacher_food_per_minute,retained_ratio,finite,retained90percent\n";
        std::array<double,4> reference_rates{};std::vector<bool> retained(policies.size(),true);
        for(size_t policy_index=0;policy_index<policies.size();++policy_index) for(int cell=0;cell<4;++cell) {
            const auto seed=cell<2?food_old_seed:food_fresh_seed;const float current=cell%2?.0075f:0;
            const auto result=food_rate(policy_index==0?original.motor:policies[policy_index].parameters,policy_index==0,source,seed,current);
            if(policy_index==0) {need(result.finite,"Independent original food teacher failed finite playback");reference_rates[cell]=result.rate;}
            const double ratio=result.rate/std::max(1e-12,reference_rates[cell]);const bool pass=result.finite && ratio>=.9;retained[policy_index]=retained[policy_index]&&pass;
            food_file<<policies[policy_index].name<<','<<seed<<','<<current<<','<<result.rate<<','<<reference_rates[cell]<<','<<ratio<<','<<result.finite<<','<<pass<<'\n';
        }
        std::ofstream results(output/"final-results.csv");results<<"policy,heldout_food_fit,heldout_cover_fit,heldout5of16,heldout10of16,nominal5of4,nominal10of4,invalid_cases,all20safe10s,food_retention_all4cells,combined_diagnostic_gate,adopted\n";
        for(size_t i=0;i<policies.size();++i) {const auto& s=safety[i];const auto& fit=heldout_metrics[i];results<<policies[i].name<<','<<fit.food<<','<<fit.cover<<','<<s.heldout5<<','<<s.heldout10<<','<<s.nominal5<<','<<s.nominal10<<','<<s.invalid<<','<<s.pass()<<','<<retained[i]<<','<<(fit.food&&fit.cover&&s.pass()&&retained[i])<<",0\n";}
        need(checkpoint::read(argv[1])==source_bytes && checkpoint::read(argv[2])==original_bytes,"Protected selected student or independently read original teacher changed");
        std::ofstream verification(output/"verification.json");verification<<std::setprecision(17)<<"{\"completed\":true,\"arms\":3,\"gradient_steps_each\":1000,\"student_and_zero_adam_initial_bytes_identical\":true,\"fresh_supervised_actor_optimizer_readback_identical\":true,\"same_batch_indices_each_arm\":true,\"index_payload_fingerprint\":"<<hash_bytes(index_payload.data)<<",\"train_payload_fingerprint\":"<<hash_bytes(train_payload)<<",\"heldout_payload_fingerprint\":"<<hash_bytes(test_payload)
            <<",\"train_rows\":"<<train.rows.size()<<",\"heldout_rows\":"<<test.rows.size()<<",\"raw_start_state_bytes\":"<<sizeof(sm::State)<<",\"teacher55_normalized59_max_absolute_error\":"<<equivalence_maximum<<",\"food_task_scales\":["<<units.food[0]<<','<<units.food[1]<<','<<units.food[2]<<','<<units.food[3]
            <<"],\"independent_original_teacher_identity_verified\":true,\"source_files_unchanged\":true,\"physical_MSE_gain_finite_difference_passed\":true,\"heldout_student_evaluation_final_only\":true,\"closed_loop_deterministic_physical_means\":true,\"supervised_files_are_not_PPO_resume_files\":true,\"critic_learning\":false,\"std_learning\":false,\"ppo_updates\":0,\"production_files_modified\":false,\"adopted\":false}\n";
        for(auto* file:{&admission,&checks,&row_file,&derivatives,&progress,&prediction,&trajectory,&case_file,&food_file,&results,&verification}) {file->flush();need(bool(*file),"Could not finish supervised diagnostic artifacts");}
        std::cout<<"HOLDING_SUPERVISED_FIT_DONE steps=3000 sources_unchanged=1 PPO_updates=0 adopted=0 output="<<output.u8string()<<'\n';return 0;
    } catch(const std::exception& error) {std::cerr<<"HOLDING_SUPERVISED_FIT_ERROR "<<error.what()<<'\n';return 1;}
}

// Reuse the previously qualified CUDA rollout, contact, request and evidence
// routines in their original TU, including its fixed floating-point profile.
// Only the residual CLI below is reachable from this executable's entry point.
// The physics math profile must precede all runtime and observation headers.
#include "cuda/pond_math_profile.h"
#include "staged_schedule.h"
// Select the residual request adapter only for the shared rollout TU below.
#define pond_ppo_schedule residual_turn_schedule
#define main archived_full_motor_ppo_entry
#include "../pond-ppo/train.cu"
#undef main
#undef pond_ppo_schedule
#include "contract.h"
#include "residual_identity.h"
#include <memory>
#include <condition_variable>
#include <thread>
#include <mutex>

namespace residual_training {
namespace rp=cudarl::residual_ppo;
constexpr int Y=2, W=42;
constexpr uint32_t program=pt::signed_half_second_v1;
constexpr char classified_id[]="pond-motor-residual-classified-v2-2026-11-24";
constexpr float actor_lr=.0003f,critic_lr=.001f;
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
struct Args{bool init=false,classified=false;fs::path plan,out,resume,receipt;};
Args args(int argc,char** argv){
    Args a;require(argc>=2,"Expected --initialize-only or --run");
    a.init=std::string(argv[1])=="--initialize-only";
    require(a.init||std::string(argv[1])=="--run","Unknown residual operation");
    for(int i=2;i<argc;++i){const std::string k=argv[i];
        if(k=="--classified-clearance"){require(!a.classified,"Duplicate clearance mode");a.classified=true;continue;}
        require(++i<argc,"Missing option value");const fs::path v=argv[i];
        if(k=="--protocol")a.plan=v;else if(k=="--out")a.out=v;else if(k=="--resume")a.resume=v;else if(k=="--retention-receipt")a.receipt=v;
        else throw std::runtime_error("Unknown option "+k);}
    require(!a.plan.empty()&&!a.out.empty()&&a.init==a.resume.empty()&&a.init==a.receipt.empty(),"Invalid stage inputs");
    return a;
}
struct Study{
    std::string doc;uint64_t plan=0,evaluator=0;fs::path ledger,parent;Package motor;std::vector<float> critic;
    explicit Study(const Args& a){
        auto bytes=whole_file(a.plan);plan=cp::checksum(bytes);doc={bytes.begin(),bytes.end()};
        auto field=[&](const char* k){return scalar(doc,k);};
        auto bound=[&](const char* pk,const char* hk){const fs::path f=field(pk);require(f.is_absolute()&&sha256(whole_file(f))==field(hk),"Binding changed: "+std::string(pk));return f;};
        require(field("protocol_id")== (a.classified?classified_id:rp::protocol_id)&&field("source_sha256")==residual_source_hash&&field("max_updates")=="32"&&
                field("worlds")=="512"&&field("rollout_controls")=="64"&&std::stod(field("actor_learning_rate"))==.0003&&
                std::stod(field("critic_learning_rate"))==.001,"Residual protocol differs");
        evaluator=std::stoull(field("retention_evaluator_source_fnv1a64"));ledger=field("ledger_path_absolute");
        require(evaluator&&ledger.is_absolute(),"Missing evaluator or allocation ledger");
        parent=field("parent_policy_path_absolute");motor=load_package(parent);
        const auto cpu=bound("cpu_gate_path_absolute","cpu_gate_sha256"),cuda=bound("cuda_gate_path_absolute","cuda_gate_sha256");
        ph::validate_gate_documents(text_file(cpu),text_file(cuda),motor.checksum,cp::checksum(whole_file(cpu)));
        const auto prior=bound("critic_source_protocol_path_absolute","critic_source_protocol_sha256");
        const auto checkpoint=bound("critic_source_candidate_path_absolute","critic_source_candidate_sha256");
        require(field("critic_source_candidate_sha256")=="9b68920b345b4be6699f8b979729e07d9f9d7a451664e4ca99f34487f2ca4ded"&&
                field("critic_source_protocol_sha256")=="ee7d102f0f13222f1128aa603bcd7bfa36d5082293556f81f57fb788ed2c7eae","Require verified v6 critic");
        if(a.classified){
            require(field("max_process_wall_seconds")=="4"&&field("classified_physics_source_sha256")==ph::physics_source_sha256(),"Classified source or wall cap mismatch");
            const auto physical=text_file(bound("classified_dynamics_path_absolute","classified_dynamics_sha256"));
            const auto collection=text_file(bound("classified_collection_path_absolute","classified_collection_sha256"));
            const auto physics_binding=scalar(physical,"bindings"),collection_binding=scalar(collection,"bindings");
            require(scalar(physical,"passed")=="true"&&scalar(physics_binding,"protocol")=="pond-broadphase-dynamics-v1"&&
                    scalar(physics_binding,"runtime_sha256")==residual_runtime_hash&&scalar(physics_binding,"sweep_sha256")==residual_sweep_hash&&
                    scalar(physical,"gpu_controls")=="2056"&&scalar(physical,"cpu_controls")=="8","Classified dynamics receipt mismatch");
            require(scalar(collection,"passed")=="true"&&scalar(collection_binding,"protocol")=="pond-collection-replay-v2"&&
                    scalar(collection_binding,"optimizer_calls")=="0"&&
                    scalar(collection_binding,"source_sha256")==residual_source_hash&&scalar(collection,"physical_controls")=="32768"&&
                    scalar(collection,"physical_substeps")=="327680"&&scalar(collection,"completed_updates")=="0","Classified collection receipt mismatch");
        }else require(scalar(text_file(prior),"qualified_physics_source_sha256")==ph::physics_source_sha256(),"Qualified physical source changed");
        const auto imported=pp::read_candidate(cp::read(checkpoint.string()),motor.checksum,cp::checksum(whole_file(prior)),
            std::stoull(scalar(text_file(prior),"source_identity_fnv1a64")),sizeof(pr::World),sizeof(ps::SignedCourseTask),W);
        require(imported.counters.completed_updates==8,"Wrong auxiliary critic");critic=imported.critic;
    }
};
struct Extra{
    device_ptr<float> action{Rows*Y},mask{Rows},base{Rows*A},mb_action,mb_mask,mb_kl,pack;
    device_ptr<int> eligible;
    explicit Extra(BufferUse use=BufferUse::Training){
        if(use==BufferUse::Training){
            mb_action.allocate(MB*Y);mb_mask.allocate(MB);mb_kl.allocate(MB);pack.allocate(Rows);eligible.allocate(Rows);
        }
    }
};
__global__ void compose(const pt::Task* tasks,const float* base,float* actions,float* logp,float* mask,float* combined,float* base_history){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
    const bool active=tasks[i].kind==pt::Turn&&tasks[i].request_changed;
    mask[i]=active?1.f:0.f;
    if(!active){actions[i*2]=actions[i*2+1]=0;logp[i]=0;}
    for(int j=0;j<4;++j){
        base_history[i*4+j]=base[i*4+j];
        combined[i*4+j]=active&&j%2==0?base[i*4+j]+actions[i*2+j/2]:base[i*4+j];
    }
}
__global__ void likelihood(const float* means,const float* logs,const float* actions,const float* old,const float* mask,
                           float* kl,int n,bool check_old,int* fault){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    if(!mask[i]){kl[i]=0;if(old[i]!=0||actions[i*2]!=0||actions[i*2+1]!=0)fail(fault,i);return;}
    const float delta=rl::gaussian_logp(means+i*Y,logs,actions+i*Y,Y)-old[i];
    kl[i]=expf(delta)-1.f-delta;
    if(!p::finite_number(kl[i])||(check_old&&fabsf(delta)>1e-4f))fail(fault,i);
}
__global__ void pack(const float* values,const int* indices,int n,float* dest){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dest[i]=values[indices[i]];
}
__global__ void loss(const float* means,const float* logs,const float* actions,const float* old,const float* masks,
                     const float* advantages,const float* adv_stats,const float* values,const float* returns,float actor_scale,
                     float* dy,float* dv,float* kl,int* fault){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=MB)return;
    const float v=values[i],target=returns[i];
    dv[i]=.5f/MB*(v-target);dy[i*2]=dy[i*2+1]=0;kl[i]=0;
    if(!masks[i])return;
    const float adv=(advantages[i]-adv_stats[0])/(adv_stats[1]+1e-8f);
    float scratch[Y]{},unused=0;
    const auto terms=rl::ppo_gaussian_row(means+i*Y,logs,Y,actions+i*Y,old[i],adv,v,target,
                   {pp::clip,0,.5f},actor_scale,dy+i*Y,scratch,&unused);
    kl[i]=terms.approx_kl;
    if(!p::finite_number(kl[i])||!p::finite_number(dy[i*Y])||!p::finite_number(dy[i*Y+1]))fail(fault,i);
}
nn::AdamConfig adam_config(float lr){nn::AdamConfig cfg;cfg.lr=lr;cfg.max_grad_norm=1;return cfg;}
nn::MlpSpec residual_spec(){return {{W,16,Y},{nn::Activation::Tanh,nn::Activation::Identity}};}
struct Networks{
    nn::GpuMlp base{actor_spec(D),MB,default_seed},residual{residual_spec(),MB,default_seed+7},critic{critic_spec(),MB,default_seed+1};
    nn::GpuAdam actor_adam{int(rp::parameters),adam_config(actor_lr)},critic_adam{pp::critic_weights,adam_config(critic_lr)};
    explicit Networks(const Study& study){
        base.set_params(study.motor.weights);critic.set_params(study.critic);
        auto params=residual.get_params();std::fill(params.begin()+rp::output_offset,params.end(),0.f);residual.set_params(params);
    }
};
void write_state(const fs::path& out,const rp::Actor& a,Networks& net,Buffer& b,cudaStream_t stream){
    const auto actor=rp::write_actor(a);cp::write_atomic((out/("residual-"+std::to_string(a.update)+".bin")).string(),actor);
    cp::Writer w,am,cm;w.value(uint32_t{0x31534c52});w.value(uint32_t{1});w.array(actor);w.array(net.critic.get_params());
    net.actor_adam.save(am);net.critic_adam.save(cm);w.array(am.data);w.array(cm.data);
    w.array(raw_bytes(download(b.worlds,N,stream)));w.array(saved_tasks(b,program,stream));
    cp::write_atomic((out/("state-"+std::to_string(a.update)+".bin")).string(),w.data);
}
void restore_adam(nn::GpuAdam& adam,const std::vector<uint8_t>& bytes,float lr,int age,int count){
    cp::Reader verify{bytes};const auto c=verify.value<nn::AdamConfig>();const auto expected=adam_config(lr);
    require(c.lr==expected.lr&&c.beta1==expected.beta1&&c.beta2==expected.beta2&&c.eps==expected.eps&&c.max_grad_norm==1&&
            verify.value<int>()==age,"Optimizer age/config mismatch");
    for(int j=0;j<2;++j){const auto moment=verify.array<float>(count);require(moment.size()==size_t(count),"Moment shape");}
    verify.finish();cp::Reader r{bytes};adam.load(r);r.finish();
}
rp::Actor restore_state(const std::vector<uint8_t>& bytes,const Study& study,Networks& net,Buffer& b){
    cp::Reader r{bytes};require(r.value<uint32_t>()==0x31534c52&&r.value<uint32_t>()==1,"Unknown residual state");
    auto a=rp::read_actor(r.array<uint8_t>(16384),study.motor.checksum,study.plan);
    const auto critic=r.array<float>(pp::critic_weights);require(critic.size()==pp::critic_weights&&
        std::all_of(critic.begin(),critic.end(),[](float x){return std::isfinite(x);}),"Invalid critic");
    net.residual.set_params(a.weights);net.critic.set_params(critic);
    restore_adam(net.actor_adam,r.array<uint8_t>(pp::maximum_adam_bytes),actor_lr,int(a.actor_adam_calls),rp::parameters);
    restore_adam(net.critic_adam,r.array<uint8_t>(pp::maximum_adam_bytes),critic_lr,int(a.critic_adam_calls),pp::critic_weights);
    restore(b.worlds,r.array<uint8_t>(pp::maximum_state_bytes));restore_tasks(b,r.array<uint8_t>(pp::maximum_state_bytes),program);r.finish();return a;
}
struct TimeBudget{
    double prior=0;Clock::time_point start=Clock::now();
    void guard()const{require(prior+elapsed(start)<pp::maximum_wall_seconds-10,"Residual wall budget exhausted");}
};
struct StageAccounting{int update=0,ticks=0;uint64_t controls=0,substeps=0;bool accounted=false;};
void clear_batch(Buffer& b,Extra& e,cudaStream_t stream){
    for(auto* a:{&b.observation,&b.actor_observation,&b.latent,&b.logp,&b.value,&b.reward,&b.done,&b.truncated,
                 &b.final_value,&b.last_value,&e.mask,&e.action,&e.base})CUDA_CHECK_THROW(cudaMemsetAsync(a->get(),0,a->size_bytes(),stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(b.attempted.get(),0,b.attempted.size_bytes(),stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(b.substeps.get(),0,b.substeps.size_bytes(),stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(b.reason.get(),0,b.reason.size_bytes(),stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(b.trace.get(),0,b.trace.size_bytes(),stream));clear_fault(b,stream);
}
void check_good(Buffer& b,cudaStream_t s,const char* error){check_cuda();require(fault(b,s)==INT_MAX,error);}
struct CollectionTiming {
    double inference_ms=0;
    // A bounded replay may publish each issued tick before the GPU is submitted.
    void (*on_issue)(int)=nullptr;
    void (*on_complete)(int)=nullptr;
};
template<class Models,class Admission>
float collect(Buffer& b,Extra& e,Models& net,const Admission& study,StageAccounting& a,cudaStream_t s,const TimeBudget& budget,
        pr::ControlClearance* classes=nullptr,CollectionTiming* timings=nullptr){
    float physics_ms=0;Timer physics;
    std::unique_ptr<Timer> inference;
    if(timings)inference=std::make_unique<Timer>();
    auto mark_inference_end=[&](){if(inference)CUDA_CHECK_THROW(cudaEventRecord(inference->last,s));};
    auto account_inference=[&](){if(inference){float ms=0;CUDA_CHECK_THROW(cudaEventElapsedTime(&ms,inference->first,inference->last));timings->inference_ms+=ms;}};
    for(int t=0;t<T;++t){
        budget.guard();const int row=t*N;float* normal=b.observation.get()+row*D;float* extended=b.actor_observation.get()+row*W;
        observe_kernel<<<(N+31)/32,32,0,s>>>(b.worlds.get(),b.tasks.get(),b.schedules.get(),normal,extended,b.movement.get(),b.fault.get(),program,default_seed,true);
        check_good(b,s,"Invalid pre-control state/observation");
        if(inference)inference->start(s);
        net.base.forward(normal,N,s);net.residual.forward(extended,N,s);net.critic.forward(normal,N,s);
        CUDA_CHECK_THROW(cudaMemcpyAsync(b.value.get()+row,net.critic.output(),N*sizeof(float),cudaMemcpyDeviceToDevice,s));
        rl::sample_gaussian(net.residual.output(),b.log_std.get(),N,Y,default_seed,(a.update-1)*T+t,e.action.get()+row*Y,b.logp.get()+row,s);
        compose<<<(N+31)/32,32,0,s>>>(b.tasks.get(),net.base.output(),e.action.get()+row*Y,b.logp.get()+row,e.mask.get()+row,b.latent.get()+row*A,e.base.get()+row*A);
        mark_inference_end();
        if(timings&&timings->on_issue)timings->on_issue(t+1);
        physics.start(s);a.ticks=t+1;
        step_kernel<<<N/8,8,0,s>>>(b.worlds.get(),b.tasks.get(),b.schedules.get(),b.histories.get(),b.movement.get(),b.latent.get()+row*A,study.motor.gain,
            b.reward.get()+row,b.done.get()+row,b.truncated.get()+row,b.final_observation.get(),b.attempted.get()+row,b.substeps.get()+row,b.reason.get()+row,
            b.trace.get()+row,b.fault.get(),program,default_seed,classes?classes+row:nullptr);
        physics_ms+=physics.stop(s);check_good(b,s,"Physical collection failed before GAE or Adam");
        account_inference();
        if(timings&&timings->on_complete)timings->on_complete(t+1);
        if(inference)inference->start(s);
        net.critic.forward(b.final_observation.get(),N,s);
        CUDA_CHECK_THROW(cudaMemcpyAsync(b.final_value.get()+row,net.critic.output(),N*sizeof(float),cudaMemcpyDeviceToDevice,s));
        mark_inference_end();
        reset_terminal_kernel<<<(N+31)/32,32,0,s>>>(b.worlds.get(),b.tasks.get(),b.schedules.get(),b.histories.get(),b.movement.get(),b.done.get()+row,default_seed,program,b.fault.get());
        check_good(b,s,"Invalid episode reset/certificate");
        account_inference();
    }
    observe_kernel<<<(N+31)/32,32,0,s>>>(b.worlds.get(),b.tasks.get(),b.schedules.get(),b.last_observation.get(),nullptr,nullptr,b.fault.get(),program,default_seed,false);
    if(inference)inference->start(s);
    net.critic.forward(b.last_observation.get(),N,s);
    CUDA_CHECK_THROW(cudaMemcpyAsync(b.last_value.get(),net.critic.output(),N*sizeof(float),cudaMemcpyDeviceToDevice,s));
    mark_inference_end();
    audit_kernel<<<(Rows+255)/256,256,0,s>>>(b.observation.get(),b.latent.get(),b.logp.get(),b.value.get(),b.reward.get(),b.done.get(),b.truncated.get(),b.final_value.get(),b.fault.get());
    finite_kernel<<<(N+255)/256,256,0,s>>>(b.last_value.get(),N,b.fault.get());check_good(b,s,"Invalid collection data");account_inference();return physics_ms;
}
struct Source{std::vector<uint8_t> worlds,tasks;std::vector<float> actor,critic;};
std::vector<int> save_rollout(const fs::path& out,const Source& origin,Buffer& b,Extra& e,StageAccounting& a,cudaStream_t s,bool success,
                              const device_ptr<pr::ControlClearance>* classes=nullptr){
    const auto h=batch(b,s,a.ticks);const size_t count=h.trace.size();
    const auto residual=download(e.action,count*Y,s),mask=download(e.mask,count,s),base=download(e.base,count*A,s);
    a.controls=0;a.substeps=0;for(size_t i=0;i<count;++i){a.controls+=h.attempted[i];a.substeps+=h.substeps[i];}
    a.accounted=true;
    cp::Writer w;w.value(uint32_t{0x31425252});w.value(uint32_t(classes?2:1));w.value(a.update);w.value(a.ticks);
    w.array(origin.worlds);w.array(origin.tasks);w.array(origin.actor);w.array(origin.critic);
    w.array(h.observation);w.array(residual);w.array(mask);w.array(base);w.array(h.latent);w.array(h.logp);
    w.array(h.value);w.array(h.reward);w.array(h.done);w.array(h.truncated);w.array(h.final_value);w.array(h.last_value);
    w.array(h.attempted);w.array(h.substeps);w.array(h.reason);w.value(uint32_t(sizeof(TransitionTrace)));w.array(raw_bytes(h.trace));
    if(classes){
        // Version 2 appends an explicit bound/precision record for every row.
        // The legacy trace minimum is NaN wherever a numerical minimum was unmeasured.
        const auto records=download(*classes,count,s);w.value(uint32_t{1});w.value(uint64_t(records.size()));
        for(const auto& c:records){w.value(uint32_t(c.precise));w.value(c.lower_m);w.value(c.rejected_contacts);w.value(c.exact_steps);w.value(c.exact_queries);}
    }
    cp::write_atomic((out/((success?"batch-":"partial-batch-")+std::to_string(a.update)+".bin")).string(),w.data);
    std::vector<int> indices;if(!success)return indices;
    uint32_t starts[4][2]{},ticks[4][2]{};
    require(a.ticks==T&&a.controls==Rows&&a.substeps==10*Rows,"Incomplete physical rollout");
    for(size_t i=0;i<count;++i){
        require(h.attempted[i]==1&&h.substeps[i]==10&&(mask[i]==0||mask[i]==1),"Invalid collected row");
        if(mask[i]){
            const auto& trace=h.trace[i];
            require(trace.kind==pt::Turn&&trace.request_serial>0,"Residual enabled outside turn");indices.push_back(int(i));
            const float degrees=std::abs(trace.requested_angle)*180.f/po::pi;
            const int stage=degrees<22.5f?0:(degrees<45.f?1:(degrees<75.f?2:3));
            const int magnitudes[4]={15,30,60,90},side=trace.requested_angle>0?1:0;
            require(std::abs(degrees-magnitudes[stage])<.001f,"Unregistered training turn angle");
            ++ticks[stage][side];
            if(trace.events&ps::Onset)++starts[stage][side];
        }
        for(int j=0;j<A;++j){
            const float expected=mask[i]&&j%2==0?base[i*A+j]+residual[i*Y+j/2]:base[i*A+j];
            require(std::isfinite(expected)&&h.latent[i*A+j]==expected,"Frozen base/yaw composition changed");}
    }
    require(!indices.empty(),"Rollout contained no eligible turn actions");
    std::ofstream angles(out/("angles-"+std::to_string(a.update)+".csv"));
    angles<<"signed_degrees,starts,active_controls\n";
    const int magnitudes[4]={15,30,60,90};
    for(int stage=0;stage<4;++stage)for(int side=0;side<2;++side)
        angles<<(side?1:-1)*magnitudes[stage]<<','<<starts[stage][side]<<','<<ticks[stage][side]<<'\n';
    angles.close();require(bool(angles),"Turn-angle evidence write failed");return indices;
}
void audit_policy(Buffer& b,Extra& e,Networks& net,bool old,cudaStream_t s){
    for(int start=0;start<Rows;start+=MB){
        net.residual.forward(b.actor_observation.get()+start*W,MB,s);
        likelihood<<<(MB+255)/256,256,0,s>>>(net.residual.output(),b.log_std.get(),e.action.get()+start*Y,b.logp.get()+start,e.mask.get()+start,b.kl.get()+start,MB,old,b.fault.get());
    }
    check_good(b,s,"Masked residual log probability replay failed");
}
void finite(const float* ptr,int n,Buffer& b,cudaStream_t s){finite_kernel<<<(n+255)/256,256,0,s>>>(ptr,n,b.fault.get());}
float optimize(Buffer& b,Extra& e,Networks& net,int update,const std::vector<int>& indices,cudaStream_t s,const TimeBudget& budget){
    audit_policy(b,e,net,true,s);
    rl::compute_gae(T,N,b.reward.get(),b.value.get(),b.done.get(),b.truncated.get(),b.final_value.get(),b.last_value.get(),pp::gamma,pp::lambda,b.advantages.get(),b.returns.get(),s);
    finite(b.advantages.get(),Rows,b,s);finite(b.returns.get(),Rows,b,s);check_good(b,s,"Invalid returns/advantages before Adam");
    const int active=static_cast<int>(indices.size());
    CUDA_CHECK_THROW(cudaMemcpyAsync(e.eligible.get(),indices.data(),indices.size()*sizeof(int),cudaMemcpyHostToDevice,s));
    pack<<<(active+255)/256,256,0,s>>>(b.advantages.get(),e.eligible.get(),active,e.pack.get());
    rl::mean_std(e.pack.get(),active,b.adv_stats.get(),s);
    // Normalize policy loss across eligible turn actions only. The critic uses
    // all 512 physical worlds, including straight, settling and rest samples.
    for(uint32_t epoch=0;epoch<pp::epochs;++epoch){
        const auto perm=rl::make_permutation(Rows,default_seed^(update*1664525u+epoch*1013904223u));
        for(int first=0;first<Rows;first+=MB){
            budget.guard();auto gather=[&](int width,const float* src,float* dst){rl::gather_rows(perm,first,MB,width,src,dst,s);};
            gather(W,b.actor_observation.get(),b.mb_actor_observation.get());gather(D,b.observation.get(),b.mb_observation.get());
            gather(Y,e.action.get(),e.mb_action.get());gather(1,e.mask.get(),e.mb_mask.get());gather(1,b.logp.get(),b.mb_logp.get());
            gather(1,b.returns.get(),b.mb_return.get());gather(1,b.advantages.get(),b.mb_advantage.get());
            net.residual.forward(b.mb_actor_observation.get(),MB,s);net.critic.forward(b.mb_observation.get(),MB,s);
            loss<<<(MB+255)/256,256,0,s>>>(net.residual.output(),b.log_std.get(),e.mb_action.get(),b.mb_logp.get(),e.mb_mask.get(),b.mb_advantage.get(),
                b.adv_stats.get(),net.critic.output(),b.mb_return.get(),float(Rows)/(MB*float(active)),b.dmean.get(),b.dvalue.get(),e.mb_kl.get(),b.fault.get());
            rl::mean_std(e.mb_kl.get(),MB,b.stats.get(),s);rl::mean_std(e.mb_mask.get(),MB,b.stats.get()+2,s);
            const auto stats=download(b.stats,4,s);require(stats[2]>0&&std::isfinite(stats[0])&&stats[0]/stats[2]<=maximum_approx_kl,"Predeclared eligible-row minibatch KL stop");
            net.residual.zero_grad(s);net.critic.zero_grad(s);net.residual.backward(b.dmean.get(),MB,nullptr,s);net.critic.backward(b.dvalue.get(),MB,nullptr,s);
            finite(net.residual.grads(),rp::parameters,b,s);finite(net.critic.grads(),pp::critic_weights,b,s);check_good(b,s,"Nonfinite gradient before Adam");
            net.actor_adam.step(net.residual.params(),net.residual.grads(),s);net.critic_adam.step(net.critic.params(),net.critic.grads(),s);
            finite(net.residual.params(),rp::parameters,b,s);finite(net.critic.params(),pp::critic_weights,b,s);check_good(b,s,"Nonfinite updated weights");
        }
    }
    audit_policy(b,e,net,false,s);pack<<<(active+255)/256,256,0,s>>>(b.kl.get(),e.eligible.get(),active,e.pack.get());
    rl::mean_std(e.pack.get(),active,b.stats.get(),s);const float kl=download(b.stats,2,s)[0];
    require(std::isfinite(kl)&&kl<=maximum_approx_kl,"Final whole-rollout eligible-row KL stop");return kl;
}
struct Ledger{
    uint64_t protocol=0,state=0;uint32_t status=0,update=0;double wall=0;
    std::vector<uint8_t> bytes()const{cp::Writer w;w.value(uint32_t{0x314c5252});w.value(protocol);w.value(state);w.value(status);w.value(update);w.value(wall);return w.data;}
    static Ledger read(const fs::path& path){
        const auto bytes=cp::read(path.string());cp::Reader r{bytes};require(r.value<uint32_t>()==0x314c5252,"Residual study ledger format");
        Ledger l;l.protocol=r.value<uint64_t>();l.state=r.value<uint64_t>();l.status=r.value<uint32_t>();l.update=r.value<uint32_t>();l.wall=r.value<double>();r.finish();
        require(l.update<=32&&std::isfinite(l.wall)&&l.wall>=0,"Invalid residual study ledger");return l;
    }
};
void run(const Args& args){
    TimeBudget budget;Study study(args);cudarl::pond_math_profile::validate_host_geometry();
    cp::Lease lease(study.ledger.string()+".study");
    Ledger ledger;ledger.protocol=study.plan;
    std::vector<uint8_t> resumed;
    if(args.init)require(!fs::exists(study.ledger),"Residual allocation was already initialized");
    else{
        ledger=Ledger::read(study.ledger);resumed=cp::read(args.resume.string());
        require(ledger.protocol==study.plan&&ledger.status==2&&ledger.state==cp::checksum(resumed)&&ledger.update<32,"Resume must be the last complete unspent eight-update stage");
        budget.prior=ledger.wall;
    }
    budget.guard();require(!fs::exists(args.out)&&fs::create_directories(args.out),"Expected a fresh stage output");
    ledger.status=1;cp::write_atomic(study.ledger.string(),ledger.bytes());
    CUDA_CHECK_THROW(cudaSetDevice(0));Stream stream;Buffer b(true);Extra e;Networks net(study);
    device_ptr<pr::ControlClearance> classified_rows;if(args.classified)classified_rows.allocate(Rows);
    const auto s=stream.value;clear_fault(b,s);
    std::vector<float> logs(A,pp::fixed_log_std());b.log_std.copy_from_host(logs.data(),logs.size());
    rp::Actor a;a.parent=study.motor.checksum;a.protocol=study.plan;
    uint64_t physical=uint64_t(ledger.update)*Rows,substeps=physical*10;double physics_total=0,opt_total=0,evidence_total=0;std::string reason;
    bool ok=false;StageAccounting attempt{};Source origin{};bool started=false;uint32_t completed=ledger.update;
    try{
        if(args.init){
            initialize_kernel<<<(N+31)/32,32,0,s>>>(b.worlds.get(),b.tasks.get(),b.schedules.get(),b.histories.get(),b.movement.get(),default_seed,program,b.fault.get());
            check_good(b,s,"Initial GPU physical certificate failed");a.weights=net.residual.get_params();
            require(rp::output_layer_zero(a.weights),"Residual must start at zero output");
        }else{
            a=restore_state(resumed,study,net,b);
            require(a.update==ledger.update,"Resume stage mismatch");const auto identity=cp::checksum(rp::write_actor(a));
            const auto receipt=pp::read_retention_receipt(cp::read(args.receipt.string()),identity,study.motor.checksum,study.plan,study.evaluator,a.update==0);
            require(receipt.passed&&receipt.cases_passed==20,"Full required inherited-skill receipt missing");
            const int end=int(a.update)+8;std::ofstream metrics(args.out/"updates.csv");
            metrics<<std::setprecision(10)<<"update,attempted_controls,attempted_substeps,actor_Adam,critic_Adam,eligible_turn_actions,physics_ms,optimization_ms,evidence_ms,approx_KL,aggregate_wall_s\n";
            for(int update=int(a.update)+1;update<=end;++update){
                budget.guard();clear_batch(b,e,s);attempt={};attempt.update=update;
                origin={raw_bytes(download(b.worlds,N,s)),saved_tasks(b,program,s),net.residual.get_params(),net.critic.get_params()};
                if(args.classified)CUDA_CHECK_THROW(cudaMemsetAsync(classified_rows.get(),0,classified_rows.size_bytes(),s));
                started=true;const double physics_ms=collect(b,e,net,study,attempt,s,budget,classified_rows.get());physics_total+=physics_ms;
                const auto save_start=Clock::now();const auto indices=save_rollout(args.out,origin,b,e,attempt,s,true,args.classified?&classified_rows:nullptr);
                const double evidence_ms=1000*elapsed(save_start);evidence_total+=evidence_ms;
                physical+=attempt.controls;substeps+=attempt.substeps;started=false;
                const auto opt_start=Clock::now();const float kl=optimize(b,e,net,update,indices,s,budget);
                const double opt_ms=1000*elapsed(opt_start);opt_total+=opt_ms;
                require(net.base.get_params()==study.motor.weights,"Frozen motor changed during optimization");
                completed=uint32_t(update);a.update=completed;a.training_controls=physical;a.training_substeps=substeps;
                a.actor_adam_calls=net.actor_adam.steps();a.critic_adam_calls=net.critic_adam.steps();
                a.weights=net.residual.get_params();
                metrics<<completed<<','<<physical<<','<<substeps<<','<<a.actor_adam_calls<<','<<a.critic_adam_calls<<','<<indices.size()<<','
                       <<physics_ms<<','<<opt_ms<<','<<evidence_ms<<','<<kl<<','<<budget.prior+elapsed(budget.start)<<'\n';metrics.flush();
                require(bool(metrics),"Update evidence write failed");
            }
        }
        require(sha256(whole_file(study.parent))==pp::parent_package_sha256&&net.base.get_params()==study.motor.weights,"Protected base changed");
        write_state(args.out,a,net,b,s);
        ledger.state=cp::checksum(cp::read((args.out/("state-"+std::to_string(a.update)+".bin")).string()));
        ledger.update=a.update;ledger.status=a.update==32?3:2;ok=true;
        reason=args.init?"zero_residual_initialized":"chunk_complete_retention_required";
    }catch(const std::exception& ex){
        reason=ex.what();ledger.status=4;
        if(started){
            try{if(!attempt.accounted)save_rollout(args.out,origin,b,e,attempt,s,false,args.classified?&classified_rows:nullptr);physical+=attempt.controls;substeps+=attempt.substeps;}
            catch(const std::exception& recovery){reason+="; incomplete batch recovery: "+std::string(recovery.what());}
        }
        try{
            cp::Writer w,am,cm;w.value(uint32_t{0x31465252});w.value(completed);w.value(physical);w.value(substeps);
            w.array(net.residual.get_params());w.array(net.critic.get_params());net.actor_adam.save(am);net.critic_adam.save(cm);
            w.array(am.data);w.array(cm.data);w.array(raw_bytes(download(b.worlds,N,s)));w.array(saved_tasks(b,program,s));
            cp::write_atomic((args.out/"failed-state.bin").string(),w.data);
        }catch(const std::exception& recovery){reason+="; failed snapshot: "+std::string(recovery.what());}
    }
    ledger.wall=budget.prior+elapsed(budget.start);cp::write_atomic(study.ledger.string(),ledger.bytes());
    std::ofstream result(args.out/"result.json");result<<std::setprecision(17)<<"{\"passed\":"<<(ok?"true":"false")<<",\"initialized\":"<<(args.init&&ok?"true":"false")
        <<",\"reason\":\""<<json_escape(reason)<<"\",\"protocol_id\":\""<<(args.classified?classified_id:rp::protocol_id)<<"\",\"completed_updates\":"<<completed
        <<",\"attempted_controls\":"<<physical<<",\"attempted_substeps\":"<<substeps<<",\"actor_Adam_calls\":"<<net.actor_adam.steps()
        <<",\"critic_Adam_calls\":"<<net.critic_adam.steps()<<",\"base_Adam_calls\":0,\"physics_ms\":"<<physics_total<<",\"optimization_ms\":"<<opt_total
        <<",\"batch_evidence_ms\":"<<evidence_total<<",\"stage_wall_seconds\":"<<elapsed(budget.start)<<",\"aggregate_wall_seconds\":"<<ledger.wall
        <<",\"physical_accounting_readback_complete\":"<<(!started||attempt.accounted?"true":"false")<<"}\n";result.close();require(bool(result),"Result write failed");
    require(ok,reason);
}
} // namespace residual_training
#ifndef POND_RESIDUAL_COLLECTOR_ONLY
namespace {
struct ClassifiedProcessDeadline{
    std::mutex mutex;std::condition_variable wake;bool done=false;std::thread watchdog;
    explicit ClassifiedProcessDeadline(std::chrono::steady_clock::time_point start):watchdog([this,start]{
        std::unique_lock<std::mutex> lock(mutex);
        if(!wake.wait_until(lock,start+std::chrono::seconds(4),[this]{return done;})){
            std::fputs("Classified experiment reached its four-second wall cap.\n",stderr);std::fflush(stderr);std::_Exit(3);
        }
    }){}
    ~ClassifiedProcessDeadline(){{std::lock_guard<std::mutex> lock(mutex);done=true;}wake.notify_one();watchdog.join();}
};
}
int main(int argc,char** argv){
    const auto start=std::chrono::steady_clock::now();
    try{
        namespace res=residual_training;namespace spec=cudarl::residual_ppo;
        if((argc==2||argc==3)&&std::string(argv[1])=="--describe"){
            const bool classified=argc==3;require(!classified||std::string(argv[2])=="--classified-clearance","Unknown descriptor mode");
            const std::string source=residual_source_hash;
            std::cout<<"{\"protocol_id\":\""<<(classified?res::classified_id:spec::protocol_id)<<"\",\"rollout_version\":"<<(classified?2:1)<<",\"source_sha256\":\""<<source
                <<"\",\"source_identity_fnv1a64\":\""<<cp::checksum({source.begin(),source.end()})<<"\",\"qualified_physics_source_sha256\":\""<<ph::physics_source_sha256()
                <<"\",\"actor_inputs\":42,\"critic_inputs\":41,\"residual_parameters\":722,\"base_parameters\":2532,\"world_stride\":"<<sizeof(pr::World)
                <<",\"task_stride\":"<<sizeof(ps::SignedCourseTask)<<",\"trace_stride\":"<<sizeof(TransitionTrace)<<",\"turn_practice_degrees\":[15,30,60,90],\"curriculum_ordinal\":\"lifetime_per_world_issued_requests\",\"max_updates\":32,\"worlds\":512,\"rollout_controls\":64,\"minibatch\":4096,\"epochs\":4,\"actor_learning_rate\":0.0003,\"critic_learning_rate\":0.001,\"base_Adam_calls\":0,\"physical_controls\":0,\"optimizer_calls\":0}\n";return 0;}
        const auto options=res::args(argc,argv);std::unique_ptr<ClassifiedProcessDeadline> limit;
        if(options.classified)limit=std::make_unique<ClassifiedProcessDeadline>(start);
        res::run(options);return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
#endif

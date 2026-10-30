// One frozen validation of contact A. No runtime/session/training entry point.
#include "contact_validation_cases.h"
#include "contact_validation_reference.h"
#include "contact_pilot_reference.h"
#include "physics/contact_pilot.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace p=cudarl::physics;namespace c=cudarl::contact;namespace a=cudarl::contact_pilot;
namespace s=cudarl::contact_validation_spec;namespace r=cudarl::contact_validation_reference;
namespace independent=cudarl::contact_pilot_reference;namespace cp=cudarl::checkpoint;namespace fs=std::filesystem;
static_assert(double(a::kStiffness)==r::kStiffness&&double(a::kDamping)==r::kDamping,"Continuous and executed coefficients must match");
constexpr s::Stage stages[]={s::Stage::ScalarAccuracy,s::Stage::RotatingAccuracy,s::Stage::Support,s::Stage::FreeWater,s::Stage::Passive,s::Stage::HeldOut};
struct Budget {
    int charged=0,committed=0,backend[2]{},stage[6]{},geometry=0,kinetic=0,solves=0;uint64_t reference=0;
    void charge(s::Stage which,int device) {
        const int i=static_cast<int>(which);
        if(charged>=s::kMaximumAdvances||stage[i]>=s::stage_budget(which))throw std::runtime_error("Frozen physical budget exhausted");
        ++charged;++stage[i];++backend[device];
    }
    void oracle(uint64_t count) {reference+=count;if(reference>12000)throw std::runtime_error("Continuous reference budget exhausted");}
};
struct Job {
    p::BodyState before{},after{};p::Parameters fluid{};p::V current{};
    a::ScalarState scalar_before{},scalar_after{};a::Telemetry telemetry{};
    float action[4]{},h=.005f;double mass=.003,load=0;
    bool scalar=false,legacy=false,ok=false;uint64_t serial=0,completed=0,echo=0;
};
CUDARL_HD uint32_t bits(float v) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(v);
#else
    uint32_t out;std::memcpy(&out,&v,sizeof(out));return out;
#endif
}
CUDARL_HD uint64_t bits64(double v) {
#ifdef __CUDA_ARCH__
    return static_cast<uint64_t>(__double_as_longlong(v));
#else
    uint64_t out;std::memcpy(&out,&v,sizeof(out));return out;
#endif
}
CUDARL_HD void mix(uint64_t& h,uint64_t v) {for(int i=0;i<8;++i){h^=(v>>(8*i))&255;h*=1099511628211ull;}}
CUDARL_HD uint64_t hash_input(const Job& j) {
    uint64_t h=14695981039346656037ull;const auto& b=j.before;
    for(float v:{b.position.x,b.position.y,b.position.z,b.orientation.x,b.orientation.y,b.orientation.z,b.orientation.w,b.work,b.travel,j.fluid.density,j.fluid.viscosity,j.fluid.gravity,j.current.x,j.current.y,j.current.z,j.h})mix(h,bits(v));
    for(float v:b.angle)mix(h,bits(v));for(float v:b.velocity)mix(h,bits(v));for(float v:j.action)mix(h,bits(v));
    for(double v:{j.mass,j.load,j.scalar_before.gap,j.scalar_before.velocity})mix(h,bits64(v));mix(h,j.scalar);mix(h,j.legacy);return h;
}
CUDARL_HD void advance(Job& j) {
    j.echo=hash_input(j);j.after=j.before;j.scalar_after=j.scalar_before;
    if(j.scalar)j.ok=a::scalar_step(j.scalar_after,j.mass,j.load,j.h,a::Law::Compliant,&j.telemetry);
    else if(j.legacy){c::substep(j.after,j.action,j.h,j.fluid,j.current);j.ok=p::finite_body(j.after);}
    else j.ok=a::substep(j.after,j.action,j.h,a::Law::Compliant,j.fluid,j.current,&j.telemetry);
    j.completed=j.serial;
}
__global__ void kernel(Job* j) {if(!blockIdx.x&&!threadIdx.x)advance(*j);}
void checked(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(std::string("CUDA integrity fault: ")+cudaGetErrorString(e));}
struct Executor {
    cudaStream_t stream=nullptr;Job* host=nullptr;Job* device=nullptr;uint64_t serial=0;
    Executor() {try {int n=0;checked(cudaGetDeviceCount(&n));if(!n)throw std::runtime_error("Real CUDA device required");
        checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));checked(cudaMallocHost(reinterpret_cast<void**>(&host),sizeof(Job)));
        checked(cudaMalloc(reinterpret_cast<void**>(&device),sizeof(Job)));}catch(...){release();throw;}}
    void release() noexcept {if(stream)cudaStreamSynchronize(stream);cudaFreeHost(host);cudaFree(device);if(stream)cudaStreamDestroy(stream);stream=nullptr;}
    ~Executor(){release();}
    Job step(Job j,int backend,s::Stage stage,Budget& budget) {
        j.serial=++serial;j.completed=0;j.echo=0;const auto hash=hash_input(j);
        if(backend==0){budget.charge(stage,backend);advance(j);}
        else {*host=j;checked(cudaMemcpyAsync(device,host,sizeof(Job),cudaMemcpyHostToDevice,stream));
            budget.charge(stage,backend);kernel<<<1,1,0,stream>>>(device);checked(cudaGetLastError());
            checked(cudaMemcpyAsync(host,device,sizeof(Job),cudaMemcpyDeviceToHost,stream));checked(cudaStreamSynchronize(stream));j=*host;}
        if(j.completed!=serial||j.echo!=hash)throw std::runtime_error("Nonblocking upload/input/serial check failed");
        if(j.ok)++budget.committed;return j;
    }
};
std::string quote(const std::string& v){std::string out="\"";for(char x:v){if(x=='\\'||x=='\"')out+='\\';if(x=='\n')out+="\\n";else out+=x;}return out+'\"';}
void write(const fs::path& path,const std::string& value){std::ofstream o(path,std::ios::binary);o.exceptions(std::ios::failbit|std::ios::badbit);o<<value;o.flush();}
std::array<float,23> fields(const p::BodyState& b) {
    std::array<float,23> out{};int n=0;for(float v:{b.position.x,b.position.y,b.position.z,b.orientation.x,b.orientation.y,b.orientation.z,b.orientation.w})out[static_cast<size_t>(n++)]=v;
    for(float v:b.angle)out[static_cast<size_t>(n++)]=v;for(float v:b.velocity)out[static_cast<size_t>(n++)]=v;out[21]=b.work;out[22]=b.travel;return out;
}
bool close(double x,double y,double absolute=2e-4){return std::isfinite(x)&&std::isfinite(y)&&std::abs(x-y)<=absolute+2e-3*std::abs(x);}
double tolerance(double e){return std::max(1e-10,.001*std::abs(e));}
double spring(const p::BodyState& b) {
    const auto g=p::shape(b);double e=0;
    for(int body=0;body<3;++body)for(int face=0;face<6;++face)for(int end=0;end<(body==0?1:2);++end){const auto n=c::normal(face);const double d=p::dot(n,g.link[body].center+c::support_offset(g.link[body],body,n,end))+c::offset(face);if(d<0)e+=.5*a::kStiffness*d*d;}
    return e;
}
double softstop(const p::BodyState& b){double e=0;for(float v:b.angle){const double d=v-p::clamp(v,-p::limit,p::limit);e+=.5e-4*d*d;}return e;}
double potential(const p::BodyState& b,const p::Parameters& fluid) {
    const auto geometry=p::shape(b);double e=0;
    for(int i=0;i<3;++i){const auto& l=geometry.link[i];if(fluid.density>0&&l.center.y+l.vertical_radius>0)throw std::runtime_error("Energy oracle requires fully submerged body");
        e+=(l.mass-fluid.density*l.volume)*fluid.gravity*l.center.y;
        if(i==0)e-=fluid.density*l.volume*fluid.gravity*l.q.rotate({0,.0004f,0}).y;}
    return e;
}
double energy(const p::BodyState& b,const p::Parameters& fluid){return p::energy(b)+spring(b)+softstop(b)+potential(b,fluid);}
p::V momentum(const p::BodyState& b){p::V out{};for(const auto& l:p::shape(b).link)out+=p::multiply(l.linear,b.velocity)*l.mass;return out;}
double scalar_energy(const a::ScalarState& state,double load){return .5*.003*state.velocity*state.velocity+.5*a::kStiffness*std::pow(std::min(0.,state.gap),2)+load*state.gap;}
struct Check {double gap=0,momentum=0,constitutive=0,energy=0,world_energy_change=0,drag_work=0,current_work=0;std::string error;};
Check check(const Job& j,double initial_energy,bool passive_or_loaded) {
    Check out;const auto& t=j.telemetry;
    if(j.legacy){const auto g=c::inspect(j.after);out.gap=g.min_gap;if(!j.ok||g.invalid||g.escape||g.excessive)out.error="legacy_free_failure";return out;}
    const auto body=t.pose_ready?t.trial:j.after;
    if(!t.finite||(j.scalar?(!std::isfinite(t.scalar_gap)||!std::isfinite(t.scalar_velocity)):!p::finite_body(body))){out.error="nonfinite";return out;}
    out.gap=j.scalar?t.scalar_gap:c::inspect(body).min_gap;
    if(!j.ok||!t.geometry_valid||out.gap<-.0005){out.error="geometry_or_0.5mm_safety";return out;}
    if(!t.converged||t.missed_activation||t.residual>t.tolerance){out.error="solver_or_activation";return out;}
    if(j.scalar){const double delta=j.mass*(j.scalar_after.velocity-t.scalar_free_velocity),impulse=t.total_impulse;
        out.momentum=std::abs(delta-impulse)/(1e-13+2e-5*std::max(std::abs(delta),std::abs(impulse)));
        out.energy=scalar_energy(j.scalar_after,j.load);
        if(out.energy>initial_energy+tolerance(initial_energy)){out.error="scalar_physical_energy";return out;}
    }else {
        const auto system=p::assemble(j.before,j.fluid,j.current);
        for(int i=0;i<10;++i){double delta=0,scale=0,impulse=0;for(int k=0;k<10;++k){const double term=system.M[i][k]*(double(t.after_velocity[k])-t.free_velocity[k]);delta+=term;scale+=std::abs(term);}
            for(const auto& row:t.rows)impulse+=double(row.jacobian[i])*row.impulse;
            out.momentum=std::max(out.momentum,std::abs(delta-impulse)/(1e-13+2e-5*std::max(scale,std::abs(impulse))));}
        out.energy=energy(body,j.fluid);out.world_energy_change=out.energy-energy(j.before,j.fluid);
        // Free-water drag work uses the free substage velocity, not a fictitious
        // contact force or positive-work actuator counter. Current work is the
        // translational drag-source force dotted with that same velocity.
        const auto calm=p::assemble(j.before,j.fluid,{});
        for(int i=0;i<10;++i){out.current_work+=j.h*(system.force[i]-calm.force[i])*t.free_velocity[i];
            for(int k=0;k<10;++k)out.drag_work-=j.h*double(t.free_velocity[i])*system.R[i][k]*t.free_velocity[k];}
        for(int k=0;k<4;++k){const double displacement=j.before.angle[k]-p::clamp(j.before.angle[k],-p::limit,p::limit);
            out.drag_work-=j.h*(p::torque/12.+(displacement!=0?3e-6:0))*t.free_velocity[6+k]*t.free_velocity[6+k];}
        const bool stationary_water=j.current.x==0&&j.current.y==0&&j.current.z==0;
        bool no_actuation=true;for(float command:j.action)no_actuation=no_actuation&&command==0;
        if(passive_or_loaded&&stationary_water&&no_actuation&&out.energy>initial_energy+tolerance(initial_energy)){out.error="full_mechanical_energy_ceiling";return out;}
    }
    const int n=j.scalar?1:30;
    for(int i=0;i<n;++i){const auto& row=t.rows[i];if(row.impulse/j.h< -1e-9){out.error="adhesion";return out;}
        if(row.active&&row.effective_mass>0){double velocity=j.scalar?j.scalar_after.velocity:0;if(!j.scalar)for(int k=0;k<10;++k)velocity+=double(row.jacobian[k])*t.after_velocity[k];
            const double d=1./(double(j.h)*(a::kDamping+double(j.h)*a::kStiffness));
            const double residual=row.impulse*d+velocity+a::kStiffness*double(row.gap)/(a::kDamping+double(j.h)*a::kStiffness);
            out.constitutive=std::max(out.constitutive,std::abs(std::min(double(row.impulse)*(1./row.effective_mass+d),residual)));}}
    if(out.momentum>1||out.constitutive>4*t.tolerance||std::abs(t.kinetic_delta-t.contact_work)>tolerance(t.kinetic_before)){out.error="equation_or_impulse_momentum";return out;}
    if(t.passivity_residual>tolerance(t.kinetic_before+t.elastic_before))out.error="contact_substage_passivity";
    return out;
}
bool parity(const Job& cpu,const Job& gpu){
    if(cpu.ok!=gpu.ok)return false;
    if(cpu.scalar){if(!close(cpu.scalar_after.gap,gpu.scalar_after.gap,2e-6)||!close(cpu.scalar_after.velocity,gpu.scalar_after.velocity))return false;}
    else{const auto x=fields(cpu.after),y=fields(gpu.after);for(size_t i=0;i<x.size();++i)if(!close(x[i],y[i]))return false;
        const auto a0=c::inspect(cpu.after),b0=c::inspect(gpu.after);if(a0.invalid!=b0.invalid||a0.escape!=b0.escape||a0.excessive!=b0.excessive||a0.contact!=b0.contact||a0.floor_contact!=b0.floor_contact||!close(a0.min_gap,b0.min_gap,2e-6))return false;}
    if(!cpu.legacy&&!close(cpu.telemetry.floor_impulse/cpu.h,gpu.telemetry.floor_impulse/gpu.h,2e-6))return false;
    if(!cpu.legacy&&!cpu.scalar)for(int k=0;k<10;++k){double x=0,y=0;for(int row=0;row<30;++row){x+=double(cpu.telemetry.rows[row].jacobian[k])*cpu.telemetry.rows[row].impulse/cpu.h;y+=double(gpu.telemetry.rows[row].jacobian[k])*gpu.telemetry.rows[row].impulse/gpu.h;}
        if(k>=3){x/=.072;y/=.072;}if(!close(x,y,2e-6))return false;}
    return true;
}
struct Output {
    fs::path root;std::ofstream trace,points;
    explicit Output(fs::path path):root(std::move(path)),trace(root/"trace.csv"),points(root/"contact-points.csv"){
        for(auto* f:{&trace,&points}){f->exceptions(std::ios::badbit|std::ios::failbit);*f<<std::setprecision(17);}
        trace<<"stage,id,h_s,backend,legacy,tick,time_s,ok,gap_m,scalar_velocity_m_s,reference_gap_m,reference_velocity_m_s,reference_force_N,reference_impulse_N_s,applied_floor_impulse_N_s,cumulative_floor_impulse_N_s,momentum_scaled_residual,constitutive_residual,energy_J,contact_M_delta_K_J,contact_impulse_work_J,actual_spring_before_J,actual_spring_after_J,world_energy_delta_J,water_drag_work_J,current_source_work_J,scalar_actual_load_work_J,scalar_split_kick_work_J,error";
        for(int k=0;k<23;++k)trace<<",body_"<<k;trace<<'\n';
        points<<"stage,id,h_s,backend,tick,row,body,face,end,pre_gap_m,impulse_N_s,normal_velocity_free_m_s,normal_velocity_after_m_s,effective_mass_kg\n";
    }
};
struct Result {
    s::Trial trial{};int backend=0;bool legacy=false,completed=false,passed=false;int attempts=0;
    std::string reason="blocked_by_accuracy_or_prior_gate";double min_gap=1,max_compression=0,restitution=0,impulse=0;
    double gap_error=0,velocity_error=0,impulse_error=0,compression_error=0,restitution_error=0;
    double max_momentum=0,max_constitutive=0,max_energy=0,settling_v=0,settling_w=0,settling_joint=0;
    double support_I=0,gravity_I=0,drag_I=0,actual_delta_P=0,balance_error=0;
    std::vector<Job> history;
};
void failure(Output& out,const Result& result,const Job& job) {
    cp::Writer w;w.value(uint32_t{0x43564146});w.value(uint32_t{1});w.value(static_cast<uint32_t>(result.trial.stage));w.value(result.trial.fixture.id);w.value(result.backend);w.value(result.attempts);
    cudarl::contact_pilot_spec::detail::write_body(w,job.before);cudarl::contact_pilot_spec::detail::write_body(w,job.after);cudarl::contact_pilot_spec::detail::write_body(w,job.telemetry.trial);
    for(double v:{job.scalar_before.gap,job.scalar_before.velocity,job.scalar_after.gap,job.scalar_after.velocity,job.telemetry.scalar_gap,job.telemetry.scalar_velocity,job.mass,job.load})w.value(v);
    cp::write_atomic((out.root/(std::string("first-failure-")+s::stage_name(result.trial.stage)+"-"+std::to_string(result.trial.fixture.id)+"-"+std::to_string(result.backend)+".bin")).string(),w.data);
}
void execute(Result& result,Executor& executor,Budget& budget,Output& out,const Result* cpu=nullptr,const Result* original=nullptr){
    const auto& trial=result.trial;const auto& f=trial.fixture;p::BodyState body=f.initial;a::ScalarState scalar{f.scalar_gap,f.scalar_velocity};
    std::unique_ptr<r::Reference> oracle; r::Report report;
    if(f.scalar){oracle=std::make_unique<r::Reference>(static_cast<r::Case>(f.id));report=oracle->report();budget.oracle(oracle->query_count());if(!report.passed)throw std::runtime_error("Continuous reference invalid");}
    const double initial=f.scalar?scalar_energy(scalar,-f.scalar_force):energy(body,f.fluid);
    result.reason="";result.max_energy=initial;bool contact_seen=false;p::V window_start_P{};
    for(int tick=1;tick<=trial.steps;++tick){
        Job input;input.before=body;input.scalar_before=scalar;input.scalar=f.scalar;input.legacy=result.legacy;input.fluid=f.fluid;input.current=f.current;input.h=trial.h;
        input.mass=f.scalar_mass;input.load=-f.scalar_force;for(int i=0;i<4;++i)input.action[i]=f.action[static_cast<size_t>(i)];
        const auto job=executor.step(input,result.backend,trial.stage,budget);++result.attempts;const auto proof=check(job,initial,trial.stage==s::Stage::Passive||trial.stage==s::Stage::Support||trial.stage==s::Stage::HeldOut);
        auto error=proof.error;result.min_gap=std::min(result.min_gap,proof.gap);result.max_compression=std::max(result.max_compression,std::max(0.,-proof.gap));
        result.impulse+=job.telemetry.floor_impulse;result.max_momentum=std::max(result.max_momentum,proof.momentum);result.max_constitutive=std::max(result.max_constitutive,proof.constitutive);result.max_energy=std::max(result.max_energy,proof.energy);
        r::Sample continuous;const double time=tick*double(trial.h);
        if(f.scalar){continuous=oracle->sample(std::min(.25,time));budget.oracle(1);
            result.gap_error=std::max(result.gap_error,std::abs(job.scalar_after.gap-continuous.gap_m));result.velocity_error=std::max(result.velocity_error,std::abs(job.scalar_after.velocity-continuous.velocity_m_s));
            if(job.telemetry.total_impulse>0)contact_seen=true;if(contact_seen&&f.id==1)result.restitution=std::max(result.restitution,std::max(0.,job.scalar_after.velocity)/.05);}
        if(error.empty()&&cpu){if(static_cast<size_t>(tick)>cpu->history.size()||!parity(cpu->history[static_cast<size_t>(tick-1)],job))error="CPU_CUDA_parity";}
        if(error.empty()&&original){const auto x=fields(original->history[static_cast<size_t>(tick-1)].after),y=fields(job.after);
            if(std::memcmp(x.data(),y.data(),sizeof(x))!=0)error="free_water_not_bit_exact_to_original";}
        const bool support=trial.stage==s::Stage::Support||trial.stage==s::Stage::HeldOut;
        if(support&&tick>trial.steps-200){
            if(tick==trial.steps-199)window_start_P=momentum(job.before);
            result.support_I+=job.telemetry.floor_impulse;const auto system=p::assemble(job.before,f.fluid,f.current);
            result.gravity_I+=job.h*system.force[1];double drag=0;for(int k=0;k<10;++k)drag-=double(system.R[1][k])*job.telemetry.free_velocity[k];result.drag_I+=job.h*drag;
            result.actual_delta_P=momentum(job.after).y-window_start_P.y;
            const auto geometry=p::shape(job.after);
            for(const auto& l:geometry.link){result.settling_v=std::max(result.settling_v,double(p::norm(p::multiply(l.linear,job.after.velocity))));result.settling_w=std::max(result.settling_w,double(p::norm(l.omega)));}
            for(int k=6;k<10;++k)result.settling_joint=std::max(result.settling_joint,std::abs(double(job.after.velocity[k])));
        }
        const auto scalar_proposal=job.telemetry.pose_ready?a::ScalarState{job.telemetry.scalar_gap,job.telemetry.scalar_velocity}:job.scalar_after;
        auto& log=out.trace;log<<s::stage_name(trial.stage)<<','<<f.id<<','<<trial.h<<','<<result.backend<<','<<result.legacy<<','<<tick<<','<<time<<','<<job.ok<<','<<proof.gap<<','<<scalar_proposal.velocity<<','<<continuous.gap_m<<','<<continuous.velocity_m_s<<','<<continuous.normal_force_N<<','<<continuous.normal_impulse_N_s<<','<<job.telemetry.floor_impulse<<','<<result.impulse<<','<<proof.momentum<<','<<proof.constitutive<<','<<proof.energy<<','<<job.telemetry.kinetic_delta<<','<<job.telemetry.contact_work<<','<<job.telemetry.elastic_before<<','<<job.telemetry.elastic_actual_after<<','<<proof.world_energy_change<<','<<proof.drag_work<<','<<proof.current_work<<','<<(-input.load*(scalar_proposal.gap-scalar.gap))<<','<<job.telemetry.external_work<<','<<error;
        const auto proposed=job.telemetry.pose_ready?job.telemetry.trial:job.after;for(float v:fields(proposed))log<<','<<v;log<<'\n';
        if(!result.legacy)for(int k=0;k<(f.scalar?1:30);++k){const auto& row=job.telemetry.rows[k];if(!row.active&&row.impulse==0)continue;
            out.points<<s::stage_name(trial.stage)<<','<<f.id<<','<<trial.h<<','<<result.backend<<','<<tick<<','<<k<<','<<row.body<<','<<row.face<<','<<row.end<<','<<row.gap<<','<<row.impulse<<','<<row.velocity_free<<','<<row.velocity_after<<','<<row.effective_mass<<'\n';}
        result.history.push_back(job);
        if(!error.empty()){result.reason=error;failure(out,result,job);break;}body=job.after;scalar=job.scalar_after;
    }
    result.completed=result.reason.empty()&&result.attempts==trial.steps;
    if(result.completed&&f.scalar){result.compression_error=std::abs(result.max_compression-report.max_compression_m);result.restitution_error=std::abs(result.restitution-report.restitution);result.impulse_error=std::abs(result.impulse-report.normal_impulse_at_horizon_N_s);
        if(result.compression_error>r::compression_tolerance(report.max_compression_m)||result.restitution_error>r::kRestitutionAbsolute||result.impulse_error>r::impulse_tolerance(report.normal_impulse_at_horizon_N_s)||result.gap_error>r::kTrajectoryGapAbsolute||result.velocity_error>r::kTrajectoryVelocityAbsolute)result.reason="continuous_law_accuracy";}
    if(result.completed&&(trial.stage==s::Stage::Support||trial.stage==s::Stage::HeldOut)){
        result.balance_error=std::abs(result.actual_delta_P-(result.support_I+result.gravity_I+result.drag_I));
        if(result.settling_v>.0005||result.settling_w>.05||result.settling_joint>.05||result.balance_error>1e-8+.05*std::abs(result.gravity_I)||std::abs(result.support_I+result.gravity_I)>1e-8+.1*std::abs(result.gravity_I))result.reason="settling_or_actual_impulse_force_balance";
    }
    result.passed=result.completed&&result.reason.empty();if(result.passed)result.reason="passed_registered_case";out.trace.flush();out.points.flush();
}
std::string registration(){std::ostringstream o;o<<std::setprecision(17)<<"{\"protocol\":\"contact-A-validation-v1\",\"physical_ceiling\":18200,\"law\":\"unchanged contact_pilot physics4 A\",\"k_N_m\":"<<a::kStiffness<<",\"c_N_s_m\":"<<a::kDamping<<",\"runtime_physics\":2,\"safety_overlap_m\":0.0005,\"continuous_query_cap\":12000,\"stage_caps\":[2100,1500,6000,400,200,8000],\"trials\":[";
    bool first=true;for(auto stage:stages)for(const auto& t:s::trials(stage)){if(!first)o<<',';first=false;o<<"{\"stage\":"<<quote(s::stage_name(stage))<<",\"id\":"<<t.fixture.id<<",\"name\":"<<quote(t.fixture.name)<<",\"h_s\":"<<t.h<<",\"steps_per_model_backend\":"<<t.steps<<",\"scalar\":"<<t.fixture.scalar<<",\"scalar_mass_gap_velocity_signed_force\":["<<t.fixture.scalar_mass<<','<<t.fixture.scalar_gap<<','<<t.fixture.scalar_velocity<<','<<t.fixture.scalar_force<<"],\"body\":[";auto b=fields(t.fixture.initial);for(size_t i=0;i<b.size();++i){if(i)o<<',';o<<b[i];}o<<"],\"fluid\":["<<t.fixture.fluid.density<<','<<t.fixture.fluid.viscosity<<','<<t.fixture.fluid.gravity<<"],\"current\":["<<t.fixture.current.x<<','<<t.fixture.current.y<<','<<t.fixture.current.z<<"],\"action\":[";for(size_t i=0;i<4;++i){if(i)o<<',';o<<t.fixture.action[i];}o<<"]}";}
    o<<"],\"scalar_tolerances\":{\"compression_abs_m\":2e-6,\"compression_rel\":0.05,\"restitution_abs\":0.02,\"impulse_abs_N_s\":1e-8,\"impulse_rel\":0.05,\"trajectory_gap_m\":1e-5,\"trajectory_speed_m_s\":0.001},\"rotation_tolerances\":{\"head_and_endpoints_m\":0.0001,\"orientation_rad\":0.02,\"joint_rad\":0.02,\"head_speed_m_s\":0.002,\"angular_and_joint_speed_rad_s\":0.5,\"impulse_abs_N_s\":1e-8,\"impulse_rel\":0.05},\"long_stage_prerequisite\":\"all 5ms scalar accuracy and rotating convergence pass; no automatic finer timestep adoption\",\"counts\":\"charged before each actual CPU call or CUDA launch; failures count; no retries/reallocation\"}\n";return o.str();}
std::vector<uint8_t> binding(){cp::Writer w;w.array(s::serialize_protocol());const auto json=registration();for(char value:json)w.value(static_cast<uint8_t>(value));return w.data;}
void prepare(const fs::path& root){if(fs::exists(root/"protocol.bin")||fs::exists(root/"run-started.json"))throw std::runtime_error("Refusing existing registration");
    int total=0;for(auto stage:stages){int count=0;for(const auto& t:s::trials(stage))count+=t.steps*2*s::model_count(stage);if(count!=s::stage_budget(stage))throw std::runtime_error("Registered stage does not match its cap");total+=count;}
    if(total!=s::kMaximumAdvances||s::fixture_geometry_queries()!=4)throw std::runtime_error("Registered total or initial-placement query count mismatch");
    fs::create_directories(root);write(root/"protocol.json",registration());cp::write_atomic((root/"protocol.bin").string(),binding());std::cout<<"Frozen18200 attempts; zero physical calls;4 static initial-fixture queries.\n";}
double qdistance(p::Q x,p::Q y){double dot=double(x.x)*y.x+double(x.y)*y.y+double(x.z)*y.z+double(x.w)*y.w;const double nx=std::sqrt(double(x.x)*x.x+double(x.y)*x.y+double(x.z)*x.z+double(x.w)*x.w),ny=std::sqrt(double(y.x)*y.x+double(y.y)*y.y+double(y.z)*y.z+double(y.w)*y.w);return 2*std::acos(std::clamp(std::abs(dot)/(nx*ny),0.,1.));}
struct Refinement {int backend=0,level=0;double position=0,endpoints=0,gap=0,compression=0,orientation=0,joints=0,speed=0,rates=0,impulse_error=0,impulse_path_error=0,energy_error=0;bool passed=false;};
Refinement compare(const Result& coarse,const Result& fine,int level){Refinement x;x.backend=coarse.backend;x.level=level;if(!coarse.completed||!fine.completed)return x;
    const size_t ratio=fine.history.size()/coarse.history.size();double coarse_I=0,fine_I=0;for(size_t tick=0;tick<coarse.history.size();++tick){const auto& b=coarse.history[tick].after;const auto& d=fine.history[(tick+1)*ratio-1].after;
        coarse_I+=coarse.history[tick].telemetry.floor_impulse;for(size_t k=tick*ratio;k<(tick+1)*ratio;++k)fine_I+=fine.history[k].telemetry.floor_impulse;
        x.impulse_path_error=std::max(x.impulse_path_error,std::abs(coarse_I-fine_I));
        x.gap=std::max(x.gap,std::abs(double(c::inspect(b).floor_gap)-c::inspect(d).floor_gap));
        x.energy_error=std::max(x.energy_error,std::abs(energy(b,coarse.trial.fixture.fluid)-energy(d,fine.trial.fixture.fluid)));
        x.position=std::max(x.position,double(p::norm((b.position-d.position)*p::scale)));x.orientation=std::max(x.orientation,qdistance(b.orientation,d.orientation));
        for(int k=0;k<4;++k)x.joints=std::max(x.joints,std::abs(double(b.angle[k])-d.angle[k]));
        x.speed=std::max(x.speed,double(p::norm(p::V{b.velocity[0]-d.velocity[0],b.velocity[1]-d.velocity[1],b.velocity[2]-d.velocity[2]})));
        for(int k=3;k<10;++k)x.rates=std::max(x.rates,std::abs(double(b.velocity[k])-d.velocity[k]));
        const auto bg=p::shape(b),dg=p::shape(d);for(int part=0;part<3;++part)for(int end=0;end<(part==0?1:2);++end){const auto n=c::normal(2);const auto bp=bg.link[part].center+c::support_offset(bg.link[part],part,n,end),dp=dg.link[part].center+c::support_offset(dg.link[part],part,n,end);x.endpoints=std::max(x.endpoints,double(p::norm(bp-dp)));}}
    x.compression=std::abs(coarse.max_compression-fine.max_compression);x.impulse_error=std::abs(coarse.impulse-fine.impulse);x.passed=x.position<=.0001&&x.endpoints<=.0001&&x.gap<=.0001&&x.orientation<=.02&&x.joints<=.02&&x.speed<=.002&&x.rates<=.5&&x.impulse_path_error<=std::max(1e-8,.05*std::abs(fine.impulse));return x;}
void summary(const fs::path& root,const Budget& b,const std::vector<Result>& cases,const std::vector<Refinement>& convergence,bool accuracy,const std::string& error){
    bool all=accuracy&&error.empty();for(const auto& v:cases)all=all&&v.passed;std::ostringstream o;o<<std::setprecision(17)<<std::boolalpha<<"{\"accepted_A_at_5ms\":"<<(all?"true":"false")<<",\"accuracy_passed\":"<<(accuracy?"true":"false")<<",\"fault\":"<<quote(error)<<",\"physical_attempts\":"<<b.charged<<",\"committed\":"<<b.committed<<",\"ceiling\":18200,\"backend_attempts\":["<<b.backend[0]<<','<<b.backend[1]<<"],\"continuous_queries\":"<<b.reference<<",\"static_counts\":["<<b.geometry<<','<<b.kinetic<<','<<b.solves<<"],\"stage_attempts\":[";for(int i=0;i<6;++i){if(i)o<<',';o<<b.stage[i];}o<<"],\"cases\":[";
    for(size_t i=0;i<cases.size();++i){if(i)o<<',';const auto& v=cases[i];o<<"{\"stage\":"<<quote(s::stage_name(v.trial.stage))<<",\"id\":"<<v.trial.fixture.id<<",\"h_s\":"<<v.trial.h<<",\"backend\":"<<v.backend<<",\"legacy\":"<<v.legacy<<",\"attempts\":"<<v.attempts<<",\"completed\":"<<v.completed<<",\"passed\":"<<v.passed<<",\"reason\":"<<quote(v.reason)<<",\"min_gap_m\":"<<v.min_gap<<",\"max_compression_m\":"<<v.max_compression<<",\"restitution\":"<<v.restitution<<",\"impulse_N_s\":"<<v.impulse<<",\"max_gap_error_m\":"<<v.gap_error<<",\"max_speed_error_m_s\":"<<v.velocity_error<<",\"compression_error_m\":"<<v.compression_error<<",\"restitution_error\":"<<v.restitution_error<<",\"impulse_error_N_s\":"<<v.impulse_error<<",\"momentum_scaled_residual\":"<<v.max_momentum<<",\"constitutive_residual_m_s\":"<<v.max_constitutive<<",\"max_energy_J\":"<<v.max_energy<<",\"settling_link_speed_m_s\":"<<v.settling_v<<",\"settling_link_rotation_rad_s\":"<<v.settling_w<<",\"settling_joint_rate_rad_s\":"<<v.settling_joint<<",\"support_impulse_N_s\":"<<v.support_I<<",\"gravity_buoyancy_impulse_N_s\":"<<v.gravity_I<<",\"drag_impulse_N_s\":"<<v.drag_I<<",\"actual_COM_delta_P_N_s\":"<<v.actual_delta_P<<",\"force_balance_residual_N_s\":"<<v.balance_error<<'}';}
    o<<"],\"rotating_refinement\":[";for(size_t i=0;i<convergence.size();++i){if(i)o<<',';const auto& x=convergence[i];o<<"{\"backend\":"<<x.backend<<",\"level\":"<<x.level<<",\"passed\":"<<x.passed<<",\"head_position_m\":"<<x.position<<",\"endpoints_m\":"<<x.endpoints<<",\"gap_error_m\":"<<x.gap<<",\"compression_error_m\":"<<x.compression<<",\"orientation_rad\":"<<x.orientation<<",\"joint_angles_rad\":"<<x.joints<<",\"head_speed_m_s\":"<<x.speed<<",\"max_angular_joint_rate_rad_s\":"<<x.rates<<",\"contact_impulse_error_N_s\":"<<x.impulse_error<<",\"contact_impulse_path_error_N_s\":"<<x.impulse_path_error<<",\"mechanical_energy_error_J\":"<<x.energy_error<<'}';}o<<"]}\n";write(root/"summary.json",o.str());}
int run(const fs::path& root){
    if(fs::exists(root/"run-started.json")||fs::exists(root/"trace.csv"))throw std::runtime_error("One-shot validation already attempted; no retries");
    if(cp::read((root/"protocol.bin").string())!=binding())throw std::runtime_error("Frozen protocol mismatch");
    std::ifstream file(root/"protocol.json",std::ios::binary);const std::string json((std::istreambuf_iterator<char>(file)),{});if(json!=registration())throw std::runtime_error("Numeric registration differs");
    write(root/"run-started.json","{\"one_shot\":true,\"ceiling\":18200}\n");Budget budget;Output out(root);std::vector<Result> results;std::vector<Refinement> convergence;std::string error;bool accuracy=true;
    try{
        const auto states=cudarl::contact_pilot_spec::reference_states();const auto reference=independent::reference_report({states.begin(),states.end()});write(root/"static-reference.json",independent::to_json(reference));
        if(s::fixture_geometry_queries()!=4)throw std::runtime_error("Run initial-fixture query count mismatch");
        budget.geometry=reference.geometry_queries+8;budget.kinetic=reference.kinetic_evaluations;budget.solves=reference.matrix_solves;
        if(budget.geometry>256||budget.kinetic>256||budget.solves>96)throw std::runtime_error("Static prerequisite query cap exceeded");
        if(!reference.passed)throw std::runtime_error("Independent geometry/M prerequisite failed");
        if(reference.cases.empty()||!reference.cases[0].energy_bound.passed)throw std::runtime_error("Rolling calibration energy prerequisite failed");
        for(const auto& state:reference.cases)for(const auto& row:state.floor_rows)if(row.mass_M_reference>a::kMassCap)throw std::runtime_error("Calibration normal mass cap exceeded");
        std::ostringstream truth;truth<<std::setprecision(17)<<"{\"independent\":true,\"cases\":[";
        for(int id=0;id<3;++id){r::Reference oracle(static_cast<r::Case>(id));budget.oracle(oracle.query_count());const auto v=oracle.report();if(!v.passed)throw std::runtime_error("Continuous oracle prerequisite failed");if(id)truth<<',';
            truth<<"{\"id\":"<<id<<",\"mass_kg\":"<<v.mass_kg<<",\"k_N_m\":"<<v.stiffness_N_m<<",\"c_N_s_m\":"<<v.damping_N_s_m<<",\"max_compression_m\":"<<v.max_compression_m<<",\"max_compression_time_s\":"<<v.max_compression_time_s<<",\"force_release_time_s\":"<<v.force_release_time_s<<",\"release_gap_m\":"<<v.release_gap_m<<",\"restitution\":"<<v.restitution<<",\"impulse_N_s\":"<<v.normal_impulse_at_horizon_N_s<<",\"unrecovered_spring_release_J\":"<<v.unrecovered_spring_at_release_J<<'}';}
        truth<<"]}\n";write(root/"continuous-reference.json",truth.str());
        Executor executor;
        for(auto stage:stages){
            if(stage==s::Stage::Support){
                for(int backend=0;backend<2;++backend){const Result* fine=nullptr;std::vector<const Result*> ladder;
                    for(const auto& v:results)if(v.trial.stage==s::Stage::RotatingAccuracy&&v.backend==backend)ladder.push_back(&v);
                    if(ladder.size()!=4)throw std::runtime_error("Incomplete rotating accuracy ladder");fine=ladder.back();
                    for(size_t level=0;level+1<ladder.size();++level){const auto x=compare(*ladder[level],*fine,static_cast<int>(level));convergence.push_back(x);if(level==0||level==2)accuracy=accuracy&&x.passed;}
                }
                for(const auto& v:results)if(v.trial.stage==s::Stage::ScalarAccuracy&&v.trial.h==.005f)accuracy=accuracy&&v.passed;
            }
            for(const auto& t:s::trials(stage))for(int model=0;model<s::model_count(stage);++model){
                const size_t cpu_index=results.size();for(int backend=0;backend<2;++backend){Result v;v.trial=t;v.backend=backend;v.legacy=stage==s::Stage::FreeWater&&model==0;
                    if(!accuracy&&stage!=s::Stage::ScalarAccuracy&&stage!=s::Stage::RotatingAccuracy){results.push_back(std::move(v));continue;}
                    const Result* original=nullptr;if(stage==s::Stage::FreeWater&&model==1)original=&results[cpu_index-2+static_cast<size_t>(backend)];
                    const Result* cpu=backend?&results[cpu_index]:nullptr;execute(v,executor,budget,out,cpu,original);
                    const bool diagnostic=stage==s::Stage::ScalarAccuracy||stage==s::Stage::RotatingAccuracy;
                    const bool fatal=!v.completed||(!diagnostic&&!v.passed);results.push_back(std::move(v));if(fatal)throw std::runtime_error("Physical/numerical prerequisite failed: "+results.back().reason);
                }
            }
        }
    }catch(const std::exception& e){error=e.what();accuracy=false;}
    summary(root,budget,results,convergence,accuracy,error);std::cout<<"Validation attempts "<<budget.charged<<"/18200; accuracy="<<accuracy<<"; fault="<<error<<'\n';
    return error.empty()?(accuracy?0:3):2;
}
}
int main(int argc,char** argv){try{if(argc!=3)throw std::runtime_error("Use --prepare DIR or --run DIR (no implicit execution)");const std::string mode=argv[1];if(mode=="--prepare"){prepare(argv[2]);return 0;}if(mode=="--run")return run(argv[2]);throw std::runtime_error("Unknown mode");}catch(const std::exception& e){std::cerr<<"Validation refused: "<<e.what()<<'\n';return 1;}}

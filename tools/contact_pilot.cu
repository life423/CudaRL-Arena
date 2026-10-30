// Opt-in, one-shot 2,024-advance contact pilot. No production/session/save API.
#include "contact_pilot_cases.h"
#include "contact_pilot_reference.h"
#include "physics/contact_pilot.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace p=cudarl::physics;namespace c=cudarl::contact;namespace old=cudarl::contact_response;
namespace law=cudarl::contact_pilot;namespace spec=cudarl::contact_pilot_spec;
namespace ref=cudarl::contact_pilot_reference;namespace cp=cudarl::checkpoint;namespace fs=std::filesystem;
constexpr spec::Stage stages[]={spec::Stage::Parity,spec::Stage::Scalar,spec::Stage::FreeWater,spec::Stage::Passive,spec::Stage::Rolling,spec::Stage::Fresh};
constexpr double kMomentumAbsolute=1e-13,kMomentumRelative=2e-5,kOracleTolerance=1e-6,kResidualMultiple=4;
const char* model_name(int model) {return model==0?"legacy-v2":(model==1?"implicit-v3":(model==2?"calibrated-compliance-A":"hard-contact-B"));}
law::Law selected_law(int model) {return model==2?law::Law::Compliant:law::Law::Hard;}
struct Budget {
    int charged=0,dispatched=0,committed=0,by_stage[6]{};
    int geometry=0,kinetic=0,solves=0;
    void charge(spec::Stage stage) {
        const auto i=static_cast<unsigned>(stage);
        if(charged>=spec::kMaximumAdvances||by_stage[i]>=spec::stage_budget(stage)) throw std::runtime_error("Pilot physical budget exhausted");
        ++charged;++by_stage[i];
    }
    void algebra() {if(solves>=spec::kStaticSolveCap) throw std::runtime_error("Static matrix/LCP query cap exhausted");++solves;}
};
struct Job {
    p::BodyState before{},after{};p::Parameters fluid{};p::V current{};
    law::ScalarState scalar_before{},scalar_after{};
    float action[4]{},h=spec::kStep;double scalar_mass=.003,scalar_force=0;
    int model=0;bool scalar=false,ok=false,committed=false;
    old::Telemetry prior{};law::Telemetry pilot{};
    uint64_t serial=0,completed_serial=0,echo_hash=0;
};
struct Oracle {
    double matrix[30][30]{},q[30]{},impulse[30]{};bool active[30]{};
    int count=0,model=2;bool ok=false;law::SolverStats stats{};
    uint64_t serial=0,completed_serial=0;
};
CUDARL_HD inline uint32_t float_bits(float value) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(value);
#else
    uint32_t out;std::memcpy(&out,&value,sizeof(out));return out;
#endif
}
CUDARL_HD inline uint64_t double_bits(double value) {
#ifdef __CUDA_ARCH__
    return static_cast<uint64_t>(__double_as_longlong(value));
#else
    uint64_t out;std::memcpy(&out,&value,sizeof(out));return out;
#endif
}
CUDARL_HD inline void mix(uint64_t& hash,uint64_t bits) {for(int i=0;i<8;++i) {hash^=(bits>>(8*i))&255;hash*=1099511628211ull;}}
CUDARL_HD uint64_t input_hash(const Job& x) {
    uint64_t hash=14695981039346656037ull;
    const auto& b=x.before;
    const float fields[]={b.position.x,b.position.y,b.position.z,b.orientation.x,b.orientation.y,b.orientation.z,b.orientation.w,b.work,b.travel,x.fluid.density,x.fluid.viscosity,x.fluid.gravity,x.current.x,x.current.y,x.current.z,x.h};
    for(float value:fields) mix(hash,float_bits(value));for(float value:b.angle) mix(hash,float_bits(value));for(float value:b.velocity) mix(hash,float_bits(value));
    for(float value:x.action) mix(hash,float_bits(value));mix(hash,static_cast<uint64_t>(x.model));mix(hash,x.scalar?1:0);
    const double scalar_fields[]={x.scalar_before.gap,x.scalar_before.velocity,x.scalar_mass,x.scalar_force};
    for(double value:scalar_fields) mix(hash,double_bits(value));
    return hash;
}
CUDARL_HD double dot10(const float* a,const float* b) {double value=0;for(int i=0;i<10;++i) value+=double(a[i])*b[i];return value;}
CUDARL_HD old::Telemetry legacy_contacts(const Job& job) {
    old::Telemetry out;out.finite=p::finite_body(job.after);out.converged=out.finite;
    const auto shape=p::shape(job.before);int index=0;
    for(int body=0;body<3;++body) {
        const auto& link=shape.link[body];const int ends=body==0?1:2;const float weight=1.f/ends;
        for(int face=0;face<6;++face) for(int end=0;end<ends;++end) {
            auto& row=out.rows[index++];row.body=body;row.face=face;row.end=end;const auto n=c::normal(face),offset=c::support_offset(link,body,n,end);
            row.gap=p::dot(n,link.center+offset)+c::offset(face);row.k=c::kContactStiffness*weight;row.c=2*c::kContactDampingRatio*sqrtf(c::kContactStiffness*link.mass)*weight;
            p::V jacobian[10];for(int j=0;j<10;++j) {jacobian[j]=link.linear[j]+p::cross(link.angular[j],offset);row.jacobian[j]=p::dot(n,jacobian[j]);}
            row.velocity_free=p::dot(p::multiply(jacobian,job.before.velocity),n);row.velocity_after=static_cast<float>(dot10(row.jacobian,job.after.velocity));row.active=row.gap<0;
            if(row.active) {row.impulse=static_cast<float>(job.h*(-double(row.k)*row.gap-(row.velocity_free<0?double(row.c)*row.velocity_after:0)));out.total_impulse+=row.impulse;if(face==2) out.floor_impulse+=row.impulse;++out.active_rows;}
        }
    }return out;
}
CUDARL_HD void advance(Job& job) {
    job.echo_hash=input_hash(job);job.after=job.before;job.scalar_after=job.scalar_before;
    if(job.scalar) {job.ok=law::scalar_step(job.scalar_after,job.scalar_mass,-job.scalar_force,job.h,job.model==2?law::Law::Compliant:law::Law::Hard,&job.pilot);job.committed=job.ok;}
    else if(job.model==0) {c::substep(job.after,job.action,job.h,job.fluid,job.current);job.committed=true;job.ok=p::finite_body(job.after);job.prior=legacy_contacts(job);}
    else if(job.model==1) {job.ok=old::substep(job.after,job.action,job.h,job.fluid,job.current,&job.prior);job.committed=job.ok;}
    else {job.ok=law::substep(job.after,job.action,job.h,job.model==2?law::Law::Compliant:law::Law::Hard,job.fluid,job.current,&job.pilot);job.committed=job.ok;}
    job.completed_serial=job.serial;
}
__global__ void advance_kernel(Job* job) {if(!blockIdx.x&&!threadIdx.x) advance(*job);}
__global__ void oracle_kernel(Oracle* oracle) {
    if(blockIdx.x||threadIdx.x) return;
    oracle->ok=law::solve_contact(oracle->count,oracle->matrix,oracle->q,oracle->active,oracle->model==2?law::Law::Compliant:law::Law::Hard,oracle->impulse,oracle->stats,kOracleTolerance);
    oracle->completed_serial=oracle->serial;
}
void checked(cudaError_t error) {if(error!=cudaSuccess) throw std::runtime_error(std::string("Shared CUDA/stream fault: ")+cudaGetErrorString(error));}
struct Device {
    cudaStream_t stream=nullptr;Job* device=nullptr;Job* host=nullptr;Oracle* device_oracle=nullptr;Oracle* host_oracle=nullptr;uint64_t serial=0;
    Device() {try {
        int count=0;checked(cudaGetDeviceCount(&count));if(!count) throw std::runtime_error("Actual CUDA hardware is required");
        checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));checked(cudaMalloc(reinterpret_cast<void**>(&device),sizeof(Job)));
        checked(cudaMallocHost(reinterpret_cast<void**>(&host),sizeof(Job)));new(host) Job{};
        checked(cudaMalloc(reinterpret_cast<void**>(&device_oracle),sizeof(Oracle)));checked(cudaMallocHost(reinterpret_cast<void**>(&host_oracle),sizeof(Oracle)));new(host_oracle) Oracle{};
    }catch(...) {release();throw;}}
    void release() noexcept {if(stream) cudaStreamSynchronize(stream);cudaFree(device);cudaFreeHost(host);cudaFree(device_oracle);cudaFreeHost(host_oracle);if(stream) cudaStreamDestroy(stream);stream=nullptr;}
    ~Device() {release();}
    Job step(Job input,spec::Stage stage,Budget& budget) {
        input.serial=++serial;input.completed_serial=0;input.echo_hash=0;*host=input;const auto expected=input_hash(input);
        checked(cudaMemcpyAsync(device,host,sizeof(Job),cudaMemcpyHostToDevice,stream));budget.charge(stage);
        advance_kernel<<<1,1,0,stream>>>(device);checked(cudaGetLastError());++budget.dispatched;
        checked(cudaMemcpyAsync(host,device,sizeof(Job),cudaMemcpyDeviceToHost,stream));checked(cudaStreamSynchronize(stream));
        if(host->completed_serial!=input.serial||host->echo_hash!=expected) throw std::runtime_error("Shared nonblocking-stream input/serial integrity fault");
        if(host->committed) ++budget.committed;return *host;
    }
    Oracle algebra(Oracle input,Budget& budget) {
        input.serial=++serial;input.completed_serial=0;*host_oracle=input;
        checked(cudaMemcpyAsync(device_oracle,host_oracle,sizeof(Oracle),cudaMemcpyHostToDevice,stream));budget.algebra();
        oracle_kernel<<<1,1,0,stream>>>(device_oracle);checked(cudaGetLastError());checked(cudaMemcpyAsync(host_oracle,device_oracle,sizeof(Oracle),cudaMemcpyDeviceToHost,stream));checked(cudaStreamSynchronize(stream));
        if(host_oracle->completed_serial!=input.serial) throw std::runtime_error("Shared algebra stream serial fault");return *host_oracle;
    }
};
std::string quote(const std::string& text) {std::string out="\"";for(char ch:text){if(ch=='\\'||ch=='\"')out+='\\';if(ch=='\n')out+="\\n";else out+=ch;}return out+'\"';}
void number(std::ostream& out,double value) {if(std::isfinite(value)) out<<value;else out<<"null";}
void write_text(const fs::path& path,const std::string& text) {std::ofstream out(path,std::ios::binary|std::ios::trunc);out.exceptions(std::ios::badbit|std::ios::failbit);out<<text;out.flush();}
void vector_json(std::ostream& out,p::V v) {out<<'['<<v.x<<','<<v.y<<','<<v.z<<']';}
std::vector<spec::Trial> catalog() {std::vector<spec::Trial> out;int total=0;for(auto stage:stages){int calls=0;for(const auto& t:spec::trials(stage)){out.push_back(t);calls+=4*t.steps;}if(calls!=spec::stage_budget(stage))throw std::runtime_error("Fixture stage cap mismatch");total+=calls;}if(total!=2024)throw std::runtime_error("Fixture total cap mismatch");return out;}
std::vector<uint8_t> binding() {
    cp::Writer out;out.array(spec::serialize_protocol());out.value(law::kPhysicsVersion);
    for(int value:{law::kRows,law::kComplianceSweeps,law::kBasisIterations,law::kActivationPasses,ref::kGeometryCap,ref::kKineticCap,ref::kSolveCap})out.value(value);
    for(double value:{double(law::kStiffness),double(law::kDamping),law::kRankRelative,law::kImpulseRoundoff,law::kResidualAbsolute,law::kResidualRelative,kMomentumAbsolute,kMomentumRelative,kOracleTolerance,kResidualMultiple,
        ref::kTranslationEps[0],ref::kTranslationEps[1],ref::kAngularEps[0],ref::kAngularEps[1],ref::kGapAbs,ref::kTranslationJAbs,ref::kAngularJAbs,ref::kJRelative,ref::kMassScaledTolerance,ref::kEffectiveMassRelative,ref::kEnergyRelative,ref::kRankPivot,ref::kStep,ref::kAvailableEnergyLimit,ref::kCalibrationSpring,ref::kMaximumNormalMass})out.value(value);
    return out.data;
}
std::string protocol_json(const std::vector<spec::Trial>& trials) {
    std::ostringstream out;out<<std::setprecision(17)<<"{\"scope\":\"approved contact-only 0.25s screen; no training/adoption/refinement\",\"maximum_physical_attempts\":2024,\"static_caps\":[256,256,96],\"source_trace_sha256\":"<<quote(spec::kSourceTraceSHA256)
        <<",\"models\":[\"legacy-v2\",\"implicit-v3\",\"version4-compliance-A k320 c1.9595918 full patch\",\"version4-hard-B zero compliance/zero restitution\"],\"backend\":[\"CPU\",\"CUDA nonblocking\"],\"stop_policy\":\"shared integrity/nonfinite/static failures stop all; old parity-specific failure quarantines that old arm; A/B-specific failure stops that arm in later cases; no retry/reallocation\",\"accounting\":\"CPU call/launch attempt charged before dispatch; allocation/upload without a launch costs zero; failed launches conservatively charged; proposals and commits distinguished; static reference/LCP calls separately counted\",\"parity_definition\":\"unchanged state/gap/safety/net-floor-force limits; generalized impulse divided by h and angular components by72mm body length uses unchanged force tolerance. Hard redundant point-dual allocations are recorded, not required unique. Free A/B bitwise comparison occurs only when both arms reach a case; missing peer is unproved, no third reference rollout.\",\"energy\":\"Scalar physical E=0.5*m*v^2+A_spring-signed_force*gap is gated against initial E. Actual external work=signed_force*displacement and numerical split-kick work are distinct. Whole passive bodies use actual T+joint soft stops+model springs; B has no contact spring. Contact substage uses physical M kinetic change/work. Complete gravity/current world-work balance is unproved; current-driven fresh cases are not passive.\",\"trials\":[";
    bool first=true;for(const auto& t:trials){if(!first)out<<',';first=false;const auto& f=t.fixture;const auto& b=f.initial;
        out<<"{\"stage\":"<<quote(spec::stage_name(t.stage))<<",\"id\":"<<f.id<<",\"name\":"<<quote(f.name)<<",\"h_s\":"<<t.h<<",\"steps_per_arm_backend\":"<<t.steps<<",\"scalar\":"<<(f.scalar?"true":"false")<<",\"scalar_mass_gap_velocity_signed_force\":["<<f.scalar_mass<<','<<f.scalar_gap<<','<<f.scalar_velocity<<','<<f.scalar_force<<"],\"position_model\":";vector_json(out,b.position);
        out<<",\"orientation_xyzw\":["<<b.orientation.x<<','<<b.orientation.y<<','<<b.orientation.z<<','<<b.orientation.w<<"],\"joints_rad\":[";for(int i=0;i<4;++i){if(i)out<<',';out<<b.angle[i];}out<<"],\"velocity_SI\":[";for(int i=0;i<10;++i){if(i)out<<',';out<<b.velocity[i];}out<<"],\"work_J\":"<<b.work<<",\"travel_model\":"<<b.travel<<",\"fluid\":["<<f.fluid.density<<','<<f.fluid.viscosity<<','<<f.fluid.gravity<<"],\"current_m_s\":";vector_json(out,f.current);out<<",\"action\":[";for(int i=0;i<4;++i){if(i)out<<',';out<<f.action[static_cast<size_t>(i)];}out<<"]}";}
    out<<"]}\n";return out.str();
}
void prepare(const fs::path& root) {
    if(fs::exists(root/"protocol.bin")||fs::exists(root/"protocol.json")||fs::exists(root/"run-started.json"))throw std::runtime_error("Prepared pilot already exists; refusing overwrite");
    fs::create_directories(root);const auto json=protocol_json(catalog());write_text(root/"protocol.json",json);cp::Writer out;out.array(binding());out.value(cp::checksum(std::vector<uint8_t>(json.begin(),json.end())));cp::write_atomic((root/"protocol.bin").string(),out.data);
    std::cout<<"Prepared immutable pilot; zero physical/static queries. Maximum2024 physical attempts.\n";
}

std::array<float,23> fields(const p::BodyState& b) {
    std::array<float,23> out{};int j=0;for(float x:{b.position.x,b.position.y,b.position.z,b.orientation.x,b.orientation.y,b.orientation.z,b.orientation.w})out[static_cast<size_t>(j++)]=x;
    for(float x:b.angle)out[static_cast<size_t>(j++)]=x;for(float x:b.velocity)out[static_cast<size_t>(j++)]=x;out[21]=b.work;out[22]=b.travel;return out;
}
bool bit_equal(const p::BodyState& a,const p::BodyState& b) {const auto x=fields(a),y=fields(b);return std::memcmp(x.data(),y.data(),sizeof(x))==0;}
bool close(double a,double b,double absolute=spec::kParityAbsolute) {return std::isfinite(a)&&std::isfinite(b)&&std::fabs(a-b)<=absolute+spec::kParityRelative*std::fabs(a);}
double allowance(double energy) {return std::max(spec::kEnergyAbsolute,spec::kEnergyRelative*std::fabs(energy));}
p::BodyState proposed(const Job& job) {return job.model>=2&&job.pilot.pose_ready?job.pilot.trial:job.after;}
law::ScalarState proposed_scalar(const Job& job) {return job.pilot.pose_ready?law::ScalarState{job.pilot.scalar_gap,job.pilot.scalar_velocity}:job.scalar_after;}
double spring_energy(const p::BodyState& body,int model) {
    if(model!=2)return 0;double energy=0;const auto shape=p::shape(body);
    for(int part=0;part<3;++part)for(int face=0;face<6;++face)for(int end=0;end<(part==0?1:2);++end){const auto n=c::normal(face);const double gap=p::dot(n,shape.link[part].center+c::support_offset(shape.link[part],part,n,end))+c::offset(face);if(gap<0)energy+=.5*law::kStiffness*gap*gap;}return energy;
}
double passive_energy(const p::BodyState& body,int model) {
    double energy=p::energy(body)+spring_energy(body,model);
    for(float angle:body.angle){const double d=double(angle)-p::clamp(angle,-p::limit,p::limit);energy+=.5e-4*d*d;}return energy;
}
const old::Row* rows(const Job& job) {return job.model<2?job.prior.rows:job.pilot.rows;}
double floor_force(const Job& job) {return (job.model<2?job.prior.floor_impulse:job.pilot.floor_impulse)/job.h;}
bool finite_result(const Job& job) {
    if(job.scalar){const auto s=proposed_scalar(job);return job.pilot.finite&&std::isfinite(s.gap)&&std::isfinite(s.velocity);}
    return p::finite_body(proposed(job))&&(job.model<2?job.prior.finite:job.pilot.finite);
}
struct Files {
    fs::path root;std::ofstream trace,points,algebra;
    explicit Files(const fs::path& path):root(path),trace(path/"trace.csv"),points(path/"contact-points.csv"),algebra(path/"algebra.csv") {
        for(auto* out:{&trace,&points,&algebra}){out->exceptions(std::ios::badbit|std::ios::failbit);*out<<std::setprecision(17);}
        trace<<"stage,case,model,backend,attempt,committed,pose_ready,ok,finite,input_hash,gap_m,other_gap_m,scalar_velocity_m_s,net_floor_force_N,residual,tolerance,iterations,basis_rank,kinetic_contact_before_J,kinetic_contact_after_J,kinetic_delta_J,contact_work_J,elastic_before_J,elastic_actual_after_J,contact_passivity_residual_J,scalar_split_load_kick_work_J,scalar_actual_external_work_J,proposal_positive_motor_work_delta_J,scalar_total_or_body_K_spring_softstop_J,independent_momentum_scaled_residual,independent_constitutive_residual,scalar_split_energy_ledger_J";
        for(int j=0;j<23;++j)trace<<",proposal_body_"<<j;for(int j=0;j<4;++j)trace<<",actual_torque_"<<j<<"_N_m";trace<<'\n';
        points<<"stage,case,model,backend,attempt,row,body,face,end,pre_gap_m,active,impulse_N_s,normal_force_N,pre_normal_velocity_m_s,after_normal_velocity_m_s,row_residual,effective_mass_kg\n";
        algebra<<"case,model,backend,row,impulse_N_s,normal_residual_m_s,reported_residual_m_s,rank,iterations,passed\n";
    }
};
bool algebra_oracles(Device& device,Budget& budget,Files& files,bool valid[4]) {
    bool all=true;
    for(int model=2;model<4;++model)for(int test=0;test<4;++test){
        Oracle input;input.model=model;input.count=test<2?1:2;const double inverse=1./.003;
        for(int i=0;i<input.count;++i){input.active[i]=true;input.q[i]=test==1?.05:(i==0?-.05:-.03);for(int j=0;j<input.count;++j)input.matrix[i][j]=test==3?inverse:(i==j?inverse:(inverse*.2));}
        if(test==3)input.q[1]=input.q[0];
        if(model==2)for(int i=0;i<input.count;++i)input.matrix[i][i]+=1./(double(spec::kStep)*(double(law::kDamping)+double(spec::kStep)*law::kStiffness));
        // Closed-form double reference; no pose/time is advanced.
        double expected[2]{};if(test==0)expected[0]=.05/input.matrix[0][0];
        if(test==2){const double d=input.matrix[0][0]*input.matrix[1][1]-input.matrix[0][1]*input.matrix[1][0];expected[0]=(-input.q[0]*input.matrix[1][1]+input.matrix[0][1]*input.q[1])/d;expected[1]=(-input.q[1]*input.matrix[0][0]+input.matrix[1][0]*input.q[0])/d;}
        if(test==3){if(model==2)expected[0]=expected[1]=.05/(input.matrix[0][0]+input.matrix[0][1]);else expected[0]=.05/inverse;}
        Oracle outputs[2];outputs[0]=input;budget.algebra();outputs[0].ok=law::solve_contact(input.count,outputs[0].matrix,outputs[0].q,outputs[0].active,selected_law(model),outputs[0].impulse,outputs[0].stats,kOracleTolerance);outputs[1]=device.algebra(input,budget);
        for(int backend=0;backend<2;++backend){const auto& result=outputs[backend];bool passed=result.ok&&result.stats.finite&&result.stats.converged;double total=0,expected_total=0;
            for(int i=0;i<input.count;++i){double residual=input.q[i];for(int j=0;j<input.count;++j)residual+=input.matrix[i][j]*result.impulse[j];const double error=result.impulse[i]>0?std::fabs(residual):std::max(0.0,-residual);total+=result.impulse[i];expected_total+=expected[i];passed=passed&&result.impulse[i]>=0&&error<=kResidualMultiple*kOracleTolerance;
                if(!(test==3&&model==3))passed=passed&&close(expected[i]/spec::kStep,result.impulse[i]/spec::kStep,spec::kGapForceAbsolute);
                files.algebra<<test<<','<<model<<','<<backend<<','<<i<<','<<result.impulse[i]<<','<<error<<','<<result.stats.residual<<','<<result.stats.basis_rank<<','<<result.stats.iterations<<','<<passed<<'\n';}
            passed=passed&&close(expected_total/spec::kStep,total/spec::kStep,spec::kGapForceAbsolute);if(!passed){valid[model]=false;all=false;}
        }
    }files.algebra.flush();return all;
}
struct StaticResult {bool passed=false,a_calibrated=true;std::string json;};
StaticResult static_reference(Budget& budget) {
    const auto inputs=spec::reference_states();auto report=ref::reference_report(std::vector<p::BodyState>(inputs.begin(),inputs.end()));
    StaticResult result;bool calibrated=report.passed;std::ostringstream fresh;fresh<<std::setprecision(17)<<'[';bool first=true;
    if(report.passed){calibrated=report.cases.size()==3&&report.cases[0].energy_bound.passed;for(const auto& state:report.cases)for(const auto& row:state.floor_rows)calibrated=calibrated&&row.mass_M_reference<=law::kMassCap;
        for(const auto& t:spec::trials(spec::Stage::Fresh)){const auto bound=ref::fresh_energy_bound(report,0,t.fixture.initial,t.fixture.current);calibrated=calibrated&&bound.fully_submerged&&bound.safe_floor&&std::isfinite(bound.available)&&bound.available<=law::kEnergyCap;
            if(!first)fresh<<',';first=false;fresh<<"{\"id\":"<<t.fixture.id<<",\"available_initial_J\":"<<bound.available<<",\"calm\":"<<(bound.calm?"true":"false")<<",\"calm_bound_applicable\":"<<(bound.applicable?"true":"false")<<",\"calm_bound_passed\":"<<(bound.passed?"true":"false")<<'}';}
    }
    budget.geometry=report.geometry_queries;budget.kinetic=report.kinetic_evaluations;budget.solves=report.matrix_solves;
    fresh<<']';result.passed=report.passed;result.a_calibrated=calibrated;result.json="{\"reference\":"+ref::to_json(report)+",\"A_calibration_valid\":"+(calibrated?"true":"false")+",\"fresh_initial_bounds\":"+fresh.str()+"}";
    if(budget.geometry>spec::kGeometryQueryCap||budget.kinetic>spec::kKineticQueryCap||budget.solves>spec::kStaticSolveCap)throw std::runtime_error("Shared static query cap exceeded");
    return result;
}

struct Check {
    bool finite=true,numeric=true,geometry=true,energy=true,nonadhesive=true;
    double gap=0,other_gap=0,momentum=0,constitutive=0,scalar_ledger=0,actual_external_work=0,passive=0;
};
Check inspect(const Job& job,double initial_passive) {
    Check out;out.finite=finite_result(job);if(!out.finite)return out;
    const auto* contact=rows(job);
    if(job.scalar){const auto s=proposed_scalar(job);out.gap=s.gap;
        const double u=job.pilot.scalar_free_velocity,pulse=job.pilot.total_impulse,delta=job.scalar_mass*(s.velocity-u);
        out.momentum=std::fabs(delta-pulse)/(kMomentumAbsolute+kMomentumRelative*std::max(std::fabs(delta),std::fabs(pulse)));
        out.constitutive=std::fabs((s.gap-job.scalar_before.gap)/job.h-s.velocity);
        const double before_elastic=job.model==2?.5*law::kStiffness*std::pow(std::min(0.0,job.scalar_before.gap),2):0;
        const double after_elastic=job.model==2?.5*law::kStiffness*std::pow(std::min(0.0,s.gap),2):0;
        out.scalar_ledger=.5*job.scalar_mass*(s.velocity*s.velocity-job.scalar_before.velocity*job.scalar_before.velocity)+after_elastic-before_elastic-job.pilot.external_work;
        out.actual_external_work=job.scalar_force*(s.gap-job.scalar_before.gap);
        out.passive=.5*job.scalar_mass*s.velocity*s.velocity+after_elastic-job.scalar_force*s.gap;
        out.energy=out.scalar_ledger<=allowance(.5*job.scalar_mass*job.scalar_before.velocity*job.scalar_before.velocity+before_elastic)&&out.passive<=initial_passive+allowance(initial_passive);
    }else{const auto body=proposed(job);const auto proof=c::inspect(body);out.gap=proof.floor_gap;out.other_gap=proof.min_other_gap;out.geometry=!proof.invalid&&!proof.excessive&&!proof.escape;
        out.passive=passive_energy(body,job.model);
        if(job.model>=2&&job.pilot.pose_ready){const auto system=p::assemble(job.before,job.fluid,job.current);
            for(int i=0;i<10;++i){double delta=0,impulse=0,scale=0;for(int j=0;j<10;++j){const double x=double(system.M[i][j])*(double(job.pilot.after_velocity[j])-job.pilot.free_velocity[j]);delta+=x;scale+=std::fabs(x);}for(int row=0;row<30;++row)impulse+=double(contact[row].jacobian[i])*contact[row].impulse;
                out.momentum=std::max(out.momentum,std::fabs(delta-impulse)/(kMomentumAbsolute+kMomentumRelative*std::max(scale,std::fabs(impulse))));}
        }
        if(job.fluid.density==0&&job.fluid.viscosity==0&&job.fluid.gravity==0&&job.current.x==0&&job.current.y==0&&job.current.z==0)
            out.energy=out.passive<=initial_passive+allowance(initial_passive);
    }
    if(job.model>=2){out.geometry=out.geometry&&job.pilot.geometry_valid&&out.gap>=-(job.model==3?spec::kHardGeometryGap:spec::kSafetyGap);
        const int count=job.scalar?1:30;
        for(int i=0;i<count;++i){const auto& row=contact[i];if(double(row.impulse)/job.h < -spec::kNegativeNormalForce)out.nonadhesive=false;
            if(row.active&&row.effective_mass>0){const double v=job.scalar?proposed_scalar(job).velocity:dot10(row.jacobian,job.pilot.after_velocity);
                const double d=job.model==2?1./(double(job.h)*(double(law::kDamping)+double(job.h)*law::kStiffness)):0;
                const double bias=job.model==2?double(law::kStiffness)*row.gap/(double(law::kDamping)+double(job.h)*law::kStiffness):std::max(0.0,double(row.gap))/job.h;
                const double residual=row.impulse*d+v+bias;
                out.constitutive=std::max(out.constitutive,std::fabs(std::min(double(row.impulse)*(1./row.effective_mass+d),residual)));}
        }
        const double kinetic_closure=std::fabs(job.pilot.kinetic_delta-job.pilot.contact_work);
        out.numeric=job.pilot.finite&&job.pilot.converged&&!job.pilot.missed_activation&&job.pilot.residual<=job.pilot.tolerance&&out.constitutive<=kResidualMultiple*job.pilot.tolerance&&out.momentum<=1&&kinetic_closure<=allowance(job.pilot.kinetic_before);
        out.energy=out.energy&&job.pilot.passivity_residual<=allowance(job.pilot.kinetic_before+job.pilot.elastic_before);
    }else{const auto proof=c::inspect(job.after);out.geometry=!proof.invalid&&!proof.excessive&&!proof.escape;
        if(job.model==1)out.numeric=job.ok&&job.prior.converged&&!job.prior.missed_activation&&job.prior.projected_residual<=job.prior.tolerance;
    }
    return out;
}
bool parity(const Job& cpu,const Job& gpu) {
    if(cpu.ok!=gpu.ok||cpu.committed!=gpu.committed)return false;
    if(cpu.scalar){const auto a=proposed_scalar(cpu),b=proposed_scalar(gpu);if(!close(a.gap,b.gap,spec::kGapForceAbsolute)||!close(a.velocity,b.velocity))return false;}
    else{const auto a=proposed(cpu),b=proposed(gpu);const auto x=fields(a),y=fields(b);for(size_t i=0;i<x.size();++i)if(!close(x[i],y[i]))return false;
        const auto ca=c::inspect(a),cb=c::inspect(b);if(ca.invalid!=cb.invalid||ca.excessive!=cb.excessive||ca.escape!=cb.escape||ca.floor_contact!=cb.floor_contact||ca.contact!=cb.contact||!close(ca.min_gap,cb.min_gap,spec::kGapForceAbsolute))return false;
    }
    if(!close(floor_force(cpu),floor_force(gpu),spec::kGapForceAbsolute))return false;
    // Hard impulses may be nonunique. Compare observable generalized reaction,
    // not a particular dual allocation among redundant contact rows.
    if(!cpu.scalar){const auto* a=rows(cpu);const auto* b=rows(gpu);float length=0;for(int i=0;i<3;++i)length+=p::body_geometry(i).length;
        for(int j=0;j<10;++j){double x=0,y=0;for(int i=0;i<30;++i){x+=double(a[i].jacobian[j])*a[i].impulse/cpu.h;y+=double(b[i].jacobian[j])*b[i].impulse/gpu.h;}if(j>=3){x/=length;y/=length;}if(!close(x,y,spec::kGapForceAbsolute))return false;}
    }
    return true;
}
void record(Files& files,const spec::Trial& trial,int backend,int tick,const Job& job,const Check& check) {
    const auto body=proposed(job);const auto scalar=proposed_scalar(job);const auto& t=job.pilot;
    const bool pose_ready=job.model<2||t.pose_ready;
    files.trace<<spec::stage_name(trial.stage)<<','<<trial.fixture.id<<','<<job.model<<','<<backend<<','<<tick<<','<<job.committed<<','<<pose_ready<<','<<job.ok<<','<<check.finite<<','<<job.echo_hash<<','<<check.gap<<','<<check.other_gap<<','<<(job.scalar?scalar.velocity:0)<<','<<floor_force(job)<<','<<(job.model<2?job.prior.projected_residual:t.residual)<<','<<(job.model<2?job.prior.tolerance:t.tolerance)<<','<<(job.model<2?job.prior.sweeps:t.iterations)<<','<<t.basis_rank<<','<<t.kinetic_before<<','<<t.kinetic_after<<','<<t.kinetic_delta<<','<<t.contact_work<<','<<t.elastic_before<<','<<t.elastic_actual_after<<','<<t.passivity_residual<<','<<t.external_work<<','<<check.actual_external_work<<','<<double(body.work)-job.before.work<<','<<check.passive<<','<<check.momentum<<','<<check.constitutive<<','<<check.scalar_ledger;
    for(float value:fields(body))files.trace<<','<<value;for(float value:job.action)files.trace<<','<<p::torque*p::clamp(value,-1,1);files.trace<<'\n';
    const auto* contact=rows(job);for(int i=0;i<(job.scalar?1:30);++i){const auto& row=contact[i];if(!row.active&&row.impulse==0&&row.gap>0)continue;
        files.points<<spec::stage_name(trial.stage)<<','<<trial.fixture.id<<','<<job.model<<','<<backend<<','<<tick<<','<<i<<','<<row.body<<','<<row.face<<','<<row.end<<','<<row.gap<<','<<row.active<<','<<row.impulse<<','<<row.impulse/job.h<<','<<(job.scalar?job.scalar_before.velocity:(job.model==0?double(row.velocity_free):dot10(row.jacobian,job.before.velocity)))<<','<<row.velocity_after<<','<<row.residual<<','<<row.effective_mass<<'\n';}
}
void retain_failure(Files& files,const spec::Trial& trial,int backend,int tick,const Job& job,const std::string& reason) {
    cp::Writer out;out.value(uint32_t{0x43504641});out.value(uint32_t{1});out.value(static_cast<uint32_t>(trial.stage));out.value(trial.fixture.id);out.value(job.model);out.value(backend);out.value(tick);out.value(job.committed);out.value(job.pilot.pose_ready);
    spec::detail::text(out,reason.c_str());spec::detail::write_body(out,job.before);spec::detail::write_body(out,job.after);spec::detail::write_body(out,proposed(job));
    for(double value:{job.scalar_before.gap,job.scalar_before.velocity,job.scalar_after.gap,job.scalar_after.velocity,proposed_scalar(job).gap,proposed_scalar(job).velocity})out.value(value);
    const auto name=std::string("first-failure-")+spec::stage_name(trial.stage)+"-"+std::to_string(trial.fixture.id)+"-"+std::to_string(job.model)+"-"+std::to_string(backend)+".bin";
    cp::write_atomic((files.root/name).string(),out.data);
}

struct Outcome {
    spec::Trial trial{};int model=0,backend=0,attempts=0,committed=0;bool passed=false,free_equivalence=false;
    std::string status="not_run",reason="prerequisite_or_global_stop";
    double minimum_gap=1e6,max_passive=0,max_restitution=0,max_momentum=0,max_constitutive=0;
};
void execute(Outcome& result,Device& device,Budget& budget,Files& files,std::vector<Job>& history,
    const std::vector<Job>* cpu,const std::vector<Job>* free_A) {
    const auto& trial=result.trial;const auto& fixture=trial.fixture;result.status="running";result.reason="";
    p::BodyState body=fixture.initial;law::ScalarState scalar{fixture.scalar_gap,fixture.scalar_velocity};
    const double initial_passive=fixture.scalar?.5*double(fixture.scalar_mass)*scalar.velocity*scalar.velocity+(result.model==2?.5*law::kStiffness*std::pow(std::min(0.0,scalar.gap),2):0)-double(fixture.scalar_force)*scalar.gap:passive_energy(body,result.model);
    result.max_passive=initial_passive;bool had_contact=false;
    for(int tick=1;tick<=trial.steps;++tick){
        if(cpu&&static_cast<size_t>(tick)>cpu->size()){result.reason="cpu_cuda_termination_mismatch";break;}
        Job input;input.before=body;input.scalar_before=scalar;input.scalar=fixture.scalar;input.model=result.model;input.fluid=fixture.fluid;input.current=fixture.current;input.h=trial.h;input.scalar_mass=fixture.scalar_mass;input.scalar_force=fixture.scalar_force;
        for(int j=0;j<4;++j)input.action[j]=fixture.action[static_cast<size_t>(j)];
        Job job;
        if(result.backend==0){job=input;budget.charge(trial.stage);++budget.dispatched;advance(job);if(job.committed)++budget.committed;}
        else job=device.step(input,trial.stage,budget);
        ++result.attempts;result.committed+=job.committed?1:0;const auto check=inspect(job,initial_passive);record(files,trial,result.backend,tick,job,check);history.push_back(job);
        result.minimum_gap=std::min(result.minimum_gap,check.gap);result.max_passive=std::max(result.max_passive,check.passive);result.max_momentum=std::max(result.max_momentum,check.momentum);result.max_constitutive=std::max(result.max_constitutive,check.constitutive);
        std::string failure;
        if(!check.finite)failure="shared_nonfinite";
        else if(!check.numeric)failure="model_equation_or_solver_failure";
        else if(!check.geometry||!job.ok)failure=result.model==3?"hard_actual_gap_or_geometry":"geometry_or_step_failure";
        else if(!check.nonadhesive&&result.model>=2)failure="negative_normal_impulse";
        else if(!check.energy&&result.model>=2)failure="energy_or_passivity_failure";
        else if(cpu&&!parity((*cpu)[static_cast<size_t>(tick-1)],job))failure="model_cpu_cuda_parity";
        if(failure.empty()&&job.scalar){had_contact=had_contact||job.pilot.total_impulse>0;if(had_contact&&fixture.id==1){result.max_restitution=std::max(result.max_restitution,std::max(0.0,proposed_scalar(job).velocity)/spec::kImpactSpeed);if(result.max_restitution>spec::kMaximumRestitution)failure="scalar_restitution";}}
        if(failure.empty()&&trial.stage==spec::Stage::FreeWater&&result.model==3&&free_A&&static_cast<size_t>(tick)<=free_A->size()){
            if(!bit_equal(proposed((*free_A)[static_cast<size_t>(tick-1)]),proposed(job)))failure="freewater_not_bit_exact";
            else result.free_equivalence=true;
        }
        if(!failure.empty()){result.reason=failure;retain_failure(files,trial,result.backend,tick,job,failure);if(failure=="shared_nonfinite")throw std::runtime_error("Shared nonfinite result; remaining comparisons stopped");break;}
        body=job.after;scalar=job.scalar_after;
    }
    if(result.reason.empty()&&cpu&&history.size()!=cpu->size())result.reason="model_cpu_cuda_count_mismatch";
    result.passed=result.reason.empty()&&result.attempts==trial.steps;
    result.free_equivalence=result.free_equivalence&&result.passed&&free_A&&free_A->size()==history.size();
    result.status=result.passed?"completed_short_screen":"failed";
    if(result.passed)result.reason="completed_registered_short_screen_not_full_contact_acceptance";
    files.trace.flush();files.points.flush();
}
void summary(const fs::path& root,const Budget& budget,const std::vector<Outcome>& outcomes,const bool valid[4],const std::string& fault) {
    std::ostringstream out;out<<std::setprecision(17)<<"{\"scope\":\"2,024-call contact-only screen; not sustained support, validated motor or adoption\",\"global_fault\":"<<quote(fault)<<",\"charged_physical_attempts\":"<<budget.charged<<",\"dispatched_calls\":"<<budget.dispatched<<",\"committed_advances\":"<<budget.committed<<",\"maximum\":2024,\"static_geometry_queries\":"<<budget.geometry<<",\"static_kinetic_evaluations\":"<<budget.kinetic<<",\"static_matrix_LCP_queries\":"<<budget.solves<<",\"static_caps\":[256,256,96],\"stage_charges\":[";
    for(int i=0;i<6;++i){if(i)out<<',';out<<budget.by_stage[i];}out<<"],\"arm_remaining_eligible\":[";for(int i=0;i<4;++i){if(i)out<<',';out<<(valid[i]?"true":"false");}out<<"],\"cases\":[";
    bool first=true;for(const auto& item:outcomes){if(!first)out<<',';first=false;out<<"{\"stage\":"<<quote(spec::stage_name(item.trial.stage))<<",\"id\":"<<item.trial.fixture.id<<",\"model\":"<<quote(model_name(item.model))<<",\"backend\":"<<item.backend<<",\"status\":"<<quote(item.status)<<",\"reason\":"<<quote(item.reason)<<",\"attempts\":"<<item.attempts<<",\"committed\":"<<item.committed<<",\"registered\":"<<item.trial.steps<<",\"screen_passed\":"<<(item.passed?"true":"false")<<",\"free_equivalence_observed\":"<<(item.free_equivalence?"true":"false")<<",\"minimum_gap_m\":";number(out,item.minimum_gap);out<<",\"max_passive_energy_J\":";number(out,item.max_passive);out<<",\"max_restitution\":";number(out,item.max_restitution);out<<",\"max_momentum_scaled_residual\":";number(out,item.max_momentum);out<<",\"max_constitutive_residual_m_s\":";number(out,item.max_constitutive);out<<'}';}
    out<<"]}\n";write_text(root/"summary.json",out.str());
}
int run(const fs::path& root) {
    for(const char* name:{"run-started.json","summary.json","trace.csv","contact-points.csv","algebra.csv","static-reference.json"})if(fs::exists(root/name))throw std::runtime_error("Pilot was already attempted; no retries or result replacement");
    const auto bytes=cp::read((root/"protocol.bin").string());cp::Reader reader{bytes};const auto saved=reader.array<uint8_t>(2000000);const auto hash=reader.value<uint64_t>();reader.finish();if(saved!=binding())throw std::runtime_error("Prepared pilot configuration differs from this executable");
    std::ifstream input(root/"protocol.json",std::ios::binary);if(!input)throw std::runtime_error("Missing prepared numeric JSON");const std::string json((std::istreambuf_iterator<char>(input)),std::istreambuf_iterator<char>());if(cp::checksum(std::vector<uint8_t>(json.begin(),json.end()))!=hash)throw std::runtime_error("Prepared JSON was modified");
    const auto list=catalog();std::vector<Outcome> outcomes;
    for(const auto& trial:list)for(int model=trial.stage==spec::Stage::Parity?0:2;model<(trial.stage==spec::Stage::Parity?2:4);++model)for(int backend=0;backend<2;++backend){Outcome row;row.trial=trial;row.model=model;row.backend=backend;outcomes.push_back(row);}
    write_text(root/"run-started.json","{\"one_shot\":true,\"physical_cap\":2024,\"training_calls\":0}\n");Budget budget;bool valid[4]={true,true,true,true};std::string fault;
    try{
        Files files(root);const auto reference=static_reference(budget);write_text(root/"static-reference.json",reference.json);
        if(!reference.passed)throw std::runtime_error("Shared independent static reference gate failed");valid[2]=reference.a_calibrated;
        Device device;algebra_oracles(device,budget,files,valid);size_t index=0;
        for(const auto& trial:list){std::vector<Job> free_A[2];
            for(int model=trial.stage==spec::Stage::Parity?0:2;model<(trial.stage==spec::Stage::Parity?2:4);++model){std::vector<Job> cpu;const bool run_pair=valid[model];
                for(int backend=0;backend<2;++backend){auto& result=outcomes[index++];
                    if(!run_pair){result.reason="arm_quarantined_by_earlier_prerequisite";continue;}
                    std::vector<Job> history;execute(result,device,budget,files,history,backend?&cpu:nullptr,trial.stage==spec::Stage::FreeWater&&model==3?&free_A[backend]:nullptr);
                    if(!result.passed)valid[model]=false;
                    if(trial.stage==spec::Stage::FreeWater&&model==3&&result.free_equivalence)for(auto& peer:outcomes)if(peer.trial.stage==spec::Stage::FreeWater&&peer.trial.fixture.id==trial.fixture.id&&peer.model==2&&peer.backend==backend)peer.free_equivalence=true;
                    if(!backend)cpu=history;if(trial.stage==spec::Stage::FreeWater&&model==2)free_A[backend]=std::move(history);
                }
            }
        }
    }catch(const std::exception& error){
        fault=error.what();
        for(bool& eligible:valid)eligible=false;
        for(auto& outcome:outcomes)if(outcome.status=="running"){
            outcome.status="stopped_global";
            if(outcome.reason.empty())outcome.reason=fault;
        }
    }
    summary(root,budget,outcomes,valid,fault);std::cout<<"Pilot charged "<<budget.charged<<"/2024; static "<<budget.geometry<<'/'<<budget.kinetic<<'/'<<budget.solves<<"; A="<<valid[2]<<" B="<<valid[3]<<"; fault="<<fault<<'\n';
    return !fault.empty()?1:((valid[2]||valid[3])?0:2);
}
} // namespace
int main(int argc,char** argv) {
    try{if(argc!=3||(std::string(argv[1])!="--prepare"&&std::string(argv[1])!="--run"))throw std::runtime_error("Usage: contact_pilot --prepare DIR | --run DIR");const fs::path root=fs::absolute(fs::u8path(argv[2]));if(std::string(argv[1])=="--prepare"){prepare(root);return 0;}return run(root);}
    catch(const std::exception& error){std::cerr<<"Pilot refused/failed: "<<error.what()<<'\n';return 1;}
}

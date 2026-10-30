// Opt-in, one-shot physical comparison. No world/session/learner/save ownership.
#include "contact_probe_cases.h"
#include "physics/contact_response.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace {
namespace p=cudarl::physics;namespace c=cudarl::contact;namespace r=cudarl::contact_response;
namespace spec=cudarl::contact_probe;namespace cp=cudarl::checkpoint;namespace fs=std::filesystem;
constexpr spec::Stage stages[]={spec::Stage::Primary,spec::Stage::Refinement,spec::Stage::HeldOut,spec::Stage::FreeWater,spec::Stage::Passivity};
constexpr double kEquationTolerance=4e-6,kImpulseOracleTolerance=1e-9;
constexpr double kMomentumAbsolute=1e-13,kMomentumRelative=2e-5;
struct Budget {
    int charged=0,dispatched=0,committed=0,stage[5]{};
    void charge(spec::Stage s) {
        const auto i=static_cast<unsigned>(s);
        if(charged>=spec::kMaximumSubsteps||stage[i]>=spec::stage_budget(s)) throw std::runtime_error("Registered physical-call budget exhausted");
        ++charged;++stage[i];
    }
};
struct Step {
    p::BodyState before{},after{};p::Parameters fluid{};p::V current{};
    float action[4]{},h=.005f;int model=0;uint64_t serial=0,completed_serial=0;
    bool ok=false,committed=false;r::Telemetry response{};
};
struct Oracle {
    float matrix[r::kRows][r::kRows]{},q[r::kRows]{},impulse[r::kRows]{};
    bool active[r::kRows]{};int count=0;uint64_t serial=0,completed_serial=0;
    bool ok=false;r::SolverStats stats{};
};
CUDARL_HD r::Telemetry legacy_contacts(const Step& step);
__global__ void advance_kernel(Step* step) {
    if(blockIdx.x||threadIdx.x) return;
    step->after=step->before;
    if(step->model==0) {
        c::substep(step->after,step->action,step->h,step->fluid,step->current);
        step->committed=true;step->ok=p::finite_body(step->after);
        step->response=legacy_contacts(*step);
    }
    else {step->ok=r::substep(step->after,step->action,step->h,step->fluid,step->current,&step->response);step->committed=step->ok;}
    step->completed_serial=step->serial;
}
__global__ void oracle_kernel(Oracle* oracle) {
    if(blockIdx.x||threadIdx.x) return;
    oracle->ok=r::solve_lcp(oracle->count,oracle->matrix,oracle->q,oracle->active,oracle->impulse,oracle->stats);
    oracle->completed_serial=oracle->serial;
}
void cuda_check(cudaError_t error) {if(error!=cudaSuccess) throw std::runtime_error(std::string("CUDA/stream failure: ")+cudaGetErrorString(error));}
struct Device {
    cudaStream_t stream=nullptr;Step* device_step=nullptr;Step* host_step=nullptr;Oracle* device_oracle=nullptr;Oracle* host_oracle=nullptr;
    uint64_t serial=0;
    Device() {
        try {
            int count=0;cuda_check(cudaGetDeviceCount(&count));if(count<1) throw std::runtime_error("A real CUDA device is required; no GPU skip is accepted");
            cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_step),sizeof(Step)));
            cuda_check(cudaMallocHost(reinterpret_cast<void**>(&host_step),sizeof(Step)));
            new(host_step) Step{};
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_oracle),sizeof(Oracle)));
            cuda_check(cudaMallocHost(reinterpret_cast<void**>(&host_oracle),sizeof(Oracle)));
            new(host_oracle) Oracle{};
        } catch(...) {release();throw;}
    }
    void release() noexcept {
        if(stream) cudaStreamSynchronize(stream);
        cudaFree(device_step);cudaFreeHost(host_step);cudaFree(device_oracle);cudaFreeHost(host_oracle);
        if(stream) cudaStreamDestroy(stream);
        device_step=nullptr;host_step=nullptr;device_oracle=nullptr;host_oracle=nullptr;stream=nullptr;
    }
    ~Device() {release();}
    Step advance(Step input,spec::Stage stage,Budget& budget) {
        input.serial=++serial;input.completed_serial=0;*host_step=input;
        cuda_check(cudaMemcpyAsync(device_step,host_step,sizeof(Step),cudaMemcpyHostToDevice,stream));
        budget.charge(stage); // Conservative charge immediately before launch, never retry.
        advance_kernel<<<1,1,0,stream>>>(device_step);cuda_check(cudaGetLastError());++budget.dispatched;
        cuda_check(cudaMemcpyAsync(host_step,device_step,sizeof(Step),cudaMemcpyDeviceToHost,stream));
        cuda_check(cudaStreamSynchronize(stream));
        if(host_step->completed_serial!=input.serial) throw std::runtime_error("Nonblocking-stream output serial is stale");
        if(host_step->committed) ++budget.committed;
        return *host_step;
    }
    Oracle equation(Oracle input) {
        input.serial=++serial;input.completed_serial=0;*host_oracle=input;
        cuda_check(cudaMemcpyAsync(device_oracle,host_oracle,sizeof(Oracle),cudaMemcpyHostToDevice,stream));
        oracle_kernel<<<1,1,0,stream>>>(device_oracle);cuda_check(cudaGetLastError());
        cuda_check(cudaMemcpyAsync(host_oracle,device_oracle,sizeof(Oracle),cudaMemcpyDeviceToHost,stream));
        cuda_check(cudaStreamSynchronize(stream));
        if(host_oracle->completed_serial!=input.serial) throw std::runtime_error("Equation oracle stream serial is stale");
        return *host_oracle;
    }
};
std::string quote(const std::string& input) {
    std::string out="\"";for(char ch:input) {if(ch=='\\'||ch=='\"') out+='\\';if(ch=='\n') out+="\\n";else out+=ch;}return out+'\"';
}
void number(std::ostream& out,double value) {if(std::isfinite(value)) out<<value;else out<<"null";}
void json_vector(std::ostream& out,p::V value) {out<<'['<<value.x<<','<<value.y<<','<<value.z<<']';}
void text_file(const fs::path& path,const std::string& value) {
    std::ofstream out(path,std::ios::binary|std::ios::trunc);out.exceptions(std::ios::badbit|std::ios::failbit);out<<value;out.flush();
}
std::vector<spec::Trial> catalog() {
    std::vector<spec::Trial> out;int total=0;
    for(auto stage:stages) {
        int calls=0;for(const auto& trial:spec::trials(stage)) {out.push_back(trial);calls+=trial.steps*4;}
        if(calls!=spec::stage_budget(stage)) throw std::runtime_error("Compiled fixture stage does not match registered budget");total+=calls;
    }
    if(total!=spec::kMaximumSubsteps) throw std::runtime_error("Compiled fixture total differs");return out;
}
std::vector<uint8_t> binding() {
    cp::Writer out;out.array(spec::serialize_protocol());out.value(r::kPhysicsVersion);out.value(r::kRows);
    out.value(r::kMaxSweeps);out.value(r::kMaxActivationPasses);out.value(r::kResidualAbsolute);out.value(r::kResidualRelative);
    out.value(kEquationTolerance);out.value(kImpulseOracleTolerance);out.value(kMomentumAbsolute);out.value(kMomentumRelative);
    return out.data;
}
std::string protocol_json(const std::vector<spec::Trial>& trials) {
    std::ostringstream out;out<<std::setprecision(17);
    out<<"{\"scope\":\"bounded contact physics, not learned movement\",\"protocol_version\":1,\"legacy_physics\":"<<c::kPhysicsVersion
       <<",\"candidate_physics\":"<<r::kPhysicsVersion<<",\"maximum_charged_substep_calls\":"<<spec::kMaximumSubsteps
       <<",\"stream\":\"real CUDA nonblocking; all copies and kernels on that stream\",\"run_policy\":\"one run; no gains/sweep/retry; finite case failure stops only that model-case; nonfinite/parity/stream/solver numerical failure stops all; all fixed independent stages otherwise continue\","
       <<"\"accounting\":\"charge each CPU attempt and GPU launch attempt before dispatch; allocation/upload failure before launch costs zero; rejected launches retain a conservative charge; committed poses reported separately; zero-integration equation kernels are not physical substeps\","
       <<"\"source_catalog_sha256\":"<<quote(spec::kArchivedCatalogSha256)<<",\"equation_residual_limit_m_s\":"<<kEquationTolerance
       <<",\"impulse_oracle_limit_N_s\":"<<kImpulseOracleTolerance<<",\"momentum_absolute\":"<<kMomentumAbsolute<<",\"momentum_relative\":"<<kMomentumRelative
       <<",\"candidate_sweeps\":"<<r::kMaxSweeps<<",\"activation_passes\":"<<r::kMaxActivationPasses
       <<",\"residual_absolute\":"<<r::kResidualAbsolute<<",\"residual_relative\":"<<r::kResidualRelative
       <<",\"overlap_limit_m\":"<<spec::kOverlapTolerance<<",\"state_parity_absolute\":"<<spec::kPhysicalParityAbs<<",\"parity_relative\":"<<spec::kParityRelative
       <<",\"gap_force_parity_absolute\":"<<spec::kGapForceParityAbs<<",\"normal_force_negative_allowance_N\":"<<spec::kNormalForceNegativeAllowance
       <<",\"freewater_clearance_m\":"<<spec::kFreeWaterClearance<<",\"support_load_fraction\":"<<spec::kSupportLoadFraction<<",\"support_vertical_speed_m_s\":"<<spec::kSupportVerticalSpeed
       <<",\"independent_dynamic_residual_multiple\":4,\"stage_budgets\":[9600,21600,9600,1600,1200],\"model_labels\":[\"legacy explicit spring/closing dashpot\",\"predictive unilateral implicit Kelvin-Voigt\"],\"backend_labels\":[\"CPU\",\"CUDA\"],\"support_window\":\"final third; actual floor contact every sample; mean actual net floor force>=registered fraction of deficit; max head|vy|<=registered speed\","
       <<"\"work_definition\":\"candidate contact_work is frozen-A free-to-constrained work; legacy contact_work is backward-endpoint impulse work; neither is a complete gravity/fluid energy balance. Passive-energy ceiling applies only to passivity stage.\","
       <<"\"passivity_absolute_J\":"<<spec::kEnergyAbsoluteAllowance<<",\"passivity_relative\":"<<spec::kEnergyRelativeAllowance
       <<",\"refinement_position_m\":"<<spec::kTerminalPositionRefinement<<",\"refinement_penetration_m\":"<<spec::kPenetrationRefinement<<",\"trials\":[";
    bool first=true;
    for(const auto& trial:trials) {
        if(!first) out<<',';first=false;const auto& f=trial.fixture;const auto& b=f.initial;
        out<<"{\"stage\":"<<quote(spec::stage_name(trial.stage))<<",\"id\":"<<f.id<<",\"source_case\":"<<f.source_case<<",\"name\":"<<quote(f.name)
           <<",\"support\":"<<(f.support?"true":"false")<<",\"h_s\":"<<trial.h<<",\"steps_per_solver_backend\":"<<trial.steps<<",\"position_model\":";json_vector(out,b.position);
        out<<",\"quaternion_xyzw\":["<<b.orientation.x<<','<<b.orientation.y<<','<<b.orientation.z<<','<<b.orientation.w<<"],\"joint_angles_rad\":[";
        for(int j=0;j<4;++j) {if(j) out<<',';out<<b.angle[j];}out<<"],\"generalized_velocity_SI\":[";
        for(int j=0;j<10;++j) {if(j) out<<',';out<<b.velocity[j];}out<<"],\"initial_positive_work_J\":"<<b.work<<",\"initial_travel_model\":"<<b.travel;
        out<<",\"fluid\":["<<f.fluid.density<<','<<f.fluid.viscosity<<','<<f.fluid.gravity<<"],\"current_m_s\":";json_vector(out,f.current);
        out<<",\"normalized_action\":[";for(int j=0;j<4;++j) {if(j) out<<',';out<<f.action[static_cast<size_t>(j)];}out<<"]}";
    }
    out<<"]}\n";return out.str();
}
void prepare(const fs::path& root) {
    if(fs::exists(root/"protocol.bin")||fs::exists(root/"protocol.json")||fs::exists(root/"run-started.json")) throw std::runtime_error("Refusing to replace a prepared/executed comparison");
    fs::create_directories(root);const auto trials=catalog();const auto json=protocol_json(trials);
    text_file(root/"protocol.json",json);cp::Writer out;out.array(binding());out.value(cp::checksum(std::vector<uint8_t>(json.begin(),json.end())));
    cp::write_atomic((root/"protocol.bin").string(),out.data);
    std::cout<<"Prepared "<<trials.size()<<" fixed trials; maximum43600 charged physical calls; no dynamics or training.\n";
}

std::array<float,23> body_fields(const p::BodyState& b) {
    std::array<float,23> out{};int i=0;
    for(float v:{b.position.x,b.position.y,b.position.z,b.orientation.x,b.orientation.y,b.orientation.z,b.orientation.w}) out[static_cast<size_t>(i++)]=v;
    for(float v:b.angle) out[static_cast<size_t>(i++)]=v;for(float v:b.velocity) out[static_cast<size_t>(i++)]=v;
    out[21]=b.work;out[22]=b.travel;return out;
}
bool same_body(const p::BodyState& a,const p::BodyState& b) {
    const auto x=body_fields(a),y=body_fields(b);return std::memcmp(x.data(),y.data(),sizeof(x))==0;
}
double body_error(const p::BodyState& a,const p::BodyState& b,bool& close) {
    const auto x=body_fields(a),y=body_fields(b);double maximum=0;close=true;
    for(size_t i=0;i<x.size();++i) {maximum=std::max(maximum,std::fabs(double(x[i])-y[i]));close=close&&spec::parity_close(x[i],y[i]);}return maximum;
}
CUDARL_HD inline double product(const float* a,const float* b) {double out=0;for(int j=0;j<p::dofs;++j) out+=double(a[j])*b[j];return out;}
CUDARL_HD r::Telemetry legacy_contacts(const Step& step) {
    r::Telemetry out;out.finite=p::finite_body(step.after);out.converged=out.finite;
    const auto shape=p::shape(step.before);int i=0;
    for(int body=0;body<3;++body) {
        const auto& link=shape.link[body];const int points=body==0?1:2;const float weight=1.f/points;
        for(int face=0;face<6;++face) for(int end=0;end<points;++end) {
            auto& row=out.rows[i++];row.body=body;row.face=face;row.end=end;
            const auto n=c::normal(face),offset=c::support_offset(link,body,n,end);
            row.gap=p::dot(n,link.center+offset)+c::offset(face);row.k=c::kContactStiffness*weight;
            row.c=2.f*c::kContactDampingRatio*sqrtf(c::kContactStiffness*link.mass)*weight;
            p::V point_jacobian[p::dofs];
            for(int j=0;j<p::dofs;++j) {point_jacobian[j]=link.linear[j]+p::cross(link.angular[j],offset);row.jacobian[j]=p::dot(n,point_jacobian[j]);}
            // Match the old FLOAT branch exactly, independent double force sums.
            const float before=p::dot(p::multiply(point_jacobian,step.before.velocity),n);
            const double after=product(row.jacobian,step.after.velocity);row.velocity_free=before;
            row.velocity_after=static_cast<float>(after);row.active=row.gap<0;
            if(row.active) {
                const double force=-double(row.k)*row.gap-(before<0?double(row.c)*after:0);
                row.impulse=static_cast<float>(step.h*force);++out.active_rows;
                out.total_impulse+=row.impulse;out.linear_impulse+=n*row.impulse;if(face==2) out.floor_impulse+=row.impulse;
                out.contact_work+=double(row.impulse)*after;
                const double g0=row.gap<0?double(row.gap):0.0;
                const double endpoint=double(row.gap)+step.h*after,g1=endpoint<0?endpoint:0.0;
                out.elastic_before+=.5*row.k*g0*g0;out.elastic_linear_after+=.5*row.k*g1*g1;
            }
        }
    }
    out.passivity_residual=out.contact_work+out.elastic_linear_after-out.elastic_before;return out;
}
double momentum_residual(const Step& step,const r::Telemetry& contact) {
    auto free=p::assemble(step.before,step.fluid,step.current);
    for(int j=0;j<4;++j) {
        const float displacement=step.before.angle[j]-p::clamp(step.before.angle[j],-p::limit,p::limit);
        free.force[6+j]+=p::torque*p::clamp(step.action[j],-1,1)-1e-4f*displacement;
        free.R[6+j][6+j]+=p::torque/12.f+(displacement!=0?3e-6f:0.f);
    }
    double maximum=0;
    for(int i=0;i<p::dofs;++i) {
        double rhs=double(step.h)*(double(free.force[i])-free.C[i]),lhs=0,load=0,scale=0;
        for(int j=0;j<p::dofs;++j) {
            rhs+=double(free.M[i][j])*step.before.velocity[j];
            const double term=(double(free.M[i][j])+double(step.h)*free.R[i][j])*step.after.velocity[j];lhs+=term;scale+=std::fabs(term);
        }
        for(const auto& row:contact.rows) load+=double(row.jacobian[i])*row.impulse;
        scale=std::max(scale,std::fabs(rhs)+std::fabs(load));
        maximum=std::max(maximum,std::fabs(lhs-rhs-load)/(kMomentumAbsolute+kMomentumRelative*scale));
    }
    return maximum;
}
double equation_residual(const Step& step,const r::Telemetry& contact,bool& missed) {
    double maximum=0;missed=false;
    for(const auto& row:contact.rows) {
        const double velocity=product(row.jacobian,step.after.velocity);
        if(row.active) {
            const double value=double(row.impulse)/(double(step.h)*(row.c+double(step.h)*row.k))+velocity+double(row.k)*row.gap/(row.c+double(step.h)*row.k);
            const double residual=row.impulse>0?std::fabs(value):std::max(0.0,-value);maximum=std::max(maximum,residual);
        } else if(double(row.gap)+step.h*velocity < -4*double(step.h)*contact.tolerance-1e-8) missed=true;
    }
    return maximum;
}
struct Files {
    std::ofstream steps,points,oracles;
    explicit Files(const fs::path& root):steps(root/"trace.csv"),points(root/"contact-points.csv"),oracles(root/"equation-oracle.csv") {
        for(auto* out:{&steps,&points,&oracles}) {out->exceptions(std::ios::badbit|std::ios::failbit);*out<<std::setprecision(17);}
        steps<<"stage,case,source_case,h_s,model,backend,substep,time_s,ok,finite,floor_gap_m,other_gap_m,contact,net_floor_force_N,net_total_force_N,min_normal_force_N,lcp_residual_m_s,lcp_tolerance_m_s,independent_equation_residual_m_s,scaled_momentum_residual,missed_activation,active_rows,sweeps,activation_passes,contact_work_J,elastic_before_J,elastic_linear_after_J,contact_passivity_residual_J,passive_mechanical_energy_J,positive_motor_work_delta_J,head_speed_m_s,com_speed_m_s,angular_speed_rad_s,joint_rate_max_rad_s";
        for(int j=0;j<23;++j) steps<<",body_"<<j;
        for(int j=0;j<3;++j) steps<<",link"<<j<<"_x_m,link"<<j<<"_y_m,link"<<j<<"_z_m,link"<<j<<"_vx_m_s,link"<<j<<"_vy_m_s,link"<<j<<"_vz_m_s,link"<<j<<"_omega_rad_s";
        for(int j=0;j<4;++j) steps<<",actual_torque"<<j<<"_N_m";steps<<'\n';
        points<<"stage,case,h_s,model,backend,substep,row,body,face,end,pre_gap_m,active,impulse_N_s,force_N,pre_normal_v_m_s,post_normal_v_m_s,linear_gap_after_m,k_N_m,c_N_s_m,row_solver_residual_m_s,effective_mass_kg\n";
        oracles<<"case,backend,row,impulse_N_s,independent_double_expected_N_s,residual_m_s,reported_residual_m_s,converged,passed\n";
    }
};
void preflight(Device& device,Files& files) {
    for(int test=0;test<3;++test) {
        Oracle input;input.count=test==0?1:2;
        // Independent hand-declared KV equations, with double closed-form solve.
        const double h=.005,k[2]={.8,.4},damping[2]={.08,.04},gap[2]={-.0001,.00005};
        const double W[2][2]={{.5,.1},{.1,.4}},velocity[2]={-.02,test==2?.03:-.03};
        for(int i=0;i<input.count;++i) {
            input.active[i]=true;input.q[i]=static_cast<float>(velocity[i]+k[i]*gap[i]/(damping[i]+h*k[i]));
            for(int j=0;j<input.count;++j) input.matrix[i][j]=static_cast<float>(W[i][j]+(i==j?1/(h*(damping[i]+h*k[i])):0));
        }
        double expected[2]{};
        if(test==0||test==2) expected[0]=-double(input.q[0])/input.matrix[0][0];
        else {
            const double determinant=double(input.matrix[0][0])*input.matrix[1][1]-double(input.matrix[0][1])*input.matrix[1][0];
            expected[0]=(-double(input.q[0])*input.matrix[1][1]+double(input.matrix[0][1])*input.q[1])/determinant;
            expected[1]=(-double(input.q[1])*input.matrix[0][0]+double(input.matrix[1][0])*input.q[0])/determinant;
        }
        Oracle outputs[2];outputs[0]=input;outputs[0].ok=r::solve_lcp(input.count,outputs[0].matrix,outputs[0].q,outputs[0].active,outputs[0].impulse,outputs[0].stats);outputs[1]=device.equation(input);
        for(int backend=0;backend<2;++backend) for(int i=0;i<input.count;++i) {
            const auto& result=outputs[backend];double value=input.q[i];for(int j=0;j<input.count;++j) value+=double(input.matrix[i][j])*result.impulse[j];
            const double residual=result.impulse[i]>0?std::fabs(value):std::max(0.0,-value);
            const bool passed=result.ok&&result.stats.finite&&result.stats.converged&&result.impulse[i]>=0&&std::fabs(result.impulse[i]-expected[i])<=kImpulseOracleTolerance&&residual<=kEquationTolerance;
            files.oracles<<test<<','<<backend<<','<<i<<','<<result.impulse[i]<<','<<expected[i]<<','<<residual<<','<<result.stats.residual<<','<<result.stats.converged<<','<<passed<<'\n';
            if(!passed) throw std::runtime_error("Independent zero-integration CPU/CUDA equation oracle failed");
        }
    }
    files.oracles.flush();
}
struct Outcome {
    spec::Trial trial;int model=0,backend=0,done=0,committed=0,window_count=0;bool safe=true,support_contact=true,accepted=false;
    p::BodyState final{};double minimum_gap=1e6,window_force=0,window_vy=0,max_energy=0,initial_energy=0,max_parity=0;
    std::string reason="not_run";
};
void trace(Files& files,const spec::Trial& trial,int model,int backend,int tick,const Step& step,const r::Telemetry& contact,double equation,double momentum,bool missed) {
    const auto proof=c::inspect(step.after);const bool finite=p::finite_body(step.after);const auto geometry=p::shape(step.after);
    p::V com{};float mass=0,joint=0;double min_force=0;
    for(const auto& link:geometry.link) {com+=p::multiply(link.linear,step.after.velocity)*link.mass;mass+=link.mass;}
    for(int j=6;j<10;++j) joint=std::max(joint,std::fabs(step.after.velocity[j]));
    for(const auto& row:contact.rows) min_force=std::min(min_force,double(row.impulse)/step.h);
    auto& out=files.steps;out<<spec::stage_name(trial.stage)<<','<<trial.fixture.id<<','<<trial.fixture.source_case<<','<<trial.h<<','<<model<<','<<backend<<','<<tick<<','<<double(tick-(step.committed?0:1))*trial.h<<','<<step.ok<<','<<finite<<','<<proof.floor_gap<<','<<proof.min_other_gap<<','<<proof.contact<<','<<contact.floor_impulse/step.h<<','<<contact.total_impulse/step.h<<','<<min_force<<','<<contact.projected_residual<<','<<contact.tolerance<<','<<equation<<','<<momentum<<','<<missed<<','<<contact.active_rows<<','<<contact.sweeps<<','<<contact.activation_passes<<','<<contact.contact_work<<','<<contact.elastic_before<<','<<contact.elastic_linear_after<<','<<contact.passivity_residual<<','<<spec::passive_energy(step.after)<<','<<double(step.after.work)-step.before.work<<','<<p::norm({step.after.velocity[0],step.after.velocity[1],step.after.velocity[2]})<<','<<p::norm(com/mass)<<','<<p::norm({step.after.velocity[3],step.after.velocity[4],step.after.velocity[5]})<<','<<joint;
    for(float value:body_fields(step.after)) out<<','<<value;
    for(const auto& link:geometry.link) {const auto v=p::multiply(link.linear,step.after.velocity);out<<','<<link.center.x<<','<<link.center.y<<','<<link.center.z<<','<<v.x<<','<<v.y<<','<<v.z<<','<<p::norm(link.omega);}
    for(float value:step.action) out<<','<<p::torque*p::clamp(value,-1,1);out<<'\n';
    for(int index=0;index<r::kRows;++index) {
        const auto& row=contact.rows[index];const double vn=product(row.jacobian,step.after.velocity),linear_gap=double(row.gap)+step.h*vn;
        if(!row.active&&row.impulse==0&&linear_gap>0) continue;
        files.points<<spec::stage_name(trial.stage)<<','<<trial.fixture.id<<','<<trial.h<<','<<model<<','<<backend<<','<<tick<<','<<index<<','<<row.body<<','<<row.face<<','<<row.end<<','<<row.gap<<','<<row.active<<','<<row.impulse<<','<<row.impulse/step.h<<','<<(model==0?double(row.velocity_free):product(row.jacobian,step.before.velocity))<<','<<vn<<','<<linear_gap<<','<<row.k<<','<<row.c<<','<<row.residual<<','<<row.effective_mass<<'\n';
    }
}

void execute(const spec::Trial& trial,int model,int backend,Device& device,Budget& budget,Files& files,
    Outcome& result,std::vector<Step>& history,const std::vector<Step>* cpu_reference,const std::vector<Step>* legacy_reference) {
    result.trial=trial;result.model=model;result.backend=backend;result.final=trial.fixture.initial;
    result.initial_energy=spec::passive_energy(result.final);result.max_energy=result.initial_energy;result.reason="running";
    const auto initial=c::inspect(result.final);result.minimum_gap=initial.min_gap;
    if(initial.invalid) {result.reason="initial_nonfinite";throw std::runtime_error("A frozen initial fixture is invalid");}
    if(initial.excessive||initial.escape) {result.safe=false;result.reason="initial_unsafe";return;}
    for(int tick=1;tick<=trial.steps;++tick) {
        if(cpu_reference&&static_cast<size_t>(tick)>cpu_reference->size()) {result.reason="cpu_cuda_termination_mismatch";throw std::runtime_error("GPU trajectory would continue past the CPU's first failure");}
        Step input;input.before=result.final;input.after=input.before;input.fluid=trial.fixture.fluid;input.current=trial.fixture.current;input.h=trial.h;input.model=model;
        for(int j=0;j<4;++j) input.action[j]=trial.fixture.action[static_cast<size_t>(j)];
        Step step;
        if(backend==0) {
            budget.charge(trial.stage);++budget.dispatched;step=input;
            if(model==0) {c::substep(step.after,step.action,step.h,step.fluid,step.current);step.committed=true;step.ok=p::finite_body(step.after);}
            else {step.ok=r::substep(step.after,step.action,step.h,step.fluid,step.current,&step.response);step.committed=step.ok;}
            if(step.committed) ++budget.committed;
        } else step=device.advance(input,trial.stage,budget);
        if(model==0&&backend==0) step.response=legacy_contacts(step);
        result.done=tick;result.committed+=step.committed?1:0;result.final=step.after;
        bool missed=false;const double equation=model==1&&step.ok?equation_residual(step,step.response,missed):0;
        const double momentum=step.ok?momentum_residual(step,step.response):0;
        trace(files,trial,model,backend,tick,step,step.response,equation,momentum,missed);history.push_back(step);
        const auto proof=c::inspect(step.after);
        if(!p::finite_body(step.after)||!step.response.finite) {result.reason="nonfinite";throw std::runtime_error("Nonfinite physical/solver output; stopping entire comparison");}
        if(model==1&&(!step.ok||!step.response.converged||step.response.missed_activation||missed||
            equation>4*step.response.tolerance||step.response.projected_residual>step.response.tolerance)) {
            result.reason="solver_numerical_failure";throw std::runtime_error("Candidate solver/equation/activation failure; stopping entire comparison");
        }
        if(momentum>1||!std::isfinite(momentum)) {result.reason="momentum_equation_failure";throw std::runtime_error("Independent physical momentum equation check failed");}
        if(model==1) for(const auto& row:step.response.rows) if(double(row.impulse)/step.h < -spec::kNormalForceNegativeAllowance) {
            result.reason="adhesive_impulse";throw std::runtime_error("Candidate contact impulse violated nonadhesion");
        }
        if(cpu_reference&&static_cast<size_t>(tick)<=cpu_reference->size()) {
            const auto& reference=(*cpu_reference)[static_cast<size_t>(tick-1)];bool close=false;
            result.max_parity=std::max(result.max_parity,body_error(reference.after,step.after,close));
            const auto reference_proof=c::inspect(reference.after);
            close=close&&reference.ok==step.ok&&reference_proof.invalid==proof.invalid&&reference_proof.excessive==proof.excessive&&reference_proof.escape==proof.escape
                &&reference_proof.contact==proof.contact&&reference_proof.floor_contact==proof.floor_contact
                &&spec::parity_close(reference.response.floor_impulse/reference.h,step.response.floor_impulse/step.h,spec::kGapForceParityAbs)
                &&spec::parity_close(reference_proof.min_gap,proof.min_gap,spec::kGapForceParityAbs);
            for(int row=0;row<r::kRows;++row) close=close&&spec::parity_close(reference.response.rows[row].impulse/reference.h,step.response.rows[row].impulse/step.h,spec::kGapForceParityAbs);
            if(!close) {result.reason="cpu_cuda_parity";throw std::runtime_error("CPU/CUDA state/gap/net-force mismatch; stopping entire comparison");}
        }
        result.minimum_gap=std::min(result.minimum_gap,double(proof.min_gap));result.max_energy=std::max(result.max_energy,spec::passive_energy(step.after));
        if(tick-1>=spec::support_window_begin(trial)) {
            ++result.window_count;result.support_contact=result.support_contact&&proof.floor_contact;
            result.window_force+=double(step.response.floor_impulse)/step.h;result.window_vy=std::max(result.window_vy,std::fabs(double(step.after.velocity[1])));
        }
        if(proof.excessive||proof.escape) {result.safe=false;result.reason=proof.escape?"escaped_volume":"overlap_limit";break;}
        if(trial.stage==spec::Stage::FreeWater) {
            if(proof.min_gap<=spec::kFreeWaterClearance) {result.safe=false;result.reason="freewater_clearance";break;}
            if(model==1&&(!legacy_reference||static_cast<size_t>(tick)>legacy_reference->size()||!same_body((*legacy_reference)[static_cast<size_t>(tick-1)].after,step.after))) {
                result.safe=false;result.reason="freewater_not_bit_exact";break;
            }
        }
        if(trial.stage==spec::Stage::Passivity&&result.max_energy>result.initial_energy+spec::passive_energy_allowance(result.initial_energy)) {
            result.safe=false;result.reason="passivity_energy_ceiling";break;
        }
    }
    if(cpu_reference&&history.size()!=cpu_reference->size()) {result.reason="cpu_cuda_count_mismatch";throw std::runtime_error("CPU/CUDA attempted trajectory lengths differ");}
    result.accepted=result.safe&&result.done==trial.steps;
    if(result.accepted&&trial.fixture.support) {
        const double mean=result.window_count?result.window_force/result.window_count:0;
        result.accepted=result.support_contact&&result.window_count==trial.steps/3&&mean>=spec::kSupportLoadFraction*spec::known_buoyancy_deficit(trial.fixture)&&result.window_vy<=spec::kSupportVerticalSpeed;
        if(!result.accepted) result.reason="support_load_contact_or_settling";
    }
    if(result.accepted) result.reason="passed_bounded_case";
    files.steps.flush();files.points.flush();
}
void refinement_gate(std::vector<Outcome>& outcomes) {
    for(auto& refined:outcomes) if(refined.trial.stage==spec::Stage::Refinement&&refined.trial.h<.002f) {
        const Outcome* primary=nullptr;for(const auto& item:outcomes) if(item.trial.stage==spec::Stage::Primary&&item.trial.fixture.source_case==refined.trial.fixture.source_case&&item.model==refined.model&&item.backend==refined.backend) primary=&item;
        if(!primary||primary->done!=primary->trial.steps||refined.done!=refined.trial.steps||!primary->safe||!refined.safe) {
            if(refined.accepted) refined.reason="refinement_missing_safe_terminal";
            refined.accepted=false;continue;
        }
        const double position=p::norm((primary->final.position-refined.final.position)*p::scale);
        const double penetration=std::fabs(std::max(0.0,-primary->minimum_gap)-std::max(0.0,-refined.minimum_gap));
        if(position>spec::kTerminalPositionRefinement||penetration>spec::kPenetrationRefinement) {if(refined.accepted) refined.reason="refinement_position_or_penetration";refined.accepted=false;}
    }
}
bool summary(const fs::path& root,const Budget& budget,const std::vector<Outcome>& outcomes,size_t expected,const std::string& fault) {
    bool accepted=fault.empty()&&outcomes.size()==expected;for(const auto& item:outcomes) if(item.model==1&&!item.accepted) accepted=false;
    std::ostringstream out;out<<std::setprecision(17)<<"{\"scope\":\"bounded contact physics only; no learned actor\",\"candidate_accepted\":"<<(accepted?"true":"false")
        <<",\"global_fault\":"<<quote(fault)<<",\"charged_substep_attempts\":"<<budget.charged<<",\"dispatched_substep_calls\":"<<budget.dispatched<<",\"committed_pose_advances\":"<<budget.committed
        <<",\"maximum\":"<<spec::kMaximumSubsteps<<",\"expected_model_backend_cases\":"<<expected<<",\"recorded_cases\":"<<outcomes.size()<<",\"stage_charges\":[";
    for(int i=0;i<5;++i) {if(i) out<<',';out<<budget.stage[i];}out<<"],\"cases\":[";
    bool first=true;for(const auto& item:outcomes) {
        if(!first) out<<',';first=false;out<<"{\"stage\":"<<quote(spec::stage_name(item.trial.stage))<<",\"case\":"<<item.trial.fixture.id<<",\"source_case\":"<<item.trial.fixture.source_case
            <<",\"h_s\":"<<item.trial.h<<",\"model\":"<<item.model<<",\"backend\":"<<item.backend<<",\"attempted_steps\":"<<item.done<<",\"committed_steps\":"<<item.committed<<",\"registered_steps\":"<<item.trial.steps
            <<",\"accepted\":"<<(item.accepted?"true":"false")<<",\"reason\":"<<quote(item.reason)<<",\"minimum_gap_m\":";number(out,item.minimum_gap);
        out<<",\"support_window_samples\":"<<item.window_count<<",\"support_window_all_floor_contact\":"<<(item.support_contact?"true":"false")<<",\"support_mean_net_force_N\":";number(out,item.window_count?item.window_force/item.window_count:0);
        out<<",\"known_buoyancy_deficit_N\":"<<spec::known_buoyancy_deficit(item.trial.fixture)<<",\"support_max_head_abs_vy_m_s\":";number(out,item.window_vy);
        out<<",\"initial_passive_energy_J\":";number(out,item.initial_energy);out<<",\"maximum_passive_energy_J\":";number(out,item.max_energy);out<<",\"maximum_cpu_cuda_defined_field_error\":";number(out,item.max_parity);out<<'}';
    }
    out<<"]}\n";text_file(root/"summary.json",out.str());return accepted;
}
int run(const fs::path& root) {
    if(fs::exists(root/"run-started.json")||fs::exists(root/"summary.json")||fs::exists(root/"trace.csv")||fs::exists(root/"contact-points.csv")||fs::exists(root/"equation-oracle.csv")) throw std::runtime_error("This prepared comparison has already been attempted; no retry/overwrite is allowed");
    const auto bytes=cp::read((root/"protocol.bin").string());cp::Reader reader{bytes};const auto saved=reader.array<uint8_t>(2000000);const auto json_hash=reader.value<uint64_t>();reader.finish();
    if(saved!=binding()) throw std::runtime_error("Prepared numeric fixtures/criteria/candidate binding differ from this executable");
    std::ifstream source(root/"protocol.json",std::ios::binary);if(!source) throw std::runtime_error("Prepared JSON protocol is missing");
    const std::string json((std::istreambuf_iterator<char>(source)),std::istreambuf_iterator<char>());
    if(cp::checksum(std::vector<uint8_t>(json.begin(),json.end()))!=json_hash) throw std::runtime_error("Prepared numeric JSON was modified");
    const auto trials=catalog();text_file(root/"run-started.json","{\"one_shot\":true,\"training_calls\":0}\n");
    Budget budget;std::vector<Outcome> outcomes;std::string fault;
    try {
        Files files(root);Device device;preflight(device,files);
        for(const auto& trial:trials) {
            std::vector<Step> legacy[2];
            for(int model=0;model<2;++model) {
                std::vector<Step> cpu;
                for(int backend=0;backend<2;++backend) {
                    std::vector<Step> history;history.reserve(static_cast<size_t>(trial.steps));outcomes.emplace_back();
                    execute(trial,model,backend,device,budget,files,outcomes.back(),history,backend==1?&cpu:nullptr,model==1?&legacy[backend]:nullptr);
                    if(backend==0) cpu=history;
                    if(model==0) legacy[backend]=std::move(history);
                }
            }
        }
        refinement_gate(outcomes);
    } catch(const std::exception& error) {fault=error.what();if(!outcomes.empty()&&outcomes.back().reason=="running") outcomes.back().reason="global_dispatch_or_io_fault";}
    const bool accepted=summary(root,budget,outcomes,trials.size()*4,fault);
    std::cout<<"Charged "<<budget.charged<<"/43600 physical attempts; dispatched "<<budget.dispatched<<"; committed "<<budget.committed<<"; candidate accepted="<<accepted<<"; global fault="<<fault<<'\n';
    return !fault.empty()?1:(accepted?0:2);
}
} // namespace
int main(int argc,char** argv) {
    try {
        if(argc!=3||(std::string(argv[1])!="--prepare"&&std::string(argv[1])!="--run")) throw std::runtime_error("Usage: contact_response_probe --prepare DIR | --run DIR");
        const fs::path root=fs::absolute(fs::u8path(argv[2]));
        if(std::string(argv[1])=="--prepare") {prepare(root);return 0;}
        return run(root);
    } catch(const std::exception& error) {std::cerr<<"Contact comparison refused/failed: "<<error.what()<<'\n';return 1;}
}

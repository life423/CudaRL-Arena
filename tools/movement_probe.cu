#include "motor/movement_model.h"
#include "motor/forward_task.h"
#include "core/checkpoint.h"
#include "cuda/cuda_memory.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <array>
namespace {
namespace m=cudarl::movement;namespace p=cudarl::physics;namespace s=cudarl::simulation;namespace cp=cudarl::checkpoint;namespace fs=std::filesystem;
constexpr int cases=6,controls=160,cap=22400;
const char* names[cases]={"legacy","new_servo_old_water","new_model_wet","new_model_dry","new_model_rest","new_model_fine"};
struct State {
    p::BodyState body{};uint64_t attempts=0;bool failed=false;int failure_code=0;
    float gap=1e6f,peak=0,requested_peak=0,residual=0,angle_peak[4]{};
    double signed_work=0,positive_work=0,fluid_loss=0,external_work=0,numerical_loss=0,energy_error=0;
    p::V com_at2{},com_at6{};float tracking_sq[4]{};int tracking_count=0;
};
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
CUDARL_HD p::V com(const State& state,bool composite){const auto shape=composite?m::shape(state.body):p::shape(state.body);p::V result;float mass=0;
    for(const auto& l:shape.link){result+=l.center*l.mass;mass+=l.mass;}return result/mass;}
CUDARL_HD void control(State& state,int id,int tick){const bool composite=id>=2;const auto initial=m::prescribed(float(tick)*.05f);const int steps=id==5?20:10;const float h=.05f/float(steps);p::Parameters fluid;if(id==3)fluid={0,0,0};
    for(int k=0;k<steps&&!state.failed;++k){++state.attempts;const float time=float(tick)*.05f+float(k)*h;auto reference=m::prescribed(time);if(id==4)reference={};
        m::Telemetry t;
        if(id==0){const s::Gains gains;float normalized[4];for(int j=0;j<4;++j){t.requested[j]=gains.kp[j]*(initial.angle[j]-state.body.angle[j])-gains.kd[j]*state.body.velocity[6+j];t.torque[j]=p::clamp(t.requested[j],-p::torque,p::torque);normalized[j]=t.torque[j]/p::torque;}
            const float before=state.body.work;cudarl::contact::substep(state.body,normalized,h,fluid);t.positive_work=state.body.work-before;t.minimum_gap=m::clearance(state.body,false);
            state.failed=!p::finite_body(state.body)||t.minimum_gap<.01f;
        }else {state.failed=!m::substep(state.body,reference,h,fluid,t,composite);state.failure_code=t.failure_code;}
        state.gap=fminf(state.gap,t.minimum_gap);state.residual=fmaxf(state.residual,t.solve_residual);
        if(state.failed)return; // failed candidate work is not committed work
        for(int j=0;j<4;++j){state.peak=fmaxf(state.peak,fabsf(t.torque[j]));state.requested_peak=fmaxf(state.requested_peak,fabsf(t.requested[j]));state.angle_peak[j]=fmaxf(state.angle_peak[j],fabsf(state.body.angle[j]));
            if(time>=2&&time<6){auto expected=m::prescribed(time+h);if(id==4)expected={};const float error=expected.angle[j]-state.body.angle[j];state.tracking_sq[j]+=error*error;}}
        if(time>=2&&time<6)++state.tracking_count;
        state.signed_work+=t.signed_work;state.positive_work+=t.positive_work;state.fluid_loss+=t.fluid_loss;state.external_work+=t.external_work;state.numerical_loss+=t.numerical_loss;
        if(id!=0){const double error=m::kinetic(state.body,composite)-state.signed_work+state.fluid_loss+state.numerical_loss-state.external_work;
            state.energy_error=fmax(state.energy_error,fabs(error));if(fabs(error)>.02*(state.positive_work+state.fluid_loss)+2e-10){state.failed=true;state.failure_code=9;}}
    }
    if(tick==39)state.com_at2=com(state,composite);if(tick==119)state.com_at6=com(state,composite);
}
__global__ void kernel(State* state,int tick){const int i=threadIdx.x;if(i<cases)control(state[i],i,tick);}
std::vector<uint8_t> binding(){cp::Writer w;w.value(uint32_t{0x4D4F5636});w.value(m::kVersion);for(int x:{cases,controls,cap,10})w.value(x);
    for(float x:{.6f,1.f,8.f,2.f,3.f,.9f,.01f,.0072f,.48f,.12f,.05f,.02f,2e-10f})w.value(x);return w.data;}
void static_checks(){p::BodyState body;const auto geometry=m::shape(body);float mass=0;for(const auto& link:geometry.link)mass+=link.mass;
    require(fabsf(mass-.00253744692532f)<1e-9f,"Independent composite mass check failed");require(fabsf(m::body_geometry(2).half_height-.00635f)<1e-8f,"Fin span mismatch");
    body.velocity[4]=1;body.velocity[6]=-.5f;const auto a=m::assemble(body);double loss=0;
    for(int i=0;i<10;++i)for(int j=0;j<10;++j){require(p::finite_number(a.M[i][j])&&p::finite_number(a.R[i][j]),"Invalid matrices");loss+=body.velocity[i]*a.R[i][j]*body.velocity[j];}
    require(loss>0,"Rotating-point drag must dissipate energy");require(fabsf(m::clearance(p::BodyState{})-.0736f)<1e-6f,"Independent initial clearance mismatch");}
int prepare(const fs::path& root){static_checks();fs::create_directories(root);require(!fs::exists(root/"protocol.bin")&&!fs::exists(root/"run-started"),"Already prepared/attempted");cp::write_atomic((root/"protocol.bin").string(),binding());
    std::ofstream out(root/"protocol.json");out<<"{\"model\":6,\"scope\":\"six fixed mechanical comparisons; NO learning/runtime adoption\",\"cases\":[\"legacy\",\"servo_old_water\",\"new_wet\",\"new_dry\",\"new_rest\",\"new_fine\"],\"amplitude_rad\":0.6,\"Hz\":1,\"seconds\":8,\"substep_cap\":22400,\"wall_seconds_cap\":10,\"minimum_speed_m_s\":0.0072,\"minimum_amplitude_rad\":0.48,\"tracking_RMS_max_rad\":0.12,\"refinement_relative_max\":0.05,\"retries\":0}\n";return 0;}
int run(const fs::path& root){static_checks();require(cp::read((root/"protocol.bin").string())==binding(),"Protocol changed");require(fs::create_directory(root/"run-started"),"Already attempted; no retry");const auto start=std::chrono::steady_clock::now();
    std::array<State,cases> cpu{},gpu{};State initial;const auto origin=com(initial,true);int charged=0,ticks=0;float parity=0;std::string fault;
    try{int count=0;CUDA_CHECK_THROW(cudaGetDeviceCount(&count));require(count>0,"Real CUDA required");device_ptr<State> device(cases);cudaStream_t stream=nullptr;
        std::ofstream trace(root/"trace.csv");trace.exceptions(std::ios::badbit|std::ios::failbit);trace<<std::setprecision(10)<<"backend,case,tick,time_s,forward_m,lateral_m,depth_m,j0,j1,j2,j3,peak_torque_Nm,positive_work_J,min_gap_m,energy_error_J\n";
        for(int tick=0;tick<controls;++tick){require(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<10,"Wall-clock cap reached");require(charged+140<=cap,"Substep cap reached");charged+=140;
            for(int id=0;id<cases;++id){control(cpu[id],id,tick);require(!cpu[id].failed,"CPU physical/solver/energy/tracking-domain failure");}
            device.copy_from_host(gpu.data(),cases);kernel<<<1,32,0,stream>>>(device.get(),tick);CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaStreamSynchronize(stream));device.copy_to_host(gpu.data(),cases);
            for(int id=0;id<cases;++id){require(!gpu[id].failed,"CUDA physical/solver/energy/tracking-domain failure");const auto& a=cpu[id].body;const auto& b=gpu[id].body;
                const float av[]={a.position.x,a.position.y,a.position.z,a.orientation.x,a.orientation.y,a.orientation.z,a.orientation.w};const float bv[]={b.position.x,b.position.y,b.position.z,b.orientation.x,b.orientation.y,b.orientation.z,b.orientation.w};
                for(int j=0;j<7;++j){const float score=fabsf(av[j]-bv[j])/(2e-4f+2e-3f*fabsf(av[j]));parity=fmaxf(parity,score);require(score<=1,"Backend pose mismatch");}
                for(int j=0;j<10;++j)require(fabsf(a.velocity[j]-b.velocity[j])<=2e-4f+2e-3f*fabsf(a.velocity[j]),"Backend velocity mismatch");
                for(int j=0;j<4;++j)require(fabsf(a.angle[j]-b.angle[j])<=2e-4f+2e-3f*fabsf(a.angle[j]),"Backend joint mismatch");require(cpu[id].attempts==gpu[id].attempts,"Backend count mismatch");
                for(int backend=0;backend<2;++backend){const auto& state=backend?gpu[id]:cpu[id];const auto delta=com(state,id>=2)-origin;
                    trace<<(backend?"CUDA":"CPU")<<','<<names[id]<<','<<tick+1<<','<<float(tick+1)*.05f<<','<<delta.x<<','<<delta.z<<','<<delta.y;for(float q:state.body.angle)trace<<','<<q;
                    trace<<','<<state.peak<<','<<state.positive_work<<','<<state.gap<<','<<state.energy_error<<'\n';}
            }++ticks;
        }
    }catch(const std::exception& e){fault=e.what();}
    bool accepted=fault.empty();const float speed=(cpu[2].com_at6.x-cpu[2].com_at2.x)/4;
    const float fine_speed=(cpu[5].com_at6.x-cpu[5].com_at2.x)/4;float refinement=fabsf(speed-fine_speed)/fmaxf(fabsf(fine_speed),1e-6f);
    accepted=accepted&&speed>=.0072f&&refinement<=.05f;
    for(int id:{2,3})for(int j:{0,2})accepted=accepted&&cpu[id].angle_peak[j]>=.48f&&sqrtf(cpu[id].tracking_sq[j]/float(cpu[id].tracking_count?cpu[id].tracking_count:1))<=.12f;
    std::ofstream out(root/"summary.json");out<<std::setprecision(10)<<"{\"model\":6,\"fault\":"<<std::quoted(fault)<<",\"accepted_for_learning\":"<<(accepted?"true":"false")<<",\"ticks\":"<<ticks<<",\"charged_substeps\":"<<charged<<",\"substep_cap\":22400,\"max_pose_parity_score\":"<<parity<<",\"central_speed_m_s\":"<<speed<<",\"speed_refinement_relative\":"<<refinement<<",\"states\":[";uint64_t actual=0;
    for(int backend=0;backend<2;++backend)for(int id=0;id<cases;++id){if(backend||id)out<<',';const auto& state=backend?gpu[id]:cpu[id];actual+=state.attempts;const auto delta=com(state,id>=2)-origin;
        out<<"{\"backend\":"<<std::quoted(backend?"CUDA":"CPU")<<",\"case\":"<<std::quoted(names[id])<<",\"substeps\":"<<state.attempts<<",\"failure_code\":"<<state.failure_code<<",\"max_solve_residual\":"<<state.residual<<",\"forward_m\":"<<delta.x<<",\"lateral_m\":"<<delta.z<<",\"depth_m\":"<<delta.y<<",\"central_speed_m_s\":"<<(state.com_at6.x-state.com_at2.x)/4<<",\"peak_torque_Nm\":"<<state.peak<<",\"requested_peak_Nm\":"<<state.requested_peak<<",\"positive_work_J\":"<<state.positive_work<<",\"fluid_loss_J\":"<<state.fluid_loss<<",\"BE_loss_J\":"<<state.numerical_loss<<",\"max_energy_error_J\":"<<state.energy_error<<",\"min_gap_m\":"<<state.gap<<",\"tracking_RMS_rad\":[";
        for(int j=0;j<4;++j){if(j)out<<',';out<<sqrtf(state.tracking_sq[j]/float(state.tracking_count?state.tracking_count:1));}out<<"]}";}
    out<<"],\"actual_attempted_substeps\":"<<actual<<",\"wall_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<",\"closed\":true,\"retries\":0}\n";
    std::cout<<"Movement model6 closed: charged="<<charged<<"/22400, actual="<<actual<<", speed="<<speed*1000<<"mm/s, accepted="<<accepted<<", fault="<<fault<<'\n';return fault.empty()?0:3;
}
struct Numerical {double raw_residual=0,symmetric_residual=0,asymmetry=0;bool failed=false;};
CUDARL_HD Numerical numerical_case(int id){p::BodyState body;body.orientation=p::axis_rotation({0,1,0},.017f*float(id+1))*p::axis_rotation({0,0,1},.031f*float(id));
    for(int j=0;j<4;++j)body.angle[j]=.2f*sinf(float(id+j));for(int j=0;j<10;++j)body.velocity[j]=.03f*cosf(float(id+j));
    const auto system=m::assemble(body);double raw[10][10]{},sym[10][10]{},discard[10]{},rhs[10]{},fixed_rhs[10]{},x[10]{},fixed_x[10]{};Numerical out;
    m::discrete_system(system,body,.0025f,sym,discard);
    for(int i=0;i<10;++i)for(int j=0;j<10;++j)raw[i][j]=system.M[i][j]+.0025f*system.R[i][j];
    for(int i=0;i<10;++i)for(int j=0;j<10;++j){const double exact=.01*sin(double(id+j));rhs[i]+=raw[i][j]*exact;fixed_rhs[i]+=sym[i][j]*exact;out.asymmetry=fmax(out.asymmetry,fabs(raw[i][j]-raw[j][i]));}
    out.failed=!m::solve(raw,rhs,x,10)||!m::solve(sym,fixed_rhs,fixed_x,10);
    for(int i=0;i<10;++i){double r=-rhs[i],symmetric_r=-fixed_rhs[i],scale=fabs(rhs[i])+1e-15,symmetric_scale=fabs(fixed_rhs[i])+1e-15;
        for(int j=0;j<10;++j){r+=raw[i][j]*x[j];symmetric_r+=sym[i][j]*fixed_x[j];scale+=fabs(raw[i][j]*x[j]);symmetric_scale+=fabs(sym[i][j]*fixed_x[j]);}
        out.raw_residual=fmax(out.raw_residual,fabs(r)/scale);out.symmetric_residual=fmax(out.symmetric_residual,fabs(symmetric_r)/symmetric_scale);}
    return out;}
__global__ void numerical_kernel(Numerical* out){const int id=threadIdx.x;if(id<16)out[id]=numerical_case(id);}
int numerical_check(){static_checks();device_ptr<Numerical> device(16);numerical_kernel<<<1,32>>>(device.get());CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaDeviceSynchronize());std::vector<Numerical> gpu(16);device.copy_to_host(gpu.data(),16);
    double raw=0,fixed=0,asymmetry=0;for(int id=0;id<16;++id){const auto cpu=numerical_case(id);for(const auto& result:{cpu,gpu[id]}){require(!result.failed,"Static SPD factorization failed");require(result.symmetric_residual<=1e-8,"Symmetric matrix residual failed unchanged gate");raw=fmax(raw,result.raw_residual);fixed=fmax(fixed,result.symmetric_residual);asymmetry=fmax(asymmetry,result.asymmetry);}}
    std::cout<<std::setprecision(12)<<"32/32 CPU/CUDA static matrix solves passed; raw residual="<<raw<<", corrected residual="<<fixed<<", max matrix asymmetry="<<asymmetry<<"; ZERO physical advances\n";return 0;}
}
int main(int argc,char** argv){try{require(argc==3,"Use --prepare/--run DIR or --numeric-check UNUSED");const fs::path root=fs::absolute(argv[2]);if(std::string(argv[1])=="--numeric-check")return numerical_check();if(std::string(argv[1])=="--prepare")return prepare(root);require(std::string(argv[1])=="--run","Unknown mode");return run(root);}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}

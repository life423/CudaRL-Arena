// Small gated correction probe. Reuses the existing scalar oracle and fixtures.
#include "physics/contact_events.h"
#include "contact_validation_reference.h"
#include "contact_pilot_cases.h"
#include "contact_probe_cases.h"
#include "core/checkpoint.h"
#include "contact_momentum_audit.h"
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
namespace e=cudarl::contact_events;namespace r=cudarl::contact_validation_reference;namespace fs=std::filesystem;
namespace audit=cudarl::contact_momentum_audit;namespace p=cudarl::physics;namespace cp=cudarl::checkpoint;
constexpr int kScalarCap=3000,kReferenceCap=256;
struct ScalarJob {e::ScalarState state{},before{};double load=0;e::Stats stats{};e::Counter counter{};bool ok=false;uint64_t serial=0,echo=0;};
struct AuditJob {p::BodyState state{},before{};audit::Audit result{};bool ok=false;};
struct BodyJob {p::BodyState state{},before{};e::BodyStats stats{};e::Counter counter{};bool ok=false;};
__global__ void scalar_kernel(ScalarJob* job){if(!blockIdx.x&&!threadIdx.x){job->before=job->state;job->echo=job->serial;job->ok=e::scalar_step(job->state,.003,job->load,.005,job->stats,job->counter);}}
__global__ void audit_kernel(AuditJob* job){if(!blockIdx.x&&!threadIdx.x){job->before=job->state;const float zero[4]{};job->ok=audit::substep(job->state,zero,.00125f,{990,.001f,9.81f},{},job->result);}}
__global__ void body_kernel(BodyJob* job){if(!blockIdx.x&&!threadIdx.x){job->before=job->state;const float zero[4]{};job->ok=e::body_step(job->state,zero,.005,{990,.001f,9.81f},{},job->stats,job->counter);}}
void launch(ScalarJob* job,cudaStream_t stream){scalar_kernel<<<1,1,0,stream>>>(job);}
void launch(AuditJob* job,cudaStream_t stream){audit_kernel<<<1,1,0,stream>>>(job);}
void launch(BodyJob* job,cudaStream_t stream){body_kernel<<<1,1,0,stream>>>(job);}
void checked(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
template<class T> struct Device {
    cudaStream_t stream=nullptr;T* input=nullptr;T* output=nullptr;
    Device(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));checked(cudaMalloc(reinterpret_cast<void**>(&input),sizeof(T)));checked(cudaMallocHost(reinterpret_cast<void**>(&output),sizeof(T)));}
    ~Device(){if(stream)cudaStreamSynchronize(stream);cudaFree(input);cudaFreeHost(output);if(stream)cudaStreamDestroy(stream);}
    T step(T x){*output=x;checked(cudaMemcpyAsync(input,output,sizeof(x),cudaMemcpyHostToDevice,stream));launch(input,stream);checked(cudaGetLastError());checked(cudaMemcpyAsync(output,input,sizeof(x),cudaMemcpyDeviceToHost,stream));checked(cudaStreamSynchronize(stream));return *output;}
};
struct Record {int id=0,backend=0,outer=0,inner=0,events=0;double gap0=0,compression=0,restitution=0,impulse=0,gap_error=0,speed_error=0,compression_error=0,restitution_error=0,impulse_error=0;bool passed=false;};
Record scalar_case(int id,int backend,int& spent,int& queries,std::ofstream& trace,Device<ScalarJob>* device=nullptr){
    Record result;result.id=id;result.backend=backend;const bool impact=id==1||id>=3;
    result.gap0=id==1?25e-6:(id==3?.05*.0002:(id==4?.05*.0017:(id==5?.05*.0043:0)));
    const auto type=impact?r::Case::Impact:(id==0?r::Case::Support:r::Case::Separation);
    r::Reference oracle(type,result.gap0);queries+=static_cast<int>(oracle.query_count());
    if(!oracle.report().passed)throw std::runtime_error("Invalid independent oracle");
    ScalarJob job;bool physical_valid=true;job.load=id==0?.0003:0;job.state={result.gap0,impact?-.05:(id==2?.05:0)};
    for(int tick=1;tick<=4;++tick){
        if(spent>=kScalarCap||queries>=kReferenceCap)throw std::runtime_error("Frozen scalar/reference budget exhausted");
        ++spent;++result.outer;job.counter={0,static_cast<uint64_t>(kScalarCap-spent+1)};job.serial=static_cast<uint64_t>(id*100+backend*10+tick);job.echo=0;const auto before=job.state;
        if(backend)job=device->step(job);else{job.before=job.state;job.echo=job.serial;job.ok=e::scalar_step(job.state,.003,job.load,.005,job.stats,job.counter);}
        if(job.before.gap!=before.gap||job.before.velocity!=before.velocity||job.echo!=job.serial)throw std::runtime_error("Scalar upload/consumer integrity failure");
        spent+=static_cast<int>(job.counter.used)-1;result.inner+=static_cast<int>(job.stats.trials);
        result.events+=job.stats.arrivals+job.stats.releases;
        result.compression=std::max(result.compression,job.stats.maxcompression);
        if(impact)result.restitution=std::max(result.restitution,std::max(0.,job.state.velocity)/.05);
        result.impulse+=job.stats.impulse;
        const double energy=.5*.003*job.state.velocity*job.state.velocity+.5*320*std::pow(std::min(0.,job.state.gap),2)+job.load*job.state.gap;
        const double initial=impact?.5*.003*.05*.05:(id==2?.5*.003*.05*.05:0);
        const auto expected=oracle.sample(tick*.005);++queries;
        result.gap_error=std::max(result.gap_error,std::abs(job.state.gap-expected.gap_m));
        result.speed_error=std::max(result.speed_error,std::abs(job.state.velocity-expected.velocity_m_s));
        trace<<id<<','<<backend<<','<<tick<<','<<job.state.gap<<','<<job.state.velocity<<','<<expected.gap_m<<','<<expected.velocity_m_s<<','<<job.stats.impulse<<','<<job.stats.maxcompression<<','<<job.stats.trials<<','<<job.stats.arrivals<<','<<job.stats.releases<<','<<job.ok<<'\n';trace.flush();
        if(!job.ok||!std::isfinite(job.state.gap)||!std::isfinite(job.state.velocity)||result.compression>.0005||std::max(energy,job.stats.max_energy)>initial+std::max(1e-10,.001*initial)){physical_valid=false;break;}
        if(result.gap_error>r::kTrajectoryGapAbsolute||result.speed_error>r::kTrajectoryVelocityAbsolute)break;
    }
    const auto report=oracle.report();const auto endpoint=oracle.sample(result.outer*.005);++queries;
    result.compression_error=std::abs(result.compression-(impact?report.max_compression_m:endpoint.max_compression_m));
    result.restitution_error=std::abs(result.restitution-(impact?report.restitution:0));result.impulse_error=std::abs(result.impulse-endpoint.normal_impulse_N_s);
    result.passed=physical_valid&&job.ok&&result.outer==4&&result.gap_error<=r::kTrajectoryGapAbsolute&&result.speed_error<=r::kTrajectoryVelocityAbsolute&&result.compression_error<=r::compression_tolerance(impact?report.max_compression_m:endpoint.max_compression_m)&&result.restitution_error<=r::kRestitutionAbsolute&&result.impulse_error<=r::impulse_tolerance(endpoint.normal_impulse_N_s);
    return result;
}
void record(std::ofstream& out,const Record& v){out<<std::setprecision(17)<<"{\"id\":"<<v.id<<",\"backend\":"<<v.backend<<",\"passed\":"<<(v.passed?"true":"false")<<",\"outer_calls\":"<<v.outer<<",\"inner_trials\":"<<v.inner<<",\"events\":"<<v.events<<",\"initial_gap_m\":"<<v.gap0<<",\"compression_m\":"<<v.compression<<",\"restitution\":"<<v.restitution<<",\"impulse_N_s\":"<<v.impulse<<",\"max_gap_error_m\":"<<v.gap_error<<",\"max_speed_error_m_s\":"<<v.speed_error<<",\"compression_error_m\":"<<v.compression_error<<",\"restitution_error\":"<<v.restitution_error<<",\"impulse_error_N_s\":"<<v.impulse_error<<"}\n";out.flush();}
int scalar(const fs::path& root,bool first_only){
    fs::create_directories(root);const auto tag=first_only?"impact":"remaining";
    if(fs::exists(root/(std::string(tag)+".jsonl")))throw std::runtime_error("One-shot stage already exists; no retries");
    std::ofstream out(root/(std::string(tag)+".jsonl")),trace(root/(std::string(tag)+".csv"));out.exceptions(std::ios::badbit|std::ios::failbit);trace.exceptions(std::ios::badbit|std::ios::failbit);trace<<std::setprecision(17)<<"id,backend,tick,gap_m,velocity_m_s,reference_gap_m,reference_velocity_m_s,impulse_N_s,max_compression_m,inner_trials,arrivals,releases,ok\n";
    int spent=0,queries=0;bool passed=true;
    if(first_only){const auto v=scalar_case(1,0,spent,queries,trace);record(out,v);passed=v.passed;}
    else{
        std::ifstream prior(root/"impact-counts.txt");int previous_spent=0,previous_queries=0,gate=0;prior>>previous_spent>>previous_queries>>gate;if(!prior||!gate)throw std::runtime_error("First impact did not pass");spent=previous_spent;queries=previous_queries;
        Device<ScalarJob> device;for(int id=0;id<6&&passed;++id)for(int backend=0;backend<2&&passed;++backend){if(id==1&&backend==0)continue;const auto v=scalar_case(id,backend,spent,queries,trace,&device);record(out,v);passed=v.passed;}
    }
    std::ofstream counts(root/(std::string(tag)+"-counts.txt"));counts<<spent<<' '<<queries<<' '<<passed<<'\n';
    std::cout<<tag<<": pass="<<passed<<", cumulative charged="<<spent<<"/3000, reference="<<queries<<"/256\n";return passed?0:3;
}
p::BodyState read_failure(const fs::path& path,int body_index=0){const auto bytes=cp::read(path.string());cp::Reader reader{bytes};
    if(reader.value<uint32_t>()!=0x43564146||reader.value<uint32_t>()!=1)throw std::runtime_error("Wrong saved failure schema");
    for(int k=0;k<4;++k)(void)reader.value<int>();for(int k=0;k<23*body_index;++k)(void)reader.value<float>();p::BodyState b;float v[23]{};for(float& value:v)value=reader.value<float>();
    b.position={v[0],v[1],v[2]};b.orientation={v[3],v[4],v[5],v[6]};for(int k=0;k<4;++k)b.angle[k]=v[7+k];for(int k=0;k<10;++k)b.velocity[k]=v[11+k];b.work=v[21];b.travel=v[22];return b;}
int momentum(const fs::path& root){
    if(fs::exists(root/"momentum.csv"))throw std::runtime_error("Audit already attempted; no retries");
    fs::create_directories(root);std::ofstream details(root/"momentum.csv"),values(root/"solve-values.csv"),out(root/"momentum.json");
    for(auto* f:{&details,&values,&out}){f->exceptions(std::ios::badbit|std::ios::failbit);*f<<std::setprecision(17);}
    details<<"backend,matrix_origin,coordinate,raw_delta,raw_impulse,raw_residual,stored_delta,stored_impulse,stored_residual,raw_scale,stored_scale,raw_allowance,old_allowance,rounding_velocity_signed,rounding_impulse_signed,velocity_ulp_bound,impulse_ulp_bound,arithmetic_bound,bounded_allowance,old_normalized,raw_normalized,bounded_normalized,passed\n";
    values<<"backend,kind,index,raw,stored,half_ulp\n";
    const auto before=read_failure("build/contact-validation/2026-10-30-v1/first-failure-rotating-accuracy-0-1.bin");
    AuditJob jobs[2];jobs[0].state=jobs[0].before=before;const float zero[4]{};
    {std::ofstream charged(root/"audit-charges.txt");charged<<"2\n";} // entry + full legacy replay, charged before call
    jobs[0].ok=audit::substep(jobs[0].state,zero,.00125f,{990,.001f,9.81f},{},jobs[0].result);
    Device<AuditJob> device;jobs[1].state=before;{std::ofstream charged(root/"audit-charges.txt");charged<<"4\n";}jobs[1]=device.step(jobs[1]);
    bool passed=true;int rejected=0;double host_gpu_old=0,host_gpu_raw=0,max_M_difference=0;
    for(int backend=0;backend<2;++backend){auto& a=jobs[backend].result;
        if(!jobs[backend].ok||!a.captured||!audit::certify(a))passed=false;
        if(std::memcmp(&jobs[backend].before,&before,sizeof(before))!=0)throw std::runtime_error("Audit input-state echo differs");
        for(int k=0;k<10;++k){values<<backend<<",velocity,"<<k<<','<<a.raw_velocity[k]<<','<<a.stored_velocity[k]<<','<<a.velocity_half_ulp[k]<<'\n';values<<backend<<",free_velocity,"<<k<<','<<a.free_velocity[k]<<','<<a.telemetry.free_velocity[k]<<",0\n";}
        for(int k=0;k<30;++k)values<<backend<<",impulse,"<<k<<','<<a.raw_impulse[k]<<','<<a.stored_impulse[k]<<','<<a.impulse_half_ulp[k]<<'\n';
        for(int i=0;i<10;++i)for(int j=0;j<10;++j)values<<backend<<",mass,"<<(i*10+j)<<','<<a.mass[i][j]<<','<<static_cast<float>(a.mass[i][j])<<",0\n";
        for(int i=0;i<30;++i)for(int j=0;j<10;++j){values<<backend<<",jacobian,"<<(i*10+j)<<','<<a.jacobian[i][j]<<','<<static_cast<float>(a.jacobian[i][j])<<",0\n";values<<backend<<",response,"<<(i*10+j)<<','<<a.response[i][j]<<','<<static_cast<float>(a.response[i][j])<<",0\n";}
        for(int origin=0;origin<2;++origin){auto copy=a;if(origin)for(int i=0;i<10;++i)for(int j=0;j<10;++j){max_M_difference=std::max(max_M_difference,std::abs(a.mass[i][j]-jobs[0].result.mass[i][j]));copy.mass[i][j]=jobs[0].result.mass[i][j];}
            audit::certify(copy);if(backend==1&&origin){host_gpu_old=copy.maximum_old_normalized;host_gpu_raw=copy.maximum_raw_normalized;}
            for(int k=0;k<10;++k){const auto& c=copy.coordinate[k];details<<backend<<','<<origin<<','<<k<<','<<c.raw_delta<<','<<c.raw_impulse<<','<<c.raw_residual<<','<<c.stored_delta<<','<<c.stored_impulse<<','<<c.stored_residual<<','<<c.raw_scale<<','<<c.stored_scale<<','<<c.raw_allowance<<','<<c.old_stored_allowance<<','<<c.velocity_rounding_signed<<','<<c.impulse_rounding_signed<<','<<c.velocity_half_ulp_bound<<','<<c.impulse_half_ulp_bound<<','<<c.arithmetic_bound<<','<<c.bounded_stored_allowance<<','<<c.old_stored_normalized<<','<<c.raw_normalized<<','<<c.bounded_normalized<<','<<c.bounded_passed<<'\n';}}
        // Each coordinate: a corrupted stored output and a corrupted raw update
        // with a matching cast must BOTH be rejected without another advance.
        for(int k=0;k<10;++k){auto bad=a;bad.stored_velocity[k]+=.0001f;if(!audit::certify(bad))++rejected;bad=a;bad.raw_velocity[k]+=.0001;bad.stored_velocity[k]=static_cast<float>(bad.raw_velocity[k]);if(!audit::certify(bad))++rejected;}
    }
    const auto expected=read_failure("build/contact-validation/2026-10-30-v1/first-failure-rotating-accuracy-0-1.bin",1);
    const bool replay_exact=std::memcmp(&expected,&jobs[1].state,sizeof(expected))==0;
    out<<"{\"saved_GPU_replay_bit_exact\":"<<(replay_exact?"true":"false")<<",\"native_CPU_old_score\":"<<jobs[0].result.maximum_old_normalized<<",\"native_CUDA_old_score\":"<<jobs[1].result.maximum_old_normalized<<",\"CPU_M_CUDA_stored_old_score\":"<<host_gpu_old<<",\"CPU_M_CUDA_raw_score\":"<<host_gpu_raw<<",\"maximum_M_difference\":"<<max_M_difference<<",\"CPU_raw_score\":"<<jobs[0].result.maximum_raw_normalized<<",\"CUDA_raw_score\":"<<jobs[1].result.maximum_raw_normalized<<",\"CPU_bounded_score\":"<<jobs[0].result.maximum_bounded_normalized<<",\"CUDA_bounded_score\":"<<jobs[1].result.maximum_bounded_normalized<<",\"negative_controls_rejected\":"<<rejected<<",\"negative_controls_total\":40,\"physical_charges\":4,\"passed\":"<<((passed&&rejected==40)?"true":"false")<<"}\n";
    std::cout<<"Audit: native raw="<<jobs[0].result.maximum_raw_normalized<<'/'<<jobs[1].result.maximum_raw_normalized<<", CPU-M/GPU stored="<<host_gpu_old<<", negatives="<<rejected<<"/40\n";return passed&&rejected==40?0:3;
}
std::array<float,23> body_fields(const p::BodyState& b){std::array<float,23> x{};int n=0;for(float v:{b.position.x,b.position.y,b.position.z,b.orientation.x,b.orientation.y,b.orientation.z,b.orientation.w})x[static_cast<size_t>(n++)]=v;for(float v:b.angle)x[static_cast<size_t>(n++)]=v;for(float v:b.velocity)x[static_cast<size_t>(n++)]=v;x[21]=b.work;x[22]=b.travel;return x;}
std::vector<uint8_t> rolled_binding(){
    const auto fixture=cudarl::contact_probe::detail::archived(5);cp::Writer out;
    out.value(uint32_t{0x43524F4C});out.value(uint32_t{1});cudarl::contact_pilot_spec::detail::write_body(out,fixture.initial);
    for(float v:{fixture.fluid.density,fixture.fluid.viscosity,fixture.fluid.gravity,fixture.current.x,fixture.current.y,fixture.current.z})out.value(v);
    for(float v:fixture.action)out.value(v);
    out.value(6000);out.value(50);
    for(double v:{.005,.25,e::kMaximumInternalStep,e::kSpring,e::kDamper,.0005,1e-10,.001,2e-4,2e-3,2e-6})out.value(v);
    out.value(e::kReleaseBisections);out.value(e::kMaximumEvents);return out.data;
}
int prepare_rolled(const fs::path& root){
    fs::create_directories(root);
    if(fs::exists(root/"protocol.bin")||fs::exists(root/"protocol.json")||fs::exists(root/"run-started"))throw std::runtime_error("Rolled protocol already prepared or attempted; no overwrite");
    const auto fixture=cudarl::contact_probe::detail::archived(5);const auto bytes=rolled_binding();
    cp::write_atomic((root/"protocol.bin").string(),bytes);
    std::ofstream out(root/"protocol.json");out.exceptions(std::ios::badbit|std::ios::failbit);
    out<<std::setprecision(17)<<"{\"study\":\"original-rolled-quarter-second-1\",\"fixture_source_case\":5,\"physical_attempt_cap_total\":6000,\"allocation\":\"one shared cap, CPU then CUDA, no reallocation from closed studies\",\"backends\":[\"CPU\",\"CUDA nonblocking\"],\"outer_steps_each\":50,\"outer_step_s\":0.005,\"horizon_s_each\":0.25,\"internal_max_s\":"<<e::kMaximumInternalStep<<",\"event_bisections\":"<<e::kReleaseBisections<<",\"initial_body_fields\":[";
    const auto fields=body_fields(fixture.initial);for(size_t i=0;i<fields.size();++i){if(i)out<<',';out<<fields[i];}
    out<<"],\"fluid_density_viscosity_gravity\":["<<fixture.fluid.density<<','<<fixture.fluid.viscosity<<','<<fixture.fluid.gravity<<"],\"current_m_s\":[0,0,0],\"action\":[0,0,0,0],\"stiffness_N_m\":"<<e::kSpring<<",\"damping_N_s_m\":"<<e::kDamper<<",\"safety_gap_m\":0.0005,\"energy_absolute_J\":1e-10,\"energy_relative\":0.001,\"state_parity_absolute\":0.0002,\"parity_relative\":0.002,\"gap_force_parity_absolute\":0.000002,\"certificate\":\"unchanged raw physical gate plus derived half-ULP storage bound\",\"stop\":\"any budget, finite, solver, momentum, geometry, nonadhesion, energy, integrity or parity failure ends this sole run; no retries or tuning\",\"binding_checksum\":"<<cp::checksum(bytes)<<"}\n";
    std::cout<<"Prepared original rolled fixture; zero advances; total cap6000.\n";return 0;
}
int rotating(const fs::path& root,bool original=false){
    int scalar_charges=0,reference_queries=0,gate=0;
    const int cap=original?6000:4000,steps=original?50:4;
    const std::string prefix=original?"rolled":"rotating";
    if(original){
        if(cp::read((root/"protocol.bin").string())!=rolled_binding())throw std::runtime_error("Frozen rolled protocol differs from executable");
        if(!fs::create_directory(root/"run-started"))throw std::runtime_error("Rolled check already attempted; no retries");
    }else{
        std::ifstream scalar_counts(root/"remaining-counts.txt");scalar_counts>>scalar_charges>>reference_queries>>gate;
        std::ifstream audit_file(root/"momentum.json");const std::string audit_json((std::istreambuf_iterator<char>(audit_file)),{});
        if(!scalar_counts||!gate||scalar_charges>3000||reference_queries>256||audit_json.find("\"passed\":true")==std::string::npos)throw std::runtime_error("Scalar or audit prerequisite failed");
    }
    if(fs::exists(root/(prefix+".csv")))throw std::runtime_error("Rotating stage already attempted; no retries");
    std::ofstream trace(root/(prefix+".csv")),report(root/(prefix+".json"));trace.exceptions(std::ios::badbit|std::ios::failbit);report.exceptions(std::ios::badbit|std::ios::failbit);
    trace<<std::setprecision(17)<<"backend,tick,ok,charged,accepted_segments,events,bisections,min_gap_m,energy_start_J,energy_end_J,max_energy_J,energy_gain_J,raw_momentum_score,stored_momentum_score,storage_bound,solver,momentum,geometry,energy,negativeforce,floor_impulse_N_s";for(int k=0;k<23;++k)trace<<",body_"<<k;trace<<'\n';
    const auto initial=original?cudarl::contact_probe::detail::archived(5).initial:cudarl::contact_pilot_spec::reference_states()[0];std::vector<BodyJob> cpu(static_cast<size_t>(steps));int spent=0,outer=0;bool passed=true;std::string failure;
    double minimum=1,maximum_raw=0,maximum_stored=0,max_energy_gain=-1e300,initial_energy=0,max_parity=0;int accepted=0,events=0,bisections=0,negatives=0;
    Device<BodyJob> device;const float zero[4]{};
    for(int backend=0;backend<2&&passed;++backend){BodyJob job;job.state=initial;
        for(int tick=0;tick<steps&&passed;++tick){if(spent>=cap){passed=false;failure="physical_attempt_cap";break;}++spent;++outer;job.counter={0,static_cast<uint64_t>(cap-spent+1)};const auto before=job.state;
            if(backend)job=device.step(job);else{job.before=job.state;job.ok=e::body_step(job.state,zero,.005,{990,.001f,9.81f},{},job.stats,job.counter);}
            spent+=static_cast<int>(job.counter.used)-1;accepted+=job.stats.accepted_segments;events+=job.stats.events;bisections+=job.stats.bisections;negatives+=job.stats.negative_controls;
            if(std::memcmp(&job.before,&before,sizeof(before))!=0)throw std::runtime_error("Body input echo failed");
            if(tick==0)initial_energy=job.stats.energy_start;
            const auto& s=job.stats;minimum=std::min(minimum,s.min_gap);maximum_raw=std::max(maximum_raw,s.momentum_raw_score);maximum_stored=std::max(maximum_stored,s.momentum_stored_score);max_energy_gain=std::max(max_energy_gain,s.max_energy-initial_energy);
            passed=job.ok&&s.finite&&s.completed&&s.solver&&s.momentum&&s.geometry&&s.energy&&!s.negativeforce&&s.min_gap>=-.0005&&s.max_energy<=initial_energy+std::max(1e-10,.001*std::abs(initial_energy));
            if(!passed){failure=s.budget_exhausted?"physical_attempt_cap":(!s.finite?"nonfinite":(!s.solver?"solver":(!s.momentum?"momentum":(!s.geometry?"geometry":(!s.energy?"energy":(s.negativeforce?"negative_normal_force":(s.event_limit?"event_limit":"native_physical_or_numerical_gate")))))));}
            if(backend&&passed){const auto a=body_fields(cpu[tick].state),b=body_fields(job.state);for(size_t k=0;k<a.size();++k){const double score=std::abs(double(a[k])-b[k])/(2e-4+2e-3*std::abs(a[k]));max_parity=std::max(max_parity,score);if(score>1)passed=false;}
                const auto ca=cudarl::contact::inspect(cpu[tick].state),cb=cudarl::contact::inspect(job.state);if(ca.invalid!=cb.invalid||ca.excessive!=cb.excessive||ca.escape!=cb.escape||ca.floor_contact!=cb.floor_contact||ca.contact!=cb.contact)passed=false;
                const double gap_score=std::abs(cpu[tick].stats.min_gap-s.min_gap)/(2e-6+2e-3*std::abs(cpu[tick].stats.min_gap));max_parity=std::max(max_parity,gap_score);if(gap_score>1)passed=false;
                const double force=cpu[tick].stats.impulsefloor/.005,other=s.impulsefloor/.005;const double force_score=std::abs(force-other)/(2e-6+2e-3*std::abs(force));max_parity=std::max(max_parity,force_score);if(force_score>1)passed=false;
                if(!passed)failure="CPU_CUDA_parity";}
            if(!backend)cpu[tick]=job;
            trace<<backend<<','<<(tick+1)<<','<<job.ok<<','<<job.counter.used<<','<<s.accepted_segments<<','<<s.events<<','<<s.bisections<<','<<s.min_gap<<','<<s.energy_start<<','<<s.energy_end<<','<<s.max_energy<<','<<(s.max_energy-initial_energy)<<','<<s.momentum_raw_score<<','<<s.momentum_stored_score<<','<<s.storage_bound<<','<<s.solver<<','<<s.momentum<<','<<s.geometry<<','<<s.energy<<','<<s.negativeforce<<','<<s.impulsefloor;
            for(float v:body_fields(s.completed?job.state:s.trial))trace<<','<<v;trace<<'\n';trace.flush();
            if(!passed){cp::Writer out;out.value(uint32_t{0x43455646});out.value(uint32_t{1});cudarl::contact_pilot_spec::detail::write_body(out,before);cudarl::contact_pilot_spec::detail::write_body(out,job.state);cudarl::contact_pilot_spec::detail::write_body(out,s.trial);out.value(s);cp::write_atomic((root/(prefix+"-first-failure.bin")).string(),out.data);}
        }
    }
    report<<std::setprecision(17)<<"{\"passed\":"<<(passed?"true":"false")<<",\"failure\":\""<<failure<<"\",\"original_start\":"<<(original?"true":"false")<<",\"rotation_charged\":"<<spent<<",\"outer_calls\":"<<outer<<",\"accepted_segments\":"<<accepted<<",\"events\":"<<events<<",\"bisections\":"<<bisections<<",\"negative_controls_rejected\":"<<negatives<<",\"total_charged\":"<<(scalar_charges+(original?0:4)+spent)<<",\"total_cap\":"<<(original?6000:7004)<<",\"minimum_gap_m\":"<<minimum<<",\"maximum_raw_momentum_score\":"<<maximum_raw<<",\"maximum_stored_momentum_score\":"<<maximum_stored<<",\"maximum_energy_gain_J\":"<<max_energy_gain<<",\"maximum_parity_score\":"<<max_parity<<"}\n";
    std::cout<<prefix<<": pass="<<passed<<", spent="<<spent<<'/'<<cap<<", failure="<<failure<<'\n';return passed?0:3;
}
}
int main(int argc,char** argv){try{if(argc!=3)throw std::runtime_error("Use --impact/--remaining/--audit/--rotate/--prepare-rolled/--rolled DIR");const std::string mode=argv[1];if(mode=="--impact")return scalar(argv[2],true);if(mode=="--remaining")return scalar(argv[2],false);if(mode=="--audit")return momentum(argv[2]);if(mode=="--rotate")return rotating(argv[2]);if(mode=="--prepare-rolled")return prepare_rolled(argv[2]);if(mode=="--rolled")return rotating(argv[2],true);throw std::runtime_error("Unknown mode");}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}

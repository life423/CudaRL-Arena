#include "envs/swimmer/motor_contact_v2.h"
#include "envs/swimmer/paired_motor.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

namespace {
using namespace cudarl;namespace mc=motor_contact_v2;namespace fs=std::filesystem;
constexpr int cases=8,variants=3,worlds=cases*variants,controls=100;
constexpr float intervals[3]={.005f,.0025f,.00125f};
void need(bool okay,const std::string& message) {if(!okay) throw std::runtime_error(message);}
std::string read_text(const fs::path& path) {std::ifstream file(path,std::ios::binary);need(bool(file),"Cannot read contact evidence");return {std::istreambuf_iterator<char>(file),{}};}
void write_new(const fs::path& path,const std::string& value) {need(!fs::exists(path),"Refusing to overwrite contact evidence");std::ofstream file(path,std::ios::binary);file<<value;file.flush();need(bool(file),"Cannot write contact evidence");}
struct Fixture {deep::BodyState body{};deep::Physics physical{};bool support=false;};
struct Protocol {
    std::array<Fixture,8> fixture{};uint64_t source_hash=0;
    std::vector<uint8_t> serialize() const {checkpoint::Writer out;out.value(uint32_t{0x4D435032u});out.value(mc::kPhysicsVersion);out.value(mc::physics_fingerprint());out.value(uint32_t{sizeof(Fixture)});out.value(source_hash);for(const auto& f:fixture) out.value(f);for(float h:intervals) out.value(h);return out.data;}
    uint64_t hash() const {return mc::fingerprint(serialize());}
    static Protocol read(const std::vector<uint8_t>& bytes) {checkpoint::Reader in{bytes};need(in.value<uint32_t>()==0x4D435032u&&in.value<uint32_t>()==mc::kPhysicsVersion&&in.value<uint64_t>()==mc::physics_fingerprint()&&in.value<uint32_t>()==sizeof(Fixture),"Contact preregistration layout differs");Protocol out;out.source_hash=in.value<uint64_t>();for(auto& f:out.fixture) f=in.value<Fixture>();for(float h:intervals) need(in.value<float>()==h,"Contact dt changed");in.finish();return out;}
};
void align_floor(deep::BodyState& body,float gap) {body.position.y+=(gap-mc::inspect(body).floor_gap)/deep::scale;}
Protocol prepare(const std::vector<uint8_t>& source) {
    Protocol out;out.source_hash=mc::fingerprint(source);checkpoint::Reader in{source};need(in.value<uint32_t>()==0x504D4953u&&in.value<uint32_t>()==1,"Expected protected first-unsafe v1 state");const auto failed=in.value<paired_motor::State>();in.finish();out.fixture[0].body=failed.first_unsafe_task.body;
    need(fabsf(mc::inspect(out.fixture[0].body).floor_gap+.00019335001707077026f)<2e-7f,"Protected failure is not the registered fin case");
    for(int i=1;i<8;++i) {auto& f=out.fixture[i];f.body.position={0,-2,0};f.support=i==2||i==3||i==4||i==5||i==7;if(f.support) f.physical.density=990;}
    out.fixture[1].body.velocity[1]=-.02f;out.fixture[6].body.velocity[1]=-.02f;
    auto& bent=out.fixture[3].body;bent.orientation=deep::axis_rotation({0,1,0},.3f)*deep::axis_rotation({0,0,1},.1f);bent.angle[0]=.2f;bent.angle[1]=.2f;bent.angle[2]=-.2f;bent.angle[3]=-.2f;
    out.fixture[4].body.orientation=deep::axis_rotation({0,0,1},.45f);out.fixture[5].body.orientation=deep::axis_rotation({1,0,0},1.5707963267948966f);
    out.fixture[6].physical.water_velocity={.005f,0,0};out.fixture[7].physical.water_velocity={.005f,0,0};
    for(int i=1;i<8;++i) align_floor(out.fixture[i].body,i==1||i==6?.0001f:-.00001f);
    for(const auto& f:out.fixture) need(!mc::inspect(f.body).excessive&&!mc::inspect(f.body).escape&&!mc::inspect(f.body).invalid&&mc::supported_physics(f.physical),"Unsafe initial contact fixture");return out;
}
std::string manifest(const Protocol& p) {
    const auto body=deep::shape(p.fixture[2].body);float max_mass=0;for(const auto& link:body.link) max_mass=fmaxf(max_mass,link.mass);
    const double compliance=max_mass*9.81*.01/.8,travel=2.*.005*.02,rounding=.000002;
    need(compliance+travel+rounding<=mc::kOverlapTolerance,"Overlap tolerance not justified by declared physical scale");
    std::ostringstream out;out<<std::setprecision(17)<<"{\n\"physics_version\":2,\n\"physics_hash\":\""<<mc::physics_fingerprint()<<"\",\n\"protocol_hash\":\""<<p.hash()<<"\",\n\"source_failure_hash\":\""<<p.source_hash<<"\",\n\"overlap_tolerance_m\":"<<mc::kOverlapTolerance<<",\n"
        <<"\"tolerance_derivation\":\"round up maximum link's static 1% density-mismatch deflection at0.8N/m + TWO5ms travels at20mm/s +2um float margin; chosen before dynamics, not from old0.193mm breach\",\n\"largest_link_mass_kg\":"<<max_mass<<",\n\"compliance_allowance_m\":"<<compliance<<",\n\"two_step_travel_m\":"<<travel<<",\n\"float_margin_m\":"<<rounding
        <<",\n\"tolerance_bodylength_fraction\":"<<mc::kOverlapTolerance/.072f<<",\n\"tolerance_finhalfheight_fraction\":"<<mc::kOverlapTolerance/.006f<<",\n\"normal_stiffness_N_m\":0.8,\n\"normal_damping\":\"2*sqrt(0.8*link_mass), only overlapping/approaching contact; two tail endpoints split stiffness/damping half each; head analytic ellipsoid\",\n"
        <<"\"geometry\":\"exact directional support of head ellipsoid plus elliptical cylindrical tail/fin at BOTH endpoints; same intended geometry for contact/validation; independent CPU-double plane oracle\",\n\"semantics\":\"overlap in[-0.5mm,0] acceptable contact, below-0.5mm excessive; all body beyond any plane by0.5mm escape; invalid/excessive/escape rejected; floor contact never unsupported-hover credit\",\n"
        <<"\"noncontact_model\":\"identical deep::assemble with legacy contact disabled plus copied identical semiimplicit solve/integrator, unchanged mass/drag/buoyancy/softstops/actuatorlimits; rectangle only\",\n\"substeps_s\":[0.005,0.0025,0.00125],\n\"dynamic_cases\":8,\n\"duration_each_s\":5,\n\"backends\":[\"CPU\",\"CUDA\"],\n\"dynamic_substep_cap\":112000,\n\"freewater_old_new_substep_cap\":1600,\n\"total_substep_cap\":113600,\n\"nominal_50ms_duration_equivalent\":4960,\n"
        <<"\"gates\":\"finite, no excessive/escape at every microstep; all8cases upwardfloorresponse; density990 support cases last1s floorforce>=80%known buoyancy deficit, headverticalspeed<=0.5mm/s and actualfloorcontact; 5ms-vs1.25ms max penetration difference<=0.1mm, final headposition difference<=0.5mm; CPU/CUDA physicalfloat parity2e-4+2e-3abs except gap2um+2e-3abs; freewater old/new bitexact on eachbackend\",\n\"no_gain_sweep_or_retry\":true,\n\"PPO_updates\":0,\n\"cases\":[";
    const char* names[8]={"protected_v1_fin_breach","neutral_20mm_s_floor_impact","1pct_negative_buoyancy_floor_support","bent_floor_support","pitched_head_floor_support","rolled_fin_floor_support","neutral_20mm_s_impact_5mm_s_current","1pct_negative_buoyancy_support_5mm_s_current"};
    for(int i=0;i<8;++i) {if(i) out<<',';const auto& f=p.fixture[i];out<<"{\"id\":"<<i<<",\"name\":\""<<names[i]<<"\",\"density\":"<<f.physical.density<<",\"support_test\":"<<(f.support?"true":"false")<<",\"initial_floor_gap_m\":"<<mc::inspect(f.body).floor_gap<<'}';}out<<"]\n}\n";return out.str();
}
double oracle_gap(const deep::BodyState& body,int face) {
    const auto shape=deep::shape(body);double answer=1e9;const auto n=mc::normal(face);
    for(int b=0;b<3;++b) {const auto& l=shape.link[b];const auto v=l.q.inverse(n);const double nx=v.x,ny=v.y,nz=v.z,half=l.length*.5;
        const double support=b==0?std::sqrt(half*half*nx*nx+.0064*.0064*ny*ny+.008*.008*nz*nz):std::abs(nx)*half+std::sqrt((b==2?.006:.0015)*(b==2?.006:.0015)*ny*ny+.0015*.0015*nz*nz);
        const double gap=double(n.x)*l.center.x+double(n.y)*l.center.y+double(n.z)*l.center.z+mc::offset(face)-support;answer=std::min(answer,gap);
    }return answer;
}
float legacy_proxy_gap(const deep::BodyState& body) {
    const auto shape=deep::shape(body);float answer=1e6f;for(int b=0;b<3;++b) for(int sample=0;sample<3;++sample) {const auto& l=shape.link[b];const auto point=l.center+l.q.rotate({(sample-1)*l.length/3.f,0,0});answer=fminf(answer,point.y+4.f*deep::scale-(b==0?.0064f:.002f));}return answer;
}
bool geometry_checks(const Protocol& p,int& count,double& error,std::ostream& log) {
    bool okay=true;auto check=[&](bool yes,const std::string& what) {++count;log<<what<<','<<yes<<'\n';okay=okay&&yes;};
    for(int pose=0;pose<16;++pose) {deep::BodyState body;body.position={.2f,-2.f,.1f};body.orientation=deep::axis_rotation({0,1,0},pose*.31f)*deep::axis_rotation({0,0,1},(pose-7)*.13f)*deep::axis_rotation({1,0,0},pose*.21f);body.angle[0]=.1f*(pose%3);body.angle[1]=-.08f*(pose%4);body.angle[2]=-.1f*(pose%5);body.angle[3]=.06f*(pose%6);
        const auto shape=deep::shape(body);for(int face=0;face<6;++face) {float actual=1e6f;for(int b=0;b<3;++b) for(int end=0;end<(b==0?1:2);++end) actual=fminf(actual,deep::dot(mc::normal(face),shape.link[b].center+mc::support_offset(shape.link[b],b,mc::normal(face),end))+mc::offset(face));const double e=std::abs(double(actual)-oracle_gap(body,face));error=std::max(error,e);check(e<=2e-6,"independent_plane_pose"+std::to_string(pose)+"_face"+std::to_string(face));}
        align_floor(body,-.00025f);const auto contact=mc::inspect(body);check(contact.floor_contact&&!contact.excessive&&!contact.escape,"acceptable_contact_pose"+std::to_string(pose));align_floor(body,-.001f);check(mc::inspect(body).excessive&&!mc::inspect(body).escape,"excessive_pose"+std::to_string(pose));
    }
    auto escaped=p.fixture[2].body;escaped.position.y=-7;check(mc::inspect(escaped).escape&&mc::inspect(escaped).excessive,"whole_body_below_floor_escape");auto invalid=escaped;invalid.velocity[0]=std::numeric_limits<float>::quiet_NaN();check(mc::inspect(invalid).invalid,"nonfinite_rejection");
    const auto reproduced=mc::inspect(p.fixture[0].body);mc::Telemetry force;const auto a=mc::assemble(p.fixture[0].body,p.fixture[0].physical,&force);(void)a;
    check(reproduced.floor_gap<0&&legacy_proxy_gap(p.fixture[0].body)>0,"reproduced_full_fin_vs_legacy_proxy_disagreement");check(force.floor_force>0&&force.points>0,"corrected_contact_responds_to_reproduced_fin");check(!reproduced.excessive,"reproduced_overlap_is_within_independently_registered_tolerance");
    return okay;
}
struct Metrics {
    int steps=0,window_samples=0;float min_gap=1e6f,min_floor=1e6f,max_floor_force=0,window_force=0,max_window_vy=0;
    bool invalid=false,excessive=false,escape=false,any_floor=false,window_floor=true;
};
struct World {deep::BodyState body{};Metrics metric{};mc::Telemetry contact{};};
CUDARL_HD void advance_control(World& world,const Fixture& fixture,float h,int control) {
    if(world.metric.invalid||world.metric.excessive||world.metric.escape) return;const float zero[4]{};const int micro=h>.004f?10:(h>.002f?20:40);
    for(int sub=0;sub<micro;++sub) {
        mc::substep(world.body,zero,h,fixture.physical,&world.contact);++world.metric.steps;const auto status=mc::inspect(world.body);
        world.metric.invalid=world.metric.invalid||status.invalid;world.metric.excessive=world.metric.excessive||status.excessive;world.metric.escape=world.metric.escape||status.escape;world.metric.any_floor=world.metric.any_floor||world.contact.floor_force>0;
        world.metric.min_gap=fminf(world.metric.min_gap,status.min_gap);world.metric.min_floor=fminf(world.metric.min_floor,status.floor_gap);world.metric.max_floor_force=fmaxf(world.metric.max_floor_force,world.contact.floor_force);
        if(control>=80) {++world.metric.window_samples;world.metric.window_force+=world.contact.floor_force;world.metric.max_window_vy=fmaxf(world.metric.max_window_vy,fabsf(world.body.velocity[1]));world.metric.window_floor=world.metric.window_floor&&status.floor_contact;}
        if(world.metric.invalid||world.metric.excessive||world.metric.escape) return;
    }
}
__global__ void advance_kernel(World* world,const Fixture* fixture,int control) {const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<worlds) advance_control(world[i],fixture[i/3],i%3==0?.005f:(i%3==1?.0025f:.00125f),control);}
template<class T> std::vector<T> download(const device_ptr<T>& p) {std::vector<T> out(p.size());p.copy_to_host(out.data(),out.size());return out;}
struct Parity {uint64_t fields=0,failed=0;double maximum=0;};
void compare(Parity& p,float a,float b,double absolute=2e-4) {++p.fields;const double ratio=std::abs(double(a)-b)/(absolute+2e-3*std::abs(double(a)));p.maximum=std::max(p.maximum,ratio);if(!std::isfinite(ratio)||ratio>1) ++p.failed;}
void compare(Parity& p,const World& a,const World& b) {
    for(const auto pair:{std::array<float,2>{a.body.position.x,b.body.position.x},{a.body.position.y,b.body.position.y},{a.body.position.z,b.body.position.z},{a.body.orientation.x,b.body.orientation.x},{a.body.orientation.y,b.body.orientation.y},{a.body.orientation.z,b.body.orientation.z},{a.body.orientation.w,b.body.orientation.w},{a.body.work,b.body.work},{a.body.travel,b.body.travel}}) compare(p,pair[0],pair[1]);
    for(int j=0;j<4;++j) compare(p,a.body.angle[j],b.body.angle[j]);for(int j=0;j<10;++j) compare(p,a.body.velocity[j],b.body.velocity[j]);compare(p,a.metric.min_floor,b.metric.min_floor,2e-6);compare(p,a.contact.floor_force,b.contact.floor_force,2e-6);
    if(a.metric.invalid!=b.metric.invalid||a.metric.excessive!=b.metric.excessive||a.metric.escape!=b.metric.escape||a.metric.steps!=b.metric.steps) ++p.failed;
}
void frame(std::vector<mc::ContactFrame>& frames,int backend,int id,int control,float h,const World& before,const World& after,const Fixture& fixture) {
    const int steps=after.metric.steps-before.metric.steps;if(!steps) return;mc::ContactFrame out;out.backend=backend;out.case_id=id;out.step=control+1;out.substeps=steps;out.h=h;out.time_s=after.metric.steps*h;out.before=before.body;out.after=after.body;out.physical=fixture.physical;out.inspection=mc::inspect(after.body);out.contact=after.contact;frames.push_back(out);
}
bool failure(const World& world) {return world.metric.invalid||world.metric.excessive||world.metric.escape;}
struct FreePair {deep::BodyState old_body{},new_body{};int steps=0;bool exact=true;};
CUDARL_HD void free_control(FreePair& pair,int id) {
    deep::Physics physical;if(id&1) physical.water_velocity={.005f,0,0};const float actions[4]={id? .01f:0.f,id?-.02f:0.f,id?.015f:0.f,id?-.005f:0.f};
    for(int sub=0;sub<10;++sub) {deep::substep(pair.old_body,actions,.005f,physical);mc::substep(pair.new_body,actions,.005f,physical);++pair.steps;
        const auto& a=pair.old_body;const auto& b=pair.new_body;bool same=a.position.x==b.position.x&&a.position.y==b.position.y&&a.position.z==b.position.z&&a.orientation.x==b.orientation.x&&a.orientation.y==b.orientation.y&&a.orientation.z==b.orientation.z&&a.orientation.w==b.orientation.w&&a.work==b.work&&a.travel==b.travel;
        for(int j=0;j<4;++j) same=same&&a.angle[j]==b.angle[j];for(int j=0;j<10;++j) same=same&&a.velocity[j]==b.velocity[j];pair.exact=pair.exact&&same&&mc::finite_body(a)&&mc::finite_body(b)&&!mc::inspect(b).contact;
    }
}
__global__ void free_kernel(FreePair* pairs) {const int id=threadIdx.x;if(id<4) free_control(pairs[id],id);}
}
int main(int argc,char** argv) {
    try {
        if(argc!=4) {std::cerr<<"Usage: motor_contact_v2_check PROTECTED-FIRST-UNSAFE NEW-OUTPUT --prepare|--run\n";return 2;}
        const fs::path input=fs::absolute(argv[1]),output=fs::absolute(argv[2]);const std::string mode=argv[3];need(mode=="--prepare"||mode=="--run","Explicit prepare/run required");const auto source=checkpoint::read(input.string());
        if(mode=="--prepare") {need(!fs::exists(output),"Choose a new contact study directory");const auto protocol=prepare(source);fs::create_directories(output);checkpoint::write_atomic((output/"protected-first-unsafe.bin").string(),source);checkpoint::write_atomic((output/"protocol.bin").string(),protocol.serialize());write_new(output/"protocol.json",manifest(protocol));std::cout<<"CONTACT_V2_PREPARED physics_steps=0 protocol="<<protocol.hash()<<'\n';return 0;}
        const auto protocol_bytes=checkpoint::read((output/"protocol.bin").string());const auto protocol=Protocol::read(protocol_bytes);need(protocol.source_hash==mc::fingerprint(source)&&checkpoint::read((output/"protected-first-unsafe.bin").string())==source&&read_text(output/"protocol.json")==manifest(protocol),"Contact source or preregistration changed");need(!fs::exists(output/"run-started.json"),"No repeat or gain sweep permitted");write_new(output/"run-started.json","{\"substep_cap\":113600,\"PPO_updates\":0}\n");
        mc::ContactGate gate;gate.source_failure_hash=protocol.source_hash;gate.protocol_hash=protocol.hash();std::ofstream checks(output/"geometry-checks.csv");checks<<"check,passed\n";double oracle_error=0;gate.geometry=geometry_checks(protocol,gate.static_checks,oracle_error,checks);checks.flush();
        std::vector<World> cpu(worlds),gpu(worlds);for(int i=0;i<worlds;++i) {cpu[i].body=protocol.fixture[i/3].body;const auto status=mc::inspect(cpu[i].body);cpu[i].metric.min_gap=status.min_gap;cpu[i].metric.min_floor=status.floor_gap;}gpu=cpu;
        device_ptr<World> d_world(worlds);device_ptr<Fixture> d_fixture(8);d_world.copy_from_host(gpu.data(),gpu.size());d_fixture.copy_from_host(protocol.fixture.data(),8);CUDA_CHECK_THROW(cudaDeviceSynchronize());
        std::vector<mc::ContactFrame> frames;frames.reserve(5000);Parity parity;bool aborted=!gate.geometry;std::ofstream trace(output/"contact-trace.csv");trace<<std::setprecision(9)<<"backend,case,dt_s,time_s,steps,head_x,head_y,head_z,vy_m_s,floor_gap_m,min_other_gap_m,overlap_m,floor_force_N,contact,excessive,escape,invalid\n";
        int completed_controls=0;
        for(int control=0;control<controls&&!aborted;++control) {
            const auto before_cpu=cpu,before_gpu=gpu;for(int i=0;i<worlds;++i) advance_control(cpu[i],protocol.fixture[i/3],intervals[i%3],control);
            advance_kernel<<<1,32>>>(d_world.get(),d_fixture.get(),control);CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaDeviceSynchronize());gpu=download(d_world);++completed_controls;
            for(int i=0;i<worlds;++i) {frame(frames,0,i,control,intervals[i%3],before_cpu[i],cpu[i],protocol.fixture[i/3]);frame(frames,1,i,control,intervals[i%3],before_gpu[i],gpu[i],protocol.fixture[i/3]);compare(parity,cpu[i],gpu[i]);for(int backend=0;backend<2;++backend) {const auto& w=backend?gpu[i]:cpu[i];const auto status=mc::inspect(w.body);trace<<backend<<','<<i/3<<','<<intervals[i%3]<<','<<w.metric.steps*intervals[i%3]<<','<<w.metric.steps<<','<<w.body.position.x<<','<<w.body.position.y<<','<<w.body.position.z<<','<<w.body.velocity[1]<<','<<status.floor_gap<<','<<status.min_other_gap<<','<<fmaxf(0,-status.min_gap)<<','<<w.contact.floor_force<<','<<status.contact<<','<<status.excessive<<','<<status.escape<<','<<status.invalid<<'\n';if(failure(w)) aborted=true;}}
            if(parity.failed) aborted=true;
        }
        gate.response=!aborted;gate.support=!aborted;gate.timestep=!aborted;
        std::ostringstream metrics;metrics<<std::setprecision(17)<<'[';
        for(int backend=0;backend<2;++backend) for(int i=0;i<worlds;++i) {const auto& w=backend?gpu[i]:cpu[i];const auto& f=protocol.fixture[i/3];const auto& m=w.metric;gate.response=gate.response&&m.any_floor&&!failure(w);float mass=0;for(const auto& l:deep::shape(f.body).link) mass+=l.mass;const float expected=mass*f.physical.gravity*(1-f.physical.density/1000.f),mean=m.window_samples?m.window_force/m.window_samples:0;
            const bool supported=!f.support||(m.window_samples>0&&m.window_floor&&mean>=.8f*expected&&m.max_window_vy<=.0005f);gate.support=gate.support&&supported;
            if(i%3==0) {const auto& fine=backend?gpu[i+2]:cpu[i+2];gate.timestep=gate.timestep&&fabsf(m.min_floor-fine.metric.min_floor)<=.0001f&&deep::norm((w.body.position-fine.body.position)*deep::scale)<=.0005f;}
            if(backend||i) metrics<<',';metrics<<"{\"backend\":"<<backend<<",\"case\":"<<i/3<<",\"h\":"<<intervals[i%3]<<",\"steps\":"<<m.steps<<",\"min_gap_m\":"<<m.min_gap<<",\"min_floor_m\":"<<m.min_floor<<",\"max_floor_force_N\":"<<m.max_floor_force<<",\"window_mean_floor_force_N\":"<<mean<<",\"known_buoyancy_deficit_N\":"<<expected<<",\"max_last_second_head_vy_m_s\":"<<m.max_window_vy<<",\"support_gate\":"<<(supported?"true":"false")<<",\"excessive\":"<<(m.excessive?"true":"false")<<",\"escape\":"<<(m.escape?"true":"false")<<",\"invalid\":"<<(m.invalid?"true":"false")<<'}';if(backend) gate.cuda_substeps+=m.steps;else gate.cpu_substeps+=m.steps;
        }metrics<<']';
        gate.freewater=!aborted;std::vector<FreePair> free_cpu(4),free_gpu(4);
        if(!aborted) {
            for(int i=0;i<4;++i) {deep::BodyState body;body.position={.2f,-2.f,.1f};body.orientation=deep::axis_rotation({0,1,0},i*.4f)*deep::axis_rotation({0,0,1},.1f*i);body.angle[0]=.1f*i;body.angle[2]=-.1f*i;free_cpu[i].old_body=free_cpu[i].new_body=body;}
            free_gpu=free_cpu;device_ptr<FreePair> d_free(4);d_free.copy_from_host(free_gpu.data(),4);CUDA_CHECK_THROW(cudaDeviceSynchronize());
            for(int control=0;control<10;++control) {
                const auto old_cpu=free_cpu,old_gpu=free_gpu;for(int i=0;i<4;++i) free_control(free_cpu[i],i);free_kernel<<<1,4>>>(d_free.get());CUDA_CHECK_THROW(cudaGetLastError());CUDA_CHECK_THROW(cudaDeviceSynchronize());free_gpu=download(d_free);
                for(int backend=0;backend<2;++backend) for(int i=0;i<4;++i) {
                    mc::ContactFrame out;out.backend=backend;out.case_id=24+i;out.step=control+1;out.substeps=10;out.h=.005f;out.time_s=.05f*(control+1);out.before=(backend?old_gpu:old_cpu)[i].new_body;out.after=(backend?free_gpu:free_cpu)[i].new_body;
                    if(i&1) out.physical.water_velocity={.005f,0,0};out.action[0]=i?.01f:0;out.action[1]=i?-.02f:0;out.action[2]=i?.015f:0;out.action[3]=i?-.005f:0;out.inspection=mc::inspect(out.after);frames.push_back(out);
                }
            }
            for(int i=0;i<4;++i) {gate.freewater=gate.freewater&&free_cpu[i].exact&&free_gpu[i].exact;gate.cpu_substeps+=free_cpu[i].steps*2;gate.cuda_substeps+=free_gpu[i].steps*2;}
        }
        gate.parity=parity.failed==0;gate.completed=completed_controls==100&&gate.cpu_substeps==56800&&gate.cuda_substeps==56800;
        checkpoint::Writer frame_file;frame_file.value(uint32_t{0x4D434632u});frame_file.value(mc::kPhysicsVersion);frame_file.value(uint32_t{sizeof(mc::ContactFrame)});frame_file.value(gate.physics_hash);frame_file.value(gate.protocol_hash);frame_file.array(frames);checkpoint::write_atomic((output/"contact-frames.bin").string(),frame_file.data);checkpoint::write_atomic((output/"contact-gate.bin").string(),gate.serialize());
        need(checkpoint::read(input.string())==source&&checkpoint::read((output/"protocol.bin").string())==protocol_bytes,"Protected source/protocol changed");trace.flush();
        std::ostringstream summary;summary<<std::setprecision(17)<<"{\"physics_version\":2,\"physics_hash\":\""<<gate.physics_hash<<"\",\"protocol_hash\":\""<<gate.protocol_hash<<"\",\"completed\":"<<(gate.completed?"true":"false")<<",\"passed\":"<<(gate.passed()?"true":"false")<<",\"geometry\":"<<(gate.geometry?"true":"false")<<",\"response\":"<<(gate.response?"true":"false")<<",\"support\":"<<(gate.support?"true":"false")<<",\"timestep\":"<<(gate.timestep?"true":"false")<<",\"CPU_CUDA_parity\":"<<(gate.parity?"true":"false")<<",\"freewater_old_new_exact\":"<<(gate.freewater?"true":"false")<<",\"geometry_checks\":"<<gate.static_checks<<",\"oracle_max_abs_m\":"<<oracle_error<<",\"parity_fields\":"<<parity.fields<<",\"parity_failed\":"<<parity.failed<<",\"parity_max_over_tolerance\":"<<parity.maximum<<",\"cpu_substeps\":"<<gate.cpu_substeps<<",\"cuda_substeps\":"<<gate.cuda_substeps<<",\"recorded_frames\":"<<frames.size()<<",\"aborted\":"<<(aborted?"true":"false")<<",\"PPO_updates\":0,\"cases\":"<<metrics.str()<<"}\n";write_new(output/"summary.json",summary.str());std::cout<<"CONTACT_V2_FINISHED "<<summary.str();return gate.passed()?0:3;
    } catch(const std::exception& error) {std::cerr<<"CONTACT_V2_STOPPED "<<error.what()<<'\n';return 1;}
}

#include "motor_benchmark.h"
#include "rl/command_motor_policy.h"
#include "cuda/device.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

// Bounded open-loop torque feasibility research. No PPO, physical parameter
// changes, deployment, final-suite evaluation, controller tuning or rescue.
// --prepare is CPU-only and registers everything before --run may use CUDA.
namespace {
using namespace cudarl;
namespace fs=std::filesystem;
namespace bench=motor_benchmark;
namespace cm=command_motor;
constexpr uint32_t protocol=1,seed=0x4D072610u;
constexpr int population=64,rounds=4,elites=8,knots=12,hold=10,horizon=knots*hold,dimensions=knots*4,replay_ticks=400;
constexpr int search_budget=61440,replay_budget=1600;
using Sequence=std::array<float,dimensions>;
void need(bool yes,const std::string& why) {if(!yes) throw std::runtime_error(why);}
uint64_t hash_bytes(const std::vector<uint8_t>& bytes) {
    uint64_t value=14695981039346656037ull;for(uint8_t b:bytes) {value^=b;value*=1099511628211ull;}return value;
}
std::string read_text(const fs::path& path) {
    std::ifstream in(path,std::ios::binary);need(static_cast<bool>(in),"Cannot read "+path.string());return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
}
void new_text(const fs::path& path,const std::string& text) {
    need(!fs::exists(path),"Refusing to replace existing probe evidence: "+path.string());std::ofstream out(path,std::ios::binary);out<<text;out.flush();need(static_cast<bool>(out),"Cannot write "+path.string());
}
struct Input {std::vector<uint8_t> bytes;std::array<bench::Fixture,2> starts;uint64_t source=0;};
Input load_input(const fs::path& path) {
    Input result;result.bytes=checkpoint::read(path.string());checkpoint::Reader in{result.bytes};
    need(in.value<uint32_t>()==0x434D4649u&&in.value<uint32_t>()==1,"Expected saved reference fixture format1");
    // Immutable captures were made under task protocol1. Protocol2 changes
    // only terminal/data-safety handling, not State layout, forces or torques.
    need(in.value<uint64_t>()==10511261149222959219ull,"Expected immutable protocol1 reference captures");result.source=in.value<uint64_t>();
    const auto fixtures=in.array<bench::Fixture>(16);in.finish();need(fixtures.size()==16,"Expected existing16 reference fixtures");
    for(int i=0;i<2;++i) {
        result.starts[i]=fixtures[static_cast<size_t>(4+i)];const auto& f=result.starts[i];const auto initial=bench::measure(f.initial);
        need(f.id==4+i&&f.seed==bench::reference_seed&&f.kind==bench::Kind::Brake&&f.qualified&&f.needs_warmup&&f.warmup_ticks>0&&f.initial.steps==0,"Expected exact previously captured braking fixture4/5");
        need(initial.finite&&initial.clearance>=0&&f.actual_entry_speed>=.01f&&f.actual_entry_speed<=.03f,"Captured initial body invalid or outside registered entry speed");
        need(deep::norm(f.initial.request.velocity)==0&&deep::norm(f.initial.request.position-f.initial.body.position)<1e-7f,"Captured stop request was altered");
        need(f.pulse_start<0&&f.pulse_ticks==0,"Braking probe must use the existing calm, undisturbed start");
    }
    return result;
}
std::string manifest(const Input& input) {
    std::ostringstream out;out<<std::setprecision(17);
    out<<"{\n\"protocol\":1,\"task\":\"bounded direct internal torque braking feasibility; no neural learning\",\"source_reference_fixture_payload_fnv1a64\":\""<<hash_bytes(input.bytes)
       <<"\",\"source_boulder_fingerprint\":\""<<input.source<<"\",\"benchmark_rules_fingerprint\":\""<<bench::rules_fingerprint()<<"\",\"environment_fingerprint\":\""<<rl::command_motor_fingerprint()
       <<"\",\"seed\":"<<seed<<",\"fixture_ids\":[4,5],\"entry_speeds_m_s\":["<<input.starts[0].actual_entry_speed<<','<<input.starts[1].actual_entry_speed
       <<"],\"complete_captured_state_preserved\":true,\"population\":64,\"rounds_per_start\":4,\"elite_count\":8,\"knots\":12,\"commands_per_knot\":4,\"ticks_per_knot\":10,\"control_dt_s\":0.05,\"horizon_ticks\":120,\"horizon_s\":6,\"search_candidates\":512,\"search_physical_ticks\":61440,\"replay_cases\":4,\"replay_ticks_each\":400,\"replay_physical_ticks\":1600,\"initial_mean\":0,\"initial_sd\":0.15,\"minimum_sd\":0.005,\"torque_command_bounds\":[-1,1],\"internal_torque_limit_Nm\":0.00001,\"slot0\":\"pure zero\",\"slot1\":\"best saved incumbent; initially zero duplicate, still counted\",\"other_slots\":\"62 independent counter-Gaussian samples, clipped once to actuator bounds\",\"sampling_index\":\"rng.normal(seed,start_index*4+round,candidate*48+parameter)\",\"cem_update\":\"top8 safe-first then lowest cost then lowest candidate index; direct coordinate mean and population variance, sd floor0.005; no smoothing or sweep\",\n"
       <<"\"objective\":{\"running\":\"sum(dt*(0.5*(head_speed^2+COM_speed^2)/(2*0.02^2)+0.25*RMS216_speed^2/0.02^2+0.10*body_omega^2/0.5^2+0.10*mean4(joint_rate^2)/4^2+0.25*(original_heading_error/(10deg))^2+0.10*(original_goal_error/0.072)^2+0.002*mean4(command^2)))\",\"terminal\":\"5*mean(last20 poststep ticks)((head_speed^2+COM_speed^2)/(2*0.002^2)+RMS216_speed^2/0.001^2+body_omega^2/0.02^2+mean4(joint_rate^2)/0.05^2)\",\"safety\":\"every safe finite sequence ranks before every penetrating sequence; nonfinite aborts entire probe; unchanged analytic whole-body clearance\"},\n"
       <<"\"replay\":\"deterministic best and passive for each start, no noise; chosen12 knots run t0..6s then EXACT ZERO torque for t6..20s; persistent original request unchanged\",\"research_gate\":{\"scope\":\"two captured starts only, not a reusable controller or final-envelope acceptance\",\"safe\":\"finite and nonnegative unchanged whole-body clearance throughout all20s\",\"stop\":\"head AND COM <=0.002m/s for11 consecutive observations spanning0.5s; completion >=0.5s earlier than paired passive; distance <=passive AND <=0.072m\",\"quiet\":\"fixed201 observations t10.00..20.00s inclusive, achieved-position anchor captured once exactlyt10s; no physical goal retarget, recentering or window selection\",\"bands\":\"original heading<=10deg; head/COM<=0.0005m/s; max216point<=0.001m/s; bodyomega<=0.02rad/s; each jointrate<=0.05rad/s; each joint peak-to-peak<=0.05rad; anchor drift<=0.0036m\"},\"original_gates_reported_separately\":\"original7.2mm stop-request goal band, unchanged ten-poststep benchmark stop definition and3s/72mm brake gate, original settle/hold gate; never relaxed by research label\",\"no_PPO_updates\":true,\"no_original_gate_modification\":true,\"no_deployment\":true\n}\n";
    return out.str();
}
bench::Four action_at(const Sequence& sequence,int tick) {
    bench::Four out{};if(tick<horizon) for(int j=0;j<4;++j) out[j]=sequence[(tick/hold)*4+j];return out;
}
struct Cost {double running=0,terminal=0;double total() const {return running+5*terminal/20.;}};
void accumulate(Cost& c,const cm::State& state,const bench::Sample& s,const bench::Four& action,int tick) {
    double rates=0,torques=0;for(int j=0;j<4;++j) {rates+=double(state.body.velocity[6+j])*state.body.velocity[6+j]/4.;torques+=double(action[j])*action[j]/4.;}
    const double v2=double(s.head_speed)*s.head_speed+double(s.com_speed)*s.com_speed,rms2=double(s.point_rms_speed)*s.point_rms_speed,omega2=double(s.angular_speed)*s.angular_speed;
    c.running+=deep::dt*(.5*v2/(2*.02*.02)+.25*rms2/(.02*.02)+.10*omega2/(.5*.5)+.10*rates/16.+.25*std::pow(s.heading_error/bench::heading_band,2)+.10*std::pow(s.goal_error/.072,2)+.002*torques);
    if(tick>=100) c.terminal+=v2/(2*.002*.002)+rms2/(.001*.001)+omega2/(.02*.02)+rates/(.05*.05);
}
struct Candidate {Sequence sequence{};Cost cost{};bool safe=true;float minimum_clearance=1e6f;int first_failure=-1;};
bool better(const Candidate& a,const Candidate& b) {return a.safe!=b.safe?a.safe:a.cost.total()<b.cost.total();}
struct Frame {int stage=0,start=0,round=0,candidate=0;cm::State state{};bench::Four action{};};
void save_frames(const fs::path& path,const std::vector<Frame>& frames) {
    checkpoint::Writer out;out.value(uint32_t{0x42525054u});out.value(protocol);out.value(static_cast<uint32_t>(sizeof(Frame)));out.array(frames);checkpoint::write_atomic(path.string(),out.data);
}
void trace_header(std::ostream& out) {
    out<<"stage,start_fixture,round,candidate,tick,time_s,head_x_m,head_y_m,head_z_m,goal_error_m,original_heading_error_rad,head_m_s,COM_m_s,tail_m_s,max216point_m_s,RMS216point_m_s,body_omega_rad_s,max_joint_rad_s,clearance_m,finite";
    for(int j=0;j<4;++j) out<<",applied"<<j;for(int j=0;j<10;++j) out<<",velocity"<<j;for(int j=0;j<4;++j) out<<",joint"<<j;out<<'\n';
}
void trace_row(std::ostream& out,const char* stage,int fixture,int round,int candidate,const cm::State& state,const bench::Sample& s,const bench::Four& a) {
    out<<stage<<','<<fixture<<','<<round<<','<<candidate<<','<<state.steps<<','<<state.steps*deep::dt<<','<<s.head_position.x<<','<<s.head_position.y<<','<<s.head_position.z<<','<<s.goal_error<<','<<s.heading_error<<','<<s.head_speed<<','<<s.com_speed<<','<<s.tail_speed<<','<<s.point_speed<<','<<s.point_rms_speed<<','<<s.angular_speed<<','<<s.joint_rate<<','<<s.clearance<<','<<s.finite;
    for(float value:a) out<<','<<value;for(float value:state.body.velocity) out<<','<<value;for(float value:state.body.angle) out<<','<<value;out<<'\n';
}
void upload(device_ptr<float>& device,const std::vector<float>& host) {CUDA_CHECK_THROW(cudaMemcpy(device.get(),host.data(),host.size()*sizeof(float),cudaMemcpyHostToDevice));}
Candidate search(const bench::Fixture& fixture,int start,const fs::path& output,std::ofstream& trace,std::ofstream& candidate_csv,std::vector<Frame>& frames,int& physical_ticks) {
    cm::Batch worlds(population,seed);Sequence mean{},sd{};sd.fill(.15f);Candidate incumbent;bool have_incumbent=false;
    for(int round=0;round<rounds;++round) {
        std::array<Candidate,population> candidates{};candidates[1].sequence=incumbent.sequence;
        for(int candidate=2;candidate<population;++candidate) for(int parameter=0;parameter<dimensions;++parameter)
            candidates[candidate].sequence[parameter]=deep::clamp(mean[parameter]+sd[parameter]*rng::normal(seed,static_cast<uint32_t>(start*rounds+round),static_cast<uint32_t>(candidate*dimensions+parameter)),-1,1);
        std::vector<Sequence> recorded;for(const auto& c:candidates) recorded.push_back(c.sequence);
        checkpoint::Writer sequences;sequences.value(uint32_t{0x42525053u});sequences.value(protocol);sequences.value(fixture.id);sequences.value(round);sequences.array(recorded);
        checkpoint::write_atomic((output/("search-start"+std::to_string(fixture.id)+"-round"+std::to_string(round)+"-sequences.bin")).string(),sequences.data);
        worlds.set_states(std::vector<cm::State>(population,fixture.initial),nullptr);std::vector<float> actions(population*4);device_ptr<float> device_actions(actions.size());
        auto states=worlds.host_states(nullptr);
        for(int i=0;i<population;++i) {
            const auto s=bench::measure(states[i]);candidates[i].safe=s.finite&&s.clearance>=0;candidates[i].minimum_clearance=s.clearance;
            const auto action=action_at(candidates[i].sequence,0);trace_row(trace,"search",fixture.id,round,i,states[i],s,action);frames.push_back({0,start,round,i,states[i],action});
        }
        for(int tick=0;tick<horizon;++tick) {
            for(int i=0;i<population;++i) {const auto action=action_at(candidates[i].sequence,tick);for(int j=0;j<4;++j) actions[i*4+j]=action[j];}
            upload(device_actions,actions);worlds.step(device_actions.get(),{},nullptr,false);physical_ticks+=population;states=worlds.host_states(nullptr);bool finite=true;
            for(int i=0;i<population;++i) {
                const auto sample=bench::measure(states[i]);const auto action=action_at(candidates[i].sequence,tick);auto& c=candidates[i];
                c.minimum_clearance=fminf(c.minimum_clearance,sample.clearance);c.safe=c.safe&&sample.finite&&sample.clearance>=0;
                if((!sample.finite||sample.clearance<0)&&c.first_failure<0) c.first_failure=states[i].steps;
                finite=finite&&sample.finite;accumulate(c.cost,states[i],sample,action,tick);finite=finite&&std::isfinite(c.cost.total());
                trace_row(trace,"search",fixture.id,round,i,states[i],sample,action);frames.push_back({0,start,round,i,states[i],action});
            }
            if(!finite||worlds.invalid_count(nullptr)!=0) {save_frames(output/"aborted-search-states.bin",frames);throw std::runtime_error("Nonfinite braking probe: stopped without extension or rescue");}
        }
        std::array<int,population> order{};std::iota(order.begin(),order.end(),0);
        std::stable_sort(order.begin(),order.end(),[&](int a,int b) {return better(candidates[a],candidates[b]);});
        if(!have_incumbent||better(candidates[order[0]],incumbent)) {incumbent=candidates[order[0]];have_incumbent=true;}
        for(int parameter=0;parameter<dimensions;++parameter) {
            double sum=0;for(int rank=0;rank<elites;++rank) sum+=candidates[order[rank]].sequence[parameter];mean[parameter]=static_cast<float>(sum/elites);
            double variance=0;for(int rank=0;rank<elites;++rank) {const double delta=candidates[order[rank]].sequence[parameter]-mean[parameter];variance+=delta*delta;}
            sd[parameter]=fmaxf(.005f,static_cast<float>(sqrt(variance/elites)));
        }
        for(int rank=0;rank<population;++rank) {const int id=order[rank];const auto& c=candidates[id];candidate_csv<<fixture.id<<','<<round<<','<<id<<','<<rank<<','<<c.safe<<','<<c.minimum_clearance<<','<<c.first_failure<<','<<c.cost.running<<','<<c.cost.terminal/20.<<','<<c.cost.total()<<'\n';}
        trace.flush();candidate_csv.flush();need(static_cast<bool>(trace)&&static_cast<bool>(candidate_csv),"Cannot write search evidence");
        std::cout<<"BRAKING_SEARCH fixture="<<fixture.id<<" round="<<round+1<<"/4 candidates=64 cumulative_ticks="<<physical_ticks<<" best_safe="<<incumbent.safe<<" best_cost="<<incumbent.cost.total()<<'\n';
    }
    checkpoint::Writer selected;selected.value(uint32_t{0x42525042u});selected.value(protocol);selected.value(seed);selected.value(fixture);selected.value(incumbent.sequence);selected.value(incumbent.cost);selected.value(incumbent.safe);
    checkpoint::write_atomic((output/("best-start"+std::to_string(fixture.id)+".bin")).string(),selected.data);return incumbent;
}
struct Replay {
    bench::Progress original;
    deep::V anchor{};
    float joint_low[4]{},joint_high[4]{};
    float stop_start=-1,stop_complete=-1,stop_distance=-1,maximum_drift=0,quiet_max_goal_error=0;
    float quiet_max_head=0,quiet_max_com=0,quiet_max_point=0,quiet_max_omega=0,quiet_max_rate=0,quiet_max_heading=0,quiet_max_excursion=0;
    int streak_start=-1,quiet_samples=0,first_quiet_failure=-1,first_quiet_bits=0;
    bool quiet=true,original_goal_band=true;
};
void replay_assess(Replay& p,const cm::State& state,const bench::Sample& sample,const bench::Fixture& fixture) {
    bench::assess(p.original,state,sample,fixture);
    if(sample.head_speed<=.002f&&sample.com_speed<=.002f) {
        if(p.streak_start<0) p.streak_start=state.steps;
        if(p.stop_complete<0&&state.steps-p.streak_start>=10) {p.stop_start=p.streak_start*deep::dt;p.stop_complete=state.steps*deep::dt;p.stop_distance=p.original.path;}
    } else p.streak_start=-1;
    if(state.steps<200) return;
    if(state.steps==200) {p.anchor=sample.head_position;for(int j=0;j<4;++j) p.joint_low[j]=p.joint_high[j]=state.body.angle[j];}
    ++p.quiet_samples;p.maximum_drift=fmaxf(p.maximum_drift,deep::norm(sample.head_position-p.anchor));p.quiet_max_goal_error=fmaxf(p.quiet_max_goal_error,sample.goal_error);
    p.quiet_max_head=fmaxf(p.quiet_max_head,sample.head_speed);p.quiet_max_com=fmaxf(p.quiet_max_com,sample.com_speed);p.quiet_max_point=fmaxf(p.quiet_max_point,sample.point_speed);
    p.quiet_max_omega=fmaxf(p.quiet_max_omega,sample.angular_speed);p.quiet_max_rate=fmaxf(p.quiet_max_rate,sample.joint_rate);p.quiet_max_heading=fmaxf(p.quiet_max_heading,sample.heading_error);
    int bits=0;if(sample.head_speed>.0005f) bits|=1;if(sample.com_speed>.0005f) bits|=2;if(sample.point_speed>.001f) bits|=4;if(sample.angular_speed>.02f) bits|=8;if(sample.joint_rate>.05f) bits|=16;
    if(sample.heading_error>bench::heading_band) bits|=32;if(p.maximum_drift>.0036f) bits|=64;
    for(int j=0;j<4;++j) {p.joint_low[j]=fminf(p.joint_low[j],state.body.angle[j]);p.joint_high[j]=fmaxf(p.joint_high[j],state.body.angle[j]);p.quiet_max_excursion=fmaxf(p.quiet_max_excursion,p.joint_high[j]-p.joint_low[j]);if(p.joint_high[j]-p.joint_low[j]>.05f) bits|=128;}
    if(sample.clearance<0) bits|=256;if(!sample.finite) bits|=512;
    if(bits&&p.first_quiet_failure<0) {p.first_quiet_failure=state.steps;p.first_quiet_bits=bits;}
    p.quiet=p.quiet&&bits==0;p.original_goal_band=p.original_goal_band&&sample.goal_error<=bench::position_band;
}
std::array<Replay,4> replay(const Input& input,const std::array<Candidate,2>& chosen,const fs::path& output,std::ofstream& trace,std::vector<Frame>& frames,int& physical_ticks) {
    cm::Batch worlds(4,seed);std::vector<cm::State> states;for(int i=0;i<4;++i) states.push_back(input.starts[i/2].initial);worlds.set_states(states,nullptr);
    std::array<Replay,4> results;device_ptr<float> device_actions(16);std::vector<float> actions(16);
    for(int i=0;i<4;++i) {const auto s=bench::measure(states[i]);const auto a=i%2==0?action_at(chosen[i/2].sequence,0):bench::Four{};replay_assess(results[i],states[i],s,input.starts[i/2]);trace_row(trace,i%2==0?"best-replay":"passive-replay",input.starts[i/2].id,-1,i,states[i],s,a);frames.push_back({1,i/2,-1,i,states[i],a});}
    for(int tick=0;tick<replay_ticks;++tick) {
        for(int i=0;i<4;++i) {const auto a=i%2==0?action_at(chosen[i/2].sequence,tick):bench::Four{};for(int j=0;j<4;++j) actions[i*4+j]=a[j];}
        upload(device_actions,actions);worlds.step(device_actions.get(),{},nullptr,false);physical_ticks+=4;states=worlds.host_states(nullptr);bool finite=true;
        for(int i=0;i<4;++i) {
            const auto s=bench::measure(states[i]);const auto a=i%2==0?action_at(chosen[i/2].sequence,tick):bench::Four{};finite=finite&&s.finite;
            replay_assess(results[i],states[i],s,input.starts[i/2]);trace_row(trace,i%2==0?"best-replay":"passive-replay",input.starts[i/2].id,-1,i,states[i],s,a);frames.push_back({1,i/2,-1,i,states[i],a});
        }
        if(!finite||worlds.invalid_count(nullptr)!=0) {save_frames(output/"aborted-replay-states.bin",frames);throw std::runtime_error("Nonfinite frozen braking replay: stopped without rescue");}
    }
    return results;
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=4) {std::cerr<<"Usage: motor_braking_probe existing-reference-fixtures.bin NEW-output-dir --prepare\n       motor_braking_probe same-reference-fixtures.bin registered-output-dir --run\n";return 2;}
        const fs::path source=fs::absolute(argv[1]),output=fs::absolute(argv[2]);const std::string mode=argv[3];need(mode=="--prepare"||mode=="--run","Explicit prepare/run mode required");
        const auto input=load_input(source);const auto prereg=manifest(input);
        if(mode=="--prepare") {
            need(!fs::exists(output),"Preparation requires a new directory");fs::create_directories(output);
            checkpoint::write_atomic((output/"input-reference-fixtures.bin").string(),input.bytes);checkpoint::Writer selected;selected.value(uint32_t{0x42525049u});selected.value(protocol);selected.value(input.starts);checkpoint::write_atomic((output/"complete-selected-starts.bin").string(),selected.data);
            new_text(output/"protocol.json",prereg);std::cout<<prereg<<"BRAKING_PROBE_PREPARED no_cuda=1 physical_steps=0 learning=0\n";return 0;
        }
        need(fs::is_directory(output)&&read_text(output/"protocol.json")==prereg,"Run requires the exact previously prepared protocol and source");
        need(checkpoint::read((output/"input-reference-fixtures.bin").string())==input.bytes,"Prepared fixture bytes differ");
        need(!fs::exists(output/"run-started.json"),"Probe already started; no repeat or extension is permitted in this directory");
        new_text(output/"run-started.json","{\"protocol\":1,\"search_budget\":61440,\"replay_budget\":1600,\"extensions_authorized\":false}\n");
        std::ofstream trace(output/"physical-trace.csv"),candidate_csv(output/"candidate-costs.csv");need(static_cast<bool>(trace)&&static_cast<bool>(candidate_csv),"Cannot create probe evidence");trace<<std::setprecision(9);candidate_csv<<std::setprecision(17);trace_header(trace);
        candidate_csv<<"start_fixture,round,candidate,rank,safe,min_clearance_m,first_safety_failure_tick,running_cost,last_second_quiet_mean,total_cost\n";
        std::array<Candidate,2> chosen;std::vector<Frame> frames;frames.reserve(64000);int search_ticks=0,replay_used=0;
        for(int i=0;i<2;++i) chosen[i]=search(input.starts[i],i,output,trace,candidate_csv,frames,search_ticks);
        need(search_ticks==search_budget,"Search physical budget mismatch");const auto observed=replay(input,chosen,output,trace,frames,replay_used);need(replay_used==replay_budget,"Replay physical budget mismatch");
        save_frames(output/"complete-physical-states.bin",frames);trace.flush();candidate_csv.flush();need(static_cast<bool>(trace)&&static_cast<bool>(candidate_csv),"Failed to write probe evidence");
        need(checkpoint::read(source.string())==input.bytes&&read_text(output/"protocol.json")==prereg,"Protected fixture source or preregistration changed");
        std::ostringstream result;result<<std::setprecision(17);result<<"{\n\"protocol\":1,\"research_only\":true,\"source_fixture_payload_fnv1a64\":\""<<hash_bytes(input.bytes)<<"\",\"search_physical_ticks\":"<<search_ticks<<",\"replay_physical_ticks\":"<<replay_used<<",\"source_unchanged\":true,\"PPO_updates\":0,\"automatically_adopted\":false,\"original_physical_reference_gate_modified\":false,\"starts\":[";
        bool ready=true;
        for(int start=0;start<2;++start) {
            if(start) result<<',';const auto& best=observed[start*2];const auto& passive=observed[start*2+1];
            const bool safe=best.original.finite&&!best.original.penetrated,quiet=best.quiet&&best.quiet_samples==201;
            const bool earlier=best.stop_complete>=0&&passive.stop_complete>=0&&best.stop_complete+.5f<=passive.stop_complete+1e-6f;
            const bool distance=best.stop_distance>=0&&best.stop_distance<=passive.stop_distance&&best.stop_distance<=bench::length;
            const bool start_ready=safe&&quiet&&earlier&&distance;ready=ready&&start_ready;const auto final=bench::finish(best.original,input.starts[start]),passive_final=bench::finish(passive.original,input.starts[start]);
            result<<"{\"fixture\":"<<input.starts[start].id<<",\"entry_m_s\":"<<input.starts[start].actual_entry_speed<<",\"research_ready\":"<<(start_ready?"true":"false")<<",\"safe_all20s\":"<<(safe?"true":"false")<<",\"quiet_fixed10to20s\":"<<(quiet?"true":"false")<<",\"at_least_half_second_earlier\":"<<(earlier?"true":"false")<<",\"distance_no_more_than_passive_and72mm\":"<<(distance?"true":"false")
                <<",\"best_stop_dwell_start_s\":"<<best.stop_start<<",\"best_stop_dwell_complete_s\":"<<best.stop_complete<<",\"best_stop_distance_m\":"<<best.stop_distance<<",\"passive_stop_dwell_start_s\":"<<passive.stop_start<<",\"passive_stop_dwell_complete_s\":"<<passive.stop_complete<<",\"passive_stop_distance_m\":"<<passive.stop_distance
                <<",\"best_fixed_anchor_m\":["<<best.anchor.x<<','<<best.anchor.y<<','<<best.anchor.z<<"],\"best_maximum_fixed_anchor_drift_m\":"<<best.maximum_drift<<",\"best_quiet_max_original_goal_error_m\":"<<best.quiet_max_goal_error<<",\"best_original_goal_band_all10to20s\":"<<(best.original_goal_band?"true":"false")<<",\"best_first_quiet_failure_tick\":"<<best.first_quiet_failure<<",\"best_first_quiet_failure_bits\":"<<best.first_quiet_bits<<",\"best_first_safety_failure_tick\":"<<best.original.first_failure_step<<",\"best_minimum_clearance_m\":"<<best.original.minimum_clearance
                <<",\"best_quiet_maxima\":{\"head_m_s\":"<<best.quiet_max_head<<",\"COM_m_s\":"<<best.quiet_max_com<<",\"point_m_s\":"<<best.quiet_max_point<<",\"body_omega_rad_s\":"<<best.quiet_max_omega<<",\"joint_rate_rad_s\":"<<best.quiet_max_rate<<",\"original_heading_rad\":"<<best.quiet_max_heading<<",\"joint_excursion_rad\":"<<best.quiet_max_excursion<<'}'
                <<",\"passive_safe_all20s\":"<<(passive.original.finite&&!passive.original.penetrated?"true":"false")<<",\"passive_minimum_clearance_m\":"<<passive.original.minimum_clearance<<",\"passive_first_safety_failure_tick\":"<<passive.original.first_failure_step<<",\"passive_quiet_fixed10to20s\":"<<(passive.quiet&&passive.quiet_samples==201?"true":"false")<<",\"passive_first_quiet_failure_tick\":"<<passive.first_quiet_failure<<",\"passive_first_quiet_failure_bits\":"<<passive.first_quiet_bits
                <<",\"passive_maximum_fixed_anchor_drift_m\":"<<passive.maximum_drift<<",\"passive_quiet_max_original_goal_error_m\":"<<passive.quiet_max_goal_error<<",\"passive_original_goal_band_all10to20s\":"<<(passive.original_goal_band?"true":"false")<<",\"passive_unchanged_final_3s_72mm_gate\":"<<(passive_final.motion?"true":"false")<<",\"passive_unchanged_full_original_goal_gate\":"<<(passive_final.passed?"true":"false")
                <<",\"best_unchanged_final_3s_72mm_gate\":"<<(final.motion?"true":"false")<<",\"best_unchanged_full_original_goal_gate\":"<<(final.passed?"true":"false")<<",\"best_original_gate_stop_s\":"<<final.stop_time<<",\"best_original_gate_stop_distance_m\":"<<final.stop_distance<<",\"best_original_goal_completed_hold_start_s\":"<<final.settle_time<<",\"best_original_goal_longest_hold_s\":"<<final.hold_seconds<<'}';
        }
        result<<"],\"research_ready\":"<<(ready?"true":"false")<<",\"scope\":\"two exact captured starts and optimized open-loop sequences only; neither a generalized feedback motor nor final-envelope acceptance\",\"failure_bit_legend\":\"1headSpeed 2COMspeed 4maxPointSpeed 8bodyOmega 16jointRate 32originalHeading 64anchorDrift 128jointExcursion 256clearance 512nonfinite\",\"action_alignment\":\"tick0 preview; each later row action produced before the recorded resulting state\"}\n";
        new_text(output/"research-readiness.json",result.str());std::cout<<"BRAKING_PROBE_FINISHED research_ready="<<ready<<" search_ticks="<<search_ticks<<" replay_ticks="<<replay_used<<" no_deployment=1\n";return ready?0:3;
    } catch(const std::exception& e) {std::cerr<<"Braking probe stopped: "<<e.what()<<'\n';return 1;}
}

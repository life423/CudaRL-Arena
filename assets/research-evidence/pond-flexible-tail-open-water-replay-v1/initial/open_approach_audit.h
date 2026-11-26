#pragma once

// Frozen CPU evaluator only. The trainer and Godot never link this diagnostic.
namespace open_approach {
using Actor = cudarl::nn::CpuMlp<float>;
constexpr uint64_t stop_substep = 33979;
constexpr int fork_ticks[]{1200,1520,1680,1800,1840,1880,1920,1960,2000,2040,2080,2120};

struct State {
    chooser::World world{};
    int slow=0, longest_slow=0;
};

struct Trial {
    int fork_tick, action, decisions;
    bool parent;
};

struct Outcome {
    int first=-1, released=-1, fault=0, longest_slow=0;
    bool pass=false;
    uint64_t accepted=0, attempted=0;
    double speed=0, gap=0, end_gap=0, tilt=0, depth=0;
};

double clearance(const chooser::World& w) {
    return 1000.*tail::submerged_gap_m(larger_pond,w.motor.navigation.motion.body,w.pose);
}
V position_at(const chooser::World& w,int control) {
    const auto& motion=w.motor.navigation.motion;
    const auto back=motion.controls-control;
    if(!back)return w.pose.com_m;
    if(back<0 || back>tail::turn_history_controls)throw std::runtime_error("Missing stall history");
    return motion.history[(motion.history_next+tail::turn_history_controls-back)%tail::turn_history_controls];
}

chooser::Result advance(State& s,const Models& models,Clock::time_point deadline) {
    if(Clock::now()>=deadline)throw std::runtime_error("Open-water replay timed out");
    const auto before=s.world.motor.navigation.motion.controls;
    const auto r=chooser::advance(s.world,models.gait.data(),models.packed_motor.data(),larger_pond,4);
    if(r.fault && r.fault!=chooser::Clearance)throw std::runtime_error("Unexpected physical replay fault");
    if(!r.fault && r.controls!=4)throw std::runtime_error("Incomplete native macro");
    // As in run_case, a failed macro terminates and is not credited as a complete motion window.
    if(!r.fault){
        for(int k=std::max(before+1,301); k<=before+r.controls; ++k) {
            const auto speed=physics::norm(horizontal(position_at(s.world,k)-position_at(s.world,k-25)))/.5f;
            s.slow=speed<.006f?s.slow+1:0;
            s.longest_slow=std::max(s.longest_slow,s.slow);
        }
    }
    return r;
}

Outcome branch(const State& capture,const Trial& t,const Models& m,const std::vector<float>& parent,
               int seconds,Clock::time_point deadline) {
    static_assert(std::is_trivially_copyable_v<State>);
    State s;
    std::memcpy(&s,&capture,sizeof(s));
    auto& w=s.world;
    Actor actor(chooser::actor_spec());
    actor.params()=t.parent?parent:m.policy;
    Outcome o;
    int decision=0;
    const uint64_t end=w.accepted_substeps+uint64_t(seconds)*samples_per_second;
    // Score transport from COM displacements over complete half-second windows.
    double distance=0;int windows=0,next_window=t.fork_tick+25;
    auto last_pos=w.pose.com_m;
    while(!w.fault && w.accepted_substeps<end) {
        if(chooser::ready(w)) {
            const int policy=close_probe::decide(w,actor,m.distant_feedback);
            const int action=decision<t.decisions?t.action:policy;
            if(decision==0)o.first=action;
            if(decision==t.decisions)o.released=action;
            if(!chooser::accept(w,action))throw std::runtime_error("Request rejected at fork");
            ++decision;
        }
        const auto r=advance(s,m,deadline);
        if(!r.fault && w.motor.navigation.motion.controls>=next_window) {
            const auto pos=position_at(w,next_window);
            distance+=physics::norm(horizontal(pos-last_pos));
            last_pos=pos;next_window+=25;++windows;
        }
    }
    o.accepted=w.accepted_substeps-capture.world.accepted_substeps;
    o.attempted=w.attempted_substeps-capture.world.attempted_substeps;
    o.fault=w.fault;o.gap=1000.*w.min_gap_m;o.end_gap=clearance(w);
    o.tilt=w.peak_tilt_deg;o.depth=1000.*w.peak_depth_error_m;
    o.longest_slow=s.longest_slow;o.speed=windows?distance*2000./windows:0.;
    o.pass=!o.fault && w.accepted_substeps==end && o.gap>0 &&
        o.longest_slow<=407 && o.speed>=6 && o.tilt<=10 && o.depth<=1;
    return o;
}

void write(std::ostream& out,const Trial& t,const Outcome& o,const State& s) {
    out<<t.fork_tick/50.<<','<<close_probe::hash(s.world)<<','<<clearance(s.world)<<','
       <<1000.*physics::norm(horizontal(s.world.pose.com_velocity_m_s))<<','
       <<(t.parent?"parent":t.decisions?"override_then_112":"chooser112")<<','<<t.action<<','
       <<t.decisions<<','<<o.first<<','<<o.released<<','<<o.fault<<','<<o.accepted/800.<<','
       <<o.speed<<','<<o.gap<<','<<o.end_gap<<','<<o.longest_slow<<','<<o.tilt<<','<<o.depth<<','
       <<o.pass<<','<<o.attempted<<','<<o.accepted<<'\n';
}
constexpr const char* header="fork_s,state_checksum,initial_gap_mm,initial_actual_horizontal_mm_s,branch,forced_action,forced_decisions,first_action,first_action_on_release,fault,branch_elapsed_s,mean_horizontal_mm_s,whole_path_min_gap_mm,end_gap_mm,whole_path_max_stall_controls,tilt_deg,depth_error_mm,pass,attempted_substeps,accepted_substeps\n";

int run(const Models& m,const fs::path& output,const fs::path& parent_file,const fs::path& archived,
        Clock::time_point begin) {
    if(chooser::inputs!=35 || chooser::actions!=15 || m.distant_feedback!=2 ||
       !fs::create_directory(output))throw std::runtime_error("Require three-depth actor and fresh replay directory");
    const auto deadline=begin+std::chrono::seconds(145);
    int mode=-1;
    const auto parent=chooser::read_weights(parent_file.string(),&mode);
    if(mode!=m.distant_feedback)throw std::runtime_error("Parent observation mode mismatch");
    Actor actor(chooser::actor_spec());actor.params()=m.policy;
    std::ifstream trace(archived);
    if(!trace)throw std::runtime_error("Missing original Godot trace");
    std::string line;
    std::map<int,std::vector<std::string>> expected;
    std::vector<std::string> stop;
    while(std::getline(trace,line)){
        auto fields=close_probe::fields(line);
        if(fields.size()>17 && fields[12]=="decision")expected.emplace(std::stoi(fields[0]),fields);
        if(fields.size()>17 && fields[12]=="stop")stop=fields;
    }
    State s{};
    const auto start=endurance[0];const auto& pose=m.starts.at(start.archived_row);
    chooser::initialize(s.world,pose,horizontal_place(start,pose),yaw_offset(start,pose),m.motor.kind,m.config);
    std::map<int,State> captures;
    std::map<int,int> baseline_actions;
    std::ofstream prefixes(output/"prefix.csv");prefixes.exceptions(std::ios::failbit|std::ios::badbit);
    prefixes<<std::setprecision(12)<<"control,actual_s,action,com_error_m,logit_error,state_checksum\n";
    int checks=0;
    while(!s.world.fault){
        auto& w=s.world;
        if(chooser::ready(w)){
            const int tick=w.motor.navigation.motion.controls;
            const auto& fields=expected.at(tick);
            const int a=close_probe::decide(w,actor,mode);
            float logit_error=0;
            for(int i=0;i<chooser::actions;++i)logit_error=std::max(logit_error,std::abs(actor.output()[i]-std::stof(fields[17+i])));
            const float err=physics::norm(w.pose.com_m-V{std::stof(fields[2]),std::stof(fields[3]),std::stof(fields[4])});
            if(a!=std::stoi(fields[13]) || err>1e-7f || logit_error>1e-5f ||
               std::abs(clearance(w)-std::stod(fields[8]))>1e-5)throw std::runtime_error("Frozen replay diverged from archived action or state");
            ++checks;
            if(std::binary_search(std::begin(fork_ticks),std::end(fork_ticks),tick)){
                std::memcpy(&captures[tick],&s,sizeof(s));baseline_actions[tick]=a;
            }
            prefixes<<tick<<','<<tick/50.<<','<<a<<','<<err<<','<<logit_error<<','<<close_probe::hash(w)<<'\n';
            if(!chooser::accept(w,a))throw std::runtime_error("Frozen prefix action rejected");
        }
        advance(s,m,deadline);
    }
    const auto& w=s.world;
    if(w.fault!=chooser::Clearance || w.accepted_substeps!=stop_substep || w.attempted_substeps!=stop_substep+1 ||
       captures.size()!=std::size(fork_ticks) || checks!=53 || expected.size()!=53 || stop.size()<17 ||
       std::abs(std::stod(stop[1])-w.accepted_substeps/800.)>1e-9 ||
       physics::norm(w.pose.com_m-V{std::stof(stop[2]),std::stof(stop[3]),std::stof(stop[4])})>1e-7 ||
       std::abs(clearance(w)-std::stod(stop[8]))>1e-6)throw std::runtime_error("Godot stop reproduction failed");
    prefixes.close();
    std::vector<Trial> trials;
    for(const int tick:fork_ticks){
        trials.push_back({tick,-1,0,false});trials.push_back({tick,-1,0,true});
        for(int hold:{1,3})for(int a=0;a<chooser::actions;++a)trials.push_back({tick,a,hold,false});
    }
    if(trials.size()!=384)throw std::runtime_error("Open-replay workload differs");
    std::vector<Outcome> results(trials.size());
    std::vector<std::future<void>> workers;
    constexpr size_t threads=8;
    for(size_t worker=0;worker<threads;++worker)workers.emplace_back(std::async(std::launch::async,[&,worker]{
        for(size_t j=worker;j<trials.size();j+=threads){
            const auto& t=trials[j];results[j]=branch(captures.at(t.fork_tick),t,m,parent,120,deadline);
        }
    }));
    for(auto& worker:workers)worker.get();
    std::ofstream cases(output/"forks.csv");cases.exceptions(std::ios::failbit|std::ios::badbit);
    cases<<std::setprecision(12)<<header;
    int selected=-1;uint64_t attempts=w.attempted_substeps;
    auto rank=[&](size_t i){
        const auto& t=trials[i];
        const int orig=baseline_actions.at(t.fork_tick);
        const int changes=(orig/5!=t.action/5)+(orig%5!=t.action%5);
        return std::make_tuple(t.fork_tick,-t.decisions,-changes,results[i].gap,-t.action);
    };
    for(size_t i=0;i<trials.size();++i){
        const auto& t=trials[i];const auto& o=results[i];
        if(!t.parent && !t.decisions && (o.fault!=chooser::Clearance ||
           o.accepted+captures.at(t.fork_tick).world.accepted_substeps!=stop_substep))
            throw std::runtime_error("Unmodified continuation missed original stop");
        write(cases,t,o,captures.at(t.fork_tick));attempts+=o.attempted;
        if(o.pass && t.decisions && t.action!=baseline_actions.at(t.fork_tick) &&
           (selected<0 || rank(i)>rank(selected)))selected=int(i);
    }
    cases.close();
    std::ofstream summary(output/"summary.json");
    summary.exceptions(std::ios::failbit|std::ios::badbit);
    summary<<std::setprecision(12)<<"{\"reproduced_stop_s\":"<<stop_substep/800.
           <<",\"prefix_decisions_verified\":"<<checks<<",\"frozen_forks\":"<<trials.size()
           <<",\"paired_recovery\":"<<(selected>=0?"true":"false");
    if(selected>=0){
        const auto& t=trials[selected];
        const auto& saved=captures.at(t.fork_tick);
        std::ofstream proof(output/"paired-recovery.csv");
        proof.exceptions(std::ios::failbit|std::ios::badbit);proof<<std::setprecision(12)<<header;
        for(size_t i=0;i<trials.size();++i)if(trials[i].fork_tick==t.fork_tick &&
            (!trials[i].decisions || int(i)==selected))write(proof,trials[i],results[i],saved);
        proof.close();
        const auto follow=branch(saved,t,m,parent,1800,deadline);
        std::ofstream long_case(output/"long-confirmation.csv");
        long_case.exceptions(std::ios::failbit|std::ios::badbit);long_case<<std::setprecision(12)<<header;
        write(long_case,t,follow,saved);attempts+=follow.attempted;
        summary<<",\"latest_fork_s\":"<<t.fork_tick/50.<<",\"forced_action\":"<<t.action<<",\"decisions\":"<<t.decisions
               <<",\"long_pass\":"<<(follow.pass?"true":"false");
    }
    if(attempts>39000000)throw std::runtime_error("Replay budget exceeded");
    summary<<",\"attempted_substeps\":"<<attempts<<",\"optimizer_updates\":0,\"wall_s\":"
           <<std::chrono::duration<double>(Clock::now()-begin).count()<<"}\n";
    return 0;
}
}

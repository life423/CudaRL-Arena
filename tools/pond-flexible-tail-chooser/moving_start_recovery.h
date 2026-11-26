#pragma once

// Only the offline native evaluator may substitute chooser requests.
namespace moving_recovery {
using State = open_approach::State;
using Actor = open_approach::Actor;
constexpr int duration_seconds = 180;
constexpr const char* names[] = {"moving_a","moving_b","moving_c","moving_d"};

struct Plan {
    int actor; // 0 = +104, 1 = +112, 2 = three-depth parent
    int first=-1, hold=0, second=-1, next_hold=0;
};
struct Result {
    int fault=0, first=-1, release=-1, longest_stall=0;
    uint64_t accepted=0, attempted=0;
    double min_mm=0, end_mm=0, at20_mm=-1, prior_low_mm=-1, mean_mm_s=0, tilt=0, depth=0;
    bool escaped20=false, recovered=false;
};

Start fixture(const Models& m,int i,int depth) {
    constexpr float ray[] = {-63,-133,38,146}, gap[] = {38,49,44,36}, offset[] = {-15,21,-37,46};
    const int layer=depth==72?0:(depth==75?1:2);
    const float a=radians(ray[i]+11.f*layer);
    const float x=std::cos(a), z=std::sin(a);
    const double h=std::atan2(-z*larger_pond.water_radius_z_m,x*larger_pond.water_radius_x_m);
    return place_shore_fixture(m,8+(i%2),x,z,float(h*180./tail::pi)+offset[i],gap[i]*.001,names[i]);
}

State initialize(const Models& m,const Start& start,int upstream_mm) {
    const auto& source=m.starts.at(start.archived_row);
    V place=horizontal_place(start,source);
    const float yaw=yaw_offset(start,source);
    State s{};
    chooser::initialize(s.world,source,place,yaw,m.motor.kind,m.config);
    if(upstream_mm) {
        const V v=horizontal(s.world.pose.com_velocity_m_s);
        place=place-v*(.001f*upstream_mm/physics::norm(v));
        chooser::initialize(s.world,source,place,yaw,m.motor.kind,m.config);
    }
    if(s.world.fault || !(open_approach::clearance(s.world)>0))throw std::runtime_error("Invalid moving fixture");
    return s;
}

uint64_t verify(const Models& m,State s,const fs::path& trace,Clock::time_point deadline) {
    std::ifstream input(trace);
    if(!input)throw std::runtime_error("Missing archived +104 trace");
    std::string line;
    if(!std::getline(input,line))throw std::runtime_error("Empty archived trace");
    Actor actor(chooser::actor_spec());actor.params()=m.policy;
    auto& w=s.world;
    int decisions=0;bool stopped=false;
    while(std::getline(input,line)){
        if(stopped)throw std::runtime_error("Unexpected post-stop row");
        const auto row=close_probe::fields(line);
        const auto& event=row.at(12);
        if(event=="macro" || event=="stop")open_approach::advance(s,m,deadline);
        const V ref{std::stof(row[2]),std::stof(row[3]),std::stof(row[4])};
        if(std::abs(double(w.accepted_substeps)/800.-std::stod(row[1]))>1e-7 ||
           physics::norm(w.pose.com_m-ref)>1e-7 || std::abs(open_approach::clearance(w)-std::stod(row[8]))>1e-5)
            throw std::runtime_error("Archived position, clearance or clock differs in "+trace.string());
        if(event=="decision") {
            const int a=close_probe::decide(w,actor,m.distant_feedback);
            if(a!=std::stoi(row[13]) || !chooser::accept(w,a))throw std::runtime_error("Archived action differs");
            for(int j=0;j<chooser::actions;++j)
                if(std::abs(actor.output()[j]-std::stof(row[17+j]))>1e-5f)throw std::runtime_error("Archived logits differ");
            ++decisions;
        } else if(event=="stop") {
            if(w.fault!=chooser::Clearance || w.attempted_substeps!=w.accepted_substeps+1)
                throw std::runtime_error("Missing original clearance stop");
            stopped=true;
        } else if(event!="start" && event!="macro")throw std::runtime_error("Unexpected archived event");
    }
    if(!stopped || !decisions)throw std::runtime_error("Incomplete archived replay");
    return w.attempted_substeps;
}

std::vector<Plan> plans(bool original) {
    std::vector<Plan> v{{0},{1},{2}};
    if(original) {
        for(int n:{1,3,8,16})for(int a=0;a<15;++a)v.push_back({0,a,n});
        for(int speed=0;speed<3;++speed)for(int rate:{0,4})for(int n:{4,8})
            for(int exit_rate:{2,4-rate})v.push_back({0,speed*5+rate,n,speed*5+exit_rate,8});
    } else {
        for(int n:{8,16})for(int speed=0;speed<3;++speed)for(int rate:{0,4})v.push_back({0,5*speed+rate,n});
    }
    return v;
}

Result evaluate(const State& fork,const Models& m,const Plan& plan,const std::vector<float>& weights,Clock::time_point deadline) {
    State s;std::memcpy(&s,&fork,sizeof(s));
    auto& w=s.world;
    Result r;
    Actor actor(chooser::actor_spec());actor.params()=weights;
    int decisions=0, next_window=300, windows=0;
    double distance=0,low=open_approach::clearance(w);
    V start{};
    while(!w.fault && w.accepted_substeps<duration_seconds*samples_per_second) {
        if(chooser::ready(w)) {
            int a;
            if(decisions<plan.hold)a=plan.first;
            else if(decisions<plan.hold+plan.next_hold)a=plan.second;
            else a=close_probe::decide(w,actor,m.distant_feedback);
            if(!decisions)r.first=a;
            if(decisions==plan.hold+plan.next_hold)r.release=a;
            if(!chooser::accept(w,a))throw std::runtime_error("Invalid offline request");
            ++decisions;
        }
        const auto step=open_approach::advance(s,m,deadline);
        if(step.fault)break;
        const int tick=w.motor.navigation.motion.controls;
        if(tick>=next_window) {
            const V at=open_approach::position_at(w,next_window);
            if(next_window>300) {distance+=physics::norm(horizontal(at-start));++windows;}
            start=at;next_window+=25;
        }
        if(tick<=1000)low=std::min(low,1000.*w.min_gap_m);
        if(tick==1000) {
            r.at20_mm=open_approach::clearance(w);r.prior_low_mm=low;
            r.escaped20=r.at20_mm>0 && r.at20_mm>=low+5.;
        }
    }
    r.fault=w.fault;r.accepted=w.accepted_substeps;r.attempted=w.attempted_substeps-fork.world.attempted_substeps;
    r.min_mm=1000.*w.min_gap_m;r.end_mm=open_approach::clearance(w);r.tilt=w.peak_tilt_deg;
    r.depth=w.peak_depth_error_m*1000.;r.longest_stall=s.longest_slow;r.mean_mm_s=windows?2000.*distance/windows:0.;
    r.recovered=!r.fault && r.accepted==duration_seconds*samples_per_second && r.min_mm>0 &&
        r.tilt<=10 && r.depth<=1 && r.longest_stall<=407 && r.mean_mm_s>=6;
    return r;
}

int run(const Models& m,const fs::path& out,const fs::path& actor112,const fs::path& parent,const fs::path& archived,Clock::time_point begin) {
    if(!fs::create_directory(out) || m.starts.size()!=12 || chooser::inputs!=35 || chooser::actions!=15 || m.distant_feedback!=2)
        throw std::runtime_error("Require untouched three-depth recovery run directory and model");
    const int depth=int(std::lround(-1000*tail::snapshot(m.starts[8].motion.body).com_m.y));
    if(depth!=72 && depth!=75 && depth!=80)throw std::runtime_error("Invalid depth");
    std::array<std::vector<float>,3> policies{m.policy,{},{}};
    int mode=0;
    policies[1]=chooser::read_weights(actor112.string(),&mode);
    if(mode!=2)throw std::runtime_error("Incorrect +112 mode");
    policies[2]=chooser::read_weights(parent.string(),&mode);
    if(mode!=2)throw std::runtime_error("Incorrect three-depth parent mode");
    const auto deadline=begin+std::chrono::seconds(180);
    std::ofstream table(out/"branches.csv"),cases(out/"cases.csv");
    table.exceptions(std::ios::failbit|std::ios::badbit);cases.exceptions(std::ios::failbit|std::ios::badbit);
    table<<std::setprecision(12)<<"depth_mm,case,upstream_mm,cue_hash,actor,forced_action,hold_decisions,second_action,second_hold_decisions,first_actual_action,first_autonomous_action,fault,survival_s,resets,min_clearance_mm,end_clearance_mm,gap_at_20s_mm,lowest_gap_by_20s_mm,escape20,mean_speed_mm_s,max_stall_controls,tilt_deg,depth_error_mm,recovered180,branch_attempted_substeps\n";
    cases<<std::setprecision(12)<<"depth_mm,case,upstream_mm,spawn_gap_mm,cue_gap_mm,cue_actual_speed_mm_s,replicated_archived_stop_s,branches,escaped20,recovered180,best_survival_s,best_branch_id,best_min_gap_mm\n";
    uint64_t work=0;int count=0;
    for(int shift:{0,60,120})for(int i=0;i<4;++i) {
        auto state=initialize(m,fixture(m,i,depth),shift);
        const double spawn_gap=open_approach::clearance(state.world);
        const auto repro=shift?0:verify(m,state,archived/"traces"/(std::string(names[i])+".csv"),deadline);
        work+=repro;
        while(!chooser::ready(state.world))open_approach::advance(state,m,deadline);
        if(state.world.motor.navigation.motion.controls!=40)throw std::runtime_error("Wrong first chooser cue");
        const auto hash=close_probe::hash(state.world);
        work+=state.world.attempted_substeps;
        const auto trials=plans(shift==0);
        std::vector<Result> results(trials.size());
        std::array<std::future<void>,8> workers;
        for(size_t k=0;k<workers.size();++k)workers[k]=std::async(std::launch::async,[&,k]{
            for(size_t j=k;j<trials.size();j+=workers.size())
                results[j]=evaluate(state,m,trials[j],policies[trials[j].actor],deadline);
        });
        for(auto& worker:workers)worker.get();
        int escaped=0,full=0;size_t best=0;
        for(size_t j=0;j<trials.size();++j){
            const auto& t=trials[j];const auto& r=results[j];work+=r.attempted;++count;
            if(j==0 && !shift && (r.fault!=chooser::Clearance || r.accepted+1!=repro))throw std::runtime_error("Forked baseline differs");
            escaped+=r.escaped20;full+=r.recovered;
            if(r.recovered>results[best].recovered || (r.recovered==results[best].recovered &&
                (r.accepted>results[best].accepted || (r.accepted==results[best].accepted && r.min_mm>results[best].min_mm))))best=j;
            table<<depth<<','<<names[i]<<','<<shift<<','<<hash<<','<<(t.actor==0?104:(t.actor==1?112:128))<<','
                 <<t.first<<','<<t.hold<<','<<t.second<<','<<t.next_hold<<','<<r.first<<','<<r.release<<','<<r.fault<<','
                 <<r.accepted/800.<<",0,"<<r.min_mm<<','<<r.end_mm<<','<<r.at20_mm<<','<<r.prior_low_mm<<','<<r.escaped20<<','
                 <<r.mean_mm_s<<','<<r.longest_stall<<','<<r.tilt<<','<<r.depth<<','<<r.recovered<<','<<r.attempted<<'\n';
        }
        const auto& r=results[best];
        cases<<depth<<','<<names[i]<<','<<shift<<','<<spawn_gap<<','<<open_approach::clearance(state.world)<<','
             <<1000.*physics::norm(horizontal(state.world.pose.com_velocity_m_s))<<','<<(repro?(repro-1)/800.:0.)<<','
             <<trials.size()<<','<<escaped<<','<<full<<','<<r.accepted/800.<<','<<best<<','<<r.min_mm<<'\n';
        cases.flush();table.flush();
    }
    if(count!=468 || work>67400000)throw std::runtime_error("Unexpected diagnostic workload");
    std::ofstream receipt(out/"summary.json");receipt.exceptions(std::ios::failbit|std::ios::badbit);
    receipt<<"{\"depth_mm\":"<<depth<<",\"branches\":"<<count<<",\"attempted_substeps\":"<<work<<",\"optimizer_updates\":0,\"wall_s\":"
           <<std::chrono::duration<double>(Clock::now()-begin).count()<<"}\n";
    return 0;
}
} // namespace moving_recovery

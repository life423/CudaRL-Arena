// A separate evaluator keeps the active training and selection binaries immutable.
#define main existing_rate_evaluator_main
#include "../../pond-flexible-tail-chooser/rate-evaluate.cpp"
#undef main

namespace upstream {
struct Difference {
    double time_s=-1;
    int chosen=-1, reference=-1;
};
struct Measurements {
    moving_recovery::Result result;
    std::array<Difference,3> divergence{};
    std::array<int,3> peer_first{};
    double spawn_gap=0,cue_gap=0,cue_speed=0;
    uint64_t cue_hash=0;
};

Measurements swim(const Models& m,int depth,int index,int shift,const std::array<std::vector<float>,3>& peers,
                  Clock::time_point deadline) {
    Measurements data;
    auto state=moving_recovery::initialize(m,moving_recovery::fixture(m,index,depth),shift);
    auto& world=state.world;
    data.spawn_gap=open_approach::clearance(world);
    while(!chooser::ready(world))open_approach::advance(state,m,deadline);
    if(world.fault || world.motor.navigation.motion.controls!=40)throw std::runtime_error("Invalid native cue");
    data.cue_gap=open_approach::clearance(world);data.cue_hash=close_probe::hash(world);
    data.cue_speed=1000.*physics::norm(horizontal(world.pose.com_velocity_m_s));
    auto& r=data.result;
    open_approach::Actor actor(chooser::actor_spec());actor.params()=m.policy;
    std::vector<open_approach::Actor> refs;
    for(const auto& weights:peers) {
        refs.emplace_back(chooser::actor_spec());refs.back().params()=weights;
    }
    int next=300,windows=0;
    V previous{};double distance=0,low=data.cue_gap;
    while(!world.fault && world.accepted_substeps<180*samples_per_second) {
        if(chooser::ready(world)) {
            const int action=close_probe::decide(world,actor,m.distant_feedback);
            bool first=r.first<0;
            if(first)r.first=action;
            for(size_t j=0;j<refs.size();++j) {
                auto& d=data.divergence[j];
                if(d.time_s>=0)continue;
                const int value=close_probe::decide(world,refs[j],m.distant_feedback);
                if(first)data.peer_first[j]=value;
                if(value!=action)d={world.accepted_substeps/800.,action,value};
            }
            if(!chooser::accept(world,action))throw std::runtime_error("Rejected greedy request");
        }
        const auto step=open_approach::advance(state,m,deadline);
        if(step.fault)break;
        const int tick=world.motor.navigation.motion.controls;
        if(tick>=next) {
            const auto pos=open_approach::position_at(world,next);
            if(next>300) {distance+=physics::norm(horizontal(pos-previous));++windows;}
            next+=25;previous=pos;
        }
        if(tick<=1000)low=std::min(low,1000.*world.min_gap_m);
        if(tick==1000) {
            r.at20_mm=open_approach::clearance(world);r.prior_low_mm=low;
            r.escaped20=r.at20_mm>0 && r.at20_mm>=low+5.;
        }
    }
    r.fault=world.fault;r.accepted=world.accepted_substeps;r.attempted=world.attempted_substeps;
    r.min_mm=1000.*world.min_gap_m;r.end_mm=open_approach::clearance(world);
    r.tilt=world.peak_tilt_deg;r.depth=1000.*world.peak_depth_error_m;
    r.longest_stall=state.longest_slow;r.mean_mm_s=windows?2000.*distance/windows:0.;
    r.recovered=!r.fault && r.accepted==180*samples_per_second && r.min_mm>0 &&
        r.tilt<=10 && r.depth<=1 && r.longest_stall<=407 && r.mean_mm_s>=6;
    return data;
}

int run(int argc,char** argv) {
    if(argc!=9)throw std::runtime_error("upstream GAIT MOTOR SELECTED STARTS BASE104 BASE112 BASE128 OUTPUT");
    Models m;
    m.gait=tail::read_navigation_weights(argv[1],tail::NavigationMode::Gait);
    m.motor=tail::steering::read_model(argv[2]);
    m.config.study=tail::steering::Study::Rate;m.config.material=tail::steering::read_material("0");
    if(m.motor.kind<tail::steering::Kind::ProgramScalar || m.motor.material!=m.config.material)
        throw std::runtime_error("Motor mismatch");
    tail::fast::validate_frozen_gait(m.gait);
    m.packed_motor=m.motor.weights;m.packed_motor.insert(m.packed_motor.end(),m.motor.frozen_skip.begin(),m.motor.frozen_skip.end());
    m.policy=chooser::read_weights(argv[3],&m.distant_feedback);m.starts=tail::read_navigation_worlds(argv[4]);
    if(m.distant_feedback!=2 || m.starts.size()!=12 || !tail::distant::geometry_matches(larger_pond))
        throw std::runtime_error("Incorrect observations, habitat or starts");
    std::array<std::vector<float>,3> peers;
    for(int i=0;i<3;++i) {
        int mode=-1;peers[i]=chooser::read_weights(argv[5+i],&mode);
        if(mode!=m.distant_feedback)throw std::runtime_error("Incompatible archived chooser");
    }
    const auto depth=int(std::lround(-1000*tail::snapshot(m.starts[8].motion.body).com_m.y));
    if(depth!=72 && depth!=75 && depth!=80)throw std::runtime_error("Unexpected depth");
    const fs::path out=argv[8];if(!fs::create_directory(out))throw std::runtime_error("Reuse of evidence directory");
    const auto deadline=Clock::now()+std::chrono::seconds(45);
    std::ofstream csv(out/"candidate.csv");
    csv.exceptions(std::ios::failbit|std::ios::badbit);
    csv<<std::setprecision(12)<<"depth_mm,case,upstream_mm,spawn_gap_mm,cue_gap_mm,cue_speed_mm_s,cue_hash,first_action,fault,survival_s,min_gap_mm,longest_stall_controls,mean_speed_mm_s,tilt_deg,depth_error_mm,escaped20,qualified180s,attempted_substeps";
    for(int id:{104,112,128})csv<<",first_"<<id<<",different_s_"<<id<<",candidate_when_different_"<<id<<",reference_when_different_"<<id;
    csv<<'\n';
    // Four workers keep the descriptive checks small beside official qualification.
    for(int shift:{60,120}) {
        std::array<std::future<Measurements>,4> workers;
        for(int i=0;i<4;++i)workers[i]=std::async(std::launch::async,[&,i,shift]{return swim(m,depth,i,shift,peers,deadline);});
        for(int i=0;i<4;++i) {
            const auto d=workers[i].get();const auto& r=d.result;
            csv<<depth<<','<<moving_recovery::names[i]<<','<<shift<<','<<d.spawn_gap<<','<<d.cue_gap<<','<<d.cue_speed<<','
               <<d.cue_hash<<','<<r.first<<','<<r.fault<<','<<r.accepted/800.<<','<<r.min_mm<<','<<r.longest_stall<<','<<r.mean_mm_s
               <<','<<r.tilt<<','<<r.depth<<','<<r.escaped20<<','<<r.recovered<<','<<r.attempted;
            for(size_t j=0;j<peers.size();++j) {
                const auto& v=d.divergence[j];csv<<','<<d.peer_first[j]<<','<<v.time_s<<','<<v.chosen<<','<<v.reference;
            }
            csv<<'\n';
        }
    }
    return 0;
}
}
int main(int argc,char** argv) {
    try{return upstream::run(argc,argv);}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}

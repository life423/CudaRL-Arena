#ifndef CUDARL_TRANSITION_MOTOR_TOOL_H
#define CUDARL_TRANSITION_MOTOR_TOOL_H
// Bounded experiment functions inside the existing native runner. No runtime
// owner, session hierarchy, policy fallback or additional simulation equations.
namespace transition_experiment {
constexpr int control_cap=3680,collection_cap=800,evaluation_cap=1440;
struct Counter {
    int controls=0,substeps=0;
    void advance(m::World& state,const demo::SignedReference& reference){
        wall_guard();require(controls<control_cap&&substeps+10<=control_cap*10,"Transition budget exhausted");++controls;
        const auto before=state.attempted_substeps;const bool ok=demo::step_signed(state,reference);
        substeps+=static_cast<int>(state.attempted_substeps-before);
        require(ok&&state.attempted_substeps-before==10,"Transition physical/actuator safety stop");
    }
};
struct Score {bool passed=false,hold=true;float steady=0,rmse=0,hold_speed=0,hold_rmse=0,work_per_m=0,lateral=0,depth=0,tilt=0;};
// Same first80 controls and gates as the accepted task; subsequent40 controls
// have their own score and cannot repair a failed first window.
template<class Controller> Score segment(m::World& world,m::ForwardTask& task,const m::ForwardConfig& config,int count,Controller controller,Counter& budget,demo::TransitionExamples* examples=nullptr){
    Score score;const float work_start=world.physical.body.work;p::V hold_start{};double hold_error=0;
    for(int tick=0;tick<count;++tick){auto observed=task;observed.outcome=m::Outcome::Running;float obs[42]{};
        require(m::observe(observed,world,config,obs),"Invalid complete transition observation");
        demo::SignedReference reference;require(controller(obs,reference),"Invalid signed controller output");
        budget.advance(world,reference);
        if(tick<80){const auto transition=m::record_control(task,world,config);require(transition.accepted,"Rejected transition interval");}
        if(examples){demo::append_example(*examples,obs,reference);++examples->new_rows;}
        const auto metrics=m::measure(task,world);score.lateral=fmaxf(score.lateral,metrics.lateral_m);score.depth=fmaxf(score.depth,metrics.depth_m);score.tilt=fmaxf(score.tilt,metrics.tilt_deg);
        if(tick==79){score.passed=task.outcome==m::Outcome::Success;score.steady=task.steady_speed_m_s;score.rmse=task.speed_rmse;hold_start=cudarl::motor::motion(world.physical.body).com_model;}
        if(tick>=80)hold_error+=metrics.speed_error_m_s*metrics.speed_error_m_s;
    }
    const auto metrics=m::measure(task,world);score.work_per_m=(world.physical.body.work-work_start)/fmaxf(metrics.forward_m,1e-9f);
    if(count>80){const auto delta=(cudarl::motor::motion(world.physical.body).com_model-hold_start)*p::scale;const float forward=p::dot(delta,task.request.heading);const auto perpendicular=delta-task.request.heading*forward;
        score.hold_speed=forward/(float(count-80)*p::dt);score.hold_rmse=sqrtf(static_cast<float>(hold_error/(count-80)));
        score.hold=score.hold_speed>=config.minimum_speed_m_s&&score.hold_speed<=config.maximum_speed_m_s&&score.hold_rmse<=config.speed_rmse_limit&&
            score.lateral<=config.lateral_limit_m&&score.depth<=config.depth_limit_m&&score.tilt<=config.tilt_limit_deg&&
            atan2f(p::norm({perpendicular.x,0,perpendicular.z}),forward)*180/cudarl::movement::pi<=config.path_angle_limit_deg&&score.work_per_m<=config.work_per_distance_limit;
    }
    return score;
}
inline std::vector<uint8_t> protocol(const fs::path& actor,const fs::path& examples){
    cp::Writer out;out.value(uint32_t{0x32584E54});out.value(demo::kSignedCadenceMapping);out.value(seed);
    for(int value:{control_cap,collection_cap,evaluation_cap,600,256,80,120,0,3,1,2})out.value(value);
    for(float value:{demo::kBrakingAmplitude,.1f,1.f,-1.f,.7f,1.5f,.001f,.00001f,.5f,20.f})out.value(value);
    out.value(cp::checksum(cp::read(actor.string())));out.value(cp::checksum(cp::read(examples.string())));return out.data;
}
inline void progress(const fs::path& root,const char* phase,const Counter& budget,int optimizer_steps=0){
    std::ofstream out(root/"transition-status.json");out<<std::setprecision(17)<<"{\"phase\":"<<std::quoted(phase)<<",\"unix_seconds\":"<<std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()<<",\"controls\":"<<budget.controls<<",\"physical_substeps\":"<<budget.substeps<<",\"control_cap\":3680,\"optimizer_steps\":"<<optimizer_steps<<"}\n";out.flush();
}
inline int prepare(const fs::path& root,const fs::path& actor_file,const fs::path& examples_file){
    require(!fs::exists(root/"transition-protocol.bin")&&!fs::exists(root/"transition-started"),"Transition registration already exists");fs::create_directories(root);
    cudarl::motor::FrozenController actor;require(actor.load(actor_file.string()),"Accepted forward actor rejected");
    const auto reused=demo::reuse_compatible(demo::accepted_examples(cp::read(examples_file.string())));require(reused.reused_rows>=256,"Too few compatible complete source examples");
    const auto initial=demo::initialize_signed(actor.actor_parameters(),specification(2));finite(initial);
    cp::write_atomic((root/"transition-protocol.bin").string(),protocol(actor_file,examples_file));
    std::ofstream out(root/"transition-readiness.json");out<<"{\"ready\":true,\"physical_steps\":0,\"optimizer_steps\":0,\"signed_action_mapping\":2,\"actor_shape\":[42,32,32,2],\"reused_compatible_rows\":"<<reused.reused_rows<<",\"new_complete_rows_max\":480,\"teacher_training_poses\":[0,3],\"heldout_transition_poses\":[1,2],\"withheld_constant_speeds_mm_s\":[5,7],\"collection_controls\":800,\"baseline_controls\":1440,\"frozen_candidate_controls\":1440,\"control_cap\":3680,\"substep_cap\":36800,\"optimizer_steps_cap\":600,\"native_wall_seconds\":20,\"process_wall_seconds\":25,\"retries\":0,\"automatic_adoption\":false}\n";
    out.close();std::ifstream immediate(root/"transition-readiness.json");std::cout<<immediate.rdbuf()<<std::flush;return 0;
}
inline void warm(cudarl::motor::FrozenController& controller,const fs::path& actor,int pose,float target,Counter& budget){
    require(controller.load(actor.string())&&controller.set_speed(target)&&controller.reset(pose)&&controller.start(),"Accepted warm actor rejected");
    for(int tick=0;tick<80;++tick){wall_guard();require(budget.controls<control_cap&&budget.substeps+10<=control_cap*10,"Warmup budget exhausted");++budget.controls;
        const auto before=controller.world().attempted_substeps;const bool ok=controller.step();budget.substeps+=static_cast<int>(controller.world().attempted_substeps-before);require(ok,"Accepted warmup safety stop");}
    require(controller.task().outcome==m::Outcome::Success,"Accepted warmup did not pass constant-speed gate");
}
inline void collect(const fs::path& root,const fs::path& actor,demo::TransitionExamples& examples,Counter& budget){
    int profile=0;std::ofstream gates(root/"teacher-gates.csv");gates<<"pose,target_mm_s,passed,hold,steady_mm_s,RMSE_mm_s,hold_mm_s,hold_RMSE_mm_s,lateral_mm,depth_mm,work_J_per_m\n";
    for(int pose:{0,3})for(bool down:{true,false}){
        progress(root,"collecting_teacher",budget);cudarl::motor::FrozenController initial;warm(initial,actor,pose,down?.008f:.004f,budget);
        auto world=initial.world();auto task=initial.task();auto config=m::config_for_speed(down?.008f:.004f);
        require(m::request_speed(world,down?.004f:.008f,config,task),"Teacher request rejected");
        const auto score=segment(world,task,config,120,demo::signed_teacher,budget,&examples);
        gates<<pose<<','<<config.target_speed_m_s*1000<<','<<score.passed<<','<<score.hold<<','<<score.steady*1000<<','<<score.rmse*1000<<','<<score.hold_speed*1000<<','<<score.hold_rmse*1000<<','<<score.lateral*1000<<','<<score.depth*1000<<','<<score.work_per_m<<'\n';gates.flush();
        std::cout<<"TEACHER_GATE pose="<<pose<<" target="<<config.target_speed_m_s<<" passed="<<score.passed<<" hold="<<score.hold<<std::endl;
        require(score.passed&&score.hold,"Teacher transition/settling gate failed; no fitting");examples.accepted_segments|=uint32_t{1}<<profile++;
    }
    require(budget.controls==collection_cap&&examples.new_rows==480,"Collection accounting mismatch");
}
struct Evaluation {int constants=0,warmups=0,transitions=0,holds=0;float normalized_rmse=0,work8=0;};
inline Evaluation evaluate(const std::vector<float>& params,const fs::path& root,const char* phase,Counter& budget){
    const int first_control=budget.controls;Evaluation result;nn::CpuMlp<float> actor(specification(2));actor.params()=params;
    auto policy=[&](const float* obs,demo::SignedReference& reference){actor.forward(obs,1);return demo::signed_action(actor.output(),reference);};
    std::ofstream log(root/(std::string(phase)+"-evaluation.csv"));log<<"kind,pose,target_mm_s,passed,hold,steady_mm_s,RMSE_mm_s,work_J_per_m,lateral_mm\n";
    auto record=[&](const char* kind,int pose,float target,const Score& score){log<<kind<<','<<pose<<','<<target*1000<<','<<score.passed<<','<<score.hold<<','<<score.steady*1000<<','<<score.rmse*1000<<','<<score.work_per_m<<','<<score.lateral*1000<<'\n';};
    for(int pose:{0,3})for(float target:{.004f,.005f,.006f,.007f,.008f}){progress(root,phase,budget);auto world=m::initial_world(pose);auto config=m::config_for_speed(target);m::ForwardTask task;require(m::begin_forward(world,config,task),"Invalid constant evaluation start");
        const auto score=segment(world,task,config,80,policy,budget);record("constant",pose,target,score);result.constants+=score.passed?1:0;result.normalized_rmse+=score.rmse/target/10;if(target==.008f)result.work8+=score.work_per_m/2;}
    for(int pose:{1,2}){auto world=m::initial_world(pose);auto config=m::config_for_speed(.004f);m::ForwardTask task;require(m::begin_forward(world,config,task),"Invalid heldout start");
        auto score=segment(world,task,config,80,policy,budget);record("heldout_warmup",pose,.004f,score);result.warmups+=score.passed?1:0;
        for(float target:{.008f,.004f}){progress(root,phase,budget);require(m::request_speed(world,target,config,task),"Heldout persistent request rejected");score=segment(world,task,config,120,policy,budget);record("heldout_transition",pose,target,score);result.transitions+=score.passed?1:0;result.holds+=score.hold?1:0;}
    }
    require(budget.controls-first_control==evaluation_cap,"Evaluation stage accounting differs");return result;
}
inline int run(const fs::path& root,const fs::path& actor_file,const fs::path& examples_file){
    const auto frozen=protocol(actor_file,examples_file);require(cp::read((root/"transition-protocol.bin").string())==frozen,"Transition protocol/input changed");require(fs::create_directory(root/"transition-started"),"Transition run already attempted; no retry");
    wall_start=std::chrono::steady_clock::now();wall_limit=20;fit_steps=600;Counter budget;Evaluation before,after;demo::TransitionExamples examples;int completed=0;bool collected=false,fitting=false;std::string fault;
    try{
        int devices=0;CUDA_CHECK_THROW(cudaGetDeviceCount(&devices));require(devices>0,"Real CUDA required for fitting");
        examples=demo::reuse_compatible(demo::accepted_examples(cp::read(examples_file.string())));collect(root,actor_file,examples,budget);collected=true;
        cp::write_atomic((root/"accepted-transition-examples.bin").string(),demo::encode_transitions(examples,cp::checksum(frozen)));
        examples=demo::decode_transitions(cp::read((root/"accepted-transition-examples.bin").string()),cp::checksum(frozen));
        cudarl::motor::FrozenController accepted;require(accepted.load(actor_file.string()),"Accepted initialization actor rejected");const auto initial=demo::initialize_signed(accepted.actor_parameters(),specification(2));
        before=evaluate(initial,root,"initial",budget);
        const auto training=demo::balanced_training(examples);Stream stream;fitting=true;progress(root,"supervised_training",budget);
        std::cout<<"TRAINING_STARTED signed-cadence cloning; all four teacher gates and holds passed; no evaluation teacher/fallback"<<std::endl;
        auto save=[&](int step,nn::GpuMlp& actor,nn::GpuAdam& optimizer){cp::Writer out;out.value(uint32_t{0x32524E54});out.value(uint32_t{1});out.value(demo::kSignedCadenceMapping);out.value(cp::checksum(frozen));out.value(step);out.array(actor.get_params());optimizer.save(out);cp::write_atomic((root/("signed-update-"+std::to_string(step)+".bin")).string(),out.data);progress(root,"supervised_training",budget,step);};
        const auto fitted=fit_samples(training,initial,stream,root,budget.controls,completed,save);
        after=evaluate(fitted,root,"frozen_candidate",budget);require(budget.controls==control_cap&&budget.substeps==control_cap*10,"Final transition budget accounting differs");
    }catch(const std::exception& error){fault=error.what();}
    cp::write_atomic((root/"collection-attempts.bin").string(),demo::encode_transitions(examples,cp::checksum(frozen)));
    const bool accepted=fault.empty()&&after.constants==10&&after.warmups==2&&after.transitions==4&&after.holds==4&&after.normalized_rmse<=before.normalized_rmse&&after.work8<=before.work8;
    std::ofstream out(root/"transition-result.json");out<<std::setprecision(9)<<"{\"experiment\":\"signed-transitions-v1\",\"signed_mapping\":2,\"fault\":"<<std::quoted(fault)<<",\"teacher_accepted\":"<<(collected?"true":"false")<<",\"fit_started\":"<<(fitting?"true":"false")<<",\"optimizer_steps\":"<<completed<<",\"controls\":"<<budget.controls<<",\"physical_substeps\":"<<budget.substeps<<",\"frozen_constant_passes\":"<<after.constants<<",\"heldout_warmup_passes\":"<<after.warmups<<",\"heldout_transition_passes\":"<<after.transitions<<",\"heldout_hold_passes\":"<<after.holds<<",\"candidate_accepted\":"<<(accepted?"true":"false")<<",\"native_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-wall_start).count()<<",\"closed\":true,\"retries\":0,\"runtime_adopted\":false}\n";out.close();progress(root,fault.empty()?"done":"failed",budget,completed);std::ifstream immediate(root/"transition-result.json");std::cout<<immediate.rdbuf()<<std::flush;return fault.empty()?0:3;
}
}
#endif

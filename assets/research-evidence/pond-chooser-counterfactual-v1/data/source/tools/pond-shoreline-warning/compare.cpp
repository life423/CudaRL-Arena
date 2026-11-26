#define main archived_rate_evaluator_main
#include "../pond-flexible-tail-chooser/rate-evaluate.cpp"
#undef main
#include <map>

namespace counterfactual {
constexpr uint64_t work_cap=7776000;
uint64_t attempted=0,accepted=0;
Clock::time_point deadline;
void bounded(){if(attempted>work_cap||Clock::now()>=deadline)throw std::runtime_error("Counterfactual execution cap reached");}
chooser::Result step(chooser::World& w,const Models& m){
    bounded();auto result=chooser::advance(w,m.gait.data(),m.packed_motor.data(),larger_pond,4);
    attempted+=result.attempted_substeps;accepted+=result.accepted_substeps;bounded();
    if(result.fault&&result.fault!=chooser::Clearance)throw std::runtime_error("Counterfactual native/numerical fault");
    return result;
}
std::vector<std::string> split(const std::string& text){std::vector<std::string> fields;std::stringstream s(text);std::string word;
    while(std::getline(s,word,','))fields.push_back(word);return fields;}
struct Expected {V com{};int action=-1,fault=0,control=0;};
struct Fixture {const char* name;int policy,diagnostic_index;double failure_s;int accepted;};
constexpr Fixture fixtures[]{
    {"northwest_diagonal",0,10,71.495,57196},
    {"east_shelf_tangent",1,4,95.11,76088},
    {"northeast_to_center",1,8,35.32375,28259}};
struct Policy {
    cudarl::nn::CpuMlp<float> actor{chooser::actor_spec()};int mode=0;
    explicit Policy(const char* path){actor.params()=chooser::read_weights(path,&mode);if(mode!=2)throw std::runtime_error("Expected qualified35-input observation format");}
    int decide(const chooser::World& w,float* saved_input=nullptr,float* saved_logits=nullptr){
        float input[chooser::inputs];chooser::observe(w,larger_pond,input,mode);actor.forward(input,1);
        const auto logits=actor.output();if(!std::all_of(logits,logits+chooser::actions,physics::finite_number))throw std::runtime_error("Nonfinite frozen chooser");
        if(saved_input)std::copy(input,input+chooser::inputs,saved_input);if(saved_logits)std::copy(logits,logits+chooser::actions,saved_logits);
        return int(std::max_element(logits,logits+chooser::actions)-logits);
    }
};
uint64_t state_hash(const chooser::World& w){cudarl::checkpoint::Writer b;b.value(w);return cudarl::checkpoint::checksum(b.data);}
V velocity(const chooser::World& w,int controls=50){return tail::turn_course(w.motor.navigation.motion,w.pose.com_m,controls)/(controls*.02f);}
V outward(V p){V n{p.x/float(larger_pond.water_radius_x_m*larger_pond.water_radius_x_m),0,p.z/float(larger_pond.water_radius_z_m*larger_pond.water_radius_z_m)};
    const float length=physics::norm(n);return length>1e-6f?n/length:V{1,0,0};}
struct Capture {chooser::World world{};int fixture=0,lead=0;uint64_t hash=0;bool found=false;};
void save_capture(const fs::path& root,const Capture& c){cudarl::checkpoint::Writer b;b.value<uint32_t>(1);b.value(c.world);
    cudarl::checkpoint::write_atomic((root/"captures"/("fixture"+std::to_string(c.fixture)+"-lead"+std::to_string(c.lead)+".bin")).u8string(),b.data);}
void replay(const Fixture& f,int index,const Models& m,Policy& original,const fs::path& archive,const fs::path& root,
    std::vector<Capture>& captures,std::ostream& proof){
    std::map<int,Expected> macros,decisions;Expected final{};std::string line;
    const auto trace=archive/(f.policy==0?"update-008":"update-128")/"assessment"/"traces"/(std::string(f.name)+".csv");
    std::ifstream input(trace);if(!input)throw std::runtime_error("Missing archived prefix");std::getline(input,line);
    while(std::getline(input,line)){auto fields=split(line);if(fields.size()<33)throw std::runtime_error("Archived trace schema differs");
        Expected sample{{std::stof(fields[2]),std::stof(fields[3]),std::stof(fields[4])},-1,std::stoi(fields[32]),std::stoi(fields[0])};
        if(fields[12]=="macro")macros[sample.control]=sample;
        if(fields[12]=="decision"){sample.action=std::stoi(fields[13]);decisions[sample.control]=sample;}
        if(fields[12]=="stop")final=sample;
    }
    if(final.fault!=2||decisions.empty())throw std::runtime_error("Expected saved clearance failure");
    const auto start=diagnostic.at(f.diagnostic_index);const auto& archived=m.starts.at(start.archived_row);
    chooser::World w{};chooser::initialize(w,archived,horizontal_place(start,archived),yaw_offset(start,archived),m.motor.kind,m.config);
    std::array<Capture,3> points{};int cursor=0;for(int lead:{20,12,4}){points[cursor].fixture=index;points[cursor].lead=lead;++cursor;}
    const auto work_before=attempted,accepted_before=accepted;float difference=0;int matched=0,actions=0;
    while(w.fault==0&&w.motor.navigation.motion.controls<6000){
        const int control=w.motor.navigation.motion.controls;
        if(chooser::ready(w)){
            for(auto& capture:points){const int target=int(std::floor((f.failure_s-capture.lead)/.8))*40;
                if(control==target){std::memcpy(&capture.world,&w,sizeof(w));capture.hash=state_hash(w);capture.found=true;}}
            const int choice=original.decide(w);const auto expected=decisions.find(control);
            if(expected==decisions.end()||choice!=expected->second.action)throw std::runtime_error("Archived chooser action mismatch");
            ++actions;if(!chooser::accept(w,choice))throw std::runtime_error("Original prefix action rejected");
        }
        step(w,m);const auto found=macros.find(w.motor.navigation.motion.controls);
        if(!w.fault){if(found==macros.end())throw std::runtime_error("Archived COM sample missing");
            difference=std::max(difference,physics::norm(w.pose.com_m-found->second.com));++matched;
            if(difference>1e-6f)throw std::runtime_error("Archived prefix COM mismatch");}
    }
    difference=std::max(difference,physics::norm(w.pose.com_m-final.com));
    if(w.fault!=2||accepted-accepted_before!=uint64_t(f.accepted)||w.motor.navigation.motion.controls!=final.control||difference>1e-6f||actions!=int(decisions.size()))
        throw std::runtime_error("Archived prefix endpoint/work mismatch");
    proof<<index<<','<<f.name<<','<<f.policy<<','<<matched<<','<<actions<<','<<difference<<','<<attempted-work_before<<','<<accepted-accepted_before<<','<<w.fault<<'\n';proof.flush();
    for(const auto& capture:points){if(!capture.found||!chooser::ready(capture.world))throw std::runtime_error("Exact predecision state unavailable");
        save_capture(root,capture);captures.push_back(capture);}
}
struct Sequence {int kind=0,speed=-1,rate=0,turn_s=0;std::string label;};
struct Result {std::string label;int fixture=0,lead=0,fault=0,first_action=-1;float elapsed=0,gap=FLT_MAX,tilt=0,depth=0,mean=0,stall=0,inward=-1,angle=0;bool safe=false,useful=false;uint64_t attempted=0,accepted=0,hash=0;};
Result roll(const Capture& capture,const Sequence& sequence,const Models& m,Policy& source,Policy& installed,const fs::path& root,std::ostream& decisions){
    chooser::World w{};std::memcpy(&w,&capture.world,sizeof(w));if(state_hash(w)!=capture.hash)throw std::runtime_error("Counterfactual fork state differs");
    Result r;r.fixture=capture.fixture;r.lead=capture.lead;r.label=sequence.label;r.hash=capture.hash;
    const auto begin_attempted=attempted,begin_accepted=accepted;const int cue=w.motor.navigation.motion.controls;
    std::ostringstream name;name<<"f"<<capture.fixture<<"-lead"<<capture.lead<<'-'<<sequence.label;
    std::ofstream trace(root/"traces"/(name.str()+".csv"));trace.exceptions(std::ios::failbit|std::ios::badbit);
    trace<<std::setprecision(10)<<"elapsed_s,control,stage,action,x_m,y_m,z_m,COM_speed_mm_s,certified_gap_mm,tilt_deg,depth_error_mm,fault\n";
    const auto initial=velocity(w);const auto initial_direction=initial/physics::norm(initial);float previous_angle=0,total_angle=0,speed_sum=0;int samples=0,stall_controls=0,inward_controls=0;
    while(w.fault==0&&w.motor.navigation.motion.controls-cue<2000){
        const int relative=w.motor.navigation.motion.controls-cue;
        const char* stage=sequence.kind==1?"installed-policy":
            sequence.kind==2&&relative<sequence.turn_s*50?"diagnostic-turn":
            sequence.kind==2&&relative<(sequence.turn_s+8)*50?"diagnostic-straight":"source-policy";
        int choice=-1;
        if(chooser::ready(w)){
            if(sequence.kind==1){choice=installed.decide(w);stage="installed-policy";}
            else if(sequence.kind==2&&relative<(sequence.turn_s+8)*50){
                const int rate=relative<sequence.turn_s*50?sequence.rate:0;choice=sequence.speed*5+(rate+8)/4;
                stage=relative<sequence.turn_s*50?"diagnostic-turn":"diagnostic-straight";
            }else choice=source.decide(w);
            if(r.first_action<0)r.first_action=choice;
            decisions<<capture.fixture<<','<<capture.lead<<','<<sequence.label<<','<<relative*.02f<<','<<stage<<','<<choice<<'\n';
            if(!chooser::accept(w,choice))throw std::runtime_error("Counterfactual request rejected");
        }
        const int previous=w.motor.navigation.motion.controls;const auto result=step(w,m);
        r.elapsed=float(accepted-begin_accepted)/800.f;
        const auto historic=[&](int control){if(control==w.motor.navigation.motion.controls)return w.pose.com_m;
            const auto& motion=w.motor.navigation.motion;const int back=motion.controls-control;
            if(back<1||back>tail::turn_history_controls)throw std::runtime_error("Missing native COM history");
            return motion.history[(motion.history_next+tail::turn_history_controls-back)%tail::turn_history_controls];};
        if(!result.fault)for(int control=previous+1;control<=w.motor.navigation.motion.controls;++control){
            const float speed=physics::norm(horizontal(historic(control)-historic(control-25)))/.5f;
            speed_sum+=speed;++samples;stall_controls=speed<.006f?stall_controls+1:0;r.stall=fmaxf(r.stall,stall_controls*.02f);}
        const auto measured=velocity(w);const float speed=physics::norm(measured);
        const float angle=std::atan2(physics::cross(initial_direction,measured).y,physics::dot(initial_direction,measured));
        total_angle+=std::remainder(angle-previous_angle,2*tail::pi);previous_angle=angle;r.angle=total_angle*180/tail::pi;
        inward_controls=physics::dot(measured,outward(w.pose.com_m))<-.1f*speed&&speed>=.006f?inward_controls+4:0;
        if(r.inward<0&&inward_controls>=52)r.inward=r.elapsed;
        // Inherited lifetime extrema describe the shared prefix; update with
        // the accepted branch intervals without clearing any physical state.
        r.gap=fminf(r.gap,result.fault?w.min_gap_m*1000:float(tail::submerged_gap_m(larger_pond,w.motor.navigation.motion.body,w.pose)*1000));
        r.gap=fminf(r.gap,w.motor.min_gap_m*1000);r.tilt=fmaxf(r.tilt,w.pose.tilt_deg);
        r.depth=fmaxf(r.depth,w.peak_depth_error_m*1000);
        trace<<r.elapsed<<','<<w.motor.navigation.motion.controls<<','<<stage<<','<<choice<<','<<w.pose.com_m.x<<','<<w.pose.com_m.y<<','<<w.pose.com_m.z<<','
            <<speed*1000<<','<<r.gap<<','<<r.tilt<<','<<r.depth<<','<<w.fault<<'\n';
    }
    r.attempted=attempted-begin_attempted;r.accepted=accepted-begin_accepted;r.fault=w.fault;r.mean=samples?1000*speed_sum/samples:0;
    r.safe=r.fault==0&&r.elapsed>=39.99f&&r.gap>0&&r.tilt<=10&&r.depth<=1;
    r.useful=r.safe&&r.mean>=6&&r.stall<=8.14f&&r.inward>=0;
    return r;
}
int execute(int argc,char** argv){
    if(argc!=9)throw std::runtime_error("Usage: counterfactual GAIT MOTOR STARTS INSTALLED POLICY8 POLICY128 ARCHIVE_ROOT NEW_OUT");
    const auto begin=Clock::now();deadline=begin+std::chrono::seconds(90);const auto root=fs::u8path(argv[8]);
    if(!fs::create_directory(root))throw std::runtime_error("Counterfactual output must be unused");fs::create_directory(root/"captures");fs::create_directory(root/"traces");
    Models m;m.gait=tail::read_navigation_weights(argv[1],tail::NavigationMode::Gait);m.motor=tail::steering::read_model(argv[2]);m.starts=tail::read_navigation_worlds(argv[3]);
    m.config.study=tail::steering::Study::Rate;m.config.material=m.motor.material;m.packed_motor=m.motor.weights;
    m.packed_motor.insert(m.packed_motor.end(),m.motor.frozen_skip.begin(),m.motor.frozen_skip.end());
    Policy installed(argv[4]),source8(argv[5]),source128(argv[6]);Policy* policies[]{&source8,&source128};
    std::ofstream proof(root/"prefix-parity.csv");proof.exceptions(std::ios::failbit|std::ios::badbit);
    proof<<std::setprecision(10)<<"fixture,case,source_policy,COM_samples,decision_actions,max_COM_error_m,attempted,accepted,fault\n";
    std::vector<Capture> captures;
    for(int fixture=0;fixture<3;++fixture)replay(fixtures[fixture],fixture,m,*policies[fixtures[fixture].policy],fs::u8path(argv[7]),root,captures,proof);
    std::ofstream observations(root/"fork-observations.csv"),logits(root/"fork-logits.csv"),decision_log(root/"decisions.csv"),table(root/"branches.csv");
    for(auto stream:{&observations,&logits,&decision_log,&table}){stream->exceptions(std::ios::failbit|std::ios::badbit);*stream<<std::setprecision(10);}
    observations<<"fixture,lead_s,state_hash,absolute_time_s,feature,value\n";
    logits<<"fixture,lead_s,policy,action,logit,selected\n";decision_log<<"fixture,lead_s,branch,elapsed_s,stage,action\n";
    table<<"fixture,case,lead_s,state_hash,branch,first_action,elapsed_s,fault,safe_40s,useful_escape,min_clearance_mm,tilt_deg,depth_error_mm,mean_COM_speed_mm_s,max_stall_s,inward_s,course_change_deg,attempted,accepted\n";
    int branches=0,useful=0;uint64_t branch_attempted=0,branch_accepted=0;
    for(const auto& capture:captures){auto& source=*policies[fixtures[capture.fixture].policy];float input[chooser::inputs],scores[chooser::actions];
        const int action_source=source.decide(capture.world,input,scores);
        for(int i=0;i<chooser::inputs;++i)observations<<capture.fixture<<','<<capture.lead<<','<<capture.hash<<','<<capture.world.motor.navigation.motion.controls*.02f<<','<<i<<','<<input[i]<<'\n';
        for(int i=0;i<chooser::actions;++i)logits<<capture.fixture<<','<<capture.lead<<",source,"<<i<<','<<scores[i]<<','<<(i==action_source)<<'\n';
        const int action_installed=installed.decide(capture.world,nullptr,scores);
        for(int i=0;i<chooser::actions;++i)logits<<capture.fixture<<','<<capture.lead<<",installed,"<<i<<','<<scores[i]<<','<<(i==action_installed)<<'\n';
        std::vector<Sequence> sequences{{0,-1,0,0,"source-policy"},{1,-1,0,0,"installed-policy"}};
        for(int speed=0;speed<3;++speed)for(int rate:{-8,-4,4,8})for(int seconds:{4,12}){
            std::ostringstream label;label<<"speed"<<speed<<"-rate"<<rate<<"-turn"<<seconds;sequences.push_back({2,speed,rate,seconds,label.str()});}
        for(const auto& sequence:sequences){const auto r=roll(capture,sequence,m,source,installed,root,decision_log);
            ++branches;useful+=r.useful;branch_attempted+=r.attempted;branch_accepted+=r.accepted;
            table<<r.fixture<<','<<fixtures[r.fixture].name<<','<<r.lead<<','<<r.hash<<','<<r.label<<','<<r.first_action<<','<<r.elapsed<<','<<r.fault<<','<<r.safe<<','<<r.useful<<','
                <<r.gap<<','<<r.tilt<<','<<r.depth<<','<<r.mean<<','<<r.stall<<',';if(r.inward>=0)table<<r.inward;
            table<<','<<r.angle<<','<<r.attempted<<','<<r.accepted<<'\n';table.flush();decision_log.flush();}
    }
    std::ofstream summary(root/"summary.json");summary<<std::setprecision(10)<<"{\"complete\":true,\"prefixes\":3,\"captures\":"<<captures.size()<<",\"branches\":"<<branches
        <<",\"useful_branches\":"<<useful<<",\"branch_attempted\":"<<branch_attempted<<",\"branch_accepted\":"<<branch_accepted
        <<",\"attempted_substeps\":"<<attempted<<",\"accepted_substeps\":"<<accepted<<",\"substep_cap\":"<<work_cap
        <<",\"optimizer_updates\":0,\"resets_within_branch\":0,\"exact_fork_hashes\":true,\"wall_s\":"<<std::chrono::duration<double>(Clock::now()-begin).count()<<"}\n";
    std::cout<<"Complete "<<branches<<" branches; "<<useful<<" useful finite escapes; "<<attempted<<" substeps\n";return 0;
}
}
int main(int argc,char** argv){try{return counterfactual::execute(argc,argv);}catch(const std::exception& e){std::cerr<<e.what()<<"; attempted="<<counterfactual::attempted<<", accepted="<<counterfactual::accepted<<'\n';return 2;}}

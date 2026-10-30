#include "motor_benchmark.h"
#include "rl/boulder_source.h"
#include "rl/command_motor_policy.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>

// A bounded feasibility preflight, not a trainer or a deployed fallback.
// The only two evaluated controllers are passive zero and one fixed joint-PD
// wave/steering/braking reference. A failed reference leaves the pilot blocked;
// it does not prove that the unchanged articulated physics is uncontrollable.
namespace {
using namespace cudarl;
namespace fs=std::filesystem;
namespace bench=motor_benchmark;
void need(bool okay,const std::string& message) {if(!okay) throw std::runtime_error(message);}
uint64_t hash_bytes(const std::vector<uint8_t>& bytes) {
    uint64_t h=14695981039346656037ull;for(uint8_t x:bytes) {h^=x;h*=1099511628211ull;}return h;
}
std::string read_text(const fs::path& path) {
    std::ifstream input(path,std::ios::binary);need(static_cast<bool>(input),"Cannot read "+path.string());
    return {std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()};
}
std::string field(const std::string& text,const std::string& key) {
    std::smatch match;const std::regex pattern("\\\""+key+"\\\"\\s*:\\s*(true|false|[-+0-9.eE]+)");
    need(std::regex_search(text,match,pattern),"Numerical evidence missing "+key);return match[1].str();
}
struct SavedRun {std::vector<float> actor; shelter_motor::ActuatorScale scales{};int update=0;};
SavedRun selected_run(const std::vector<uint8_t>& bytes,const rl::BoulderSource& source) {
    checkpoint::Reader outer{bytes};SavedRun out;
    need(outer.value<uint32_t>()==0x534D4631u&&outer.value<uint32_t>()==1,"Expected protected full shelter motor run");
    need(outer.value<uint64_t>()==source.fingerprint,"Selected run parent fingerprint differs");
    need(outer.array<uint8_t>(128000000)==source.parent,"Selected run does not contain this exact original boulder source");
    const auto training=outer.array<uint8_t>(256000000);outer.finish();checkpoint::Reader in{training};
    need(in.value<uint32_t>()==0x534D5452u&&in.value<uint32_t>()==5&&in.value<uint32_t>()==sizeof(shelter_motor::State),"Expected protocol5 motor snapshot");
    const int worlds=in.value<int>();(void)in.value<uint32_t>();out.update=in.value<int>();out.scales=in.value<shelter_motor::ActuatorScale>();
    need(worlds>=16&&worlds<=2048&&out.update>=0,"Invalid selected motor dimensions");
    for(float x:out.scales.value) need(std::isfinite(x)&&x>=1&&x<=1048576,"Invalid selected actuator scales");
    out.actor=in.array<float>(static_cast<size_t>(rl::swimmer_specification(4,59).num_params()));
    need(out.actor.size()==static_cast<size_t>(rl::swimmer_specification(4,59).num_params()),"Selected motor parameter dimensions differ");
    for(float x:out.actor) need(std::isfinite(x),"Nonfinite selected motor actor");return out;
}
bool numerical_pass(const std::string& text,const std::vector<uint8_t>& selected,const SavedRun& run) {
    need(field(text,"passed")=="true","Independent numerical preflight did not pass");
    need(field(text,"independent_CPU_double_forward_backprop_transform_MSE_Adam")=="true","Independent CPU reference is not established");
    need(field(text,"source_inputs_unchanged")=="true","Numerical preflight did not preserve its inputs");
    need(field(text,"tolerances_were_not_changed")=="true","Numerical tolerances were changed");
    need(std::stoul(field(text,"protocol"))==5&&std::stoi(field(text,"source_update"))==run.update,"Numerical evidence is for another motor protocol/update");
    need(std::stoull(field(text,"source_payload_fnv1a64"))==hash_bytes(selected),"Numerical evidence is not for these exact selected source bytes");
    need(std::stoi(field(text,"food_rows"))==8&&std::stoi(field(text,"cover_rows"))==8&&std::stoi(field(text,"independent_CPU_selected_central_differences"))==6,"Numerical evidence has the wrong registered rows/checks");
    need(std::stoi(field(text,"physical_rollouts"))==0&&std::stoi(field(text,"PPO_updates"))==0,"Numerical evidence scope differs");
    const char* kinds[]={"physical_forward","standardized_loss","full_gradient","selected_cpu_finite_difference","selected_gpu_finite_difference","fresh_adam_change"};
    for(const char* kind:kinds) {
        const std::regex pattern(std::string("\\{[^{}]*\\\"kind\\\"\\s*:\\s*\\\"")+kind+"\\\"[^{}]*\\}");std::smatch match;
        need(std::regex_search(text,match,pattern),std::string("Missing numerical comparison ")+kind);
        const auto row=match[0].str();need(std::stoi(field(row,"failed"))==0&&std::stod(field(row,"maximum_error_over_tolerance"))<=1.,std::string("Failed numerical comparison ")+kind);
    }
    return true;
}
void trace_header(std::ostream& out) {
    out<<"controller,case,tick,time_s,head_x_m,head_y_m,head_z_m,com_x_m,com_y_m,com_z_m,tail_x_m,tail_y_m,tail_z_m";
    for(int j=0;j<4;++j) out<<",driving_raw"<<j;for(int j=0;j<4;++j) out<<",applied"<<j;
    for(int j=0;j<10;++j) out<<",physical_velocity"<<j;for(int j=0;j<4;++j) out<<",joint_angle"<<j;
    out<<",goal_error_m,heading_error_rad,head_m_s,com_m_s,tail_m_s,max_point_m_s,rms_point_m_s,body_angular_rad_s,max_joint_rad_s,whole_body_clearance_m,finite,work_J,travel_m,reward,discounted_return\n";
}
void trace_row(std::ostream& out,const char* controller,int id,const bench::State& state,const bench::Sample& s,const bench::Four& a,float reward,double total) {
    out<<controller<<','<<id<<','<<state.steps<<','<<state.steps*deep::dt;
    for(auto v:{s.head_position,s.com_position,s.tail_position}) out<<','<<v.x<<','<<v.y<<','<<v.z;
    for(float x:a) out<<','<<x;for(float x:a) out<<','<<deep::clamp(x,-1,1);
    for(float x:state.body.velocity) out<<','<<x;for(float x:state.body.angle) out<<','<<x;
    out<<','<<s.goal_error<<','<<s.heading_error<<','<<s.head_speed<<','<<s.com_speed<<','<<s.tail_speed<<','<<s.point_speed<<','<<s.point_rms_speed
       <<','<<s.angular_speed<<','<<s.joint_rate<<','<<s.clearance<<','<<s.finite<<','<<state.body.work<<','<<state.body.travel<<','<<reward<<','<<total<<'\n';
}
struct PointFrame {int controller=0,fixture=0,tick=0;bench::SurfacePoint points[bench::surface_count]{};};
struct BodyFrame {int controller=0,fixture=0;bench::State state{};bench::Four driving_action{};};
struct Totals {int evaluated=0,qualified=0,safe=0,motion=0,settle=0,hold=0,passed=0,physical_ticks=0;bool finite=true;};
}
int main(int argc,char** argv) {
    try {
        if(argc!=5) {std::cerr<<"Usage: motor_reference original-boulder-full-save.bin selected-v5-full-run.bin numerical-summary.json NEW-output-dir\n";return 2;}
        const fs::path original_path=fs::absolute(argv[1]),selected_path=fs::absolute(argv[2]),numeric_path=fs::absolute(argv[3]),output=fs::absolute(argv[4]);
        need(!fs::exists(output),"Reference output must be a new directory; prior evidence is immutable");
        const auto original_bytes=checkpoint::read(original_path.string()),selected_bytes=checkpoint::read(selected_path.string());
        const auto source=rl::read_boulder_source(original_bytes);const auto saved=selected_run(selected_bytes,source);const auto numerical=read_text(numeric_path);
        const bool numeric_ok=numerical_pass(numerical,selected_bytes,saved);
        fs::create_directories(output);{std::ofstream copy(output/"numerical-evidence.json",std::ios::binary);copy<<numerical;need(static_cast<bool>(copy),"Cannot save numerical evidence copy");}
        bench::FrozenMotor original(source.motor,bench::TargetMode::Original55);bench::ConventionalReference conventional;
        std::vector<bench::Fixture> fixtures;std::vector<BodyFrame> warmup_records;
        for(int id=0;id<bench::suite_count(bench::Suite::Reference);++id) {
            auto fixture=bench::make_fixture(bench::reference_seed,bench::Suite::Reference,id);std::vector<bench::WarmupFrame> warmup;
            bench::prepare_braking(fixture,original,&warmup);
            for(const auto& frame:warmup) warmup_records.push_back({-1,id,frame.state,frame.action});fixtures.push_back(fixture);
        }
        checkpoint::Writer fixture_payload;fixture_payload.value(uint32_t{0x434D4649u});fixture_payload.value(uint32_t{1});fixture_payload.value(rl::command_motor_fingerprint());fixture_payload.value(source.fingerprint);fixture_payload.array(fixtures);
        checkpoint::write_atomic((output/"reference-fixtures.bin").string(),fixture_payload.data);
        checkpoint::Writer warmup_payload;warmup_payload.value(uint32_t{0x434D5755u});warmup_payload.value(uint32_t{1});warmup_payload.array(warmup_records);
        checkpoint::write_atomic((output/"actual-original55-warmup.bin").string(),warmup_payload.data);
        std::ofstream trace(output/"physical-trace.csv"),cases(output/"cases.csv"),points(output/"surface-points.bin",std::ios::binary);
        need(static_cast<bool>(trace)&&static_cast<bool>(cases)&&static_cast<bool>(points),"Cannot create physical reference traces");trace<<std::setprecision(9);cases<<std::setprecision(9);trace_header(trace);
        const uint32_t point_header[]={0x434D5350u,1,static_cast<uint32_t>(sizeof(PointFrame)),bench::surface_count};points.write(reinterpret_cast<const char*>(point_header),sizeof(point_header));
        cases<<"controller,case,kind,qualified,safety,motion,settle,hold,passed,warmup_ticks,evaluation_ticks,requested_entry_m_s,actual_entry_m_s,half_s,reach_s,heading_s,stop_s,stop_distance_m,completed_hold_start_s,first_brief_settle_s,longest_hold_s,minimum_clearance_m,path_m,overshoot_m,cross_track_m,first_safety_failure_tick,actual_evaluation_ticks,path_efficiency,goal_error_slope_reversals,heading_error_slope_reversals,maximum_translation_m\n";
        Totals totals[2];std::vector<BodyFrame> body_records;size_t point_frames=0;
        bool abort_physics=false;
        for(int controller=0;controller<2&&!abort_physics;++controller) for(const auto& fixture:fixtures) {
            const char* label=controller==0?"passive_zero":"conventional_joint_PD";auto state=fixture.initial;bench::Progress progress;int used=0;double total=0,discount=1;
            bench::Four action=controller==0?bench::Four{}:conventional.action(state);
            PointFrame frame;frame.controller=controller;frame.fixture=fixture.id;frame.tick=0;
            auto sample=bench::measure(state,frame.points);bench::assess(progress,state,sample,fixture);trace_row(trace,label,fixture.id,state,sample,action,0,total);
            points.write(reinterpret_cast<const char*>(&frame),sizeof(frame));++point_frames;body_records.push_back({controller,fixture.id,state,action});
            // Tick0 action is a preview; every subsequent row pairs the action
            // produced from the preceding state with its resulting physical state.
            if(fixture.qualified) for(int tick=0;tick<fixture.evaluation_ticks&&progress.finite;++tick) {
                action=controller==0?bench::Four{}:conventional.action(state);const auto result=bench::advance(state,action.data(),fixture);++used;
                total+=discount*result.reward;discount*=.99;frame.tick=state.steps;sample=bench::measure(state,frame.points);
                if(result.invalid) sample.finite=false;bench::assess(progress,state,sample,fixture);trace_row(trace,label,fixture.id,state,sample,action,result.reward,total);
                points.write(reinterpret_cast<const char*>(&frame),sizeof(frame));++point_frames;body_records.push_back({controller,fixture.id,state,action});
            }
            const auto result=bench::finish(progress,fixture);auto& t=totals[controller];++t.evaluated;t.qualified+=result.qualified;t.safe+=result.safety;t.motion+=result.motion;t.settle+=result.settle;t.hold+=result.hold;t.passed+=result.passed;
            t.finite=t.finite&&progress.finite;t.physical_ticks+=used+fixture.warmup_ticks;
            need(used+fixture.warmup_ticks<=bench::total_budget_ticks,"Reference exceeded registered per-case physical budget");
            cases<<label<<','<<fixture.id<<','<<bench::name(fixture.kind)<<','<<result.qualified<<','<<result.safety<<','<<result.motion<<','<<result.settle<<','<<result.hold<<','<<result.passed
                <<','<<fixture.warmup_ticks<<','<<fixture.evaluation_ticks<<','<<fixture.requested_speed<<','<<fixture.actual_entry_speed<<','<<result.half_time<<','<<result.reach_time<<','<<result.heading_time
                <<','<<result.stop_time<<','<<result.stop_distance<<','<<result.settle_time<<','<<progress.settle_time<<','<<result.hold_seconds<<','<<progress.minimum_clearance<<','<<progress.path
                <<','<<progress.max_overshoot<<','<<progress.max_cross_track<<','<<progress.first_failure_step<<','<<used<<','<<bench::path_efficiency(progress)<<','<<progress.reversals<<','<<progress.heading_reversals<<','<<progress.max_translation<<'\n';
            std::cout<<label<<" case="<<fixture.id<<" kind="<<bench::name(fixture.kind)<<" qualified="<<result.qualified<<" passed="<<result.passed<<" hold_s="<<result.hold_seconds<<'\n';
            if(!progress.finite) {abort_physics=true;break;}
        }
        trace.flush();cases.flush();points.flush();need(static_cast<bool>(trace)&&static_cast<bool>(cases)&&static_cast<bool>(points),"Failed writing physical evidence");
        checkpoint::Writer body_payload;body_payload.value(uint32_t{0x434D4254u});body_payload.value(uint32_t{1});body_payload.array(body_records);checkpoint::write_atomic((output/"complete-physical-trace.bin").string(),body_payload.data);
        need(checkpoint::read(original_path.string())==original_bytes&&checkpoint::read(selected_path.string())==selected_bytes&&read_text(numeric_path)==numerical,"Protected preflight inputs changed");
        const bool physical_ok=totals[1].qualified==16&&totals[1].passed==16&&totals[0].finite&&totals[1].finite;
        rl::CommandMotorGate gate;gate.source_fingerprint=source.fingerprint;gate.benchmark_fixture_fingerprint=hash_bytes(fixture_payload.data);gate.numerical_passed=numeric_ok;gate.physical_reference_passed=physical_ok;
        // The helper's versioned gate also binds the fixed acceptance rules.
        gate.benchmark_rules_fingerprint=bench::rules_fingerprint();
        checkpoint::write_atomic((output/"preflight-gate.bin").string(),gate.serialize());
        std::ofstream summary(output/"summary.json");need(static_cast<bool>(summary),"Cannot write reference summary");summary<<std::setprecision(17);
        summary<<"{\n\"protocol\":1,\"numerical_passed\":"<<(numeric_ok?"true":"false")<<",\"physical_reference_passed\":"<<(physical_ok?"true":"false")
            <<",\"source_fingerprint\":\""<<source.fingerprint<<"\",\"selected_run_payload_fnv1a64\":\""<<hash_bytes(selected_bytes)<<"\",\"source_selected_update\":"<<saved.update
            <<",\"rules_fingerprint\":\""<<bench::rules_fingerprint()<<"\",\"fixture_fingerprint\":\""<<gate.benchmark_fixture_fingerprint<<"\",\"reference_seed\":"<<bench::reference_seed
            <<",\"controllers\":[";
        for(int i=0;i<2;++i) {if(i) summary<<',';const auto& t=totals[i];summary<<"{\"name\":\""<<(i==0?"passive_zero":"conventional_joint_PD")<<"\",\"registered_cases\":16,\"evaluated_cases\":"<<t.evaluated<<",\"qualified\":"<<t.qualified<<",\"safe\":"<<t.safe<<",\"motion\":"<<t.motion<<",\"settled\":"<<t.settle<<",\"held\":"<<t.hold<<",\"passed\":"<<t.passed<<",\"budgeted_control_ticks\":"<<t.physical_ticks<<'}';}
        summary<<"],\"reference_trials\":"<<totals[0].evaluated+totals[1].evaluated<<",\"maximum_registered_reference_trials\":32,\"aborted_nonfinite\":"<<(abort_physics?"true":"false")<<",\"maximum_registered_control_ticks\":19200,\"whole_body_points_per_frame\":216,\"point_frames\":"<<point_frames
            <<",\"point_binary_record_bytes\":"<<sizeof(PointFrame)<<",\"complete_body_record_bytes\":"<<sizeof(BodyFrame)
            <<",\"source_inputs_unchanged\":true,\"physical_parameters_changed\":false,\"learning_updates\":0,\"automatic_adoption\":false,\"failed_reference_is_not_uncontrollability_proof\":true,\"terminal_heading_scope\":\"level yaw, 3D position and pitched approaches\",\"action_alignment\":\"tick0 preview; tick>=1 action generated from preceding state and resulting state\",\"surface_binary_schema\":\"uint32 magic/version/record_bytes/points; records int32 controller,fixture,tick then216*(position_xyz_m,velocity_xyz_m_s) float32; native compiler padding included in record_bytes\"}\n";
        summary.flush();need(static_cast<bool>(summary),"Failed writing reference summary");
        std::cout<<"MOTOR_REFERENCE_FINISHED conventional="<<totals[1].passed<<"/16 physical_gate="<<physical_ok<<" source_preserved=1 no_training=1\n";return physical_ok?0:3;
    } catch(const std::exception& e) {std::cerr<<"Motor reference failed: "<<e.what()<<'\n';return 1;}
}

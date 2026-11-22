from pathlib import Path
import json,hashlib,shutil,subprocess
r=Path(r'C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation');local=Path(__file__).parent
pres=r/'build/review-preservation/2026-11-04-before-rest-hold'
assert not pres.exists(),'Preparation identity exists';pres.mkdir(parents=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
files=[p for folder in ['assets','build/ppo','build/diagnostics'] for p in (r/folder).rglob('*') if p.is_file()]
(pres/'prior-evidence-hashes.json').write_text(json.dumps([{'path':str(p.relative_to(r)),'sha256':sha(p),'bytes':p.stat().st_size} for p in files],indent=2))
unrelated=json.loads((r/'build/review-preservation/2026-11-04-before-frozen-brake/unrelated-hashes.json').read_text())
for h in unrelated:assert sha(r/h['path'])==h['sha256']
(pres/'unrelated-hashes.json').write_text(json.dumps(unrelated,indent=2))
changed=['CMakeLists.txt','src/nn/motor/straight_task.h','src/rl/motor_rollout.cuh','tools/ppo_motor.cu','tools/append_hold8.cu','tools/hold_scale_study.cu']
for name in changed:
 dst=pres/'source-before'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(r/name,dst)
(pres/'preexisting-diff.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=r))
def edit(name,old,new):
 p=r/name;s=p.read_text();assert s.count(old)==1,(name,old[:60],s.count(old));p.write_text(s.replace(old,new),newline='\n')
edit('src/nn/motor/straight_task.h','    bool balanced_hold = false;','    bool balanced_hold = false;\n    bool rest_hold = false; // Explicit approved hold-from-rest research mode.')
edit('src/nn/motor/straight_task.h','return hold_episode(environment) && completed_controls >= 640 ? holding_reward_gain : 1.f;','return hold_episode(environment) && (rest_hold || completed_controls >= 640) ? holding_reward_gain : 1.f;')
edit('src/nn/motor/straight_task.h','        if (!fixed_depth_offsets)\n            return 0;','        if (!fixed_depth_offsets || (rest_hold && hold_episode(environment)))\n            return 0;')
edit('src/nn/motor/straight_task.h','(controls == 160 || controls == 640 || controls == 1280) && (!stopping || controls == 1280) &&','(controls == 160 || controls == 640 || controls == 1280 || (rest_hold && controls == 320)) && (!stopping || controls == 1280) &&')
edit('src/nn/motor/straight_task.h','(!balanced_hold || (holding && appended_hold && controls == 1280));','(!balanced_hold || (holding && appended_hold && (controls == 1280 || (rest_hold && controls == 320)))) &&\n               (!rest_hold || (holding && balanced_hold && appended_hold && controls == 320 && !stopping && !startup_speed));')
edit('src/nn/motor/straight_task.h','return controls >= 640 ? long_swim_arena() : contact::Arena{};','return controls >= 640 || rest_hold ? long_swim_arena() : contact::Arena{};')
edit('src/nn/motor/straight_task.h','    CUDARL_HD ForwardConfig task_config(int completed_controls = 0, int environment = 0) const {','''    CUDARL_HD ForwardConfig task_config(int completed_controls = 0, int environment = 0) const {
        if (rest_hold && hold_episode(environment)) {
            auto c = config_for_speed(.008f);c.maximum_controls = controls;
            c.observation_schema = observation_schema;c.target_speed_m_s = 0;
            c.stationary_request = true;c.position_hold = true;return c;
        }''')
edit('src/rl/motor_rollout.cuh','        atomicExch(fault, 1);\n}\n__global__ void observe','''        atomicExch(fault, 1);
    if (!*fault && episode.rest_hold && episode.hold_episode(i) &&
        !m::update_velocity_request(worlds[i], episode.task_config(0,i), tasks[i]))
        atomicExch(fault, 1);
}
__global__ void observe''')
edit('tools/append_hold8.cu','std::vector<uint8_t> append_protocol() {','''#ifdef CUDARL_REST_HOLD_STUDY
std::vector<uint8_t> rest_protocol();
#endif
std::vector<uint8_t> append_protocol() {
#ifdef CUDARL_REST_HOLD_STUDY
    return rest_protocol();
#else''')
edit('tools/append_hold8.cu','    out.array(std::vector<uint8_t>(reward.begin(),reward.end()));return out.data;','    out.array(std::vector<uint8_t>(reward.begin(),reward.end()));return out.data;\n#endif')
edit('tools/hold_scale_study.cu','int main(int argc,char** argv){try{','#ifndef CUDARL_REST_HOLD_STUDY\nint main(int argc,char** argv){try{')
p=r/'tools/hold_scale_study.cu';p.write_text(p.read_text()+'\n#endif\n',newline='\n')
edit('tools/ppo_motor.cu','// Shared by production training and the save/reload verification: no shadow updater.','''#ifdef CUDARL_REST_HOLD_STUDY
uint32_t rest_action_seed();
uint32_t rest_action_counter(int update,int t);
void rest_verify_minibatch(int update,int epoch,int first,Rollout& data,Batches& batch,
                          const rl::Permutation& permutation,cudaStream_t stream);
#endif
// Shared by production training and the save/reload verification: no shadow updater.''')
edit('tools/ppo_motor.cu','                               actions, correlation, seed + 2,\n                               static_cast<uint32_t>(static_cast<uint64_t>(update) * horizon + t),','''                               actions, correlation,
#ifdef CUDARL_REST_HOLD_STUDY
                               rest_action_seed(),rest_action_counter(update,t),
#else
                               seed + 2,static_cast<uint32_t>(static_cast<uint64_t>(update) * horizon + t),
#endif''')
edit('tools/ppo_motor.cu','            if (horizon >= 640 && update == starting_update() && epoch == 0 && first == 0 &&','''#ifdef CUDARL_REST_HOLD_STUDY
            rest_verify_minibatch(update,epoch,first,data,batch,permutation,stream);
#endif
            if (horizon >= 640 && update == starting_update() && epoch == 0 && first == 0 &&''')
edit('CMakeLists.txt','    add_executable(ppo_hold_scale tools/hold_scale_study.cu)','''    add_executable(ppo_rest_hold tools/rest_hold_study.cu)
    target_compile_definitions(ppo_rest_hold PRIVATE CUDARL_APPEND_HOLD41=1)
    target_link_libraries(ppo_rest_hold PRIVATE creature_gpu frozen_controller)
    project_warnings(ppo_rest_hold)
    add_executable(ppo_hold_scale tools/hold_scale_study.cu)''')
shutil.copy2(local/'rest_hold_study.cu',r/'tools/rest_hold_study.cu')
print(json.dumps({'preserved_evidence_files':len(files),'unrelated_files_verified':len(unrelated),'implemented_opt_in_rest_mode':True,'physical_controls':0,'policy_optimizer_steps':0}))

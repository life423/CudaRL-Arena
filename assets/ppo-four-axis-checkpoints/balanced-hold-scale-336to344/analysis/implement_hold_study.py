from pathlib import Path
import json, hashlib, shutil, datetime
r=Path(r'C:\Users\aiand\Documents\programming\CudaRL-Arena\.worktrees\minimal-foundation')
pres=r/'build/review-preservation/2026-11-04-before-balanced-scale'
assert not pres.exists(), 'Preservation identity exists'
pres.mkdir(parents=True)
files=['CMakeLists.txt','src/nn/motor/straight_task.h','src/rl/motor_rollout.cuh','tools/ppo_motor.cu','tools/append_hold8.cu','README.md','live/project.godot','live/main.gd.uid','live/addons/frozen_swimmer/frozen_swimmer.gdextension.uid']
files += [str(x.relative_to(r)) for x in (r/'assets/ppo-smooth-exploration-v1').rglob('*') if x.is_file()]
manifest=[]
for name in files:
 p=r/name
 if not p.exists(): continue
 dst=pres/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
 manifest.append({'path':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest().upper()})
baseline=[]
for p in (r/'assets/ppo-four-axis-checkpoints').rglob('*.bin'):
 baseline.append({'path':str(p.relative_to(r)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest().upper()})
(pres/'manifest.json').write_text(json.dumps(manifest,indent=2))
(pres/'prior-checkpoint-hashes.json').write_text(json.dumps(baseline,indent=2))
def edit(name,old,new):
 p=r/name;s=p.read_text();assert s.count(old)==1,(name,old[:80],s.count(old));p.write_text(s.replace(old,new),newline='\n')
edit('src/nn/motor/straight_task.h','    bool appended_hold = false;','''    bool appended_hold = false;
    // Explicit research protocol; historical training defaults are unchanged.
    bool balanced_hold = false;
    float holding_reward_gain = 1.f;
    CUDARL_HD bool hold_episode(int environment) const {
        return holding && (balanced_hold ? (environment / 2) % 2 : environment % 2);
    }
    CUDARL_HD float reward_gain(int completed_controls, int environment) const {
        return hold_episode(environment) && completed_controls >= 640 ? holding_reward_gain : 1.f;
    }''')
edit('src/nn/motor/straight_task.h','(controls == 160 || controls == 640 || controls == 1280) && (!stopping || controls == 1280);','''(controls == 160 || controls == 640 || controls == 1280) && (!stopping || controls == 1280) &&
               (holding_reward_gain == 1.f || holding_reward_gain == 1.f / 1024) &&
               (!balanced_hold || (holding && appended_hold && controls == 1280));''')
edit('src/nn/motor/straight_task.h','if (stopping || (holding && environment % 2))','if (stopping || hold_episode(environment))')
edit('src/rl/motor_rollout.cuh','    rewards[i] = interval.reward;','''    const float gain = episode.reward_gain(tasks[i].controls - 1, i);
    rewards[i] = interval.reward * gain;''')
edit('src/rl/motor_rollout.cuh','    const float cost[5] = {interval.costs.speed, interval.costs.lane, interval.costs.depth,\n                           interval.costs.posture, interval.costs.effort};','''    const float cost[5] = {gain * interval.costs.speed, gain * interval.costs.lane, gain * interval.costs.depth,
                           gain * interval.costs.posture, gain * interval.costs.effort};''')
edit('src/rl/motor_rollout.cuh','    const float detail[4] = {interval.costs.horizontal, interval.costs.recovery, interval.costs.roll,\n                             interval.costs.pitch};','''    const float detail[4] = {gain * interval.costs.horizontal, gain * interval.costs.recovery, gain * interval.costs.roll,
                             gain * interval.costs.pitch};''')
edit('tools/append_hold8.cu','    cp::Writer out;out.value(uint32_t{16});out.value(uint32_t{41});out.value(uint32_t{9});','''#ifdef CUDARL_BALANCED_HOLD_STUDY
    cp::Writer out;out.value(uint32_t{17});out.value(uint32_t{41});out.value(uint32_t{9});
    out.value(training_episode.balanced_hold);out.value(training_episode.holding_reward_gain);
    out.value(uint32_t{349184});out.value(uint32_t{698368});out.value(uint32_t{0});
#else
    cp::Writer out;out.value(uint32_t{16});out.value(uint32_t{41});out.value(uint32_t{9});
#endif''')
edit('tools/append_hold8.cu','    for(int v:{336,8,32,1280,256,2,all_control_cap,all_control_cap*10,maximum_replays})out.value(v);','''#ifdef CUDARL_BALANCED_HOLD_STUDY
    for(int v:{336,8,32,1280,256,2,349184,3491840,0})out.value(v);
#else
    for(int v:{336,8,32,1280,256,2,all_control_cap,all_control_cap*10,maximum_replays})out.value(v);
#endif''')
edit('tools/append_hold8.cu','int main(int argc,char** argv){try{','#ifndef CUDARL_BALANCED_HOLD_STUDY\nint main(int argc,char** argv){try{')
p=r/'tools/append_hold8.cu';p.write_text(p.read_text()+'\n#endif\n',newline='\n')
edit('tools/ppo_motor.cu','    if (actor_lr_override && update == starting_update()) {','    if (actor_lr_override && (update == starting_update() || training_episode.balanced_hold)) {')
edit('tools/ppo_motor.cu','        std::ofstream proof(diagnostics / "old-policy-rescore.json");','''        std::ofstream proof(diagnostics / (training_episode.balanced_hold ?
            "old-policy-rescore-" + std::to_string(update) + ".json" : "old-policy-rescore.json"));''')
# Include the exact56s endpoint in the existing fixed-anchor acceptance check.
edit('tools/ppo_motor.cu','            if (control > 1120) {\n                anchor_final_error', '            if (control >= 1120) {\n                anchor_final_error')
edit('tools/ppo_motor.cu','                hold_speed += speed;\n                hold_vy += motion.velocity_m_s.y;','''                if (control > 1120) { hold_speed += speed; hold_vy += motion.velocity_m_s.y; }''')
edit('CMakeLists.txt','    add_executable(ppo_append_hold tools/append_hold8.cu)','''    add_executable(ppo_hold_scale tools/hold_scale_study.cu)
    target_compile_definitions(ppo_hold_scale PRIVATE CUDARL_APPEND_HOLD41=1)
    target_link_libraries(ppo_hold_scale PRIVATE creature_gpu frozen_controller)
    project_warnings(ppo_hold_scale)
    add_executable(ppo_append_hold tools/append_hold8.cu)''')
shutil.copy2(Path(__file__).with_name('hold_scale_study.cu'),r/'tools/hold_scale_study.cu')
print(json.dumps({'preserved_files':len(manifest),'prior_checkpoint_files':len(baseline),'source_implemented':True}))

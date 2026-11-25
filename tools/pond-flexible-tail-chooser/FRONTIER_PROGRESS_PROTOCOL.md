# Unvisited-water progress — November 25, 2026

The user authorizes continued food-free GPU PPO exploration, protected motor,
and Godot adoption only after safer/improved coverage. No mouse or keyboard
automation is authorized. The current appended16/motor88 package stays protected.

Reuse the recorded appended16 baseline (12+4 and four1800s starts). Its default
trace has a238.72s maximum interval between new scored cells despite swimming.
One learning change: add discounted state-potential shaping at each0.8s chooser
transition. Phi=2 exp(-d/0.25m), where d is the horizontal distance to the nearest
unvisited cell center in the existing338-cell80mm geometric sensing map. If all
mapped cells are visited,Phi=2. Unsupported depth returns0. Reward addition is
0.994 Phi(after)-Phi(before), using the actual PPO discount; true failures use
Phi(after)=0 before any training reset. Complete body and visit state determine
both values. A stationary state and a discounted closed cycle cannot farm this
term. It is a scalar reward, not a selected destination, action override, route,
scripted turn or force. Existing novelty/revisit/transport/stall/safety scoring,
all35 inputs, actions, PPO parameters, physics and motor remain unchanged.

Explicitly import full appended-v3 checkpoint16 with its recorded task identity
and35-input format, including actor, critic, both Adam/fresh-column ages, worlds,
servo/history, visitation, episodes and RNG. Require byte-identical parent
roundtrip before rebasing only study counters. Require zero-physics reward
algebra tests (toward/away/stationary/cycle/terminal/complete/unsupported depth),
GPU/CPU potential agreement<=1e-6 across1024 saved states and the existing
observer checks. One three-actual-update exact full-resume diagnostic is allowed.
Do not repeat failed studies automatically. Preserve every attempted failure.

One new native CUDA session:1024 worlds,128 new PPO updates,8 decisions/world,
1024-row minibatches,four passes,1048576 main transitions maximum. Same1200s
native/1800s external wall limits and1342177280 main substep cap. Diagnostic
limit remains three actual updates and31457280 substeps. Numerical/native faults
stop affected work; bank guard failures remain scored training terminals.
Save complete checkpoints, curves, execution hashes and source snapshots.

Unchanged12x120s+conditional4x300s assessments every8 (17 max,2112000 substeps
and30s wall each). Up to four qualified candidates receive unchanged4x1800s
continuous assessments,5760000 substeps and60/65s per assessment. No budget
extension, extra candidate or criteria change after seeing results. Baseline
reuse spends0 new physics. Count diagnostic and all evaluation work separately.

Adoption requires12+4,4/4 long physical passes,zero guards/faults/resets,mean
individual coverage>=45.91304348%,worst>=42.60869565%,default>=43.76811594%,and
max slow interval<=8.14s. Rank qualified candidates by worst then mean coverage.
Report each start and the unchanged80% milestone separately. Union coverage
cannot substitute for one swimmer. No claim of infinite/whole-pond mastery.

If qualified, preserve the old actor file and use a separately named new actor.
No physics or observer runtime change is needed. Without any desktop input,
qualify the selected model in an isolated actual project through one300s native
replay (240000 substeps max), rendered UI and package hashes before staging.
Do not overwrite a loaded DLL or take the mouse; no native DLL update is needed.
If no candidate qualifies, leave the entire working Godot package unchanged.

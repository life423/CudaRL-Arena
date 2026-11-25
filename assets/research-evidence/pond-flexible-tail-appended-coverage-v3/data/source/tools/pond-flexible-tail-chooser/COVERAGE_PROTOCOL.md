# Individual pond exploration — November 25, 2026

The user authorizes improving whole-pond exploration. Preserve the deployed
motor88 and chooser88, all previous checkpoints and the current Godot UI.
No food, scripted turns, physical changes or automatic deployed resets.

First measure the deployed pair on the four existing endurance starts for
1,800 simulated seconds each, uninterrupted. The new explicit evaluator mode
has a 60-second internal / 65-second external limit and at most 5,760,000
physical substeps. Record every case, failures, individual visited cells and
coverage maps, speed, clearance, posture, stalls and action choices. Existing
12×120s + 4×300s qualification remains unchanged. Its union is not individual
coverage. All maps use the same 50mm fixed-posture reachable-cell estimate;
report cells outside that estimate separately. 80% per case is the new first
coverage milestone, not proof of complete coverage or infinite survival.

After measurement, make one evidence-based chooser learning correction.
The baseline completed all four 1,800s runs safely, with individual coverage
21.45–29.57%. Three starts select positive-rate turns on every decision.
The correction changes only the positive safe-transport reward: 0.04/s is
multiplied by 0.05+0.95×existing cell novelty. It remains 0.04/s in unvisited
water and 0.002/s in recently credited water. First visits remain +0.6,
revisits up to +0.15 with the existing 90–300s aging, stalls −0.10/s,
approaching certificate loss up to −0.10/s and scored guard failures −5.
Observations, actions, dynamics, task mix and all other PPO settings stay fixed.
The explicit parent is selected session-88 from reliable-chooser-v1-corrected.
Verify a bit-identical full-state import under its original bound task ID,
then rebase study update/work/sampling counters while retaining actor, critic,
Adam ages/moments, physical state and visitation memory. This is a declared
task continuation, not a repeat or extension of the old closed session.
Use a separate native CUDA PPO session with 1,024 worlds and at most 128 new
updates, 8 decisions/world/update, 1,024-row minibatches and four passes.
Save full actor, critic, optimizer, worlds and RNG state. Reuse completed
unchanged physics/motor validation; verify any explicit checkpoint migration
and full resume before the new session. Up to one three-update resume check.
Stop on numerical/native faults or work/runtime budget exhaustion. Bank guard
stops are scored training failures, never hidden deployed recoveries.

The session has 1,200s native / 1,800s external wall limits and the existing
1,342,177,280 training substep ceiling. Fixed assessments every eight updates
retain all original criteria and caps (17×2,112,000 maximum substeps).
At most four post-training continuous assessments (four starts each) have the
same 5,760,000-substep, 60/65-second caps. Count baseline, diagnostic, training
and all evaluations separately. No automatic repeated failed study.

Adoption requires original 12+4 safety/movement passes and improved individual
coverage without stops/resets. The 80% milestone must be reported separately;
do not publish a weaker safety policy to raise union coverage. Keep the
existing Godot pair unless a candidate qualifies and its native replay matches.

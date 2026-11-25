# Phase-aware motor: mixed gentle/30-degree practice

November24,2026. The user authorized the recommended next PPO session after
the phase-aware v4 results. Goal: retain the learned30-degree motion while
recovering complete gentle-turn/rhythm keys. Native CUDA PPO, one existing
motor and no food, scripted steering or artificial forces. This is a new
bounded study; the completed v4 allowance is not silently resumed or extended.

## One learning change

Use balanced left/right15/30-degree practice instead of only30 degrees.
Keep the policy features, nine trainable weights, four frozen base coefficients,
fitted gait, rewards, PPO configuration and physics unchanged. Preserve the
existing complete-beat criteria; do not add or change a rhythm reward in this
experiment. Mixed practice tests retention before a separate reward change.

Fork complete state from phase-aware-v4 update112: best6/8 complete30 case
passes,3/8 complete15, all8 raw30 movement passes and12/12 swimming. Parent
task identity0e71bac8a6e09daac2a284d7935eafc93fd99a296bc5f754c6f32e3d07375c95.
Parent session SHA25607D5841E9EA028F4BC087574ED2F78802A5D75E4CFAC7A37976C7B164D2001AB;
actor SHA2564080F07BF6F62E51F73D14EE2ACCA1A20958BE2A3A16DA86D85958072B6BED79.
Bind these files, the prior manifest, frozen gait/source56 and current sources.

Import actor, critic, both Adam states, RNG progress, body, servo, applied trim,
course/beat history and per-lane episode counters. No physical state reset at
the study boundary. Continue Adam ages from7168. Fresh study update/accounting
counters start at0 and report their imported origins explicitly. Initial new
actor must match the parent export exactly. Existing active30-degree episodes
finish without interruption; the mixed lesson applies when the next normal
training episode begins. Both turn directions and angle counts are reported.

## Execution and limits

Use1024 CUDA worlds,64 decisions/world,80ms/decision, minibatches4096 and4passes.
Maximum128 new updates/8,388,608 samples/536,870,912 attempted substeps.
Cooperative training wall cap1200seconds; external process cap1800. No retry or
extension. Save immutable actor exports and complete FTSSv5 state after each
update. Preserve the old FTSSv4 files and binaries. The new format binds the
curriculum and parent identity, and prevents changed-task implicit resume.

Reuse unchanged physics/observer validation. Run a zero-step fork-equivalence
check plus exact fork/resume verification using3 actually executed PPO updates,
196,608 samples and at most12,582,912 substeps;600second external cap. Verify
imported full body/history/optimizer state and counters before collection, not
just actor weights. CPU/CUDA features must agree within1e-6 and the two resumed
complete payloads must match exactly. Count diagnostic optimization separately.

Run one pretraining physical baseline and all16 eight-update assessments with
the existing evaluator. Each contains32 fixed15/30/60/90 turns and12 straight/
from-rest cases per arm:88 trajectories and at most1,100,800 substeps. Baseline
plus16 assessments:1496 trajectories,18,713,600 substeps;30seconds external per
assessment. The whole-study substep allowance is568,167,424 including resume
and evaluation. Actual partial
terminal decisions must be counted, not rounded up or silently dropped.

Keep every original criterion:<=5deg COM course error,>=6mm/s projected and
rolling speed,<=10deg tilt,<=1mm depth error,positive certified whole-body
clearance,>=80% matched root/passive complete-beat frequency and swing; final
2s turn and first2s onward windows. Lesson success requires8/8 complete30 plus
8/8 complete15 and12/12 swimming. Report raw motion passes separately without
waiving rhythm. Report larger-angle generalization separately; neither60 nor90
is trained. Stop on native/numerical faults or required swimming failure.
Incomplete learning or gentle-turn failures remain measured progress results.

Report left/right and angle exposure, every learning curve, COM speed/course,
depth/tilt, stroke frequency/swing and timing. Preserve failed cases. Distinguish
GPU physics/inference/PPO, CPU checks, transfers/waits, evidence writing, build
and total time. A frozen gait does not guarantee the executed rhythm is unchanged.

Do not train the chooser or publish automatically. Even a fully passing motor
needs continuous repeated-turn and bank/endurance qualification in its exact
runtime before Godot autonomous adoption. Working Godot assets stay protected.

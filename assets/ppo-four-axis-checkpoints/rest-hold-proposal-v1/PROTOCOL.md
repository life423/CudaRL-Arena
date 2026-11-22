# Proposed hold-from-rest experiment — approval required

Status: PROTOCOL ONLY. No physical rollout, critic fitting, PPO update or runtime
change has been performed for this proposal. The accompanying static arithmetic
probe constructs observations and evaluates rewards/networks without advancing
physics or applying an optimizer. Previous study budgets remain closed.

## Goal and starting policy

Teach the existing single actor to hold from rest at a fixed point while retaining
swimming. This experiment contains no moving-start braking, timed32s switch,
scripted assistance, posture clamp, joint freeze or second skill network. All four
joint targets remain learned; moving joints while holding is allowed.

Start from a NEW copy of preserved checkpoint336, migrated40->41 using the
already-verified zero-column actor/critic/moment migration. Do not resume rejected
B344. Original336SHA256:
810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0.
Retain all40 original observation meanings, appended signed forward error
tanh(e_forward/.02m), initial new-column weights/moments0, old Adam ages21700,
new-column ages0. Inherited actor/critic/sigma/AR state and optimizer moments
remain intact except the explicit actor-rate override below. Do not install a
runtime policy even if this finite experiment succeeds.

## Measured static checks and their limits

The native static probe called the existing initialization, observation,
reward_costs() and CPUMLP functions. For poses0/3 x depth0,+/-.5mm it verified
zero physical/substep/optimizer counts, all velocities0, original held targets
matching original joint angles, filter rates0, unchanged first40observations,
zero forward anchor feature and elapsed-time feature1. Copied336 and migrated
actor/critic outputs match exactly in all6states. No physics integration ran.

At centered rest, immediate reward is0. Copied336 critic predictions are
-0.279538 and-0.276703; B344 predicts-0.267584 and-0.266580. These are predictions,
not measured returns. At+/-.5mm depth error with zero velocity, native cost is
about.075 and reward-.00375/control unscaled or-.00000366211 with gain1/1024.
The original target remains-80mm rather than being rebased to each depth offset.

Independent constant-cost16s arithmetic(gamma.99,320controls) illustrates scale:

| Hypothetical state held at a constant cost | Return gain1 | Return gain1/1024 |
|---|---:|---:|
| Exact rest at target |0 |0 |
|.5mm/s at anchor |-1.199862 |-.001171741 |
|1mm/s at anchor |-4.799450 |-.004686962 |
|8mm/s at anchor |-307.164774 |-.299965600 |
|.5mm depth error, zero velocity |-.359959 |-.000351522 |
|2mm horizontal anchor error, zero velocity |-5.039422 |-.004921310 |

These are numerical illustrations, NOT physically self-consistent trajectories,
policy rollouts or predictions of actual training returns. The frozen actor at
rest already produces nonzero joint targets; instantaneous zero reward does not
mean it will remain stationary. Actual rest-start return distributions remain
unknown until the separately approved frozen calibration batch below.

Do not spoof elapsed controls640 to activate the old stop mode. A synthetic query
changing only time feature1->0 changes336's centered critic to about-.848/-.840,
showing that reusing the old stop-time feature would change the starting problem.
The real rest lesson starts at control0 with feature34=1, then follows its original
decay to0 at2s. Input34 is not replaced by an episode countdown.

## Explicit rest-state construction

Add an opt-in research episode mode; preserve every historical mode/default.
The existing runner does NOT yet implement this mode. Its present stop/holding
schedule and gain gate both wait until control640, and valid horizons exclude320.
Implementation must explicitly support rest holding from control0 and the320
control horizon. These are environment/protocol changes, not new actor inputs.

For each true episode:

1. Use original initial_world(pose), pose0 or3, unchanged model6, default calm water
   and long1.2x.8m arena. All10 generalized velocities are0 because this is an
   original rest initialization. Preserve initial relative joint angles(+/-.02
   on the original yaw axes), held targets equal those angles, filter rates0,
   empty AR history, original actuator/filter/torque parameters and time0.
2. Holding episodes start CENTERED at the original target depth-80mm, at their
   horizontal hold point. Set stationary_request=true,target_speed=0,
   position_hold=true,schema4,max_controls320 BEFORE begin_forward(). Capture
   horizontal COM anchor once before the first observation/action, with anchorY
   fixed to original target depth. Retain original lane and heading references.
   First40features keep schema2 meanings; input40 starts0; input34 starts1.
3. Do not set velocities, pose, targets, reference or heading again during the
   episode. Do not advance a hidden settling simulation. The NN selects all four
   outputs from the first control onward. Reset only at true16s episode ends.
4. Swimming rehearsal uses original8mm/s configuration, observation meanings,
  2s horizontal warmup and original depth0,+/-.5mm start offsets. A rest-state
   initialization is not a runtime clamp or an action supplied to the policy.

Training holding starts are exactly at target depth/point. The four+/-.5mm depth
cases below are separate robustness assessments, not hidden perturbed training
starts. Every assessment reference is chosen before applying its depth offset,
so an offset never silently redefines desired depth.

## Fixed episode mix, reward and learning settings

32worlds x320controls(16s). pose=i%2?3:0; hold=(i/2)%2==1. Each batch therefore
contains8pose0swim,8pose3swim,8pose0hold,8pose3hold. Hold goal iszero throughout;
swim goal is8throughout. Exactly50% of all training controls are holding, versus
25% in the previous swim32s/hold32s mixture. No pose/goal confounding.

Hold depth offsets are0. Swim depth offset retains(i/2+global_update%3)%3 mapping
0,-.5,+.5mm. Across8PPOupdates each pose receives64centered holding episodes and
64swim episodes with22center,21below,21above starts. No tilt curriculum or moving
starts in this trial.

Keep the previous holding formula and fixed gain1/1024 to change coverage without
another reward-weight experiment. Apply the gain to ALL holding reward components
from the FIRST control, based on actual rest-mode zero goal, not controls>=640.
Holding horizontal and vertical speed/recovery costs are charged from the first
control even while the original elapsed-time feature decays. All swimming reward
terms and their existing masks remain unchanged. No new success bonus, early
behavioral termination, reward clipping, per-goal advantage normalization,
posture penalty increase or adaptive gain selection.

Seed1247037243; sigma=exp(-1),rho.8; actorLR.000125,criticLR.001; inherited Adam
moments/eps1e-5,global gradient clip.5; PPOclip.2,gamma.99,lambda1,minibatch256,
maximum2epochs/update, existing epoch KL stop>.01. Global advantage normalization
continues across the full batch. The shorter horizon and earlier goal activation
are explicit curriculum changes, not a repeat of the earlier scale-isolation test.

Use nonoverlapping new stochastic counters: PPO action RNG seed=source_seed+2,
stream_id=430080+(update-336)*320+t for updates336..343,t0..319. 430080=336*1280,
after all counters used to produce the64s336parent. Calibration uses its own fixed
action RNG seed2718281828; its320time keys and32world dimensions are logged. Keep
the existing PPO permutation rule. No seed search or replacement draws.

## Bounded calibration before actor learning

This entire section is PROPOSED TRAINING/PREPARATION AFTER APPROVAL. None has run.
Critic fitting is explicitly counted as learning, not called a zero-step check.

First run the baseline assessments and64control CPU/CUDA axis parity fixture below.
If the copied336 already passes all new holding gates and existing swim regression
gates, stop successfully without training. Do not consume unused budget.

Otherwise collect ONE frozen-actor32x320 calibration batch with the same50/50mix.
No policy updates; this costs10240controls. Save all states/actions/ARhistory,
actual costs, rewards, terminal flags, critic values and MonteCarlo returns.
Before any optimizer step, require all returns finite, per-goal99th percentile
absolute return<=5 and maximum absolute return<=20. Verify native sampled oldlogp,
CPU/GPU observation/request agreement, zero-goal activation and gain from control0,
all320samples, true terminal only at319 and independent GAE/return arithmetic.
If a gate fails, stop with the evidence; do not silently rescale or retrain.

To adapt the inherited critic to the new starts, keep the actor and BOTH actor
Adam moment arrays/counters frozen. Split by WORLD to prevent train/validation
episode leakage: world%8<4 is critic-fit(16worlds/5120rows); the other16worlds are
held-out validation. Each split has4worlds of each pose/goal combination.

Fit only the critic for EXACTLY4epochs x20minibatches=80Adam steps, batch256,
same criticLR.001/eps/clip and existing value-loss coefficient.5. No reset of its
old moments; new columns retain their own age. Use fixed permutation seeds
3141592653+epoch over the5120fit rows, then scatter/gather those actual indices.
No actor gradient/step, no critic fit on the validation rows, no extra epoch.
The calibration data are NEVER reused for an actor PPO update.

After80critic steps, require held-out rest-goal RMSE to satisfy BOTH
RMSE_after<=max(.05,.75*RMSE_before) and
RMSE_after<=max(.05,.5*R_RMS), and absolute mean bias<=max(.02,.25*R_RMS).
Held-out swim RMSE must stay<=max(.10,1.10*RMSE_before). These are proposed
conditioning decision thresholds in return units, not physical acceptance limits
or claims that80steps are sufficient. Record per-pose/goal values, variance,
bias, normalized error and explained variance. If a criterion fails, stop before
actor training; preserve calibrated checkpoint and do not retry/reset the critic.

Then collect the first new PPO batch using the unchanged frozen actor and distinct
PPO RNG counters. Recompute values/GAE using the calibrated critic. Before the
first actor update, require holding rows to contribute10..90% of total squared
globally normalized advantage. Log group means/stds/signs; a positive advantage
is not a physical pass. If this signal-allocation gate fails, stop without actor
updates; no automatic per-task normalization or reward gain change. This first
batch counts as the first of the eight PPO batches, not an extra rollout.

If every gate passes, run exactly8PPOupdates336->344, absent hard failure. At most
640Adam steps EACH during PPO. Actor old/new ages end at21700+actual_PPO_steps /
actual_PPO_steps; critic at21780+actual_PPO_steps /80+actual_PPO_steps. Maximum
actor22340/new640; critic22420/new720. Do not assume actor/critic clocks are equal
or erase critic calibration ages. Independent-clock checkpoint reload is required.

## Integrity and assessment gates

Before physical work freeze executable, source, copied336, protocol/config and
initial-model hashes. Nonphysical CPU/CUDA tests must cover all32world initial
requests/gains/horizons, fixed references, zero velocities/initial targets/rates,
unchanged40features, positive/negative input40, initial actor/critic equivalence,
new-column moments/ages, schema rejection and asymmetric post-calibration clocks.
Preserve schema5/41inputs with a NEW explicit lesson protocol checksum that binds
rest mode/horizon/mix/gain/calibration/counters; no legacy loader fallback.

The frozen calibration batch can provide actual-state CPU/GPU old40/new41 output
comparisons without extra physics: old/new<=1e-6,CPU/GPU<=2e-5. New input columns
are stillzero before critic fitting. Check actual sample/counter accounting,
reward component sum and independent GAE with existing.002+.0002*abs(return)
tolerance, oldlogp rescore<1e-4, every training row once per epoch, all four action
outputs, finite parameters and correct independent Adam clocks. Save every batch,
calibration split/diagnostic, complete checkpoint, assessment and failed attempt.

Baseline and final EACH comprise:

- Four original8s swims: speed7.2..8.8mm/s,3DvelocityRMSE<=2.5mm/s,lane<=2mm,
  depth<=1mm,tilt<=10degrees,path<=5degrees,work<=.001J/m; require4/4.
- Two original60s swims: retain all28previously passing4..60s windows. Report
  full30windows/formal cases separately; the two known startup-speed misses remain
  failures, not full formal long-swim acceptance. New failures are regressions.
- Six64s HOLD-FROM-REST cases: poses0/3 xdepth offsets0,-.5,+.5mm. There is NO
  initial32s swimming phase and NO grace period. Actual3DCOMspeed<=.5mm/s,
  3Dfixed-anchor error<=2mm,depth<=1mm,lane<=2mm,combinedtilt<=10degrees at the
  initial state and EVERY50ms endpoint for ALL64s. Also require final56..64s
  forward excursion from56s<=2mm. Joints may move. Require6/6, with centered2/2
  and four depth-robustness results reported separately.

Report speed ranges/RMS,maximum and signed depth error, roll/pitch/combined tilt,
lane,anchor/drift/overshoot/reversal, all4joint outputs/angles/rates/torques and
physical work. No intermediate physical candidate assessments; per-update saved
training diagnostics are allowed within the existing batch budget. Passing means
bounded64s holding plus retained swim regressions; it proves no indefinite result.
Only a fully passing result can justify PROPOSING a later moving-start lesson;
no progression or adoption is automatic.

## Hard budget and stop rules

| Work | Controls | Physics substeps |
|---|---:|---:|
| CPU/CUDA axis parity |64 |640 |
| Baseline assessment |10720 |107200 |
| Frozen calibration batch |10240 |102400 |
|8PPO batches,32x320 |81920 |819200 |
| Final assessment |10720 |107200 |
| MAXTOTAL |113664 |1136640 |

Maximum8PPOupdates;80critic-only steps plus<=640actor/critic PPOsteps; total<=640
actor steps and<=720critic steps. All partial/back-end attempts count. Native and
own-process wall caps120s for the entire physical/calibration/PPO/assessment run;
preparation build cap300s and zero-physics readiness cap60s. Local existing GPU,
no new paid resource reservation. Actual runtime is unmeasured; budget is a cap,
not a promised duration. No retries, extra seeds, extensions, additional physical
replays or reuse of unused controls.

First integrity/nonfinite/numerical/physics fault stops the entire attempt. Defined
scale/critic/signal gates stop at their phase boundary. Ordinary behavioral failures
are recorded without a pose clamp, scripted recovery or tolerance relaxation and
do not authorize additional updates. A failed final hold or swim gate rejects the
candidate. Preserve all checkpoints/evidence and original336; keep changes local.

## Remaining uncertainties

Actual frozen rest-start returns, critic validation fit, exploration stability and
time/history aliasing remain unmeasured. Static reward0 does not mean the seeded
actor holds still. Gain1/1024 may underweight near-target differences; calibration
and the signal-allocation gate diagnose this without silently tuning the objective.
The finite horizon and critic-only fitting are explicit new protocol elements.
Depth-offset generalization was not included in centered holding training, so the
four robustness cases may fail. Eight PPO updates may be insufficient. None of
these possibilities authorizes more training or weaker gates. Braking from motion
and coordinated braking authority remain separate, unvalidated future questions.

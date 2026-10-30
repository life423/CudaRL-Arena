# Single calm-water forward pilot, version 1

User authorized a quick bounded demonstration after local checkpoint3e0e4ab.
This is actual native GPU PPO with fixed-variance Gaussian latent actions,
not a scripted gait, servo-only swimming claim, or production controller.
Freeze this file, all first-party source, numeric binding and executable before
the sole run in `build/forward-demo/2026-10-30-v1/`. No retries or tuning.

## Physical scope and count

Retain physics2, original fluid1000kg/m3/.001Pa-s/9.81m/s2, zero current,
dimensions/inertia/water forces, fixed existing joint-target gains, four axes,
and10microNm limits. One native control is50ms with ten5ms substeps. Contact
remains enabled but MUST remain inactive. All native sampled envelope clearances
must be>=10mm. Any contact, partial interval, nonfinite state/torque, unsafe flag,
clearance breach, rejected task transition or numerical failure stops the whole
pilot before further collection/GAE/Adam. There is no rescue or failure reset.
This does not validate contact, free-water temporal accuracy or the full model.

Hard cap:32 readiness controls +2048 training +640 evaluation =2720 controls,
27200 native slots, both backends/failures included. Charge whole batch before
launch even if it fails; report actually executed substeps separately. There
are no internal adaptive physical steps/retries in this physics2 path. Unused
slots close with the run and are unavailable automatically.

Four starts id0..3: default stationary body position(0,-2,0) model units,
override x=(id-1.5)*.02; yaw=(id-1.5)*.05rad about worldY; joint0=-.02rad for
even ids,+.02 for odd, joint2=-joint0, others zero. No initial thrust/velocity.
Readiness uses these four starts on CPU and CUDA, four controls each: commands
(-.1,0,+.1,0),(+.1,0,-.1,0), repeating. All defined state fields use the existing
2e-4+2e-3*abs(CPU) numerical comparison; min gaps use2e-6+2e-3*abs(CPU).
Flags, control/substep counts/time agree exactly. This is a short no-contact
backend check, not independent physical-model validation.

## Task, learning and evaluation

Use task version1 in `src/motor/forward_task.h`: latch whole-COM goal20mm ahead,
terminal velocity10mm/s and initial heading.36 observations, four latent actions
mapped by tanh to normalized existing joint targets. Reward coefficients and
units are fixed in that source. Physical states remain separate from task state.

Actor36-32-32-4, critic36-32-32-1, tanh hidden layers, linear outputs. Orthogonal
gains sqrt(2),sqrt(2),actor .01/critic1. Seeds0x46574431/seed+1; sample seed+2,
streamid=update*32+tick. Fixed log_std=-1.5, no variance optimizer. Four updates,
16 worlds*32 controls/update, time-major rows. GAE gamma .99,lambda .95;
observed80-control horizon is terminal; bootstrap only running rollout cuts.
Two deterministic shuffled epochs, two256-row minibatches each. PPO clip .2,
entropy coefficient0, critic coefficient .5; Adam lr .001,beta .9/.999,
epsilon1e-5, global gradient norm clip .5. Maximum16 Adam steps per network.
No annealing, sweeps, model selection or extra data. Finite gradients and model
weights required. Explicit terminal resets rotate id by that world's episode
count; only accepted success/timeout can reset. Save parameters, Adam moments,
world/task state and episode counters at update boundaries in an ABI-scoped
checksummed research artifact. Roundtrip bytes are checked; no resume was tested
or automatic continuation authorized. Do not load into diagnostic/user saves.

Deterministic CPU inference/evaluation of initial and final weights, four starts,
maximum80 controls/4s each (640total), with the same native CPU world equations.
Success requires COM travel>=20mm, lateral error<=2mm, velocity-vector error
<=5mm/s and heading<=10degrees for three consecutive controls. Report each case,
mean forward COM travel, goal error and all failures; reward is not acceptance.
No successful-learning claim unless these fixed criteria pass. Four updates
may be insufficient: a failed lesson closes normally without extending it.

Native CSV before/after traces and immutable weights are the first deliverable.
They are measured policy trajectories, not a live Godot demonstration. No
runtime/contact adoption, original-tree change, current/food/chooser, support
study, checkpoint overwrite, push/merge/deployment or further training follows.

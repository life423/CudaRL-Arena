# Continuous pond learning audit — November 23, 2026

**Keep prepared V2 on hold.** Its numerical checks pass, but its objective and
learning interface need correction before another PPO allocation. First measure
whether canonical336 follows a changed persistent heading through its existing
motor-request inputs; do not add another controller to answer that question.

This audit executed **0 new physical controls, 0 substeps, 0 GPU kernels,
0 PPO updates and 0 Adam steps**. It read sources, prior results and existing
zero-integration fixtures. No frozen native source, checkpoint, body limit,
protocol or acceptance threshold changed. V2 remains prepared and unexecuted.

## 1. Early failure can improve the current objective

`turning_task_v2.h:243` permits negative running reward and separately subtracts
positive work. Behavioral failure costs4 once; future inactive rows pay zero.
The collector divides the whole return by60, not survival time. Failure therefore
removes all future negative cost.

For the same prefix at time t, suppose continuation has net rate -q until60s.
Immediate failure pays -4; continuing pays -q(60-t). Quitting wins whenever
`q(60-t)>4`, by `[q(60-t)-4]/60` normalized return.

For an illustrative metric combination with forward/horizontal speed5mm/s,
depth4mm, zero side speed, tilt and work:

```
s = 5/8
motion rate = s - 2(s-1)^2 = .34375
net rate = .34375 - .05*4^2 = -.45625
```

These metrics are inside the registered5mm/20degree behavioral envelope. With50s
remaining, continuation costs22.8125 versus4 for failure, a0.313542 normalized
advantage for quitting. This is an algebraic example, not a simulated trajectory.

Bound the **complete** running utility, including work, then assign absorbing
failure utility for the registered remaining time. For quality u in[0,1], use
`dt*u` while live and subtract `max(0,T-elapsed_after_transition)` once at failure,
with T=60. A separate fixed failure penalty is optional. Clipping motion but
subtracting work outside that bound does not establish the lower-bound guarantee.
Keep padding rows/rewards zero: this is terminal accounting, not invented motion.

Two deaths on the same continued trajectory differ by
`integral(t1,t2)(u+1)dt/T >= 0`; their fixed failure penalties cancel. Early death
therefore cannot improve the same-prefix return. This alone does not ensure6mm/s
or1mm/10degree acceptance. Hard physics/numerical faults remain global stops,
never ordinary absorbing learning examples.

## 2. The study does not teach the intended requested-turn motor

`turning_task_v2.h:151` replaces request heading/velocity every tick with the
**current** head-forward direction. Reward follows that moving body frame. No
persistent desired course or requested-turn error is presented. Explicit requested
Swim bypasses the new residual altogether (`turning_policy_v2.h:56`).

This could learn an autonomous bank policy, but it does not answer whether the
existing reusable motor can execute a chooser's persistent heading requests.
It simultaneously searches for bank avoidance, turning and depth coordination
while leaving that existing capability unmeasured. Old straight-swim success only
proves a course initialized aligned with the body, not a changed moving request.

More coefficients do not remove every restriction: each output remains one
additive bounded tanh of a feature sum, with at most.75 override of an inherited
latent. The [control audit](CONTINUOUS_POND_CONTROL_AUDIT_2026-11-23.md) found yaw
latents beyond that override in about half of actual autonomous baseline rows.
This is a representational risk, not proof that reversal is required or turning
is physically impossible.

The next diagnostic should use the original motor/observer unchanged:22 cases,
poses0/3, offsets0,±5,±15,±30,±60,±90degrees relative to the **original persistent
course**. Warm80 genuine controls, latch one request, continue to320 controls.
The0degree case must keep its heading/lane byte-identical. Nonzero requests may
rebase task XZ at request while preserving original depth, world/filter state and
clock. Proposed cap:320 paired precheck plus7,040 probe controls=7,360; zero PPO.
The probe remains preparation only until separately authorized.

If that motor lacks heading tracking, investigate fine-tuning the **same** actor
from336 with per-action PPO under persistent turn requests and actual retained-
skill tests. If it is already competent, chooser/navigation learning is narrower.
Neither result automatically authorizes another residual or per-ability network.

## 3. Longer dynamics do not supply the missing policy credit

V2 samples one80-dimensional Gaussian controller vector at episode start and
scores that single decision against60seconds of return. This is valid parameter-
space PPO; the latent/probability math is correct. There is no separate policy
decision or credit assignment at each turn/joint command. Same-case leave-one-out
removes case offsets, not ambiguity about which late maneuver helped.

| Quantity | Closed V1 | Prepared V2 maximum |
|---|---:|---:|
|Coefficients|16|80|
|Updates × episodes|64×4,096|24×2,048|
|Independent controller samples|262,144|49,152|
|Samples/coefficient, descriptive only|16,384|614.4|

This ratio is not a sample-complexity bound; it shows why more physical seconds
do not automatically mean more useful learning decisions. Existing2,048 zero-only
V2 samples change initial pitch actuator commands by0.274/0.182 at the95th
percentile. Zero mean preserves the parent; sampled controllers still make
material gait changes.

Expected sampled return also differs from deterministic-mean qualification.
Integrated soft penalties poorly represent whole-episode maxima and worst motion
windows. A single50ms depth error of3mm costs only0.0225/60=0.000375 normalized
reward, while failing the1mm maximum-depth gate. Opposite successful turning
styles can have useful sampled returns without their coefficient mean being a
useful controller. First16/16 selection and one reserved validation honestly
detect those gaps; they do not repair the learning signal.

Measure requested-turn competence first, then expose persistent turn intent and
per-action motor credit if needed. Preserve original saves and movement criteria.
Do not spend another large allocation solely because longer episodes, more
features and numerically correct PPO are available.

## Evidence and source identity

Closed V1:64 PPO updates/632 Adam steps,0/16 continuous selection, with retained
skills4/28/3/148/6/6/2. Its calibration produced8/4,096 single45degree heading
events at6.31–7.10mm/s; all8 drifted4.31–5.28mm in depth. No analyzed4s window
combined a45degree event,≥6mm/s and both body gates. That supports investigating
coordination/intent, not declaring the physics incapable of turning.

Unchanged sources examined:

- `src/nn/motor/turning_task_v2.h`:SHA256
  `CBCA6B57F421155EE002BB63270001493E1A56442D47E71DC813BB1B4DF159FB`.
- `src/nn/motor/turning_policy_v2.h`:SHA256
  `4450974A5679F5670C5F41EE70CDE7DD0F95582888F359CA5838B1BFD4993A39`.
- `tools/turning_swim_v2_study.cu`:SHA256
  `7956B48CE2DAE015A65D7088DA5E73F625C61A4210112CDAA3DA748760A52209`.
- `build/ppo/continuous-pond-2026-11-23-v1/training-result.json` and saved
  calibration episode/meta/post-body records.
- `build/continuous-pond-preparation-2026-11-23/saved-reward-diagnosis.json`.
- `build/ppo/continuous-pond-v2-2026-11-23-v1/prepare-zero/initialization.bin`:
  existing coefficients evaluated on CPU only, without advancing a body.

No new training, probe execution, adoption or budget extension follows.

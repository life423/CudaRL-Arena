# Living idea register: swimming, turning and the evolving pond creature

## Faster swimming and whole-pond exploration

The user would like the creature to swim faster and explore the whole pond.
The current turning lesson keeps the inherited 8 mm/s target so that speed
and steering changes can be evaluated separately. A later speed lesson can
mix faster requests with the familiar speed and Rest. Compare measured COM
speed, turning radius and recovery at each requested speed before promoting a
candidate. Increasing the playback rate would leave the swimmer's physical
ability unchanged.

For the navigation chooser, the proposed positive reward is visiting water
that has received little recent traffic. The same perimeter circuit then
earns progressively less. Favor a small impact or sustained-stall cost and
give the chooser clearance observations so it can learn its own turn. Safe
approaches to the shallows should remain possible. Evaluate interior and
outer-water coverage separately over one continuous swim, along with contact,
stalls and retained swimming and resting. These are proposed chooser and
speed lessons. The fixed first turning run keeps its declared reward.

## Latest user direction: shoreline and autonomous decisions

The user identified that the stopped creature is far from the drawn bank.
The supported whole-body ellipse is0.20/0.12m in radii; drawn water uses
0.52/0.34m, and the intervening slope is presentation only. They explicitly
requested fixing this mismatch, reaffirmed that the creature learns which
direction to turn for itself, asked us to use all three supplied reviews,
and rejected scripted or clamped creature behavior. Do not turn that into
an if-near-bank steering rule, forced yaw, position correction, bounce or
automatic reset. A guard-driven pause is not a learned turn or navigation.

The original reviews remain intact below and under design-reviews/. Apply
their shared native geometry/sensing/presentation requirement, persistent
world requests, two-policy ownership, measured moving turns and actual-candidate
skill-retention checks. New reward formulas, physics variants, PPO schedules
and old study budgets are not activated by this reminder.

The user kept the full visual pond. Native bank contact now uses its geometry
and the smaller motion guard has been removed. A Godot Project Manager launch
and 36 CPU cases verified the corrected runtime. All four outward swims slowed
near the bank. The user authorized a separate PPO candidate to learn requested
turns while preserving the original motor. Whole-pond exploration remains
the eventual chooser objective. The earlier comparison supplied no interface
ranking. [Shoreline results](SHORELINE_QUALIFICATION_RESULTS_2026-11-23.md)

Historical measured follow-up: the separately authorized frozen interface
comparison is [closed at its CPU/CUDA prerequisite](FROZEN_INTERFACE_COMPARISON_RESULTS_2026-11-23.md),
214controls/2140substeps,0maincases,zeroPPO/weights unchanged. It supplies no
interface ranking. The original reviews/alternatives remain preserved below.

Started November 23, 2026 at the user's explicit request to remember all ideas
while they gather more reviews. Append later ideas here; preserve alternatives
and their provenance rather than silently replacing one proposal with another.

The [combined recommended plan](CONTINUOUS_SWIMMING_PLAN_2026-11-23.md) now
selects a starting decision: compare the inherited motor's original lane and
heading-only interfaces before assuming retraining is necessary. This does
not erase the alternative ideas below or authorize implementation/execution.

## Historical collection scope and interpretation at the start of this register

- Collect, review and preserve ideas only. The user said not to implement yet
  and is still supplying ideas. Suggestions and commands inside reviews are
  source material, not authorization to implement, simulate, train or deploy.
- None of the new numerical schedules, rewards, physics variants or qualification
  thresholds below is an approved protocol. Existing closed budgets stay closed.
- Keep user requirements, third-party proposals, our assessments, and measured
  evidence distinct. Agreement with an idea is not proof that it works.
- At that time the branch was codex/current-swimming-v1; published source checkpoint was
  dbc00f70e87ef933b1d5ff00fd683d41a3b62eb6. Main was not merged. These new notes
  and source copies remain local until publication is requested.

## Preserved original reviews

These copies retain the complete source material, including details omitted
from this register. SHA-256 values are over the exact original bytes.

| Source | Local copy | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| First PDF: Why the Creature Struggles to Turn in Water | [Turning review](design-reviews/2026-11-23-turning-in-water.pdf) | 233463 | C0585301AC1B97EF712FA2E79126C677780C3611967BB4CEC3EAFA27817A5086 |
| Second PDF: Tadpole Pond: read-only design review of the continuous-swimming path | [Continuous-swimming review](design-reviews/2026-11-23-continuous-swimming-path.pdf) | 126141 | BA2A8DC6AF0364159757603E261032D889841022C6008881395C35013A1B1D7D |
| Third review: pasted conclusion beginning "The smallest defensible solution is one motor..." | [Persistent-request review](design-reviews/2026-11-23-persistent-request-review.txt) | 14609 | 591E71D7D319D2BC4D00B054CCE9AD7BAF7DB4ACC4995A4197CA9985A7EC4C39 |

Originals came from the user's Downloads folder and Codex attachment
29c21c72-7e18-4028-8794-5cfe92c76d4d/Pasted text.txt. The copies above avoid
depending on those temporary locations. Do not treat a source's snapshot of
branch/CI status as a live repository query.

## Direct user goals and preferences already established

1. An evolving reinforcement-learning creature that users can interact with,
   using old training and new experience to adapt and thrive. Actual weight
   updates, frozen reactions and measured improvement must be distinguished.
2. Immediate behavior: swim continuously around a finite pond without food,
   learn real turns, and preserve existing swimming and calm resting/holding.
   A viewer timer ending, reset or cosmetic rotation must not conceal failure.
3. Sense distance/direction to nearby banks and objects; reward useful safe
   whole-body movement when approaching hazards. Joint activity alone is not
   useful travel. Avoid stalling, spinning and approach/retreat reward farming.
4. Swimming and steering in currents are core aquatic goals, with old calm
   skills preserved. Braking from motion, recovery and arbitrary mode changes
   are distinct capabilities, not implied by holding from rest.
5. Longer journey: persistent pond/open-world area, meaningful user interaction
   and adaptation, physical growth, underwater limbs, shallow crawling, climbing
   the bank out of water, land movement and eventually manipulation. No limbs
   or shore-exit capability is currently implemented by these ideas.
6. Keep native C++/CUDA PPO and make effective, measured use of the configured
   GPU. Do not substitute Python/PyTorch training or scripted movement. The
   intended architecture remains one goal chooser and one reusable motor;
   training critics are auxiliaries, not extra behavior controllers.
7. Preserve original checkpoints, full saves and evidence. Retained physical
   skills must be tested on the actual candidate, not a hidden old-policy bypass.
8. AESTHETICS IS KEY. Maintain the accepted natural rounded pond, grassy bank,
   clustered rocks/reeds, muted water and coherent Godot GUI. Preserve the close
   aerial creature view, manual zoom and readable opaque header/contrast. Keep
   the creature visible and diagnose the awkward crooked-V gait physically.
9. Update and inspect the actual Godot app when a capability is ready; use the
   correct worktree project in Godot Manager. Clearly disclose limitations.
   Current app appearance/inference is not proof of ongoing learning.
10. Avoid circular experiments and proxy-score claims. Audit the end-to-end
    observation/action/physics/reward/deployment path; use finite, discriminating
    steps. Feature publication and main merges remain separate decisions.

## Evidence boundary: what is already measured

Read [AI_STATUS](AI_STATUS.md), the [handoff](AI_HANDOFF_2026-11-23.md), and
[frozen-turn results](FROZEN_REQUEST_TURN_RESULTS_2026-11-23.md) for exact scope.

- The saved swimmer supports bounded straight swimming and learned-amplitude
  calm holding from rest. This is not qualified general navigation, current
  competence, moving-start braking or unlimited swimming.
- The 64-update continuous-pond V1 study passed 0/16 qualifying starts. It
  learned episode-persistent steering coefficients around a frozen backbone;
  short-horizon return improved while sustained swimming slowed/stalled.
- The completed frozen probe tested one persistent request per case, not
  repeated turns: 22 cases, 5782 controls and 57800 accepted 5 ms substeps.
  Small +/-5-degree requests retain 7.75-7.83 mm/s, but have roughly 3-4-degree
  late course error. +/-30-degree cases slow to 2.20-2.34 mm/s. All eight
  +/-60/90-degree cases stop at 20-degree tilt; no bank/depth termination.
- Full-actor requested-turn preparation compiles, but its fresh matched-command
  gate and independent calibration/training/optimizer auditors are unfinished.
  It has not run new full-actor physics/PPO. Actor42 has no bank/object sensors.

Additional read-only analysis during the first review used the existing
frozen-probe records, not new dynamics: no actuator saturated in any of the
57800 accepted 5 ms samples. Peak requested torque was 2.5901 microN*m,
25.9008% of the unchanged 10 microN*m limit; post-request peak was 23.813%.
Peak fractions by proximal yaw/pitch and distal yaw/pitch were approximately
0.25900836, 0.16699056, 0.15584922, 0.12680872. This rules out torque clipping
as the cause of those recorded failures, not all possible physical limits.

Reconstruction: existing verify_frozen_requested_turn_result.records(), native
696-byte world layout and checksum envelope; take only accepted*4 requested
torques per record, excluding padded substeps; requested is N*m, actual is
normalized by the float32 torque cap. All post-request probe samples: 37000.
Raw local states.bin SHA-256: D25A4A0F6092E9FD13345765157B87DAA85F2C29B1164CF825D9930C14102791.
Layout SHA-256: 59B9EF2D0EA28EA588E608954DC986148D715981F23E77C37856E3C185D99C3D.
These ignored raw files remain in build/ppo/frozen-request-turn-2026-11-23-v1;
copying the reviews does not newly publish the raw research evidence.

## Architecture and interface ideas from all three reviews

- Keep a persistent world-frame request: desired course, speed/terminal velocity,
  depth and hold intent. Relative heading selections become a world intention
  once; do not continually redefine the goal using oscillating head direction.
- Candidate motor: inherited 41-input swimmer migrated to 42 inputs with a zero
  intent column, same four joint-target outputs, all motor weights trainable,
  native per-control AR-Gaussian PPO. Preserve actual sampled-latent likelihood,
  current and previous mean gradients, held targets and filter/history state.
- Zero-column migration preserves initial same-input inference only. Changed
  observations need a matched baseline; later PPO can forget old skills.
- Reviews 1/2 favor motor training before autonomous navigation. Review 3 adds
  an important alternative: first compare original lane-following against the
  corrected heading-only interface with inherited weights. Retrain only if
  the corrected interface still lacks the required physical turning envelope.
- The completed probe retains a new straight lane with 2 mm-scaled lateral
  error after the requested turn. A curved transition can generate lane error
  while trying to turn. This is a plausible interface conflict, not a proven
  sole cause of failure. This is a learned-input conflict, not a hard 2 mm
  guard or a physical force imposing a straight line. Removing familiar
  stabilizing feedback could also hurt. The new heading-only observer removes world-X/Z lane
  error BEFORE body-frame rotation, retaining original depth and persistent
  requested velocity/course. Its physical benefit remains unmeasured.
- A learned chooser can use bank/object geometry, approach velocity, actual
  COM course, current persistent intention, motor tracking quality and task
  context. No bank inputs are necessary in the first isolated motor lesson.
  This does not establish a permanent rule against local/contact observations
  in later terrain-aware motor designs.
- Proposed chooser decisions every 0.5-1.0 seconds, initially fixed swim speed
  and bounded +/-5-degree increments; a three-choice {-5,0,+5}-degree action
  set is an alternative to a bounded continuous increment. Later allow speed/
  hold context and possibly joint fine-tuning of the same two policies.
- Initially freeze a qualified motor while teaching the chooser to isolate
  navigation failures. Bank-aware joint/end-to-end learning remains an
  alternative to compare, not something proved impossible by these reviews.
- A single +/-5-degree response does NOT qualify repeated 5-degree requests
  every second. Requests can accumulate faster than motor response. Timing,
  allowed slew, tracking error and whole-body turn radius need measurement.
- Ideal kinematic examples ONLY: at 8 mm/s, 5 degrees/s implies radius about
  91.7 mm; 10 degrees/s about 45.8 mm. Inset pond minor radius is 115 mm and
  body length about 72 mm. COM circle fit alone does not establish body fit.

## Learning and curriculum alternatives (not selected protocols)

- Existing draft/report 1: fixed 32 updates with angle caps 15,30,60,90 degrees
  in eight-update blocks, 3072 worlds and 12-second lessons. All are proposed,
  not a guarantee that learning will converge.
- Report 2: qualification-based progression 0/+/-5, then +/-10/15, then +/-30,
  followed by repeated small requests accumulating 90,180,360 degrees. Any
  future success-based curriculum still needs finite per-stage/total budgets.
- Report 3: matched interface diagnostic first; repeated smooth courses and
  turns followed by straight swimming without resets; fine-tune the existing
  motor only if needed, then teach navigation over its measured capability.
- Per-control PPO offers finer statistical temporal credit than scoring one
  parameter vector per episode. It is not causal identification or a cure for
  delayed filters, sparse rewards, exploration or incorrect observations.
- Episodic parameter-space PPO is a valid method. Missing explicit course in
  an autonomous bank task does not make avoidance inherently unlearnable;
  prior wrappers had bank sensing. The failure diagnosis is not an ablation.
- Mix freshly collected current-candidate rollouts under old swim/rest task
  definitions into practice. Do not pass historical trajectories off as current
  on-policy PPO samples. Testing/rejecting a regressed update detects damage;
  practice provides retention gradients. They are complementary mechanisms.

## Reward ideas and unresolved problems

1. Reward actual mass-weighted whole-body COM movement, useful speed/course,
   bounded depth/posture and effort. Do not reward head alignment, tail motion
   or commanded turning as if it were achieved travel. Course needs a speed
   floor; angle is unreliable near zero velocity.
2. Keep complete live utility, INCLUDING effort, bounded and nonnegative if
   relying on that bound to rule out cheap failure. Use consistent dt, power/
   work and normalization units; charge unexecuted failed time once under the
   chosen finite-task accounting. Do not reinterpret padding as real motion.
3. Report 2 proposed q_speed*q_course*q_depth*q_tilt minus lateral-speed-squared
   and work costs. q_speed clips (v_parallel-v_min)/(v_star-v_min); the other
   qualities use exponential squared errors. Its subtractive costs can make
   total utility negative, contradicting its later nonnegative guarantee.
   rho >= max(utility) does not bound the negative tail. Do not copy unchanged.
4. Report 3 notes the current narrow velocity utility initially gives an ideal
   8 mm/s swimmer on its old course about 0.22 at a 30-degree new request,
   0.003 at 60 and zero at 90 (analytical examples, not simulation). Proposed
   alternative: separate bounded continued-travel credit from command-tracking
   quality, while retaining settled-course/speed/depth/tilt qualification.
   Tradeoff: travel alone can preserve comfortable wrong-direction swimming;
   progress/tracking and failure incentives still need review.
5. Report 2 proposed predicted-clearance progress only inside a hazard warning
   band and above a COM-speed gate: clipped signed clearance difference, plus
   hazard episodes and rearming only after one body length outside the band.
   Leaving can pay while re-entry from outside is unpenalized. Clipping, gating
   and rearming do not guarantee cancellation; lockout only slows the cycle.
   An intent-based predictor can improve before physical safety improves.
6. Report 3 alternative: task-conditioned bounded potential shaping
   F = gamma*Phi(next_state,next_context) - Phi(state,context), flat outside
   the relevant hazard zone, plus hazardous-proximity cost if justified.
   Preserve both signs, consistent discount/terminal treatment and observable
   context/history. Do not claim it repairs a defective base travel objective.
7. COM arc length is better than joint activity but can count oscillation or
   tiny loops. Supplement with rolling multi-stroke net displacement, actual
   course/curvature and stall diagnostics. Avoid large whole-run net-displacement
   requirements: legitimate pond circuits can finish near their starting point.
8. Proposed stall variants: existing two-second rolling arc-speed detector;
   report 2/3 eight-second windows and multi-stroke displacement. These are
   different proposals, not silently changed existing thresholds.
9. Continuous-SWIM should not permit selecting HOLD/zero speed to evade its
   movement objective. Preserve HOLD as a separately requested task of the
   same motor; physical braking and seamless mode changes still need teaching.
10. Report 3 distinguishes genuine finite 12-second skill lessons from a
    continuing navigation task. For continuing-task collection cutoffs, propose
    value bootstrap instead of pretending the creature's future ends. Physical
    failure stays terminal. A task redesign must specify discount, value inputs,
    masks, failure accounting and rollout continuity coherently; do not just
    flip the existing finite-task terminal flag.

## Diagnostics and alternative physical investigations

Report 1 suggested the following; all remain unapproved proposals, not fixes:

- Single-joint positive/negative command steps for 0.25-1 second.
- Proximal/distal differential-yaw grid, with baseline pitch commands.
- Phase-locked yaw bias at 8-16 gait phases.
- All-four-axis 1-2-second open-loop sequence search with speed/tilt constraints.
- Local action-response Jacobian at 32-128 gait states, +/-0.05 command
  perturbations and 50-250 ms response measurements.
- Requested/applied torque, saturation fraction/margin, peak/integrated yaw,
  actual COM course/speed, depth/posture, positive work, course-change/work
  efficiency and minimum-speed/8 mm/s propulsion-retention ratio.
- Optional comparisons: torque cap 0.5/1/2 times; reference filter 1 vs 3 Hz;
  explicit angular drag 0.5/1/2 times; head buoyancy-center offsets
  0/0.2/0.4/0.8 mm. Later realism: added mass, consistent Coriolis effects and
  anisotropic rotational hydrodynamics. Preserve/version baseline physics.

Review limits: success provides a feasibility witness for the tested model;
failure of a finite sweep or local Jacobian does not prove physical inability.
Tiny explicit rotational Stokes drag is not total turning resistance, and
internal joint torque is not direct whole-body yaw torque. Distributed drag
and coordinated body motion matter. No evidence calls for blindly raising torque.

The 1-to-3 Hz filter proposal is not a harmless replacement: the acceleration
guard REJECTS rather than clips above 200 rad/s^2. At 3 Hz a one-radian target
error at zero reference rate requests about 355 rad/s^2. Filter validity must
be distinguished from torque saturation and movement capability. A buoyancy
offset adds a meaningful restoring moment to a common-center model; it needs
physical justification rather than being a hidden stability aid.

Current bank rays translate the current conservative body corners toward the
ellipse. They do not predict the rotating/deforming swept body during a turn.
Prediction requires measured response/lag and conservative body clearance.

## Proposed validation, deployment and future contexts

- Match CPU/CUDA on the exact action/physics path, including filter state,
  requested/applied torques, work/energy and first-failure states/causes.
- Verify same-input inheritance and a new baseline after observer changes.
  Independently reconstruct raw actions/likelihoods, observations, reward,
  GAE/returns, masks, optimizer updates and checkpoint identity from saved data.
- Qualify actual COM-course changes in both directions and repeated sequences,
  preserving physical/actuator state. Unwrap course only above a speed floor;
  commanded angle, head yaw or accumulated angle jitter cannot count as 360.
- Proposed new gates in reviews 2/3: >=6 mm/s average; no complete eight-second
  window below 3 mm/s (review 3 explicitly excludes startup); <=1 mm depth
  and <=10-degree tilt; <=5-degree late
  course error over final four seconds. Existing parent-skill thresholds are
  unchanged; these new gates are not yet approved or universally necessary.
- Endurance proposal: 16 initial 60-second cases then eight withheld,
  uninterrupted 30-minute runs, varied starts/headings and disclosed backend.
  No resets/rescue or tolerance relaxation. Finite passing evidence supports
  the measured duration/conditions, never a mathematical infinity claim.
- Report 2 proposes >=70% spatial coverage and hazard encounters; report 3
  also favors sustained coverage. Our review: coverage is an additional
  exploration objective, not necessary for the immediate continuous-swimming
  goal. Safe circuits can be valid. Do not silently reject them or encourage
  needless hazard approaches. Any bins must be genuinely body-reachable.
- Retain the actual candidate's 4 short swims, 28 previously passing 60-second
  windows, 3 recoveries, 148 previously passing 300-second windows and all
  14 holds, recording the known startup misses. No old-policy bypass.
- Deploy exact validated artifact into the native CPU/Godot path and verify
  rendered behavior and hash/source identity. Matched commands do not prove
  long independently closed-loop performance on both backends.
- Current Godot Rest/Swim selection resets a bounded fixture. It is not seamless
  movement/braking or continuous navigation. A future navigation mode needs
  persistent requests in one native world, not just a removed viewer timer.
- During aquatic avoidance, shore is a hazard; during an explicit future shore
  task it can be a goal. Limbs are part of the intended roadmap, but whether
  this body could exit by undulation rather than requiring limbs is untested.
  Context alone cannot create suitable anatomy, contact,
  partial immersion, water exit or land physics. Currents likewise require
  qualified behavior and bank prediction relative to the pond.

## Research references checked during review

- [Hierarchical locomotion and reusable low-level control](https://proceedings.mlr.press/v155/jain21a.html).
- [DeepMimic](https://arxiv.org/abs/1804.02717): motion/task objectives are not
  a guarantee of skill retention in this project.
- [PPO](https://arxiv.org/abs/1707.06347): clipped updates do not establish a
  strict final-policy KL bound or automatically identify causal action effects.
- [Marine dynamics](https://fossen.biz/html/marineCraftModel.html): separates
  inertia, hydrodynamic damping and hydrostatic restoring forces.
- [Fish turning analysis](https://cris.technion.ac.il/en/publications/a-hydrodynamical-analysis-of-fish-turning-manoeuvres/).
- [Potential-based shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
- [Time limits in RL](https://proceedings.mlr.press/v80/pardo18a.html).
- [Modulated locomotor controllers, including a swimming snake](https://arxiv.org/abs/1610.05182).

These references motivate designs, not the creature's demonstrated capability.
The original reviews preserve further source links that were not independently
checked. No physical/training study, implementation, adoption, push or merge
is authorized by this register. Keep collecting the user's next ideas.

## Arithmetic repair and v2 launch stop

The user authorized identical-input CPU/CUDA diagnosis, unchanged equations and
tolerances, full retained-skill checks and the same88cases, with no motor/chooser
training or optimizer. Fixed-state tracing identified implicit contraction and
CPU/CUDA cbrtf rounding; an isolated host-compatible GPU profile passes the fixed
algebra checks. The one-shot process then failed before physics while traversing
deep evidence snapshots. No dynamic parity/retention/turning case ran,0/88 and no
winner. Do not infer behavioral preservation or autonomous swimming from the
arithmetic pass. Keep the preserved sources, checkpoints and failed launch.
Nextidea: long-path-safe evidence plus a native final-sealed-tree inventory
preflight before a separately authorized unchanged comparison; no retry or
training is active. Read FROZEN_INTERFACE_COMPARISON_V2_RESULTS_2026-11-23.md.

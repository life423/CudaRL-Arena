# AI status — November 24, 2026

**Eight GPU PPO updates and all fixed skill checks finished in 10.78 seconds.**
The half-second feedback candidate stabilized five right 10° practice requests.
Its independent score stayed at 3/16, with all four short swims, 28 required
long-swim windows and 14 rests retained. The registered Godot app still plays
original motor 336. Optional steering checks are in Swim → Details.
Publishing the record and verifying Godot finished after a script repair,
49.82 seconds from the original launch. [Exercise, checkpoint and timing](POND_TURNING_HALF_SECOND_RESULTS_2026-11-24.md)

## Previous exercise with repeated heading requests

The original saved motor passed four small left/right turns from two
starting poses. A separate candidate completed 32 GPU PPO updates using
repeated heading requests. Practice increased from 138 to 274 starts per
training minute, with zero stabilized training completions. The candidate
retained all 28 required long-swim windows, four short swims and 14 rests.
It passed the same 3/16 independent turns. Reliable autonomous navigation
remains open. [Repeated-request exercise, timing and checkpoint](POND_TURNING_REPEATED_REQUESTS_RESULTS_2026-11-24.md)

## Previous exercise with a course-sensitive reward

**The course-sensitive PPO exercise completed 32 updates and qualified
three of 16 turns.** The 512 GPU worlds collected 1,048,576 transitions in
282.51 seconds of training-process time, or 3,712 transitions/s. All five
checkpoints retained 28 long-swim windows, four short swims and 14 rests.
The visible swimmer uses the original motor 336. Whole-pond navigation
remains unqualified. [Course-sensitive results and checkpoint](POND_TURNING_PPO_COURSE_RESULTS_2026-11-24.md)

## Second exercise with restored straight-swim feedback

**The second exercise completed all 32 native GPU PPO updates.** Restoring
sideways feedback brought the untrained candidate back to 28/28 required
long-swim windows. All five assessments, including update 32, retained those
windows, four short swims and 14 rests. Turning stayed at 3/16. The registered
Godot project continues to use the preserved motor 336. The trained checkpoint
is saved separately for research.
[Second exercise, measurements and evidence](POND_TURNING_PPO_V2_RESULTS_2026-11-23.md)

## First exercise with heading-only moving requests

**Food-free turning training reached update eight and stopped at its first
retention check.** In the first long straight swim the candidate traveled at
8.26 mm/s in the final window, but drifted 12.28 mm sideways in 60 s. The lane
limit was 2 mm. Before its first optimizer update, the candidate had already
drifted 11.78 mm in the same test. Its moving observation removed the horizontal
position error that the saved motor used for course correction.

The registered Godot project loads the original motor (336, with its learned
holding gain) on runtime 4. The saved actor retained four short swims, 14 rests
and all 28 previously passing long-swim windows on that runtime. All 36 CPU
cases and all eight continued CPU/CUDA trajectory comparisons passed.
[Shoreline verification](SHORELINE_RUNTIME_V4_RESULTS_2026-11-23.md)

### First candidate exercise

The standalone Release trainer `build/pt31/native/Release/pond_ppo.exe` ran
on the RTX 5070. It statically links the worktree's `pond_ppo_gpu_core.lib`
and CUDA runtime. The world-step kernel ran 16 blocks of 32 threads per control,
one thread per physical world. One shared actor served all 512 worlds. World
states, servo state, rollout buffers, networks and Adam moments resided on the GPU.
The project's tiled GEMM, batched actor/critic, GAE, PPO and Adam ran on CUDA.
The CPU coordinated launches, validated the simulation independently, and
wrote the evidence. Godot played one saved motor independently of training.

Each update collected 64 controls per world and made four passes with minibatches
of 4,096. Across eight updates this produced 262,144 transitions, 2,621,440
physical substeps and 256 actor plus 256 critic Adam calls. The first failed
retention check ended training. Candidates 0 through 8 and all eight rollout
batches are saved under `build/turn31`. The highest final-rollout approximate
KL was 0.000860 (limit 0.05).
[Candidate protocol](POND_TURNING_PPO_PROTOCOL_2026-11-23.md)

### Matched skill checks before and after PPO

| Skill | Candidate before PPO | Candidate after eight updates |
| --- | ---: | ---: |
| Short straight swims | 4/4 | 4/4 |
| Rest on command, including two five-minute holds | 14/14 | 14/14 |
| Long straight swims | 0/2 | 0/2 |
| Previously passing long-swim windows retained | 6/28 | 6/28 |
| Requested turns | 3/16 | 3/16 |

Both checkpoints passed the same turns from pose 0: −15°, −5° and +5°.
For the −30° and +30° requests from that pose, the trained candidate produced
7.71 and 7.80 mm/s of projected COM speed but missed the requested courses by
10.63° and 15.23° (limit 5°). Both +90° cases stopped at the depth limit.

The two 60-second straight swims used identical starts, requests, corrected
physics and candidate observation semantics. Forward speed below is the final
four-second mean projected COM speed. The other three measures are maxima
over the entire swim. Arrows go from update 0 to update 8.

| Starting pose | Lane error (mm) | Forward speed (mm/s) | Depth error (mm) | Tilt (degrees) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 11.778 → 12.281 | 8.225 → 8.256 | 0.461 → 0.470 | 8.925 → 8.916 |
| 3 | 9.435 → 9.930 | 8.227 → 8.258 | 0.455 → 0.464 | 9.477 → 9.464 |

Both candidates exceeded the 2 mm lane limit while meeting the final-window
speed, depth and tilt thresholds. PPO slightly increased sideways drift,
speed overshoot and depth error in both fixtures. Peak tilt fell slightly.
The skill pass counts stayed the same. The original motor on the corrected
runtime retained all 28 of its previously passing windows.

The official checkpoint-eight test stopped on the first long-swim failure
after five cases, as specified. Separate frozen-weight evaluation then
measured the 31 previously unattempted cases from the same allocation.
Together they used 36,685 controls, 366,850 substeps and 8.085 s of CPU
evaluation time. No case was repeated and the stop decision stands.
Raw before-and-after data are in `build/turn31/baseline0`,
`build/turn31/eval8` and `build/turn31/characterization8`.

The next motor experiment should restore the saved motor's lateral-error
input on straight requests, verify the inherited actor recovers its long-swim
performance and keep the turning objective learned. This targets an observed
input difference without changing body physics or extending this run.

### Timing

| Measured work | Duration |
| --- | --- |
| Eight-update training command | 3.287 s, about 79,747 world-controls/s including checkpoints |
| Rollout collection | 2.557 s, including 2.386 s of GPU physics. The remaining 0.172 s includes inference, sampling, other rollout kernels and waits. |
| PPO optimization interval | 0.141 s. The timer covers minibatches and the final KL audit. GAE ran before this interval. |
| Rollout readback and CPU reward aggregation | 0.026 s |
| Independent CPU checks | Shoreline 21.779 s. Frozen candidate baseline 8.199 s. |
| Fixed-input CPU/CUDA comparison | 16.001 s wall, including 13.413 s of GPU physics and 1.706 s of evidence work. |

Training wrote 94.88 MiB of binary batches and checkpoints. The training
driver measured no independent disk-write or inference-only interval. The CUDA
timings above overlap collection and must not be added to total elapsed time.
The 9.817 s training budget charge includes initialization and a five-second
checkpoint reserve. Raw timings are in `build/turn31/updates-1-8/updates.csv`.

Recorded training data showed zero contact controls, zero controls within
10 mm of the bed and zero stalls among 172,753 post-warmup moving controls
(stall threshold 3 mm/s). Moving episodes occupied 3–8 grid cells over the
observed portion of each episode. Whole-pond coverage, perimeter looping and
uninterrupted navigation require a separate navigation test.

### Earlier product and research history

**Latest product change: untimed Godot Swim installed and inspected.**
The saved motor and physical model are unchanged. A distinct interactive
reset_untimed_swim path removes episode completion while preserving requests,
body/servo/filter state, ordering/finite checks and original pond guards.
Legacy finite reset(false) and Rest/research scorers remain unchanged.
One [bounded product check](UNTIMED_SWIM_RESULTS_2026-11-23.md) passed:
509accepted/510attemptedcontrols,5100executedsubsteps,25.45s/188.907mm, then
the original supported-domain stop. First160poses match old CPU evidence
within4.998e-12 (2e-6limit); pause/resume preservedstate.333pure bookkeeping
and160zero-physics UI assertions pass; an independent raw-data audit passes.
Actual Godot4.4.1/updatedDLL was reopened in the correct live project, inspected
paused with Swim selected and measurements visible. Saved policy hashes match;
no optimizer or research case ran. The product check is closed. Repeated turns,
indefinite pond navigation and the blocked88-case comparison remain unverified.

**V2 CLOSED: comparison blocked before physics.**
The [v2 report](FROZEN_INTERFACE_COMPARISON_V2_RESULTS_2026-11-23.md) records
passing fixed-state arithmetic after isolated FMA/cbrtf rounding corrections,
then a0.096864s exit2 during evidence-directory enumeration. Nested paths reach
356characters and the executable lacks longPathAware. No probe-started/native
result was created:0 physical controls/substeps,0/88 interface cases, no winner.
Corrected-runtime original swimming/rest retention is unassessed, not passed.
The single-process allocation is closed; no restart or training followed.
Preserve197frozen causal files and all failed evidence. Nextprepare long-path
support plus a final-tree native inventory preflight before a separately
authorized unchanged gate/retention/comparison. No motor/chooser optimizer may run.
All10457protected original files passed the post-attempt hash check; saved
checkpoints, old sources and failed-run evidence are unchanged.
Godot4.4.1 in the correct worktree was refreshed and visually inspected paused
with the launch stop,0/88 and unassessed retention visible in Details.27parser,
26exporter and64rendered assertions passed across four sizes with0extra physics.

**Controlled comparison CLOSED at the CPU/CUDA gate.** The
[independently verified result](FROZEN_INTERFACE_COMPARISON_RESULTS_2026-11-23.md)
records214 controls/2140 substeps and0/88 main trajectories. At straight-case
control107, global-X angular-rate difference2.30819e-5rad/s exceeds the fixed
2e-5 tolerance despite bit-identical commands. No interface winner, policy
adoption or retraining exists. All10183 protected files are unchanged.
The next question is backend arithmetic agreement, not another PPO allocation.
The single attempt is closed; no retry or unused-budget reuse is authorized.

**Scope of the closed attempt:** the user approved the saved-motor
lane-following versus heading-only comparison. Its
[protocol](FROZEN_INTERFACE_COMPARISON_PROTOCOL_2026-11-23.md) fixes
88 trajectories plus a fresh CPU/CUDA gate, 35,200 controls / 352,000 substeps,
240 seconds native process and 512 MiB evidence. The gate failure closed that
allocation. Godot Details now exposes the verified stop and 0/88 main cases.
Root observed the actual worktree app paused with the updated Details;
34 parser and 60 rendered assertions passed with zero additional physics/PPO.

The [combined continuous-swimming plan](CONTINUOUS_SWIMMING_PLAN_2026-11-23.md)
records the intended route: a bounded matched interface comparison, conditional
shared-motor PPO, then a bank-aware chooser above a qualified motor. Backend
agreement must now be addressed before proposing a fresh comparison. No
retraining or future execution is authorized by that saved plan.

At the user's explicit request, the [living idea register](IDEA_REGISTER_2026-11-23.md)
preserves their goals, both PDF reviews, the third pasted review, critiques,
alternative designs and exact original source copies/hashes. Those proposals
are not blanket authorization to train, alter physics or publish. Prior
capability/results below remain unchanged.

The user-approved [requested-turn check](FROZEN_REQUEST_TURN_RESULTS_2026-11-23.md)
is closed and independently verified: 22 cases, 5,782 physical controls /
57,800 substeps, 13.7060028 seconds of process time. All four matched-command
CPU/CUDA prechecks passed. Small +/-5-degree requests maintain 7.75–7.83 mm/s
with depth below 0.67 mm and tilt below 9.36 degrees. +/-15-degree requests
slow and fail to settle accurately; +/-30-degree requests slow to 2.20–2.34 mm/s.
All eight +/-60/90-degree cases stop on the 20-degree tilt guard. There are no
hard physics faults, bank/depth terminations, PPO updates or actor replacements.
Proof: `06bf4c4dd44b597a9bdbae8f93e3708b5cf25e8ecc3bf7745973e2ea5ef3acfb`.
Godot receives the portable verified outcome; continuous pond swimming remains
unqualified. Sustained requested turns in the same motor are the next learning
need. Repeated incremental requests and whole-pond navigation are untested;
no further physical study or training is authorized by this completed check.

The [second-AI review handoff](AI_HANDOFF_2026-11-23.md) records the user's
bank/object-sensing reward idea and eventual bank climbing. The full-motor
requested-turn proposal is unexecuted source preparation: its standalone CUDA
target now compiles with warnings as errors, and its separate task passed
2,615 synthetic CPU assertions. No new full-actor CUDA numerical fixture,
physical rollout or Adam update has run. Its actor42 has no bank observations;
it cannot by itself deliver autonomous pond navigation. Fresh physical-gate
implementation and independent calibration/training/optimizer-ledger auditing
remain unfinished. The draft is not ready for physical execution or adoption.

The earlier [food-free continuous-swimming study](CONTINUOUS_POND_RESULTS_2026-11-23.md)
is closed and unqualified: 64 native GPU PPO updates / 632 Adam steps completed
without a physics fault, but the candidate passed 0/16 sixty-second starts.
It learned to slow/stall instead of making dependable moving turns. The
sixteen-second objective improved while sustained motion deteriorated.
All old native swimming/recovery gates and all 14 holding cases passed again.
Final independent evidence verification passed (qualified=false); no new policy
is adopted. Details shows the portable verified rejection alongside the active
saved swimmer. The original frozen sources/GUI remain in the causal snapshot.
No reserved 30-minute validation, ablation or viewer physical capture ran.
Total actual spend was 75,555,675 controls / 755,025,046 substeps and 232.991457 s
combined native process time. The old calm swimmer, camera, zoom and saves stay
available. The subsequent narrow turning check is reported above; no new trial
follows automatically.

The [design audit](CONTINUOUS_POND_DESIGN_AUDIT_2026-11-23.md) now holds the
unexecuted V2 proposal: early-failure reward, bounded target authority and
removed persistent heading/lane feedback remain unresolved. The next lesson
must establish requested left/right turns before whole-pond learning. This is
an audit decision, not a claim that turning is physically impossible.

The [turning-check preparation](FROZEN_REQUEST_TURN_PREPARATION_2026-11-23.md)
preserves the frozen inputs and zero-integration checks made before its explicit
approval. Actual execution stayed inside 7,360 controls / 120 seconds / 128 MiB;
its old allocation is closed. The running saved swimmer remains unchanged.

The [V2 source](CONTINUOUS_POND_V2_PROTOCOL_2026-11-23.md) was prepared
with 60-second credit,80 state/depth coefficients across all four existing
outputs and ordinary failed-episode guards for stopping/depth/tilt. Native
compilation, CUDA zero-integration checks and independent preparation proof
`d25c1d99212317160facc6822a6f445e7cfda0f606a1703b19ead4f2037d9bfa`
pass with zero physical/optimizer steps. All physical entry points reject
without a fresh protocol-bound user authorization. No such authorization exists,
and the authorization writer also refuses the explicit design-hold marker.

The product mission is an evolving reinforcement-learning creature that users
can interact with, using its training and new experience to adapt and thrive.
Swimming in currents is a core aquatic requirement alongside dependable
still-water movement. Motor research is a foundation for the interactive learner.

Latest November 23 presentation update: the aerial camera starts close to the
creature, with manual zoom and a separate side angle. The rejected whole-pond fit
is superseded. An opaque header provides measured 11.16:1 title and 5.53:1 status
contrast. Four rendered viewports pass 140 camera/UI assertions with zero physical
controls. These changes affect no policy or learned capability.

The later gait-v1 attempt is also closed: its closed-loop CPU/CUDA comparison
stopped after 960 controls / 9,600 substeps, before calibration or PPO. The first
distal-yaw-rate difference was 2.0951e-5 rad/s against the fixed 2e-5 limit;
legacy/identity CPU states were exact and physical step status remained valid.
The independently audited rejected archive is
`assets/ppo-four-axis-checkpoints/gait-pond-2026-11-22-v1-rejected`.
Goal-directed turning, corrected gait and qualified current swimming remain
outstanding; the proposed combined study has not run and no new actor is adopted.

This minimal-foundation worktree is the active motor research project. The
original checkout remains the preserved legacy pond. Native C++/CUDA owns PPO
and physical simulation; Godot presents native state. The intended architecture
remains one goal chooser and one reusable request-conditioned motor. The reduced
project currently exercises the motor with scheduled requests; it has no
integrated chooser or interactive learning worker.

The [corrected v2 current study](CURRENT_SWIM_V2_RESULTS_2026-11-22.md) is closed:
its fresh 26-fixture CPU/CUDA physics gate passed at the original tolerances,
then 32 native PPO updates improved original current-swimming cases from **4/16
to 8/16**. All eight remaining failures miss the 7.2 mm/s minimum ground speed;
five also exceed the 2 mm lane limit. Sixteen learned flow-residual means extend
the same frozen four-output motor. No critic was used. This is actual restricted
parameter learning, not a fully trainable body network or qualified current
competence. The final policy is rejected for adoption. Withheld current and
64-second current endurance suites did not run because 16/16 was not reached.
Independent saved-evidence/closure verification passes all 32 training batches,
32 current assessments, 33 selected-size policy states and 988 SHA bindings.
Final calm retention passes six holds, four short swims and the same 28/30
long-window keys. All 6,482 protected prior files and both causal seals are
unchanged; this audit pass validates the rejected result, not current competence.

The approved GPU benchmark selected **16,384 worlds**, about **500,294 control
steps/second** including evidence writes. Total spend was **89,566,320 controls /
895,663,200 substeps**, 32 PPO updates and 344 Adam steps; combined process time
was 199.267 seconds of the 900-second cap. Across 181 one-second telemetry samples,
GPU activity peaked at 100% but averaged 25.27%; evidence writes and high
register/local-memory use limit sustained use. This is not full GPU efficiency. All unused
allowances are closed. No retry, new study, production adoption or publication
is authorized by this result.

The earlier [v1 failure](CURRENT_SWIM_RESULTS_2026-11-22.md) remains preserved:
a tiny CUDA buoyancy-rounding drift exceeded the fixed agreement limit after
2,634 controls, before PPO. V2 explicitly versions that arithmetic correction
without changing the old calm path, body, torque caps or acceptance gates.
The viewer retains the qualified calm-water package and presents v2's latest
result separately in Details. Current holding, changing flow and disturbance
recovery remain unvalidated. The world's 7.5 mm/s configuration limit does not
establish motor ability; v2 study conditions were bounded to 2.5 mm/s.

## Preserved swimming baseline

### Visible gait feedback — November 22

The user wants the creature in its own persistent pond world and identified the
current "crooked V" swimming as a product problem. Saved calm-swim evidence
confirms a sustained learned bend: in both 60-second cases, from 4–60 seconds,
distal pitch averages -27.73 degrees, ranging from -37.94 to -16.92 degrees and
remaining negative throughout. The source is
`build/ppo/learned-amplitude-2026-11-22-v1/training-started/assessment-7/all-cpu-poses.csv`
(category `long`, both cases, controls 80–1200). The short-swim controller tracks
its bent target with about 0.32 degrees mean error, so poor servo tracking does
not explain it. Reset pitch is zero, and Godot displays native link quaternions
unchanged. The rectangular fin emphasizes the angular shape, but its axes match
the physical model.

The current reward constrains head posture, speed, depth and path; a sustained
distal-pitch bias is permitted. Existing acceptance results remain valid within
those declared gates; they do not establish natural gait quality. Review a
warm-started motor-learning change for balanced tail movement while retaining
travel and holding. Do not force every joint straight or hide the actual bend
with rendering. The exact zero-flow current residual cannot correct the calm
gait. This diagnosis used existing source/recordings only: no new simulation,
training, geometry, renderer or policy change occurred. The integrated pond and
interactive learner remain product work beyond the current inspector.

**Latest result:** the [learned-amplitude study](HOLD_LEARNED_AMPLITUDE_RESULTS_2026-11-22.md)
qualified calm holding from rest: 6/6 original 64-second cases, 6/6 withheld
64-second cases and both uninterrupted 300-second holds. The initial deterministic
policy passed 0/6. Seven native PPO updates learned two amplitude-distribution
parameters while the 336 body network stayed frozen. All four short swims and
the same 28 long windows remain byte-identical. Training is stopped. This is a
bounded research result, not moving-start braking, disturbance/current recovery,
a fully trainable shared network or automatic production adoption.

Canonical checkpoint 336 is unchanged, SHA-256
`810DBD593CE39475CDD6E95B709BEF3FDB34D50C9FCBBB2C345C7A8882A0BEF0`.
It passes four short swimming cases and 28 of 30 long-swimming windows. The two
startup-speed windows still fail, so this is not complete formal swimming
acceptance. The additional five-minute evaluation passed 148 of 150 windows,
with the same startup limitation. No holding actor has been accepted for the
ordinary experience. The separately authorized research inspector uses an
explicit, distinct v2 package for its frozen 41-input backbone and two learned
amplitude-distribution parameters. It must not reinterpret a historical
checkpoint as that package. The current viewer starts paused; Play advances
native physics with saved weights. This is not background learning or
production adoption.

## Completed evidence through November 21

| Study | Actual outcome | Interpretation |
|---|---|---|
| November 4 compatible 41-input holding and reward-scale comparisons | No accepted holds; scaled comparison retained swimming | Input migration and critic conditioning do not demonstrate holding. |
| November 4 hold from rest | 80 critic-only steps, eight PPO updates; 0/6 holds; 4/4 short swims and 28/30 long windows | Candidate rejected. Its report also records missed delivery of the pre-PPO calibration update. |
| Still-water feasibility | Neutral targets passed 6/6 original 64-second cases; selected small stroke also passed 6/6 | Holding is physically feasible in these starts. Neither command sequence is a learned policy or evidence of disturbance/current recovery. |
| Reset-20 calibration | 80 critic-only steps; stopped on the fixed hold-bias gate | No PPO or candidate evaluation. |
| Task-normalized critic | 80 critic-only steps; failed to beat the held-out constant comparator | No PPO or candidate evaluation. |
| Critic-free measured baseline | Phase A passed; Phase B completed eight actor updates, 0/6 holds, 0/4 short swims, 11/30 long windows | Rejected; removing critic fitting did not resolve learning/retention. |
| Paired hold durations | Three paired updates; A20 retained 4/4 short swims and 28/30 long windows; B20/60 dropped to 2/4 short swims; both 0/6 holds | Both arms stopped at B's swimming-regression gate. No winner or adoption. |
| Reward audit | Neutral scores better than the failed actor within 350 ms and over 64 seconds; drifting then stopping still incurs anchor cost | The tested reward ranks the desired behavior correctly. Exploration and policy credit assignment remain unresolved. |

The paired duration study used 156,960 controls / 1,569,600 substeps including
calibration, three actor updates and 240 Adam steps per arm, zero critic calls
or critic optimizer steps, and no retries. Each arm first breached the holding speed
limit at 0.35 seconds in all six final cases. All previous budgets are closed.

Primary evidence remains in its original locations; Git does not preserve ignored
`build/` data:

- [Hold-from-rest report](../assets/ppo-four-axis-checkpoints/rest-hold-336to344/REPORT.md)
- [Still-water feasibility](../build/diagnostics/still-water-feasibility-2026-11-21-v1/REPORT.md)
- [Reward audit](../build/diagnostics/reward-audit-2026-11-21-v1/REPORT.md)
- [Reset-20 calibration](../build/ppo/rest-hold-reset20-336to344-2026-11-21-v1/REPORT.md)
- [Task-normalized critic](../build/ppo/rest-hold-reset20-task-critic-336to344-2026-11-21-v1/REPORT.md)
- [Measured baseline Phase A](../build/ppo/rest-hold-reset20-measured-baseline-336to344-2026-11-21-v1/PHASE-A-REPORT.md)
- [Measured baseline Phase B](../build/ppo/rest-hold-reset20-measured-baseline-336to344-2026-11-21-phase-B-v1/result.json)
- [Paired duration results](../build/ppo/rest-hold-duration-pair-2026-11-21-v1/phase-B-started/rounds.jsonl)
- [Paired duration closed accounting](../build/ppo/rest-hold-duration-pair-2026-11-21-v1/phase-B-result.json)

## November 22 completed work

The user authorized a saved-data audit, current status documentation and one new
training exercise. The completed [saved-data audit](HOLD_SAVED_AUDIT_2026-11-22.md)
finds zero all-four-axis commands within +/-0.03 across 30,720 training holding
rows, with no sustained near-neutral sequence. Pooled horizon-dependent offsets
account for 60–67% of held-out holding residual energy. Independent gradients do
not demonstrate that shared-policy interference caused the swimming regression.
The audit advanced no physics and applied no optimizer updates.

The [new finite protocol](HOLD_CREDIT_PROTOCOL_2026-11-22.md) tested a single copy
of checkpoint 336 with task/pose/duration/age-conditioned baseline means and
recomputed fixed residual scales. It bundles centering and normalization; it
does not change exploration or isolate their causal effects. Maximum eight
updates, 188,640 controls maximum, retention assessment after each update, no
retries or adoption. Calibration passed. The exercise then completed two PPO
updates / 160 actor Adam steps and stopped when short swimming fell to 0/4.
Holding remained 0/6; the exact 28 passing long-window keys remained intact.
Total spend was 62,880 controls / 628,800 substeps; no physical/numerical fault
or critic update occurred. The candidate is rejected and the remaining budget
is closed. [Measured results](HOLD_CREDIT_RESULTS_2026-11-22.md) record gates,
counts, uncertainty, hashes and evidence locations.

The user then explicitly authorized addressing exploration and one additional
bounded exercise. The [exploration design](HOLD_EXPLORATION_DESIGN_2026-11-22.md)
uses a frozen 336 backbone and four fresh learned output coefficients, initially
one, plus positive stochastic amplitude retained through each true holding
segment. Swimming computation remains exact; no neutral behavior is installed
in the initial deterministic actor. Sampling and conditional AR probabilities
include the actual amplitude and previous mean. This is restricted policy
parameter learning, not a fully trainable shared network retaining skills.

The [completed exploration exercise](HOLD_EXPLORATION_RESULTS_2026-11-22.md)
passed calibration with 102 complete quiet one-/three-second segments and
86.8359% held-out low-speed occupancy. It then completed all 32 native CUDA PPO
updates and 2,560 new-parameter Adam steps. All 32 deterministic assessments
passed 0/6 holds; all four short swims and the same 28 long windows were retained
with byte-identical swimming traces. Final holding still travels at
2.4485–2.5892 mm/s during the final eight seconds, drifts 19.3023–20.4295 mm in
that window, and exceeds the depth limit in all six cases. Every acceptance
assessment used amplitude one and no sampling noise. The candidate is rejected.

The exact closed spend is 691,680 controls / 6,916,800 substeps, zero collection
or training critic calls/steps, no retry or extension. Independent saved-artifact
verification passes all 32 updates and 33 gain/Adam checkpoints. Native elapsed
accounting is 84.0649 combined seconds, below 120. The unchanged canonical
checkpoint, all 5,256 prior artifacts and all 76 frozen execution inputs are
preserved. Later authorized GUI/build changes are separate from the executed
source snapshots. No new experiment follows automatically.

Recorded exact-pose inspection and an explicit opt-in live frozen research
candidate are separately authorized GUI work. They do not adopt a failed actor
as a successful holding controller or perform background learning. GUI rendering,
interaction and package checks are reported separately. Nothing was published,
merged or deployed.

## Latest learned-amplitude qualification

After the failed short-segment exploration exercise, the user explicitly approved
learning the amplitude itself and matching training to full-duration holding.
The [new protocol](HOLD_LEARNED_AMPLITUDE_PROTOCOL_2026-11-22.md) samples one
Gaussian latent per complete 64-second episode and applies its sigmoid amplitude
to the same four frozen body-network outputs. Native PPO credits the whole
episode once. Its current-batch leave-one-out baseline excludes the episode's
own return and never conditions on amplitude. One independent global calibration
scale stays fixed. There is no additional per-control action noise or teacher.

Calibration produced 52 complete passing 64-second episodes out of 256, while
the initial deterministic amplitude 0.99 failed all six original holds. Update
7 learned deterministic amplitude **0.26211655** and first passed all six
original cases. PPO stopped immediately; six withheld cases and both 300-second
cases then passed without another update. The five-minute holds preserve the
same anchor through all 6,000 controls. Their worst peak speed is 0.091504 mm/s
and anchor error 0.385946 mm, within the unchanged 0.5 mm/s and 2 mm limits.

Exact spend: **2,734,560 controls / 27,345,600 substeps**, seven PPO updates,
112 fresh two-parameter Adam steps and zero collection/training critic calls or
updates. Combined native time was 43.3575 seconds. Independent verification
passes all seven updates, eight policy states, all acceptance gates, scalar
likelihoods, returns, LOO advantages and physical counters. All 81 frozen inputs
and canonical checkpoint 336 remain unchanged. Remaining study allowance is
closed; no extra training or production replacement follows automatically.

Only the amplitude-distribution parameters learned; swimming computation was
structurally frozen and its traces remained identical. This result supports
calm holding from rest in the declared cases. It does not prove arbitrary
arrival/braking, exact target recovery, currents or indefinite stability.
The saved qualified candidate may be inspected in the native research viewer;
frozen playback is not ongoing learning. Viewer validation is separate.

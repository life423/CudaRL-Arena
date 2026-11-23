# AI status — November 22, 2026

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

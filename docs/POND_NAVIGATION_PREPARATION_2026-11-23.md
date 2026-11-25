# Food-free pond navigation: preparation plan — November23,2026

Latest decision: the user subsequently authorized one bounded PPO **turning
motor** candidate study after the necessary physics checks. Its separate
`tools/pond-ppo` contract preserves the original saved actor and holding gain,
starts the candidate from all2532 inherited weights and interleaves fresh
holding/straight/turning practice. The candidate need not already turn
successfully before being trained to turn. That authorization supersedes this
document's earlier frozen-interface-first recommendation for the motor stage;
it does not authorize training the navigation chooser. The chooser described
below remains preparation for a measured, capable motor after retention and
moving-turn assessment. Do not reuse this proposal as another physical or
optimizer allocation.

Status: design preparation only. No navigation trainer, physics launch or
optimizer update is authorized by this document. The user's newest goal is
continuous swimming and **exploring the whole reachable pond** without food,
using learned turns while keeping the saved swimming/holding motor. The user's
explicit exploration choice supersedes the earlier idea that a safe perimeter
circuit alone would satisfy the immediate goal. Finish the bounded shoreline
runtime qualification first. All older interface/training attempts remain
closed; this plan does not reuse their budgets or results as a passing gate.

## Measuring speed, depth and exploration

November 24 independent review at `058fda5`: a synthetic evaluator reported
16/25 tiles and 0/9 interior tiles for repeated perimeter laps, versus 25/25
and 9/9 for the interior sweep. Drift, stalls and border jitter earned no
coverage. Its sampled static geometry found room for a 25 mm deep pose at
X = 400 mm, where an 80 mm deep pose failed. These checks exercise the metric
and static fit. The reachable denominator still needs connected swept-corridor
evidence and validated tracking at each depth. The depth adapter must supply
the held vertical reference used by `start_com_model.y`; changing only
`Request::position.y` leaves that motor observation unchanged.

The separate `codex/pond-exploration-evaluator` commit `a64bf10` reports
75.9% for its synthetic interior-and-shoals path, 25.5% for seven perimeter
laps and 0% for stalls and jitter. A long-absence revisit earned novelty
without adding a unique cell. Its fixed-posture map excludes 5.9% and 14.8%
of otherwise clear swim time in the first two traces. Keep that excluded time
visible until the connected reachability map is qualified. The committed
50 mm evaluator and the earlier 72 mm prototype are separate measurements.

The isolated depth-adapter patch `64acca9` on `codex/motor-depth-request`
reports eight focused observation passes and 280 unchanged legacy arrays.
It is opt-in for lane-following Swim. Its proposed 78–82 mm transitions have
yet to demonstrate sustained forward swimming and safe depth tracking.

Keep unique area separate from revisitation reward. Treat proposed cell size,
displacement, dwell, rearm time and reward scales as candidate settings.
Check the transport rule on efficient coasting as well as passive drift.
Start with the qualified depth and 8 mm/s request. Add depth and speed choices
after those controls pass independent tracking checks. This review authorizes
no chooser training or simulator changes.

The first motor lesson tests 8 mm/s at the starting depth. Later lessons should
interleave shallower and deeper depth requests, the original speed, faster
speeds and explicit Rest. The same motor receives a persistent 3-D request.
Depth changes will require demonstrated vertical control and safe whole-body
clearance below the surface and above the sloped bed.

**Faster swimming** means sustained motion of the mass-weighted center of mass
in SI units. For a requested straight or turning leg, use the final four-second
mean of horizontal COM velocity projected onto the requested course. Report
that projection, the magnitude of the mean horizontal velocity, course error,
depth error and tilt. Record the longest sub-3 mm/s stall after warmup, turn
delay, swept clearance and radius at each speed. Use the same held-out starts
and requests when comparing speeds. Tail motion and playback rate are
independent of the measured speed. In current, report speed over ground and
speed relative to local water separately. Compare resting on the same original
holding cases before promoting any faster motor.

**Coverage** is the fraction of the predeclared reachable water visited during
one uninterrupted evaluation. Divide the navigable footprint into body-scale
cells. Before training, freeze the depth band, allowed posture, cell areas and
the set connected to the evaluation starts through corridors with whole-body
clearance. Keep unresolved cells identifiable and report their area separately.
Let each reachable cell's weight be its reachable area. Divide the sum of
weights for cells with a qualified visit by the sum for all reachable cells.
Report cumulative and rolling-window coverage, interior and outer-water coverage,
and time spent repeating a perimeter circuit. For multiple requested depth bands,
also report band-specific coverage and unique reachable water volume. Counting
overlapping depth layers as additional surface area would inflate the score.

For the chooser, count a visit only after sustained COM entry and actual
displacement over multiple strokes. Rearm a cell after the creature has left
its neighborhood. A first qualified visit receives full novelty credit.
Revisit credit grows with the time spent away, with a cap below the first-visit
bonus. Neither waiting in place nor rapidly crossing a cell border rearms
credit. Measure unsafe impact, sustained grounding, prolonged stalls, and time
with insufficient room for a measured turn at the requested speed. Safe shallow
travel remains valuable. Keep exploration reward inactive on explicit Rest.
Freeze numerical reward weights and visit thresholds before the chooser trial.

The reference checkout's `src/envs/swimmer/command_motor.h` already defines
position, terminal velocity and heading in a persistent request. Its
`src/envs/swimmer/pond_selector.h` separates chooser observations from motor
observations and keeps native batch state on device. Those interface ideas
carry forward. The old food-specific scores and three-target encoding do not
measure food-free exploration. Earlier device profiles measured a different
physical kernel, so their throughput cannot predict the current pond's.

## Smallest defensible architecture

Keep the exact saved41-input/four-joint motor and its learned holding amplitude
immutable. Add one learned goal chooser above it, plus a training-only value
network. This gives the intended two behavioral policies, not a new network per
skill. Use the retained native C++/CUDA MLP, categorical PPO, GAE, Adam and
checkpoint primitives. Godot presents one native world and forwards user input;
it never rotates, translates, recenters or rescues the body.

Initially the chooser has three legal request actions: decrement the persistent
world heading byΔ, keep it, or increment it byΔ. Its categorical probabilities
come from the network. No if-near-bank action selection, hazard-dependent action
mask, nearest-safe-heading replacement, route, bounce or position correction is
permitted. A fixed action vocabulary defines requests; it does not move the
creature. The motor and physical servos must produce every actual turn.

The request persists between chooser decisions. Update the previous world-frame
intention, never the oscillating head direction. Fixed swimming terminal speed
starts at the inherited8mm/s; continuous-SWIM cannot choose HOLD or zero speed
to avoid its movement requirement. Rest remains a separate explicit request to
the same motor. Braking and seamless Swim→Rest switching remain separate skills.

## First decisive gate: can the frozen motor execute the requests?

The historical probe showed useful±5° requests, slow±30° requests and tilt stops
at±60/90°. It did not test repeated turns. The old controlled comparison ran
0/88 interface cases. A bank-aware chooser cannot repair an incapable frozen
motor merely by receiving more reward or more training updates.

Prepare a NEW frozen-motor comparison in the new shoreline runtime:

| Arm | Observation change | Preserved inputs/state |
| --- | --- | --- |
| Original lane interface | Current inherited observation | Persistent course/speed and original depth |
| Heading-only candidate | Remove horizontal WORLD-space lane error before body-frame rotation | Vertical/depth error, original depth target and all other41 inputs |

Neither interface is presumed to win. Include zero-change controls, isolated
small/medium left and right requests, repeated same-sign changes accumulating
90°,180° and360°, alternating turns, and return to straight travel. Vary saved
starts and gait phases. Preserve body, joint/filter/torque history, request depth
and elapsed counters through every change. Diagnostic request sequences are
tests, not a route to install in Godot.

Choose the action incrementΔ and chooser decision intervalτ from measured
response delay, tracking error and swept turning envelope. Candidate small
increments and0.5/1/2-second intervals are test proposals, not assumptions that
the motor can follow5° every second. Freeze the exact request programs, matching
rules, case selection and interval-selection rule before the new comparison.
Record every attempted case and failure. Do not restart the old88-case study.

Judge actual mass-weighted whole-body COM course/speed, depth, tilt, repeated-turn
completion, radius, delay and certified whole-body clearance. Head alignment,
commanded heading, summed yaw noise and reward are insufficient. Contact-free
moving turns must fit the actual pond at swimming depth; surface-water radii do
not describe available clearance above the rising bed. Log physical bank contact
separately so contact deflection cannot masquerade as learned steering.

If neither interface qualifies, stop chooser training preparation at that
boundary. Report the measured motor shortfall. Any follow-up must be a separately
reviewed bounded update of the SAME reusable motor with fresh on-policy swim/rest
practice and retention gates, not a turn-specific network or repeated chooser
training against an impossible interface. No motor update is started by this plan.

## Chooser observations and network

The smallest proposed implementation is a feed-forward categorical actor with
two32-unit tanh hidden layers and3 logits, plus a separate same-width scalar
critic. Reuse the native categorical sampler/loss; no Gaussian steering decoder,
recurrent controller or extra learned motion generator is needed initially.

Proposed physical/request observation block has73 floats, frozen only after the
interface gate:

| Features | Count | Purpose / scope |
| --- | --- | --- |
| Actual selected motor's41 observation values |41| Joint phase, velocity, orientation, held targets, rate/torque history and persistent request tracking |
| Terrain vertical gaps at8 directions and2 lookahead ranges |16| Cheap bank anticipation at actual COM depth; point probes, not whole-body or swept-turn certificates |
| Three existing certified per-link bed gaps |3| Current physical clearance from the native world query |
| Pond-normalized COM X/Z |2| Position in the actual world |
| Persistent world-heading sine/cosine |2| Preserve intention independently of the head |
| Actual horizontal COM-course sine/cosine plus validity |3| Distinguish movement from head yaw; course is invalid below the declared speed floor |
| Horizontal COM speed, request age and rolling transport speed |3| Response delay and real progress |
| Highest body clearance below water, SWIM context, stall-clock state |3| Surface scope, task intent and observable failure history |

Lookahead ranges must cover measured response time and body length; choose and
freeze them from qualification evidence. Probe heights come from the same native
terrain query as contact/rendering. There is no promise that a point ray predicts
a rotating, deforming body's swept clearance. Only the full physical query is
the acceptance certificate. All scaling, clipping of observation values,
startup-history validity and missing-data rejection need an explicit schema;
observation normalization must never alter body state or replace an action.

Append an observable visitation-memory block: per-cell ever-visited flags and
bounded last-visit ages for a small frozen geometry-based grid, plus total
coverage. With B cells the proposed actor input is74+2B, still a small native
MLP. Prefer this explicit deterministic memory over another curiosity network.
Its contents are observations/reward state, not a controller: do not convert
the least-visited cell into a scripted destination, steering direction or
hazard-dependent action mask. The chooser must learn where to go from PPO.

Freeze the grid and coverage denominator from the real terrain, body envelope,
depth/posture range and connected admissible space before looking at a trained
policy's results. Water-surface area is not fully swim-reachable at this depth.
Record feasible, infeasible and unresolved cells separately; do not shrink the
denominator around where the candidate happened to swim. A sampled feasible
pose is a feasibility witness, not proof that all omitted cells are unreachable.
Use meaningful body-scale cells and a declared sustained-visit criterion so COM
jitter along cell borders cannot collect coverage credit. Preserve the grid,
visited mask, visit ages and all histories through rollout cutoffs.

## Continuing reward and PPO data

Use useful safe transport AND discovery of less-recently-visited reachable area.
No food target or positive proximity bonus is needed. Proposed live utility is
bounded and nonnegative: multi-stroke COM transport×depth quality×tilt quality×
a weighted mixture of continued-motion, heading-tracking and visitation novelty
quality×a bounded effort factor. Keep nonzero travel credit during legitimate
turning. First visits should pay more than a capped recency-based revisit signal;
otherwise an established perimeter loop may be as profitable as discovery.
Novelty is earned only through sustained actual visits with useful transport,
never by changing an intention or waiting still for a recency timer to refill.
Integrate the complete formula in actual physical seconds and freeze all event/
time units, maximums and weights in the new protocol. Do not subtract unbounded
effort from a supposedly nonnegative utility. Independently measure coverage so
reward improvement alone cannot qualify exploration.

Transport uses rolling net COM displacement over multiple strokes alongside
speed/path length. This rejects tail wagging and tiny oscillations without
requiring a large whole-run displacement: valid pond circuits can return near
their start. Continuous stalling after a declared startup/window is task failure.
Its timer/history must be observable and recorded. Distinguish it from physical
failure, finite lesson completion and administrative rollout boundaries.

Do not initially add warning-zone clearance rewards or positive-only retreat
bonuses. If later needed, preregister a bounded signed potential term with the
same discount and correct terminal treatment; merely clipping/gating a bonus
does not prevent approach/retreat farming. No predictor change counts as an
actual safety improvement.

One chooser transition spansτ seconds of persistent-request motor execution;
the motor still evaluates every50ms and physics every5ms. Sample/log the actual
categorical action and its old probability once per chooser decision. Keep
decision duration, integrated reward and discount coherent. At a collection
cutoff preserve the world/request/filter state and bootstrap the final value;
physical/task failures are genuine terminals. Fresh training episodes may start
after genuine failures, with those failures counted. Evaluation and deployment
have no automatic reset, and a training reset never counts as continued survival.

The new chooser checkpoint format must describe its own shape/observation/action
schema, critic and Adam state, sampler RNG, persistent request and native runtime
identity. Existing `ppo_checkpoint.h` fixes old motor topologies; do not relabel
a73-input chooser as that historical motor format or overwrite original files.

## GPU prerequisite and bounded execution

The retained `creature_gpu` target supplies MLP/GEMM, Adam/loss and PPO/Gaussian
kernels. It does NOT supply a qualified CUDA version of the new pond runtime.
The current shoreline qualification is CPU-only. CUDARL_HD annotations or a
successful NVCC build are not dynamic CPU/CUDA agreement or retained behavior.
Root reports that the reusable CUDA13.2/sm120 core compiles successfully; that
build is separate from GPU physics, interface qualification and optimizer work.

Before PPO, prepare identical-input fixed-state and dynamic CPU/CUDA gates for
this exact runtime, envelope and observation schema, including the old failing
arithmetic states and genuine contact. Preserve current tolerances, coefficients,
filter/integration and checkpoint bytes. Diagnose the first intermediate mismatch;
do not copy an old arithmetic profile and assume it qualifies the new world.

The largest performance risk is the full-body terrain certificate: adaptive
search, per-thread stacks/local memory, up to131,072 query visits and warp
divergence near the bank. A large world count alone does not establish efficient
GPU use. Profile fixed-state query batches first, then a separately budgeted
small real-world workload. Measure end-to-end controls/second, query visits,
kernel/transfer/I/O time, occupancy/local-memory traffic and memory use across
representative flat-water, near-bank and contact cases. Count all dynamic
profiling controls. Never regain speed by substituting the old ellipse,
point-only contacts or an uncertified physics shortcut.

Only after physics, interface and data-path gates pass should a NEW finite PPO
pilot be proposed and frozen. A concrete sizing candidate is16updates×256worlds
×32chooser decisions, or131,072 decision transitions. Depending on qualifiedτ,
its motor-control/substep cost changes; derive exact totals before approval,
including profiling, prerequisites, retention and final evaluation. Choose the
world batch from measured GPU throughput within a fixed total work/wall-time/
evidence cap. This sizing candidate is not an allocated budget, convergence
promise or permission to run Adam. No silent continuation if it fails.

## Evidence, retention and the actual app

Proposed frozen-candidate qualification is16 one-minute pond starts, then eight
withheld uninterrupted30-minute runs with varied starts/headings. Before any
execution, freeze exact cases and finite per-backend counts, source/executable/
policy hashes, maximum runtime/bytes and failure handling. These endurance cases
alone cost307,200 controls/3,072,0005ms substeps per backend; include them in the
new allocation rather than silently adding them after training.

Candidate acceptance proposals from the reviews are mean COM transport≥6mm/s,
no complete post-startup8-second window below3mm/s, depth error≤1mm and tilt≤10°.
Supplied-turn tests additionally need meaningful achieved course change and
settling, repeated turns in BOTH directions, feasible whole-body radius/clearance
and declared response-delay limits. They remain proposals until reconciled with
the inherited skills and fixed in a protocol. Preserve the exact original
swim/rest/recovery pass keys, known startup misses and existing tolerances on the
actual deployment path. Frozen motor weights alone do not establish unchanged
behavior after observation or physics changes. No hidden old-policy bypass.

The user's latest selection requires broad exploration; a safe perimeter loop
alone cannot qualify. Proposed final exploration criteria include at least70%
area-weighted distinct coverage of the frozen certified reachable map over each
30-minute run, visits to all connected substantial pond regions, and continuing
discovery beyond the first perimeter circuit. These numbers/region definitions
remain proposals until the new protocol is frozen. Report coverage versus time,
occupancy/recency maps, time near the actual bank, inner/middle/outer reachable
region visits, moving turns in both directions and stall/contact duration.
Do not create a compulsory route or force uniform occupancy. Contacts, stalls,
physical scope stops, resets, rejected batches, actual optimizer steps and
skipped tests must all be reported. Evidence supports measured conditions and
duration, never mathematical infinite reliability.

Only an exact qualified motor/interface/chooser/runtime package may reach Godot.
Update one native request in the existing world without resetting the body,
preserve the accepted pond/zoom/aesthetics, inspect the actual correct project,
and label frozen inference versus real learning updates. Current swimming,
current adaptation, bank exit, limbs and land movement remain separate stages.

## How all three supplied reviews are used

- Review1: treat sustained coordinated turns as a physical motor capability;
  preserve old skills and use finite discriminating tests before more training.
- Review2: qualify repeated motion and long endurance, inspect reward/failure
  shortcuts, and judge the actual candidate rather than proxy reward or a bypass.
- Review3: first isolate the lane-versus-heading interface, keep persistent
  requests/depth, then prefer a learned chooser over a qualified frozen motor;
  fine-tune that same motor only if the interface evidence requires it.

All originals and unresolved alternatives remain in `docs/design-reviews/` and
[the full idea register](IDEA_REGISTER_2026-11-23.md). This proposal supersedes
no historical result and executes none of the suggestions embedded in a review.

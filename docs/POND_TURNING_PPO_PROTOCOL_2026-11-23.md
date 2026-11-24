# Native candidate turning PPO, first32updates — November23,2026

Preparation. The user explicitly authorized proceeding once necessary physics
checks pass, preserving the original motor while allowing a separate candidate
to learn turning. Already-successful turning is NOT a prerequisite for training
it. The desired next behavior is learned continuous WHOLE-POND EXPLORATION;
finishing this initial motor lesson alone does not establish that behavior.

## Prerequisites and immutable originals

The executable prerequisite is now runtime4/envelope2/query arithmetic2, with
CPU receipt `shoreline-product-v4-2026-11-23` and CUDA receipt
`shoreline-product-v4-cuda-2026-11-23`. The corrected clearance support/tangent
sign is shared by both backends. CUDA-only noinline compiler boundaries and
exact within-control POST-query reuse change neither the physical equations nor
the registered limits/tolerances. Closed v3 receipts do not admit this lesson.
Evidence root is `build/turn31`; original and failed-run evidence is preserved.

Require passing corrected shoreline CPU qualification AND fixed-input CPU/CUDA
parity before collection/Adam. Bind source, runtime, compiler profile, executable,
protocol and original package hashes before the run. Use the same shared
corrected0.52/0.34m native pond geometry/contact as Godot; no smaller training
boundary, scripted route, heading chosen from bank rules, forced body motion,
rescue or deployment resets. Keep physical limits and source reviews intact.

Original backbone336 SHA810dbd593ce39475cdd6e95b709bef3fdb34d50c9fcbbb2c345c7a8882a0bef0
and amplitude package SHA4b6400c4db65873bd346c3e0c4fed384e3a77f9e5174d0dcea5d74073b94616d
remain unchanged. Import the2532actor weights into a NEW candidate41→32→32→4
motor, plus an auxiliary41→32→32→1 critic. Preserve the learned holding
amplitude0.26211655139923096 as a fixed decoder parameter. There is one shared
motor across worlds, not512networks. No separate turn-specific behavioral brain.

## Exact first-round budget

| Item | Fixed allocation |
| --- | --- |
| Parallel GPU worlds |512|
| Fresh control samples per world/update |64|
| Fresh transitions/update |32768|
| PPO updates maximum |32|
| Training controls /5ms substeps maximum |1048576 /10485760|
| Optimizer minibatch / epochs |4096 /4|
| Actor / critic Adam calls per update |32 /32|
| Total Adam calls maximum |2048|
| Checkpoints and retention gates |update0baseline, then8/16/24/32|
| All training chunks combined |1200native seconds /4GiB evidence|

The64samples are50ms physical CONTROL intervals, each using ten5ms substeps.
The3.2s collection segment is not an episode deadline: body, filter, servo,
request and counters persist across segments. Training episodes last up to60s.
A valid task failure is a true terminal(done1,truncated0); a clean60s time
boundary is done1,truncated1 with final-observation critic bootstrap. Collection
cutoff is done0 with carried-state bootstrap. Training episode resets are
explicitly counted and are never deployed as learned navigation.

Save actor/critic, both Adam moment/age states, fixed logstd, original identities,
sampler serial, carried512world/task states and cumulative work/time/evidence.
Resume after each8-update checkpoint only with a typed receipt bound to that
exact candidate, parent, protocol and evaluator source. No per-process wall or
evidence budget resets. Save the last complete update on a budget stop; retain
any partial optimizer state/fault separately instead of overwriting last good.

## PPO and task definition

Use existing native GPU MLP, Gaussian sampling/log-probability, GAE, PPO clipped
loss, gradient clipping and Adam. Fixed unscaled latent Gaussian sigma0.10;
store original latents z and exact old logp. Commands are tanh(z) for moving
requests and tanh(frozen_gain*z) for Hold. Compute PPO probabilities in latent
space consistently. No Python/PyTorch training, inference or simulation.

Adam LR1e−5,beta0.9/0.999,epsilon1e−8,gradient norm cap1. PPOclip0.1,
gamma0.995,GAElambda0.95; four epochs with deterministic shuffled minibatches.
Fixed-variance entropy does not update a hidden variance policy. The run records
actual optimizer calls, mean reward, final whole-rollout KL and collection,
physics, transfer and optimizer timings. Stored rollout latents/old likelihoods,
collection weights and whole-update checkpoints permit later loss, clipping and
parameter-change analysis without more physics or Adam calls. Pre-clipping
gradient norms are not recorded in this first driver; the native norm cap remains1.
ApproximateKL>0.05, including the final full-rollout audit after the last Adam
call, nonfinite network/gradient/loss or malformed batch stops
training; do not tune hyperparameters after results within this allocation.

Fresh on-policy task mix:25%Hold,25%Straight,50%requested turns. Starts come
from the preserved safe fixtures. After2s warmup, a turn task changes its one
persistent request by±5/15/30/60/90degrees and preserves world/body/filter state.
These are training goals, not joint actions or a route deployed in the pond.
Hold retains original lane/depth/anchor inputs. Moving candidate observations
remove only horizontal WORLD-space lane error before body-frame rotation,
preserving depth and all other inputs. No interface winner is claimed from the
closed0/88comparison; this is an explicitly versioned candidate interface.

Reward actual whole-body COM transport toward the requested heading, sustained
speed, controlled depth/tilt and reasonable physical effort. Head alignment by
itself earns no turning success. Hold rewards fixed-anchor/low-speed behavior.
Accepted states beyond5mm depth error or30degree tilt are scored task failures;
they are not physical correction/clamps. Any invalid/unsafe physics, unknown
certificate or nonfinite collection rejects the batch BEFORE GAE/Adam and stops
the run with first PRE/candidate evidence, including all attempted substeps.

## Evaluation and stopping

Native CPU deployment-path evaluation at update0and8/16/24/32: all20original
retention cases(30400controls), plus16requested-turn cases of at most400controls
each. Per evaluation≤36800controls/368000substeps; total≤184000/1840000,
900s/1GiB, zero optimizer. Preserve all28expected long-window pass keys and all
14holds using original thresholds and known startup misses. Long straight
retention uses the explicitly separate clear-water compatibility fixture;
turn tests use the actual product pond. Never substitute the parent actor to
make candidate retention pass. Baseline0is measurement, not a requirement that
the original already turn successfully.

Turn cases:±5/15/30/60/90degrees from pose0and±30/60/90from pose3, after2s
warmup. Measure actual final4s COM projection≥6mm/s and course error≤5degrees,
trajectory depth≤1mm/tilt≤10degrees, radius/delay and minimum certified body
clearance. Report valid behavioral failures separately from numerical/physical
faults. A valid turn shortfall does not prevent continued turning training;
retention regression, hard physics fault or budget exhaustion stops continuation.

Log per-episode grid visitation, actualCOM speed/course, near-bank occupancy,
stall duration and contact. Do not combine training resets into an apparent
continuous exploration path or claim reachable-area coverage without a verified
denominator. No novelty reward or chooser optimization is smuggled into this
first MOTOR block. If turning improves with retained skills, proceed to a
separately specified PPO goal chooser for whole-pond exploration, using the same
motor; otherwise report the actual shortfall and preserved checkpoints.

## CPU/GPU use and Godot

GPU performs parallel world dynamics, batched inference, rollout math and PPO.
CPU owns orchestration, files and independent evaluation. Keep device buffers
resident; time kernel, transfer, synchronization and evidence costs. Optimize
an observed bottleneck only when it does not delay this first run or change
math/criteria. A busy percentage or compilation is not a training result.

Only a qualified physical runtime and separately qualified candidate may reach
the user-facing app. Update the existing Godot project/metadata; preserve its
pond art, zoom, opaque readable labels and save namespace. Verify the actual
Project Manager Run entry. State clearly whether the displayed motor is the
preserved reference or a qualified candidate, and that autonomous navigation
still requires measured continuous trials and coverage.

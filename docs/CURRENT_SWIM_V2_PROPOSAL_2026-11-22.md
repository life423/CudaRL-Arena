# Current swimming v2 — historical approved proposal

The user approved this corrected GPU study under its separate finite protocol.
It is now closed after 32 PPO updates, improving original current cases from
4/16 to 8/16 without qualification. See [the measured results](CURRENT_SWIM_V2_RESULTS_2026-11-22.md).
The following records the proposal before execution; it does not authorize
another run. The v1 allowance remained closed and was not reused.

## Observed v1 failure

The native result at
build/ppo/current-swim-2026-11-22-v1/physics-gate/run-started/result.json
records **2,634 attempted and completed controls / 26,340 physical substeps**,
18 completed cases, zero PPO updates and zero optimizer steps. All ten
zero-current comparison cases and eight actuated-current cases finished.

The first failed comparison occurred in zero-target-servo current case 0 at
control 77, or 3.85 simulated seconds. The four joint targets were zero through
the normal active servo; this was not a torque-free passive creature.

| Recorded quantity | CPU | CUDA |
| --- | ---: | ---: |
| World vertical velocity, m/s | 0 | 2.0137041190082527e-7 |
| Absolute backend difference, m/s | | 2.0137041190082527e-7 |
| Frozen translation agreement limit, m/s | | 2e-7 |

The difference exceeds the limit by about 0.6852%. Both integrations completed;
the stop was an agreement-gate failure, not permission to discard that fixture.
The 2,682-row state CSV includes 48 initialization rows and 2,634 actual
integration rows, matching native accounting.

The saved v1 evidence remains authoritative:

- State CSV SHA-256:
  4A8672B6BDF17551656717EE5B84FFE5EE8B4BAA8F25748F92323DD6F3E9A2BE.
- Native result SHA-256:
  C907A0E0EDC36C37149E89393C3658D4369E26253A00E2DC1989BC95FB54F2B5.
- Preserve the complete frozen source/executable/input manifest, raw states,
  first-failure binary, earlier zero-integration failures and terminal reports.

No calm/current baseline, calibration batch or PPO phase followed the failed
gate. This result does not establish current-swimming competence.

## Narrow proposed correction

In the moving-water implementation, make the binary32 rounding of displaced
water mass explicit before subtracting the stored link mass:

1. Round density times volume to binary32 with the same defined operation on CPU
   and CUDA.
2. Subtract the stored binary32 mass, then apply gravity with an explicit,
   reviewed arithmetic boundary.
3. Keep the force in its original generalized gravity/buoyancy location.

The existing density-times-volume-minus-mass expression can be contracted into
a fused operation on CUDA. For a body whose stored mass was already rounded
from the same nominal water density and volume, that contraction can retain a
small multiplication residual that separate binary32 operations cancel.
The pre-run assembly diagnosis identified a static vertical residual around
1.3829321022e-10 N. The observed CUDA-only vertical drift in the failed fixture
is consistent with that arithmetic asymmetry. This is a concrete diagnosis,
not proof that it is the only source of backend divergence.

Put the correction in a **new current-model v2 arithmetic contract**. Preserve
the frozen v1 files and proof bytes. The original Model 6 geometry, mass,
inertia, joints, water coefficients, +/-10 microN*m torque caps, joint/filter
limits, time step, solve and physical gates remain unchanged. Do not disable
fused arithmetic globally, alter buoyancy to tune behavior, clamp vertical
velocity or move the body.

The corrected moving-water path should still use the direct water-relative
energy quadrature prepared before v1 execution. Keep its independent reservoir
work, fluid-loss, solve-residual and energy-balance checks. Its exactly zero-flow
branch must continue to delegate the preserved legacy implementation, retaining
the qualified calm policy and its source contract.

## Why a new attempt needs approval

The [v1 protocol](CURRENT_SWIM_PROTOCOL_2026-11-22.md#return-calibration-and-data-integrity)
states: "A physical/numerical/data/contract fault stops the whole attempt".
Its [physical prerequisite](CURRENT_SWIM_PROTOCOL_2026-11-22.md#phases-and-stop-rules)
also states: "Failure closes the attempt with no learning."

[AGENTS.md](../AGENTS.md) requires a new finite frozen protocol before an
explicitly approved run and forbids implicit retries, tolerance relaxation or
unused-budget reuse. Changing the executable or model after this physical
failure cannot make a repeat part of v1. Zero-integration diagnosis and preparing
a v2 implementation do not reopen its physical gate or authorize PPO.

## Proposed v2 verification and execution

Before requesting execution, finish the v2 implementation and review:

- CPU and actual-CUDA zero-integration assembly checks for all preregistered
  current directions/poses, including the failing fixture's initial state.
  Check the defined neutral-buoyancy residual under the new arithmetic contract.
- Retain the original Galilean, water-relative dissipation, force-assembly,
  source/checkpoint, observation, Gaussian-likelihood/gradient, LOO baseline and
  exact-zero-flow policy checks. Preserve failed preparation identities.
- Validate all sixteen original and sixteen withheld initial configurations
  without advancing them. Record new executable/source hashes and explicit
  model, task, observation and checkpoint versions; reject v1/v2 confusion.
- Preserve canonical swimmer checkpoint 336, the full inherited optimizer,
  learned calm-holding parameters and the current working Godot package.

Only after approval and a new freeze:

1. Execute a fresh **3,760-control / 37,600-substep** gate with exactly the v1
   layout: ten zero-flow comparisons on legacy CPU/current CPU/current CUDA;
   eight actuated-flow and eight zero-target-servo current cases on CPU/CUDA.
2. Retain all existing tolerances, including 2e-7 m/s linear-velocity agreement,
   2e-5 angular/joint-rate agreement, 2e-5 model-unit position agreement,
   field-specific energy/assembly checks and all physical safety rejections.
   Any failure closes v2; do not extend or retune it.
3. Only if that gate passes, run the same finite calm/current baselines. If the
   original sixteen current cases already pass, skip sizing, calibration and PPO,
   proceed to final retention/withheld/endurance checks, and identify any
   qualification as inherited trained response.
4. Otherwise, perform the separately counted GPU sizing comparison below,
   fix the learning batch size and collect its independent calibration batch.
   After independent calibration verification, permit at most 32 native CUDA PPO
   updates at that selected size, with the same frozen backbone, sixteen new
   flow-conditioned means, persistent bounded episode coefficients and exact
   latent-action probabilities.
5. Apply the same calm-retention, sixteen withheld-current and two uninterrupted
   64-second current-endurance requirements before any qualification.
   Current holding, changing currents, arbitrary-water robustness and interactive
   learning remain separate capabilities.

## Proposed GPU sizing and performance evidence

The user explicitly wants the learner to make effective use of the GPU work
already invested in this project. Keep native C++/CUDA PPO and the existing
physics foundation. Choose the amount of parallel work from measured throughput
without treating GPU activity or power draw as proof of useful work.

Read-only hardware queries on November 22 identify an NVIDIA GeForce RTX 5070,
driver 621.01, compute capability 12.0 and **48 streaming multiprocessors**.
The CUDA query reports 12,792,365,056 bytes total GPU memory and
11,571,036,160 bytes free at that instant. An earlier NVIDIA-management snapshot
reported 12,227 MiB total and 8,938 MiB free; desktop workloads and the different
memory-accounting APIs make these snapshots unsuitable as a fixed allocation
promise. The C: volume had 84,589,481,984 bytes free, about 78.78 GiB.
The query helper launched **zero kernels, physical controls and optimizer
steps**. It changed no power, voltage, clock, cooling or system settings.

The frozen v1 learner planned 256 worlds and therefore only two 128-thread
physics blocks at a time. That setup cannot be assumed to occupy a 48-SM GPU.
It was never exercised by PPO because the prerequisite gate failed. The
serial CPU/CUDA correctness gate itself is not a throughput workload.

After a fresh physical gate passes and the baseline requires learning, compare
**2,048 / 4,096 / 8,192 / 16,384**
worlds with the same policy, original sixteen current cases, physics, recording
precision and GPU kernels. For each size, execute three independent 32-control
collections: one warmup and two timed repeats. There are no optimizer updates.
Every collection uses explicit fresh episode initialization, a fixed
preregistered seed/stream assignment and balanced sixteen-case coverage.
These partial episodes are throughput data, not current-swimming acceptance or
PPO training data. They never enter calibration or policy gradients.

All physical calls count, including warmup:

3 x 32 x (2,048 + 4,096 + 8,192 + 16,384)
= **2,949,120 controls / 29,491,200 substeps**.

Record CUDA-event times for observation/policy work and physical integration,
plus wall times for synchronization, device-to-host transfer, integrity checks,
serialization and complete evidence writes. The timed measurement ends only
after that collection's evidence is flushed; an asynchronous enqueue time is
not its runtime. Select the size with the highest aggregate end-to-end
controls/second across its two timed repeats; break an exact tie toward the
smaller size. Report repeat variability and each stage rather than advertising
only the fastest kernel interval. Use the selected size for calibration and
all PPO updates; do not resize or retune after learning begins.

The 32-control comparison is a bounded sizing estimate. Record complete
160-control calibration/training pipeline times as well, including optimizer
time, so a short comparison is not represented as full-episode performance.
Also sample GPU utilization, memory, power and temperature at approximately
one-second intervals during the actual run. Those observations help interpret
timings; they do not replace them. Reaching 250 W, filling VRAM or showing 100%
activity is not the objective. No power-limit or overclock change is proposed.

Preserve the PPO update structure when scaling:

- N worlds, where N is the selected size; N/16 independent episodes per case.
- One persistent sixteen-dimensional Gaussian draw per complete 160-control
  episode; the fixed Gaussian scale, coefficient map and body policy are
  unchanged.
- Same-case leave-one-out baseline excludes the episode's own return and divides
  the other-case sum by N/16 minus one, not the old fixed denominator 15.
- Four minibatches per epoch, each **N/4 complete episodes**, and at most four
  epochs. This retains at most sixteen Adam steps per update and **512 total**,
  rather than silently increasing steps when more worlds are added.
- Normalize every minibatch loss by its actual N/4 episode count. Keep the
  same fixed-calibration global residual RMS, PPO clip, KL stop, learning rate,
  gradient norm bound, seeded permutation and whole-episode credit assignment.

## Memory, evidence and disk limits

The current full training record uses **125 float32 values per physical
control**: observation 41, flow 3, backbone 4, joint latent 4, reward 1, metrics
16, PRE state 33 and POST state 23. At 160 controls this is 80,000 bytes per
world per batch, before case counters, policy latents and final world/task state.

| Worlds N | 128-thread physics blocks | Episodes per case | Episodes per minibatch | Full raw trace per batch | Calibration plus 32 training traces |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 16 | 128 | 512 | 156.25 MiB | 5.04 GiB |
| 4,096 | 32 | 256 | 1,024 | 312.50 MiB | 10.07 GiB |
| 8,192 | 64 | 512 | 2,048 | 625.00 MiB | 20.14 GiB |
| 16,384 | 128 | 1,024 | 4,096 | 1,250.00 MiB | 40.28 GiB |

The current World/Task/Count records occupy another 1,064 bytes per world;
latents/probabilities/advantages and neural buffers add smaller allocations.
Their actual allocation totals, local-memory/register usage and free VRAM must
be checked before execution. The table is a memory calculation, not measured
GPU occupancy or speed. Do not discard or downsample physical/PPO evidence to
make a larger configuration appear faster.

At the largest selected size, all calibration/training trace arrays total
40.2832 GiB. Including the proposed profiling traces gives **41.6565 GiB**,
before metadata, raw final states, checkpoints and filesystem overhead.
Require **at least 50 GiB free on the evidence volume before starting** and
preserve at least 8 GiB working reserve before each new collection/checkpoint.
If memory, disk or time preconditions cannot be met, stop before consuming that
phase; do not overwrite prior evidence or silently shrink the study.

## Proposed new maximum

The earlier draft's 256-world 1,466,480-control allowance cannot fund this larger
parallel learner and is superseded here **as an unexecuted proposal**, not
extended after a run. The larger bound must be reviewed and approved explicitly.

| New v2 component | Maximum controls |
| --- | ---: |
| Fresh physical current/compatibility gate | 3,760 |
| Three 32-control collections at each of four GPU sizes | 2,949,120 |
| Baseline full calm retention | 10,720 |
| Baseline sixteen current cases | 2,560 |
| Independent calibration, at most 16,384 x 160 | 2,621,440 |
| At most 32 PPO batches, at most 16,384 x 160 each | 83,886,080 |
| At most 32 original-current assessments, 16 x 160 each | 81,920 |
| Final full calm retention | 10,720 |
| One-shot withheld current suite | 2,560 |
| Two one-shot 64-second current endurance cases | 2,560 |
| **Proposed complete v2 cap, including gate and profiling** | **89,571,440** |

The corresponding proposed cap is **895,714,400 physical substeps**, 32 PPO
updates, 512 fresh mean-parameter Adam steps, zero critic updates and **900
combined active native/process seconds** across the physical gate, profiling,
baseline, calibration, learning and evaluations. Each actual failed call and
warmup counts. A passing inherited baseline skips unnecessary calibration/PPO.
Smaller selected sizes and early-stop paths consume less; unspent work closes
with the attempt. The 900-second cap may terminate the study before 32 updates;
no measured speed or completed-update count is promised in advance.

The **already spent v1 2,634 controls / 26,340 substeps remain additional**.
Do not subtract them from recorded lifetime work or describe them as free
preparation. If a full new GPU-sized v2 allowance were later approved and
consumed, the combined v1+v2 maximum would be **89,574,074 controls /
895,740,740 substeps**.

## Remaining uncertainty

The buoyancy correction directly addresses the observed vertical drift. It does
not establish that all backend differences disappear. The completed v1 driven
fixtures already reached 1.6219564713537693e-7 m/s translation disagreement and
1.5750527381896973e-5 angular/joint-rate disagreement, below but approaching their
unchanged limits. Other rounded matrix entries, nonlinear drag and coupled
solves may still accumulate differences in another direction or pose.

The entire fresh gate must therefore pass. Even a passing gate is only a
numerical/physical prerequisite; neither the correction nor the gate proves
learned current swimming. Keep the live GUI on the qualified calm policy and
report v1 as stopped until a separately counted, validated current candidate
exists.

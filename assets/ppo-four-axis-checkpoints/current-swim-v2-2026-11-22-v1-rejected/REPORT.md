# Corrected GPU current-swimming study — November 22, 2026

**The numerical correction passed. PPO learned useful current compensation,
but the final motor did not qualify: original cases improved from 4/16 to 8/16
after the full 32-update budget.** The qualified calm-water viewer policy is
unchanged. No current candidate is adopted, published or deployed.

The user explicitly approved the [corrected finite protocol](CURRENT_SWIM_V2_PROTOCOL_2026-11-22.md):
a fresh physics gate, measured 2,048–16,384-world GPU sizing, at most 32 PPO
updates, 900 combined process seconds and the stated evidence/storage cap.
The previous [v1 numerical failure](CURRENT_SWIM_RESULTS_2026-11-22.md) remains
closed. This run has its own version, identity, inputs and accounting.

## What changed and passed

V2 rounds displaced and represented mass consistently in the nonzero-flow
neutral-buoyancy calculation. This removes the CUDA FMA cancellation residual
that stopped v1. It does not change Model 6 mass, geometry, inertia, four
actuators, torque caps, target filtering or the old zero-flow computation.

All 26 fresh CPU/CUDA fixtures passed at the original tolerances: ten zero-flow
compatibility fixtures, eight driven-current fixtures and eight zero-target-servo
advection fixtures. The gate consumed 3,760 controls / 37,600 substeps. Its
largest root-position disagreement was 3.57628e-6 model units, below 2e-5.
No tolerance was relaxed and there was no physical retry.

The policy keeps checkpoint 336's 41-input body network and the qualified calm
holding parameters frozen. Native CUDA PPO learns sixteen flow-conditioned
trim/amplitude means in the same four-output motor. One Gaussian coefficient
vector is sampled for each complete eight-second episode and scored once.
There is no critic, separate ability network, scripted propulsion or Python
learning runtime. Only these sixteen means learn; this is not full-network PPO.

## Measured GPU use

The RTX 5070's actual workload was benchmarked with three 32-control prefixes
at each size, including warmup in physical accounting. Selection uses the two
timed prefixes and includes readback, validation and evidence writes.

| Parallel worlds | End-to-end control steps/second |
|---:|---:|
| 2,048 | 290,507 |
| 4,096 | 412,832 |
| 8,192 | 468,325 |
| **16,384 — selected** | **500,294** |

Across 181 one-second training telemetry samples, GPU activity peaked at 100%
but averaged 25.27%; board power peaked at 149.21 W. This is not full sustained
hardware use. The compiled physical kernel uses 255 registers and 7,376 local
bytes per thread, with 16.67% theoretical occupancy. The small sixteen-parameter
PPO optimizer is not the dominant workload. At the final collection, physical
kernels took about 1.134 seconds and evidence writing 3.513 seconds out of 5.387
seconds end to end. Evidence I/O and register/local-memory pressure are concrete
remaining optimization targets. No power, clock or driver setting was changed.
The hardware ran the study; its setup was not the cause of the behavioral misses.

## Learning result and remaining failure

Calibration collected 16,384 valid complete episodes, 1,024 per case, with a fixed
global advantage scale of 0.0600903493. Every training update then collected
another 16,384 complete eight-second episodes. Native PPO completed 32 updates
and 344 Adam steps; the fixed KL guard limited epochs within the 512-step cap.

Original cases are poses 0/3, four cardinal flow directions and 1/2 mm/s flow.
The target is 8 mm/s relative to the pond, with the original simultaneous speed,
lane, depth, posture, path, work and safety requirements.

| Outcome | Initial | Final |
|---|---:|---:|
| Complete original current cases | 4/16 | 8/16 |
| Mean current cost | 0.1119046764 | 0.0629716985 |
| Mean lane-cost component | 0.07899538 | 0.02995909 |
| Mean horizontal-speed cost | 0.02529755 | 0.02604215 |

All eight final failures miss the 7.2 mm/s minimum steady ground speed; five
also exceed the 2 mm lane limit. All sixteen pass the final depth, tilt,
path-angle, velocity-RMSE and work limits. The head learned five new complete
passes but lost one initial pass. Strong sideways 2 mm/s conditions never
passed at any assessment. The final 8/16 plateau began at update 26.

The improvement is real but uneven: total cost fell 43.73%, largely through a
62.1% reduction in lane cost, while speed cost worsened 2.94%. For pose 0 with
+X 2 mm/s flow, speed fell from 8.416 to 6.962 mm/s and that case lost its pass.
For +Z 2 mm/s, lane improved from about 6.79 to 4.68 mm while speed fell from
7.111 to 5.953 mm/s. The learned affine trim/gain head couples steering and
propulsion. These records do not prove inadequate physical authority or a hard
architectural ceiling.

Final calm retention includes all six original holds and all four short swims;
the same 28/30 long-window keys remain. Zero-flow policy computation and frozen
backbone/holding parameters are unchanged. The original two startup-window
misses remain failures. Withheld current and 64-second current endurance tests
were **not run**, because the required 16/16 original gate was never met. Their
aggregate pass counts are null, not behavioral failures or successful tests.

## Exact closed accounting

| Phase | Actual controls |
|---|---:|
| Fresh physical gate | 3,760 |
| Baseline calm retention | 10,720 |
| Baseline current cases | 2,560 |
| All GPU sizing prefixes | 2,949,120 |
| Calibration | 2,621,440 |
| 32 PPO collections | 83,886,080 |
| 32 post-update original assessments | 81,920 |
| Final calm retention | 10,720 |
| Withheld/current endurance | 0 |
| **Total** | **89,566,320** |

There were **895,663,200 physical substeps**, 32 PPO updates, 344 Adam steps and
zero critic updates/forward calls. Combined native phase time was 198.9430331
seconds; combined process time was **199.2670433 seconds**, below 900. The study
closed with gate `bounded_updates_exhausted_current_not_qualified`, no safety/data
fault, zero retries and `runtime_adopted=false`. All unused allowances are closed.

## Evidence, viewer and next decision

Raw evidence remains under `build/ppo/current-swim-v2-2026-11-22-v1`, including
all control rows, likelihoods, returns, leave-one-out advantages, checkpoints,
kernel timings and telemetry. The compact rejected archive preserves the policy
states, summaries, execution sources and SHA-bound references to the large arrays.
Independent full-row and closure verification **passed**, binding 988 SHA-256
inputs. It checked every saved training control, all 32 current assessments,
all 33 selected-size policy states, Gaussian probabilities, fixed calibration
scale, leave-one-out advantages, Adam history, calm retention and exact cumulative
accounting. This verifier pass records the rejected result accurately; it cannot
turn `qualified=false` into qualification. Offline verification executes no
physical or optimizer steps.

Both causal seals remain unchanged: 44 gate inputs and 113 full-study inputs.
All 6,482 protected prior files remain byte-identical. The active bridge was
restored exactly after private preview preparation; no new DLL or current package
was adopted. The [compact rejected archive](../assets/ppo-four-axis-checkpoints/current-swim-v2-2026-11-22-v1-rejected/README.md)
and ignored full run preserve the unsuccessful outcome and reproducibility.

Final independent report SHA-256:
`6F0C0D04C093B607D31D9AD0CF09C31C44B8180E8B4F27950F896555E32B8B70`.

The final checkpoint is
`39671E78702414A6DF146D24E320AE41DD8086C954FFD84E56BFDE14642B570C`.
The approved protocol SHA-256 is
`770342C47155E9314C7CFFE4529D2D58BAD7897588E0CD236DB6E869A635CF1E`.
The executed learner SHA-256 is
`359EEED59D6792FF62A7F5EC225E5C607EBB11559189D2A4D9009B4B28ED4D49`.

Godot Details reports the actual 4→8/16 result and 32 PPO updates while explicitly
retaining the saved calm-water policy. No current controls or unqualified actor
are exposed. Private V3 preview preparation is parked outside active bridge
source; five synthetic exporter rejections and rejection of the actual result
passed without creating a package. Frontend validation is separate from the
closed physical study: 22 parser checks and 18 rendered checks at each of four
viewport sizes passed with zero controls/substeps. Desktop, phone, tablet and
short-landscape screenshots were visually inspected. The enlarged-text,
screen-reader, real-touch and frame-pacing limits in the UI guide still apply.

The next design decision is a state/request-conditioned current extension inside
the same reusable motor, with simultaneous speed/lane success better represented
in the learning objective. The existing 41 inputs already contain relevant
request and movement errors, but this study's trainable affine head cannot
respond to them directly. Zero-initialized flow connections into the existing
network are one concrete option to review while preserving exact calm behavior.
More updates of this closed study are not authorized, and the evidence does not
justify promising that another run will solve it. Current holding, changing flow,
braking, disturbances, live interactive learning, growth and limbs remain outside
this study.

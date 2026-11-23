# Current swimming v1: stopped at the CPU/CUDA gate

The new current-swimming implementation and native PPO preparation are built,
but the single physical attempt **closed before current evaluation, calibration
or learning**. The current policy has zero PPO updates and zero Adam steps. It
is not qualified or installed in the viewer. The verified calm holding/swimming
policy remains unchanged.

The [preregistered protocol](CURRENT_SWIM_PROTOCOL_2026-11-22.md) allowed at most
1,466,480 controls / 14,664,800 physical substeps and 32 native PPO updates. The
physical agreement gate stopped after **2,634 controls / 26,340 substeps**. The
unused 1,463,846 controls are closed, not available for a retry or extension.

## What was prepared

The isolated versioned current model uses the existing Model 6 body, four joint
outputs, 1 Hz target filter and +/-10 microN*m torque limits. It evaluates drag
relative to imposed water velocity and separates relative-water dissipation from
the external work supplied by moving water. Its zero-flow branch delegates the
legacy model exactly; the twelve pinned calm controller sources remain unchanged.

The proposed learner keeps the same frozen 41-input body network and qualified
calm-holding parameters. It records three extra body-frame current observations
and learns sixteen flow-conditioned trim/amplitude means through native CUDA
PPO. One bounded coefficient vector is sampled per complete eight-second episode,
with its Gaussian likelihood scored once. This parameter extension is prepared,
not trained; the fresh mean parameters and optimizer remain at age zero.

Native zero-integration preparation passed Gaussian/vector likelihood, gradients,
finite differences, coefficient mapping, leave-one-out exclusion, CPU/CUDA body
inference, all sixteen withheld initial configurations and checkpoint round trips.
The numerical fixture contains 24 actual CUDA cases, with no physical or optimizer
steps. The final force-assembly zero checks passed on CPU and CUDA, including
Galilean force consistency, comoving zero relative-water dissipation, unchanged
zero-flow computation and the explicit finite float resistance-matrix bound.

## The physical stop

The gate completed all ten zero-flow compatibility fixtures and all eight
four-second driven-current fixtures. The next zero-target-servo advection
fixture stopped at control 77, 3.85 simulated seconds:

| Quantity | Recorded value |
| --- | ---: |
| Fixture | Pose 0, +X current 2.5 mm/s, zero joint targets |
| CPU vertical velocity | 0 m/s |
| CUDA vertical velocity | 2.0137041190082527e-7 m/s |
| Absolute CPU/CUDA disagreement | 0.2013704119 micrometres/s |
| Registered linear-velocity limit | 0.2 micrometres/s |
| Completed gate fixtures | 18 |
| Total completed/attempted physical controls | 2,634 / 2,634 |
| Physical substeps | 26,340 |

The comparison failed at the original limit. It was not rounded into a pass.
All completed physical steps remained valid; this was a backend-agreement failure,
not a torque/clearance/nonfinite/escape failure. The saved trace contains exactly
one exceeded pose/velocity comparison, at that endpoint. Other recorded pose,
velocity and cumulative energy comparisons pass their registered limits. Maximum
root-position difference was 1.0251998901367188e-5 model units, below 2e-5.

The zero-target-servo condition still uses the original active joint servo. It
must not be described as torque-free, unactuated or free-joint passive motion.
The gate is numerical/safety evidence, not an eight- or 64-second swimming test.

Independent assembly diagnostics identify a shared CUDA static vertical
gravity/buoyancy rounding force of 1.3829321022e-10 N. It is present in the legacy
and current assemblies, including the zero reference. This inherited numerical
offset is consistent with the tiny accumulated vertical drift in the advection
fixture. Correcting and revalidating that numerical cancellation requires a
separately versioned path; changing the physical agreement tolerance is not the
proposed remedy.

## Exact spend and unexecuted work

| Phase | Actual controls | Actual PPO updates |
| --- | ---: | ---: |
| Pure math, configuration, assembly and checkpoint preparation | 0 | 0 |
| Zero-flow compatibility: ten fixtures x 40 controls x three backends | 1,200 | 0 |
| Driven current: eight fixtures x 80 controls x CPU/CUDA | 1,280 | 0 |
| First zero-target-servo fixture: 77 controls x CPU/CUDA | 154 | 0 |
| **Total** | **2,634** | **0** |

Active native physical-gate time was 3.0577047 seconds; launch/process time was
3.1245796 seconds, below the fixed 900-second allowance. There was one physical
attempt and no retry. Calm baseline/final retention, sixteen original current
cases, calibration, current PPO, sixteen withheld current cases and both
64-second current endurance cases **never ran**. Their pass counts are null,
not 0/16 behavioral failures. No claim about learned current competence follows.

Two earlier zero-integration preparation failures are preserved separately:
aggregate float energy cancellation and an unrealistic bitwise nonzero-flow
CUDA resistance comparison, followed by identifying the shared static force.
They consumed zero physical controls/substeps and zero Adam steps. The final
protocol and source were fixed before the one physical gate. None of its physical
acceptance limits or budget caps changed after collection began.

## Independent verification and preservation

`tools/verify_current_swim.py --gate-failure` passed independent saved-evidence
verification. It validates native binary envelopes, both canonical checkpoint
hashes, the untouched zero-mean/age-zero initial policy, zero-assembly arithmetic,
all **2,682 CSV/binary state records** (2,634 control endpoints plus 48 resets),
the failed comparison and its first-failure snapshot, terminal counters and the
aggregate closed-study summary. Its 18 read inputs are SHA-256 bound in
`build/ppo/current-swim-2026-11-22-v1/independent-verification.json`.

The verifier's `passed=true` means the **failed and stopped result is accurately
recorded**. It does not change `qualified=false` or report any PPO progress.
Independent reading executes zero additional physical or optimizer steps.

The coordinator verified all 105 frozen execution inputs, all 6,217 protected
prior files and all twelve pinned calm controller sources unchanged. The full
ignored source/executable snapshot and raw evidence remain under
`build/ppo/current-swim-2026-11-22-v1`. A compact immutable copy is preserved in
[the rejected pre-PPO archive](../assets/ppo-four-axis-checkpoints/current-swim-2026-11-22-v1-rejected/README.md).

The user subsequently approved a separately versioned correction and bounded GPU
study. [Its v2 results](CURRENT_SWIM_V2_RESULTS_2026-11-22.md) supersede this report
as the current status: the fresh physics gate passed, then 32 PPO updates improved
4/16 to 8/16 original current cases without qualifying. V1 remains closed; its
unused allowance was not reused. Neither result authorizes another trial,
threshold relaxation, production adoption, merge or deployment.

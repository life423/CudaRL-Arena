# GPU clearance and turning results — November 24, 2026

Original motor 336 and its holding gain remain protected. Two separate native
CUDA PPO residual arms completed 32 updates each. Each collected 1,048,576
transitions and finished at 2/16 fixed turns against the original baseline's
3/16. Neither reached the declared improvement trigger. The visible swimmer
uses the original saved policy.

Straight and Rest bypass both residual heads. The evaluator verified exact
original coefficients and inactive residual behavior, then reused the archived
four short swims, 28 required long-swim windows and 14 rests. These arms ran
zero new retention controls. Their 16 turns were physically evaluated at each
eight-update checkpoint. The two completed allocations remain closed.
[Yaw-arm evidence](../assets/research-evidence/pond-residual-yaw-v1/evidence-manifest.json) · [Four-axis evidence](../assets/research-evidence/pond-residual-four-v1/evidence-manifest.json)

## Conservative contact rejection

The new opt-in path proves separation for each substep using the full body
envelopes, terrain curvature and gradient bounds, surface/wet-domain checks,
and root/joint motion bounds. The proof covers both the previous velocity and
the free-solved velocity. Uncertain cases use the existing full geometry and
contact solve. Physical equations, torque/filter limits and energy gates are
preserved. The exact query retains its 10 µm interval requirement.

Threshold-only collection reports a conservative clearance lower bound.
Unmeasured numerical minima carry an explicit validity tag and NaN. Full
quantitative inspection remains available for qualification and measurements
that require it.

The matched benchmark passed in 2.44067 seconds, including setup and readback.
It executed 2,056 GPU controls and eight CPU controls, totaling 20,640
physical substeps, with zero optimizer calls.

| Matched starting states | Worlds | Exact GPU time | Swept GPU time | Speed ratio |
| --- | ---: | ---: | ---: | ---: |
| Earlier batch 8 | 512 | 11.8413 ms | 14.7287 ms | 0.80× |
| Late batch 12 | 512 | 149.9369 ms | 2.0219 ms | 74.16× |
| Bank contact | 4 | 89.7159 ms | 90.7172 ms | 0.99× |

The comparison checked actual body motion, actuator requests, contact forces,
energy and conservative clearance classifications on CPU/CUDA. The late
control's speedup does not establish full PPO training throughput.
[Benchmark record](../assets/research-evidence/pond-broadphase-dynamics-v1/result.json)

## Full collector

The first attempt to check the complete 512-world, 64-control collection path
hit its four-second wall-clock cap before issuing its first physical control.
The watchdog recorded zero issued controls, zero completed controls/substeps
and zero optimizer calls. That allocation ended at the timeout.
[Timeout record](../assets/research-evidence/pond-collection-replay-v1/timeout.json)

The follow-up reads the frozen archive once, loads saved weights directly into
512-row inference networks, and skips 38 GPU allocations for training buffers.
It finished startup and uploads in 192.539 ms. The single v2 replay still hit
its four-second deadline inside first-control preparation, with zero controls
issued and zero optimizer updates. The original and follow-up allocations are
closed. [Timed replay](../assets/research-evidence/pond-collection-replay-v2/timeout.json)

A separate Driver API probe created a cold CUDA context and four events in
120.155 ms, then hit four seconds within `cuModuleLoad` for the exact extracted
36,119,504-byte observation module. It launched zero kernels. The same module
contains the physical step, reset, and initialization kernels, with 10.01 MB,
8.98 MB, and 8.98 MB of native instructions respectively. Native `sm_120`
code is present. The successful dynamics benchmark puts its optimized step
in a separate 4.45 MB module. The next focused change is to keep the large
geometry/physics routines out of the observation module and share the expensive
device routines across step and reset. A module load time and the existing
matched motion checks should qualify any replacement before collection.
[Module probe](../assets/research-evidence/pond-collection-replay-v2/module-probe.json)

The accelerated trainer remains gated. Full-rollout and PPO throughput are
still unmeasured.

## Build and integration checks

The current source passed 75 CPU foundation tests, 132 evaluator pure checks
and 43,405 shared-geometry assertions. Each native check completed within the
four-second execution cap. Godot's packaged library, saved motor and evidence
bindings passed their hash check. Hosted CI must complete for the exact pushed
commit before integration is verified.

Optional historical comparison tools require their preserved local snapshots
and hash-bound libraries. Restore those inputs to reproduce the old studies.
Normal CPU CI builds independently of those research artifacts.

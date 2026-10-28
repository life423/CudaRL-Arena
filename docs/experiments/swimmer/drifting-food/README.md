# Visible current and suspended food — October 28, 2026

The user requested food that moves with the current and a visible change in the water when Gentle current is enabled. The latest live pond now supplies both. Its existing saved swimming and food-selection networks remain frozen and preserved.

## Watch and interact

Choose **Tadpole Pond → Open pond**. Enable **Gentle current** to apply steady +X 0.5 cm/s flow to the swimmer and suspended food. Surface streaks and a bounded pool of underwater visual tracers make the flow visible. Disable the switch to stop imposed advection; the explanatory layer fades out. Switching preserves the body, food locations, elapsed time, counters and graph. **Show untrained policy** retains the water/food mode while restarting the same layout for comparison.

The creature reacts to changing food positions using its existing neural networks. These responses are live inference, not online weight updates. The graph reports actual session collection. Real PPO practice, retention testing and candidate promotion remain later work in the [creature plan](../../../CREATURE_PLAN.md).

## Physical model and preserved baseline

Food is modelled as neutrally buoyant, overdamped suspended particles. Interior velocity equals the imposed fluid velocity, converted from metres/second to model units with `deep::scale = 0.04`; the fixed step remains 50 ms. Depth stays fixed under horizontal flow. At the existing conservative feeding-clearance envelope, outward movement is removed and tangential movement is allowed. This simplified no-flux contact boundary keeps food underwater and reachable; it is not CFD or a mass/inertia/sinking model for pellets. Food does not respawn at a boundary. Spawn separation/relocation constraints still apply on collection; naturally advecting particles can become closer together.

The swimmer's articulated dynamics, drag, internal joint torques and body parameters are unchanged. Food advection is opt-in: old `deep_pond::advance()` and `Batch::step()` callers retain stationary food by default. Existing calm trajectories and training saves remain compatible. Passive target movement occurs before measuring the step's body-progress shaping term. No new PPO update, reward constant, observation column or network weight is introduced.

The optional CUDA module has a separate versioned food-control extension; original v1 creation/stepping and current APIs remain intact, with unchanged `deep_pond::State` layout. An older module can still play its supported baseline. Automatic food-drift requests continue on CPU without resetting state if that capability is missing; strict CUDA requires rebuilding. Actual backend identity remains visible under AI details. The local updated module runs this mode on CUDA.

`pond_reliability --drifting-food` enables the new environment explicitly. Default trials remain stationary-food references. `cases.csv` includes the mode, alongside `food-mode.json`; diagnostic snapshot version is 3. These snapshots remain diagnostics, not exact resumable CUDA challenge exports.

## Frozen-agent response

The [protocol](PROTOCOL.md) preceded the trials. Four fresh seeds on CPU and CUDA compare five-minute fixed and drifting food under the same steady flow; two seeds/backend also run fifteen-minute drifting sessions. All **20 sessions / 140 simulated minutes** pass the declared provisional gate: zero confirmed immobility, invalid states/food or skipped steps, and no food gap above twenty seconds. The largest observed food gap is 13.25 s.

| Backend | Food | Duration per case | Cases | Mean food/min | Largest food gap |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | Fixed | 5 min | 4 | 17.30 | 9.70 s |
| CPU | Drifting | 5 min | 4 | 17.75 | 10.00 s |
| CPU | Drifting | 15 min | 2 | 17.90 | 9.70 s |
| CUDA | Fixed | 5 min | 4 | 17.50 | 7.80 s |
| CUDA | Drifting | 5 min | 4 | 18.20 | 8.10 s |
| CUDA | Drifting | 15 min | 2 | 17.90 | 13.25 s |

These are small-sample descriptive results, not statistical superiority or general robustness. Food motion changes the task and can bring food toward the swimmer; a higher collection rate is not proof that the policy learned a better strategy. CPU/GPU long trajectories may diverge, so compare conditions within each backend. Historical harder bank/current failures remain separate.

## Visual implementation and checks

The surface's advected patterns and 96 pooled visual tracers read the actual native current vector. Their phase integrates physical simulation time, with metres-to-model conversion, and stays continuous across current toggles. They add no forces, food, movement commands or training data. Visual tracers recycle outside the water's visible interior; actual food uses the native bounded advection above. Existing ambient water motion remains in Calm. The shaders use Godot's documented [world-space vertex coordinates](https://docs.godotengine.org/en/4.4/tutorials/shaders/shader_reference/spatial_shader.html) for tracer transport.

The full release build and **173/173 native tests** pass with GPU required and zero skips. New tests cover exact interior speed, preserved depth/RNG counters, prolonged boundary contact, zero-current retention and CPU/CUDA drift agreement. The C++-only portable suite passes; current and preserved old modules each pass **3,309 assertions**, including a no-reset food-capability fallback.

Rendered CUDA flow checks pass **69/69**, and final CPU headless flow checks pass **53/53**, covering visible on/off state, correct speed/phase, actual simulation-clock coupling, food movement/depth, no hidden training, and native state unchanged by rendering. The count difference is sixteen screenshot checks. Overview/close views at 1280×720, 768×1024, 390×844 and 844×390 were inspected under `build/drifting-food-checks/visual/`. Existing CUDA headless regressions pass: current control 47, navigation 125, latest agent 28, untrained comparison 191, current recording 32, recovery recording 57, menu 107.

Review: interaction clarity, hierarchy, consistency, feedback and responsiveness 9/10; accessibility and performance verification 8/10 (internal heuristics). Priorities remain real assistive/touch checks before platform expansion and measured rendering/training contention once practice is connected. Windows screen-control automation remains unavailable because its helper encounters the unrelated locked D: drive; these checks use Godot's own rendered Windows runs and synthetic input. Source hashes are preserved, and no ongoing learner is launched. This work is local on `codex/pond-ui-navigation`; it is not a new pushed/main integration.

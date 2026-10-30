# CudaRL · Minimal creature foundation

One physical creature, one native world and one Godot inspection scene. This is the reduced foundation for a creature that will learn through our own C++/CUDA PPO system. **It currently runs diagnostic actuator inputs, not a learned motor.**

The product direction remains capable swimming → player interaction and measured real-time adaptation → growth and limbs → leaving water, walking and eventual manipulation. Those later stages are not implemented. [Master direction](docs/CREATURE_PLAN.md) · [Ownership/request interface](docs/CREATURE_CONTROLLER_INTERFACE.md) · [Current checkpoint](docs/SESSION_CHECKPOINT.md)

## Start here

Build in this worktree, then open `godot-project/project.godot` in Godot4.4 and run its main scene:

```powershell
cmake --preset windows
cmake --build --preset release
ctest --preset release
```

The scene starts paused. Choose joint-target or direct-torque mode, set the four axis commands, then select **Single step** or **Run**. Pause before changing modes. Reset is an explicit diagnostic reset, never a deployed recovery mechanism. The viewer displays the physical links returned by native code.

**Save state / Load state** use only `user://foundation/diagnostic-state.bin`, a new versioned/checksummed physical-state format. Old pond learner files are neither loaded nor migrated. The Godot application name remains `CudaRL Arena` to avoid repurposing existing user data.

The CUDA build requires CMake3.24+, Visual Studio2022 C++ tools and a CUDA toolkit. The default architecture is120; override `CMAKE_CUDA_ARCHITECTURES` for another GPU. Set `CUDARL_GODOT_CPP_ROOT` and `CUDARL_GODOT_CPP_LIBRARY` to matching Godot4.4 C++ bindings. This worktree can reuse the original checkout's existing `external/godot-cpp` dependency.

For headless CPU validation without Godot or CUDA build dependencies:

```powershell
cmake --preset cpu
cmake --build --preset cpu
ctest --preset cpu
```

The visible single-world diagnostic currently uses native CPU physics. CUDA runs the same physical stepping checks and retains the numerical primitives for later parallel learning. Godot rendering uses the available graphics hardware; no cloud service is involved.

## Execution path

```text
godot-project/main.tscn → main.gd → CreatureLab
                                   ↓
                         simulation::PhysicalWorld
                                   ↓
                       physics/dynamics + contact
                                   ↓
                          versioned physical state
```

| Location | Concrete responsibility |
| --- | --- |
| `src/physics/` | Articulated body, shared physical equations and complete body/fin contact envelope |
| `src/simulation/` | Physical state, current, bounded actuator interface/servo, stepping and save/load |
| `src/godot/` | One world owner exposed as `CreatureLab`; Godot value conversion |
| `godot-project/` | One scene, diagnostic controls, stable camera and native-pose mesh |
| `src/nn/`, `src/rl/`, `src/cuda/` | Retained FP32 MLP/backprop/Adam, PPO/GAE/Gaussian math and GPU utilities; no active curriculum or learner session |
| `tests/` | Focused geometry/contact, torque/state, CPU/CUDA and numerical checks |

Physics is independent of both Godot and the learning algorithm. The body still has two universal joints with two controlled axes each. Actuator torque remains limited to±10µN·m. The optional deterministic servo requests joint angles and supplies limited internal joint torques every5ms; it adds no body forces or kinematic stabilization. Water uses the existing reduced-order force model, not CFD.

The intended behavioral system has **one learned goal chooser and one reusable movement motor** receiving persistent destination, desired terminal velocity and heading requests. Critics support training. Neither behavioral policy is implemented in this reduced runtime yet. The previous task/policy/session designs are retired rather than treated as specifications that must be imitated.

## The unresolved physical gate

Full-envelope geometry and contact-force queries are aligned, but the retained contact response is **not fully validated**. In the preserved study, a rolled head lost contact and recontacted while rotating, exceeding the registered0.5mm allowance near0.185s at all three tested timesteps. Geometry133/133 and2,400 executed CPU/CUDA comparisons passed; complete support and final timestep/free-water acceptance did not finish. Both new PPO arms have zero updates.

The warning stays visible in this viewer. Cleanup does not retune that response, weaken the allowance or authorize another experiment. Diagnostic movement, servo regulation and passing component checks do not establish dependable swimming, braking or holding.

## Baseline and removed paths

Original commit: `e5e9bd1db7e442715ac8ba3a5f5030c0ede26430`.
Annotated tag: `baseline/pre-minimal-foundation-2026-10-30`.
Original branch: `codex/autonomous-pond-practice`.
This branch: `codex/minimal-foundation`.

The original checkout remains available for comparison, including its pond, old benches and ignored experiment results. Its `build/foundation-baseline/2026-10-30/` contains separate verified backups of25 personal saves and artifact manifests. Tags do not back up ignored files.

This active tree removes the historical demo scenes, inheritance ladders, duplicated motor benches, old task/policy variants, session orchestration, training tools, release website and their obsolete tests. Historical source and detailed protocols remain recoverable at the baseline tag; results and personal data in the original checkout are untouched. The deployed website/download is not changed by this branch.

No replacement learning, extended research budget, production adoption, merge or deployment is part of this cleanup. The next physical correction and motor-learning protocol require a separate bounded decision.

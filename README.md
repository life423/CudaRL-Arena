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

For local GPU verification, set `$env:CUDARL_REQUIRE_GPU=1` before `ctest --preset release`; unavailable hardware then fails instead of silently skipping GPU checks. The CPU-only preset remains independent of CUDA.

Godot checks are separate from CTest. With `Godot_console.exe` on PATH:

```powershell
Godot_console.exe --headless --path godot-project --script ../tests/godot/foundation_contract_smoke.gd
Godot_console.exe --path godot-project --script ../tests/godot/foundation_ui_smoke.gd -- --foundation-layout-only
```

The first uses frozen backend stubs to check startup and error handling without physical steps or files. The second renders the real extension at desktop, tablet and phone sizes without stepping or saving. Omitting `--foundation-layout-only` additionally tests exactly two diagnostic control intervals. Close the editor before rebuilding its loaded extension; `CUDARL_GODOT_OUTPUT_DIR` can redirect a build for isolated verification.

`foundation_persistence_smoke.gd` tests real Save/Load buttons, write failures and corrupt files with zero physical steps. Run it only in a disposable copy of the project and sibling `tests/godot/` directory. In that copy, set `application/config/use_custom_user_dir=true` and `application/config/custom_user_dir_name` to a unique test directory, then pass the exact same value as `--audit-user-dir=<directory>` after `--`. The script checks the resolved user directory and refuses to replace a pre-existing save. Do not point this test at personal data.

The latest audit passed 58 GPU-required native checks, 17 CPU-only checks, 66 frozen-backend checks, 43 rendered checks and 19 isolated persistence checks. A final layout check against the installed extension also passed 35 checks without stepping or saving. These are component/integration checks; the contact and learned-movement gates below remain unresolved. Detailed evidence is in [the checkpoint](docs/SESSION_CHECKPOINT.md).

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

The subsequent bounded contact-response comparison is isolated behind `CUDARL_BUILD_CONTACT_PROBE=ON`; it is excluded from CTest and the live world. Its prepare/run modes record immutable fixtures and enforce a single-run physical-call cap. The current registration and actual results belong in the checkpoint. Building this tool does not enable the candidate in Godot, migrate saves or start learning.

That one-shot comparison stopped after 22,649 attempts and rejected the candidate: both responses failed the bent, pitched and rolled primary cases. A legacy CPU/CUDA damping-branch mismatch ended the run before free-water and passivity checks. The candidate is retained only as reproducible experiment code, not enabled behavior. Do not resume it or treat the remaining budget as authorization for another run.

The subsequent approved calibrated-compliance versus hard-contact pilot is also isolated (`CUDARL_BUILD_CONTACT_PILOT`, default OFF). It finished at 1,626/2,024 attempted advances. Calibrated compliance passed all its short cases; hard contact failed its stricter nonlinear geometry consistency gate and was quarantined. Neither is enabled in Godot. This was a 0.25-second screen, not sustained support, motor training or production readiness. Its protocol, source snapshot and results are preserved under ignored `build/contact-pilot/2026-10-30-v1/`; see the latest checkpoint before further work.

The separately authorized fixed-A validation is opt-in (`CUDARL_BUILD_CONTACT_VALIDATION`, default OFF). Its independent continuous-law accuracy gate precedes longer support and fresh acceptance tests. Its own ceiling is18,200 total CPU/CUDA physical attempts, with no retries or budget reuse. See [the frozen protocol](docs/CONTACT_VALIDATION_PROTOCOL.md); this tool cannot start PPO or change the live contact law.

## Baseline and removed paths

Original commit: `e5e9bd1db7e442715ac8ba3a5f5030c0ede26430`.
Annotated tag: `baseline/pre-minimal-foundation-2026-10-30`.
Original branch: `codex/autonomous-pond-practice`.
This branch: `codex/minimal-foundation`.

The original checkout remains available for comparison, including its pond, old benches and ignored experiment results. Its `build/foundation-baseline/2026-10-30/` contains separate verified backups of25 personal saves and artifact manifests. Tags do not back up ignored files.

This active tree removes the historical demo scenes, inheritance ladders, duplicated motor benches, old task/policy variants, session orchestration, training tools, release website and their obsolete tests. Historical source and detailed protocols remain recoverable at the baseline tag; results and personal data in the original checkout are untouched. The deployed website/download is not changed by this branch.

No replacement learning, extended research budget, production adoption, merge or deployment is part of this cleanup. The next physical correction and motor-learning protocol require a separate bounded decision.

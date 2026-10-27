# CudaRL-Arena

[![CI](https://github.com/life423/CudaRL-Arena/actions/workflows/ci.yml/badge.svg)](https://github.com/life423/CudaRL-Arena/actions/workflows/ci.yml)

Reinforcement learning environments written as hand-written CUDA kernels and driven directly from the Godot game engine, with no Python anywhere in the loop.

Godot loads a C++ GDExtension DLL that has the CUDA code linked in. GDScript calls straight into it. There's no server, no socket, no interpreter, and no ML framework. Every kernel and learning algorithm in this repo is written from scratch.

> **Status: three demonstrations in Godot—CartPole, eight-car racing and a 3D tadpole pond—with hand-written C++/CUDA reinforcement learning.** The deep pond plays a validated PPO food selector paired with a retained swimming network. Headless training runs across 1,024 worlds with full resume; fresh evaluation measures 17.56 versus 15.36 food/minute for the original controller. The shallow pond still offers live learning. See the [food-selection experiment](docs/experiments/swimmer/deep-pond-food-choice/README.md), [deep pond guide](docs/deep_pond.md) and [eight-car write-up](docs/racing_traffic.md).

![Three-food tadpole pond with native CUDA training statistics and a measured food-per-minute learning curve](docs/images/tadpole-learning.png)

![Eight cars on a shared circuit: clean laps, contact counters and measured GPU training throughput](docs/images/racing-traffic.png)

---

## Why no Python?

The usual RL stack goes Python → framework → bindings → GPU, with the environment often on the CPU in a separate process. This project tests how far you get by deleting all of that:

- **Environments are CUDA kernels.** Each GPU thread steps one environment.
- **The game engine is the frontend.** Godot calls the kernels in-process through GDExtension and renders the results.
- **Everything is from scratch.** There's no cuBLAS, cuDNN, PyTorch, NumPy, or Gymnasium in the library. Neural networks, backpropagation, and RL algorithms are our own C++/CUDA. cuBLAS appears in exactly one place: the benchmark tool, as the yardstick our kernels are measured against.

---

## What works today

| | Status |
|---|---|
| Godot 4.4 loads the CUDA-linked GDExtension DLL | ✅ Working (Windows, RTX 5070, CUDA 13.2) |
| GPU probe: device query plus a test kernel launch, so a missing driver or wrong architecture is reported instead of crashing | ✅ Working |
| **PPO on CartPole, entirely on the GPU:** batched environment kernel, rollouts, GAE with truncation bootstrapping, Feistel-shuffled minibatches, clipped-surrogate gradient, Adam | ✅ Solves in ~0.35 s on every seed tried ([write-up](docs/ppo_cartpole.md)) |
| **Live CartPole view in Godot** (`cartpole/cartpole.tscn`): carts balancing with the newest policy, learning curve, training stats, speed control | ✅ Working |
| **PPO on a racing track, entirely on the GPU:** Gaussian policy for steering and throttle, bicycle-model car, nine raycasts, same GAE and clipped objective | ✅ Working, 3 seeds reach 3.8–3.9 laps in 50 s ([write-up](docs/ppo_racing.md)) |
| **Eight-car racing in Godot**: harder circuit, nearby-car sensing, swept collisions, separated respawns, brake lights, clean laps and lap times | ✅ Working ([validation](docs/racing_traffic.md)) |
| **Startup demo menu:** CartPole and Racing with playback/fresh training; Tadpole Pond with live learning | ✅ Working; Back to demos/Escape returns cleanly ([menu](docs/demo_menu.md), [pond](docs/three_food_learning.md)) |
| **Three-food tadpole pond:** 1,024 CUDA swimmers, continuous PPO, shared articulated C++/CUDA physics and a 3D Godot view | ✅ Working; policy sees all three foods and commands two tail torques. Fixed-trial food/min graph, regional respawns and folded-pose recovery checks ([evidence and limits](docs/three_food_learning.md)) |
| **Deep pond:** free 3D swimming, three food depths and a learned food selector over a retained motor network | ✅ Working; shared-score categorical PPO improves collection by 14.3% across 512 fresh one-minute trials; 128 new five-minute trials and a recorded-stall regression checked separately. The viewer plays the validated saved controller ([results and limits](docs/experiments/swimmer/deep-pond-food-choice/README.md)) |
| **Native headless training:** fixed-step CUDA execution without rendering, full optimizer/world resume | ✅ Working; the menu guide continues selector training into a separate candidate save. Earlier motor training remains available ([usage](docs/pond_food_selection.md)) |
| **End-to-end racing performance:** C++/CUDA benchmark, phase timing and retained reference paths | ✅ Two measured passes at 2,048 agents on RTX 5070: [traffic execution](docs/racing_performance.md) +12.7%, then [optimization graphs](docs/racing_optimizer.md) +14.2%, each with exact reference checks |
| Live curve-fitting view: watch a network trained on the GPU fit a function in real time | ✅ Working |
| 2-agent maze game in Godot; rules in plain C++, walls drawn from the same data used for collision | ✅ Working (CPU, random agents) |
| Hand-written MLP kernels (tiled GEMM, fused bias + activation, deterministic split-K), backprop, Adam with gradient clipping | ✅ Working |
| **Tensor-core bf16 GEMM** (`mma.sync`, `ldmatrix`, `cp.async` pipeline, swizzled shared memory), five versions, each profiled | ✅ Working, 91–96% of cuBLAS ([worklog](docs/gemm_worklog.md)) |
| Verification: fp64 finite-difference gradient checks, CPU/GPU references, tensor-core checks, bit-identical determinism, end-to-end learning | ✅ Working; see the test executable for the current count |
| Benchmark: roofline, cuBLAS comparison, tuning sweep, environment throughput, PPO solve time | ✅ Working (`nn_bench`) |

The DLL links the CUDA runtime statically. It runs on any machine with an NVIDIA driver, with no CUDA DLLs shipped alongside it.

---

## Results

Measured on an RTX 5070 (48 SMs, driver 620.02, CUDA 13.2) with `nn_bench`, best of 5 timing trials. Every GEMM result is checked against cuBLAS before it's timed.

### PPO on CartPole

Defaults: 1,024 environments × 32 steps per update, 4 epochs × 8 minibatches. The full design and tuning story is in the **[PPO write-up](docs/ppo_cartpole.md)**.

| | |
|---|---|
| Time to solve (mean length ≥ 475), 5 seeds | **0.33–0.38 s**, 19–21 updates, ~650K env steps |
| Stays solved afterwards (worst mean over the next 40 updates) | 457–489 |
| Training throughput, rollout + optimization | ~2 M env steps/s |
| Raw CartPole stepping, 262K environments | **15.9 G steps/s**, 143× a single-threaded CPU loop |

### Tensor-core GEMM (bf16 × bf16 → fp32)

Each version adds one optimization. The full story, with Nsight Compute evidence and the tuning sweep, is in the **[GEMM worklog](docs/gemm_worklog.md)**.

| 4096³ | TFLOP/s | % of cuBLAS | Tensor pipe active (ncu) |
|---|---|---|---|
| cuBLAS (`cublasGemmEx`) | 69.6 | 100% | |
| v0 WMMA | 23.6 | 34% | 38% |
| v1 `mma.sync` + `ldmatrix` | 42.4 | 61% | 63% |
| v2 + `cp.async` 3-stage pipeline | 52.5 | 75% | 77% |
| v3 + XOR-swizzled shared memory | 58.2 | 84% | 85% |
| **v4 tuned** (2 stages for occupancy, grouped block order) | **63.4** | **91%** | **94%** |

At 8192³, v4 reaches **96% of cuBLAS**. The swizzle cuts shared-memory bank conflicts 600× (101 M → 0.17 M), exactly the 4-way conflict predicted from the address layout. The tensor-core ceiling, measured with a register-only `mma.sync` loop, is 70.0 TFLOP/s.

### fp32 GEMM and training

The fp32 peak is **33.0 TFLOP/s**, from SMs × fp32 lanes × clock. Memory bandwidth is **543 GB/s measured** (672 GB/s spec). Here cuBLAS runs in pedantic fp32 mode (no TF32 tensor cores), so both sides use the same arithmetic.

| GEMM | m × n × k | Ours (GFLOP/s) | % of roofline | cuBLAS | % of cuBLAS |
|---|---|---|---|---|---|
| Square | 4096 × 4096 × 4096 | 7,012 | 21% | 22,034 | 32% |
| MLP forward, Y = XWᵀ | 65536 × 64 × 64 | 6,278 | 72% | 12,452 | 50% |
| MLP backward, dX = dZ·W | 65536 × 64 × 64 | 5,841 | 67% | 11,379 | 51% |
| MLP backward, dW = dZᵀX, split-K | 64 × 64 × 65536 | 3,138 | 36% | 13,866 | 23% |
| … same, split-K off | 64 × 64 × 65536 | 52 | 1% | | |

What these show:
- **Split-K is a 60× win on the weight-gradient GEMM.** Without it, a 64×64 output is a single tile, so one thread block on one of 48 SMs loops over all 65,536 samples. Slicing the batch across SMs and summing the slices in a fixed order keeps results bit-identical.
- **The MLP-shaped GEMMs reach 67–72% of the roofline.** At 16 FLOP/byte they're memory-bound, so bandwidth, not arithmetic, is the right yardstick.
- **The large square fp32 GEMM is at 21% of fp32 peak and 32% of cuBLAS.** It's a straightforward 64×64-tile SIMT kernel; the tensor-core path above is where the optimization effort went.

A full PPO-sized training step (8 → 64 → 64 → 2: forward, backward, Adam with clipping):

| Batch | ms / step | Samples / s |
|---|---|---|
| 1,024 | 0.19 | 5.3 M |
| 4,096 | 0.21 | 19.5 M |
| 16,384 | 0.31 | 52.8 M |
| 65,536 | 1.10 | 59.6 M |

The flat 0.2 ms up to batch 4,096 is **kernel-launch overhead**: about two dozen small kernels per step. CUDA Graphs or kernel fusion would remove most of it. For scale, the plain-loop CPU reference, which is single-threaded and built for correctness rather than speed, takes 15.2 ms for one step at batch 4,096.

### PPO on the original racing track

Historical single-car measurements: 2,048 independent cars × 64 steps per update, 4 epochs × 8 minibatches, 100 updates, with the earlier 10-input policy. The current default is an interacting eight-car race with a 30-input policy. Its rules and separate measured results are in **[Eight-car racing](docs/racing_traffic.md)**. The table below describes the **[original regression circuit](docs/ppo_racing.md)**.

| Seed | Laps per episode, update 1 | Laps per episode, update 100 | Share of final episodes that crashed |
|---|---|---|---|
| 1 | 0.03 | 3.91 | 0.0% |
| 2 | 0.03 | 3.81 | 8.3% |
| 3 | 0.03 | 3.91 | 0.8% |

Episodes end at 50 s, and about four laps is the most a car can cover at top speed. The learning is to avoid crashing until the time limit. Training runs at 3.2–3.6 M environment steps per second, about 38 ms per update.

![Racing: showcase cars driving the newest policy, and the laps-per-episode curve](docs/images/racing-learning.png)

---

## Architecture

```
┌──────────────────────────────────────────┐
│ Godot 4.4 (GDScript)                     │  rendering, input, scenes
│   godot-project/                         │
└───────────────┬──────────────────────────┘
                │ GDExtension: in-process C++ calls
┌───────────────▼──────────────────────────┐
│ Bindings: CudaDevice, MazeArena          │  Godot types ↔ plain C++;
│   src/godot/                             │  nothing throws into Godot
└───────────────┬──────────────────────────┘
                │ direct calls, same DLL
┌───────────────▼──────────────────────────┐
│ Core (no Godot dependency)               │  CUDA runtime helpers,
│   src/cuda/   src/envs/                  │  environments
└──────────────────────────────────────────┘
```

Some design points:
- **The core doesn't know about Godot.** Environments and CUDA code build as a separate static library, `cudarl_core`, so they can be tested and benchmarked without the engine.
- **`device_ptr<T>`** (`src/cuda/cuda_memory.h`) is a move-only RAII owner for device allocations. CUDA failures throw a `cuda_error` that carries the expression, file, and line.
- **Failures stay inside the extension.** A missing GPU or driver comes back as data (`CudaDevice.describe()`) instead of an exception that takes down the editor.

---

## Build (Windows)

**Requirements** (tested versions in parentheses)
- NVIDIA GPU + driver (RTX 5070, driver 620.02)
- CUDA Toolkit **12.8+** for Blackwell `sm_120` (13.2). The toolkit can't be newer than what your driver supports; check the `CUDA Version` shown in `nvidia-smi`.
- Visual Studio 2022 Build Tools with the C++ workload (MSVC 19.44)
- CMake 3.24+ (3.31, bundled with VS Build Tools)
- Godot 4.4 (4.4.1)

**godot-cpp:** fetch it with `git submodule update --init`. The build also needs its prebuilt `bin/libgodot-cpp.windows.template_release.x86_64.lib`. See Known limitations.

**Build the extension** from the repo root:

```bash
cmake --preset windows
```

```bash
cmake --build --preset release
```

This writes `godot-project/addons/cudarl/bin/cudarl.dll` and builds the tests and benchmark in `build/Release/`. The CUDA architecture defaults to `120`; pass `-DCMAKE_CUDA_ARCHITECTURES=<yours>` to the configure step for other GPUs. The benchmark also needs the toolkit's cuBLAS component.

**Run the tests and the benchmark:**

```bash
ctest --preset release
```

```bash
build/Release/nn_bench.exe
```

**Measure the complete eight-car training pipeline** without the renderer or
display throttle:

```powershell
build/Release/racing_bench.exe envs=2048 warmup=10 updates=40 repeats=5 compare_graphs=1
```

This alternates stream and CUDA graph optimization on the
same seed and training budget, with distributed traffic in both paths.
It reports wall-clock update distributions and
agent-steps/s. The separate `profile=1` mode records CUDA event phase spans;
`simulation=1` drives both paths with an identical fixed action tape, and
`evaluate=1` measures a 60-second showcase after training. This tool links only
the project's core and the CUDA runtime. See the [performance study](docs/racing_performance.md)
and [optimizer study](docs/racing_optimizer.md) for methodology, numerical
checks, measurements and limitations. Graph optimization is enabled by default;
`graphs=0` retains stream execution. Use `compare=1 graphs=0` to reproduce the
earlier traffic-only comparison.

`cudarl_tests.exe <filter>` runs only the tests whose names contain `<filter>`, for example `gemm`. On a machine without an NVIDIA GPU, the GPU tests report `SKIP` and the CPU tests still run; set `CUDARL_REQUIRE_GPU=1` to make a missing GPU a failure instead.

**Without Godot:** configure with `-DCUDARL_BUILD_GODOT=OFF` (or use the `ci` preset) to build only the core library, tests, and benchmark. godot-cpp isn't needed then.

**Run the live PPO view** (close Godot before rebuilding; a running game locks the DLL):

```bash
godot --path godot-project
```

The default scene is the three-demo menu. For CartPole and Racing, **Watch trained** loads a bundled actor policy and runs the showcase on the CPU without starting training. **Train from scratch** opens a paused, untrained run; press **Start learning** to train on CUDA. For **Tadpole Pond**, choose **Watch it learn**: a fresh 1,024-swimmer run begins automatically after a short untrained view and keeps learning until the scene closes. Its controls include Pause/Resume, paced/maximum training speed and Start over. **Back to demos** or Escape returns to the menu and stops that session. See [menu validation and policy regeneration](docs/demo_menu.md) and [three-food learning](docs/three_food_learning.md). Direct scenes remain available by appending `res://cartpole/cartpole.tscn`, `res://racing/racing.tscn`, `res://pond/pond.tscn`, `res://fit/fit.tscn`, or `res://maze/maze.tscn` to the command.

### Continuous integration

Every push and pull request runs [`.github/workflows/ci.yml`](.github/workflows/ci.yml) on a GitHub-hosted Windows runner. It installs the CUDA 13.2 compiler and runtime from NVIDIA's network installer, builds the core, tests, and benchmark with **warnings as errors**, and runs the test suite. Hosted runners have no GPU, so CI proves the build and the CPU reference, while the GPU tests (`SKIP` there) run locally on the RTX 5070. A self-hosted GPU runner is planned.

---

## GDScript API (current)

```gdscript
var device := CudaDevice.new()      # probes the GPU once
device.is_available()               # bool
device.describe()                   # one-line summary, or the reason CUDA is unavailable
device.get_info()                   # { available, name, compute_capability, multiprocessors, memory_bytes, error }

var lab := PpoLab.new()
lab.start(num_envs, seed, showcase_count)   # initializes paused; false + get_error() without CUDA
lab.set_paused(false)              # start learning on the background GPU thread
lab.set_updates_per_second(2.0)     # throttle so learning is watchable (<= 0: full speed)
var carts = lab.step_showcase(true) # call at 50 Hz: [x, theta, steps, failed] per showcase cart
lab.get_stats()                     # latest update: mean episode length, entropy, KL, clip fraction, ...
lab.get_learning_curve()            # PackedVector2Array of (env steps, mean episode length)

var trainer := NetworkTrainer.new()
trainer.configure(target, width, depth, activation, learning_rate)   # false + get_error() without CUDA
var s = trainer.train(steps)        # { step, loss, ms_per_step }, runs `steps` Adam steps on the GPU
var ys = trainer.predict(xs)        # PackedFloat32Array in, PackedFloat32Array out
trainer.get_target_names()          # the target functions configure() accepts, by index

var maze := MazeArena.new()
maze.set_walls(cells)               # PackedVector2Array of wall cells on the 10×10 board
maze.get_goal()                     # Vector2i
var r = maze.step([human_action, ai_action])   # 0 Up, 1 Right, 2 Down, 3 Left
# r[i] = { "state": Vector2, "reward": float, "done": bool }
maze.reset()
```

---

## Project layout

```
CMakeLists.txt        one build for the core library and the Godot extension
CMakePresets.json     `windows`/`release` for local builds, `ci` for builds without Godot
.github/workflows/    CI: build with warnings as errors, run tests
src/core/             host/device shared code: counter-based RNG, bf16 conversion
src/cuda/             CUDA runtime helpers: RAII device memory, errors, device probe
src/nn/               MLP: fp32 and bf16 tensor-core GEMM kernels, GPU forward/backward, Adam, losses, CPU reference
src/rl/               PPO: trainers (discrete and Gaussian), kernels (sampling, GAE, shuffle, losses), shared math
src/envs/cartpole/    CartPole-v1 as a batched CUDA environment (host/device physics)
src/envs/racing/      racing track: bicycle-model car, raycasts, fixed track, as a batched CUDA environment
src/demos/            live demos driven from Godot (PPO sessions + showcases, FunctionFit)
src/envs/maze/        maze rules (plain C++)
src/godot/            GDExtension classes and registration
tests/                hand-written test harness and test suites (run via ctest)
tools/nn_bench.cpp    roofline and cuBLAS benchmark (`tc`, `fp32`, `train`, `tune`, `profile`)
godot-project/        Godot project: addons/cudarl/ (extension), cartpole/ (live PPO), racing/ (live driving), fit/, maze/
docs/                 GEMM worklog, PPO write-ups (CartPole, racing), screenshots
external/godot-cpp/   godot-cpp 4.4-stable (submodule)
```

---

## Known limitations

- **Windows only for now.** The build links a Windows godot-cpp library directly, and the `.gdextension` file only lists `windows.x86_64`.
- **Traffic learning has limited evaluation.** The harder circuit has nearby-car sensing and road-relative heading, but its clean-race result is one policy/seed and a short evaluation. Multi-seed learning, sustained collision-free driving, overtaking and held-out circuits need further measurement. The maze agents still act randomly.
- **The MLP still trains in fp32.** The tensor-core GEMM is a standalone kernel for now (forward-pass layout only). The PPO-sized networks are memory- and launch-bound, so it wouldn't speed them up much; it goes into the MLP when bigger models need it. It's also at 65% of cuBLAS at 1024³, where too few blocks fill the GPU (see the worklog's next steps).
- **Small batches are launch-bound.** A training step is about two dozen kernel launches; CUDA Graphs or fusion would cut the 0.2 ms floor.
- **Building godot-cpp isn't scripted.** The prebuilt library isn't checked in, and building it runs Godot's own Python binding generator. That's a one-time upstream build tool, not part of this project's code or runtime.

---

## Roadmap

1. ~~**Neural network core.**~~ ✅ MLP forward and backward kernels and Adam, verified against finite differences and a CPU reference, benchmarked against the roofline and cuBLAS.
2. ~~**Tensor-core GEMM.**~~ ✅ bf16 `mma.sync` kernel at 91–96% of cuBLAS on large matrices, with a [worklog](docs/gemm_worklog.md) of every optimization and its profile.
3. ~~**PPO on CartPole.**~~ ✅ GPU-resident rollouts, GAE, and updates; solves in about 0.35 s and is watchable live in Godot ([write-up](docs/ppo_cartpole.md)).
4. ~~**PPO on a racing track.**~~ ✅ Gaussian policy for continuous steering and throttle, nine raycasts, a live Godot view, and a laps-per-episode learning curve ([write-up](docs/ppo_racing.md)).
5. **Measure and accelerate the complete training loop.** Current focus: reproducible
   racing benchmarks and optimizations that preserve the reference training results.
6. **Attention-based policy.** Connect the existing fused attention kernels to a
   trainable policy and compare it fairly with the current MLP.
7. **Linux build and GPU CI.** Linux work is deferred at the user's request;
   Windows/CUDA performance takes priority. A swarm with spatial hashing remains optional.

---

## Contact

Drew Clark, GPU and real-time systems engineer
- GitHub: [@life423](https://github.com/life423)
- Portfolio: https://drewclark.io
- Email: drew@drewclark.io

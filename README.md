# CudaRL-Arena

[![CI](https://github.com/life423/CudaRL-Arena/actions/workflows/ci.yml/badge.svg)](https://github.com/life423/CudaRL-Arena/actions/workflows/ci.yml)

Reinforcement learning environments written as hand-written CUDA kernels and driven directly from the Godot game engine, with no Python anywhere in the loop.

Godot loads a C++ GDExtension DLL that has the CUDA code linked in. GDScript calls straight into it. There's no server, no socket, no interpreter, and no ML framework. Every kernel and learning algorithm in this repo is written from scratch.

> **Status: neural network core and tensor-core GEMM done; PPO next.** Hand-written CUDA MLP kernels, backprop, and Adam are verified against finite differences and a CPU reference. The hand-written bf16 tensor-core GEMM reaches **91–96% of cuBLAS** on large matrices. Everything below is marked as either **working** or **planned**, and every number comes from a measurement you can reproduce from this repo.

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
| **Live training view in Godot:** watch a network trained on the GPU fit a curve in real time, with live loss and timing | ✅ Working (main scene) |
| 2-agent maze game in Godot; rules in plain C++, walls drawn from the same data used for collision | ✅ Working (CPU, random agents) |
| Hand-written MLP kernels (tiled GEMM, fused bias + activation, deterministic split-K), backprop, Adam with gradient clipping | ✅ Working |
| **Tensor-core bf16 GEMM** (`mma.sync`, `ldmatrix`, `cp.async` pipeline, swizzled shared memory), five versions, each profiled | ✅ Working, 91–96% of cuBLAS ([worklog](docs/gemm_worklog.md)) |
| Verification: fp64 finite-difference gradient checks, GPU vs. CPU reference, exact-arithmetic tensor-core checks, bit-identical determinism, end-to-end learning | ✅ Working, 45 tests |
| Benchmark: roofline, cuBLAS comparison, tuning sweep | ✅ Working (`nn_bench`) |
| PPO trained entirely on the GPU (CartPole first, then racing), watched live in Godot | ⏳ Planned |

![Live training view: a 1→32→32→1 network learning sin(3x)(0.5 + 0.15x) on the GPU, 0.3 ms per step](docs/images/live-fit.png)

*The live view: Godot calls the CUDA trainer every frame and draws the network's prediction (blue) over the target (gray), with the loss history below. Target function, activation, width, depth, learning rate, and training speed can all be changed while it runs.*

The DLL links the CUDA runtime statically. It runs on any machine with an NVIDIA driver, with no CUDA DLLs shipped alongside it.

---

## Results

Measured on an RTX 5070 (48 SMs, driver 620.02, CUDA 13.2) with `nn_bench`, best of 5 timing trials. Every result is checked against cuBLAS before it's timed.

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

`cudarl_tests.exe <filter>` runs only the tests whose names contain `<filter>`, for example `gemm`. On a machine without an NVIDIA GPU, the GPU tests report `SKIP` and the CPU tests still run; set `CUDARL_REQUIRE_GPU=1` to make a missing GPU a failure instead.

**Without Godot:** configure with `-DCUDARL_BUILD_GODOT=OFF` (or use the `ci` preset) to build only the core library, tests, and benchmark. godot-cpp isn't needed then.

**Run the live training view** (close Godot before rebuilding; a running game locks the DLL):

```bash
godot --path godot-project
```

The console prints something like `CUDA ready: NVIDIA GeForce RTX 5070 (sm_120, 48 SMs, 11.9 GiB)`. The older maze demo is still there: `godot --path godot-project res://maze/maze.tscn`.

### Continuous integration

Every push and pull request runs [`.github/workflows/ci.yml`](.github/workflows/ci.yml) on a GitHub-hosted Windows runner. It installs the CUDA 13.2 compiler and runtime from NVIDIA's network installer, builds the core, tests, and benchmark with **warnings as errors**, and runs the test suite. Hosted runners have no GPU, so CI proves the build and the CPU reference, while the GPU tests (`SKIP` there) run locally on the RTX 5070. A self-hosted GPU runner is planned.

---

## GDScript API (current)

```gdscript
var device := CudaDevice.new()      # probes the GPU once
device.is_available()               # bool
device.describe()                   # one-line summary, or the reason CUDA is unavailable
device.get_info()                   # { available, name, compute_capability, multiprocessors, memory_bytes, error }

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
src/demos/            live-training demos driven from Godot (FunctionFit)
src/envs/maze/        maze rules (plain C++)
src/godot/            GDExtension classes and registration
tests/                hand-written test harness and test suites (run via ctest)
tools/nn_bench.cpp    roofline and cuBLAS benchmark (`tc`, `fp32`, `train`, `tune`, `profile`)
godot-project/        Godot project: addons/cudarl/ (extension), fit/ (live training view), maze/
docs/                 GEMM worklog, screenshots
external/godot-cpp/   godot-cpp 4.4-stable (submodule)
```

---

## Known limitations

- **Windows only for now.** The build links a Windows godot-cpp library directly, and the `.gdextension` file only lists `windows.x86_64`.
- **No RL yet.** The neural network core trains (see the sin(x) fit test), but PPO isn't built. The maze agents act randomly; the maze is a placeholder demo for the engine bridge and gets replaced by the PPO environments.
- **The MLP still trains in fp32.** The tensor-core GEMM is a standalone kernel for now (forward-pass layout only). The PPO-sized networks are memory- and launch-bound, so it wouldn't speed them up much; it goes into the MLP when bigger models need it. It's also at 65% of cuBLAS at 1024³, where too few blocks fill the GPU (see the worklog's next steps).
- **Small batches are launch-bound.** A training step is about two dozen kernel launches; CUDA Graphs or fusion would cut the 0.2 ms floor.
- **Building godot-cpp isn't scripted.** The prebuilt library isn't checked in, and building it runs Godot's own Python binding generator. That's a one-time upstream build tool, not part of this project's code or runtime.

---

## Roadmap

1. ~~**Neural network core.**~~ ✅ MLP forward and backward kernels and Adam, verified against finite differences and a CPU reference, benchmarked against the roofline and cuBLAS.
2. ~~**Tensor-core GEMM.**~~ ✅ bf16 `mma.sync` kernel at 91–96% of cuBLAS on large matrices, with a [worklog](docs/gemm_worklog.md) of every optimization and its profile.
3. **PPO on CartPole.** GPU-resident rollouts, advantage estimation (GAE), and updates. CartPole has well-known solutions, so a bug can't hide behind the environment.
4. **PPO on a racing track.** Raycast sensors and continuous control. The first demo worth recording, with lap progress as a clear learning curve.
5. **Fused attention kernel**, then an attention-based policy.
6. **Linux build and CI**, then a swarm with spatial hashing if there's time.

---

## Contact

Drew Clark, GPU and real-time systems engineer
- GitHub: [@life423](https://github.com/life423)
- Portfolio: https://drewclark.io
- Email: drew@drewclark.io

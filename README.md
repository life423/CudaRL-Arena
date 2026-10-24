# CudaRL-Arena

Reinforcement learning environments written as hand-written CUDA kernels and driven directly from the Godot game engine, with no Python anywhere in the loop.

Godot loads a C++ GDExtension DLL that has the CUDA code linked in. GDScript calls straight into it. There's no server, no socket, no interpreter, and no ML framework. Every kernel and learning algorithm in this repo is written from scratch.

> **Status: early prototype.** The Godot ↔ CUDA bridge works on real hardware. Learning and benchmarking are the next milestones. Everything below is marked as either **working** or **planned**, and any performance number in this README will come from a measurement you can reproduce from this repo.

---

## Why no Python?

The usual RL stack goes Python → framework → bindings → GPU, with the environment often on the CPU in a separate process. This project tests how far you get by deleting all of that:

- **Environments are CUDA kernels.** Each GPU thread steps one environment.
- **The game engine is the frontend.** Godot calls the kernels in-process through GDExtension and renders the results.
- **Everything is from scratch.** There's no cuBLAS, cuDNN, PyTorch, NumPy, or Gymnasium. Neural networks, backpropagation, and RL algorithms are our own C++/CUDA.

Performance gets measured against a **CPU C++ reference implementation of the same environment**, also in this repo, so the comparison is apples to apples.

---

## What works today

| | Status |
|---|---|
| Godot 4.4 loads the CUDA-linked GDExtension DLL | ✅ Working (Windows, RTX 5070, CUDA 13.2) |
| GPU probe: device query plus a test kernel launch, so a missing driver or wrong architecture is reported instead of crashing | ✅ Working |
| 2-agent maze game in Godot; rules in plain C++, walls drawn from the same data used for collision | ✅ Working (CPU, random agents) |
| Hand-written neural network kernels, backprop, and Adam, verified by gradient checking | ⏳ Next |
| PPO trained entirely on the GPU (CartPole first, then racing) | ⏳ Planned |
| Benchmark harness: GPU vs. CPU C++ reference | ⏳ Planned |
| Live training view in Godot (steps/sec, learning curve) | ⏳ Planned |

The DLL links the CUDA runtime statically. It runs on any machine with an NVIDIA driver, with no CUDA DLLs shipped alongside it.

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
- CMake 3.21+ (3.31, bundled with VS Build Tools)
- Godot 4.4 (4.4.1)

**godot-cpp:** fetch it with `git submodule update --init`. The build also needs its prebuilt `bin/libgodot-cpp.windows.template_release.x86_64.lib`. See Known limitations.

**Build the extension** from the repo root:

```bash
cmake --preset windows
```

```bash
cmake --build --preset release
```

This writes `godot-project/addons/cudarl/bin/cudarl.dll`. The CUDA architecture defaults to `120`; pass `-DCMAKE_CUDA_ARCHITECTURES=<yours>` to the configure step for other GPUs.

**Run the demo:**

```bash
godot --path godot-project
```

The console prints something like `CUDA ready: NVIDIA GeForce RTX 5070 (sm_120, 48 SMs, 11.9 GiB)`, then `Episode over: …` as rounds finish. Press Space to reset.

---

## GDScript API (current)

```gdscript
var device := CudaDevice.new()      # probes the GPU once
device.is_available()               # bool
device.describe()                   # one-line summary, or the reason CUDA is unavailable
device.get_info()                   # { available, name, compute_capability, multiprocessors, memory_bytes, error }

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
CMakePresets.json     `windows` configure preset, `release` build preset
src/cuda/             CUDA runtime helpers: RAII device memory, errors, device probe
src/envs/maze/        maze rules (plain C++)
src/godot/            GDExtension classes and registration
godot-project/        Godot project: addons/cudarl/ (extension), maze/ (demo scene)
external/godot-cpp/   godot-cpp 4.4-stable (submodule)
```

---

## Known limitations

- **Windows only for now.** The build links a Windows godot-cpp library directly, and the `.gdextension` file only lists `windows.x86_64`.
- **No learning yet.** The maze agents act randomly. The maze is a placeholder demo for the engine bridge and gets replaced by the PPO environments.
- **Building godot-cpp isn't scripted.** The prebuilt library isn't checked in, and building it runs Godot's own Python binding generator. That's a one-time upstream build tool, not part of this project's code or runtime.
- **No automated tests yet.** They arrive with the neural network work, starting with gradient checks.

---

## Roadmap

1. **Neural network core.** Hand-written CUDA kernels for MLP forward and backward passes and Adam. Verified against finite differences and a CPU reference.
2. **PPO on CartPole.** GPU-resident rollouts, advantage estimation (GAE), and updates. CartPole has well-known solutions, so a bug can't hide behind the environment.
3. **PPO on a racing track.** Raycast sensors and continuous control. The first demo worth recording, with lap progress as a clear learning curve.
4. **Swarm with spatial hashing.** Thousands of agents in one world, using neighbour search with a hand-written GPU radix sort, trained with the same PPO.
5. **Benchmarks and live view.** Steps/sec vs. the CPU reference, an Nsight Systems profile, and Godot rendering a few environments while the rest train.

---

## Contact

Drew Clark, GPU and real-time systems engineer
- GitHub: [@life423](https://github.com/life423)
- Portfolio: https://drewclark.io
- Email: drew@drewclark.io

# CudaRL-Arena

Reinforcement learning environments written as hand-written CUDA kernels and driven directly from the Godot game engine, with no Python anywhere in the loop.

Godot loads a C++ GDExtension DLL that has the CUDA code linked in. GDScript calls straight into it. There's no server, no socket, no interpreter, and no ML framework. Every kernel and learning algorithm in this repo is written from scratch.

> **Status: early prototype.** The Godot ↔ CUDA bridge works on real hardware. Training and benchmarking aren't built yet. Everything below is marked as either **working** or **planned**, and any performance number in this README will come from a measurement you can reproduce from this repo.

---

## Why no Python?

The usual RL stack goes Python → framework → bindings → GPU, with the environment often on the CPU in a separate process. This project tests how far you get by deleting all of that:

- **Environments are CUDA kernels.** Each GPU thread steps one environment.
- **The game engine is the frontend.** Godot calls the kernels in-process through GDExtension and renders the results.
- **Everything is from scratch.** There's no cuBLAS, cuDNN, PyTorch, NumPy, or Gymnasium. Learning algorithms are written as our own C++/CUDA.

Performance gets measured against a **CPU C++ reference implementation of the same environment**, also in this repo, so the comparison is apples to apples.

---

## What works today

| | Status |
|---|---|
| Godot 4.4 loads the CUDA-linked GDExtension DLL | ✅ Working (Windows, RTX 5070, CUDA 13.2) |
| GPU kernel stepping 1,000 grid-world environments in parallel | ✅ Working (toy dynamics, fixed actions) |
| 2-agent maze game rendered in Godot, walls drawn from the same data used for collision | ✅ Working (CPU logic, random policies) |
| Runs without a GPU: CUDA features turn off instead of crashing the editor | ✅ Working |
| Shared action encoding between kernel and engine (`src/gpu/actions.h`) | ✅ Working |
| From-scratch learning (Q-learning / neural policy) on the GPU | ⏳ Planned |
| GPU-resident training loop (no per-step host copies) | ⏳ Planned |
| Benchmark harness: GPU vs. CPU C++ reference | ⏳ Planned |
| Live stats overlay (steps/sec, win rate) in Godot | ⏳ Planned |

The DLL links the CUDA runtime statically and imports only `KERNEL32.dll`. It runs on any machine with an NVIDIA driver, with no CUDA DLLs shipped alongside it.

---

## Architecture

```
┌──────────────────────────────────────────┐
│ Godot 4.4 (GDScript)                     │  rendering, input, scene
│   arena/Arena2d.gd                       │
└───────────────┬──────────────────────────┘
                │ GDExtension — in-process C++ calls
┌───────────────▼──────────────────────────┐
│ ArenaGDExtension (C++)                   │  Godot types ↔ plain C++,
│   src/godot_bindings/                    │  catches all exceptions
└───────────────┬──────────────────────────┘
                │ direct calls, same DLL
┌───────────────▼──────────────────────────┐
│ CudaArena (C++/CUDA)                     │  kernels, RAII device memory,
│   src/gpu/                               │  typed CUDA errors
└──────────────────────────────────────────┘
```

Some design points:
- **`device_ptr<T>`** (`src/gpu/cuda_memory.h`) is a move-only RAII owner for device allocations. CUDA failures throw a `cuda_error` that carries the expression, file, and line.
- **Exceptions stop at the binding layer.** Nothing escapes into Godot, so a CUDA failure becomes a Godot warning or error instead of a crashed editor.
- **One source of truth.** Action encoding lives in `actions.h` (used by `__host__ __device__` code on both sides), and maze walls are defined once in GDScript for both collision and rendering.

---

## Build (Windows)

**Requirements** (tested versions in parentheses)
- NVIDIA GPU + driver (RTX 5070, driver 620.02)
- CUDA Toolkit **12.8+** for Blackwell `sm_120` (13.2). The toolkit can't be newer than what your driver supports; check the `CUDA Version` shown in `nvidia-smi`.
- Visual Studio 2022 Build Tools with the C++ workload (MSVC 19.44)
- CMake 3.18+ (3.31, bundled with VS Build Tools)
- Godot 4.4 (4.4.1)

**godot-cpp:** the build expects godot-cpp 4.4-stable in `external/godot-cpp`, with a prebuilt `bin/libgodot-cpp.windows.template_release.x86_64.lib`. See Known limitations.

**Build the extension** from the repo root:

```bash
cmake -S src/godot_bindings -B src/godot_bindings/build -G "Visual Studio 17 2022" -A x64
```

```bash
cmake --build src/godot_bindings/build --config Release
```

This writes `godot-project/addons/cudarl_plugin/Release/cudarl_godot.dll`. The CUDA architecture is set to `120` in `src/godot_bindings/CMakeLists.txt`; change it to match your GPU.

**Run the demo:**

```bash
godot --path godot-project
```

The console should print `CUDA Ready: 1 device, using <your GPU>`, then `Episode over: …` as rounds finish. Press Space to reset.

---

## GDScript API (current)

```gdscript
var arena := ArenaGDExtension.new()

arena.has_cuda()                  # bool, false if no usable GPU/driver
arena.hello_cuda()                # String, device info (runs a test kernel)
arena.run_training(200)           # steps the 1,000-env GPU batch (needs CUDA)

arena.set_maze(cells)             # PackedVector2Array of wall cells (10×10)
arena.get_goal()                  # Vector2i
var r = arena.step_environment([human_action, ai_action])  # 0 Up, 1 Right, 2 Down, 3 Left
# r[i] = { "state": Vector2, "reward": float, "done": bool }
arena.reset()
```

---

## Project layout

```
src/gpu/              CUDA kernels, CudaArena, RAII memory, shared action encoding
src/godot_bindings/   GDExtension class + registration, CMake build
godot-project/        Godot project; arena/ holds the demo scene and script
external/godot-cpp/   godot-cpp 4.4-stable
```

---

## Known limitations

- **Windows only for now.** The CMake build links a Windows godot-cpp library directly, and the `.gdextension` file only lists `windows.x86_64`.
- **The maze game logic runs on the CPU.** The GPU batch is a separate toy grid world. Moving the playable environment onto the GPU is the next milestone.
- **No learning yet.** Both agents act randomly.
- **The godot-cpp setup isn't reproducible from a clean clone.** `external/godot-cpp` is recorded as a submodule commit without a `.gitmodules` entry, and the prebuilt library isn't checked in. Building godot-cpp uses Godot's own Python binding generator. That's a one-time build tool from upstream, not part of this project's code or runtime.
- **Synchronous host ↔ device copies** on every step of the GPU batch.

---

## Roadmap

1. **GPU-resident environments.** Environment state, resets, and RNG all stay on the device. Only the environments being rendered get copied back, asynchronously.
2. **From-scratch learning on the GPU.** Start with tabular Q-learning, then hand-written neural network kernels for richer environments.
3. **Benchmark harness.** Steps/sec and per-step latency for GPU vs. the CPU C++ reference, from 1 to 100K+ environments, with an Nsight Systems profile.
4. **Live demo.** Godot renders a few environments while the rest train, with on-screen steps/sec and learning curves.
5. **Reproducible build.** Proper submodule, scripted godot-cpp build, CI.

---

## Contact

Drew Clark, GPU and real-time systems engineer
- GitHub: [@life423](https://github.com/life423)
- Portfolio: https://drewclark.io
- Email: drew@drewclark.io

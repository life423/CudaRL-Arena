# Portable pond — October 27, 2026

**Paused at the user's request.** Keep this Windows prototype as a stopping point. Further distribution, CPU training, GPU architecture expansion, installers and Mac work are deferred unless the user explicitly resumes them. The implementation below is preserved, not a commitment to continue this roadmap.

The first portable player preserves the packaged food selector, swimming network and authoritative articulated physics. The Windows preview bundles Godot 4.4.1 and opens the deep pond directly. It needs no separate Godot, CUDA toolkit, Python or ML-framework installation.

## Open the preview

Extract **all** of `build/portable/CudaRL-Pond-Windows.zip`, then open `CudaRLPond.exe`. Keep the launcher, `engine.exe`, `player` and `licenses` together. The sidebar's **AI and physics · CUDA/CPU** reports the compute backend. Hover over it for the probe result. Camera controls and Back to demos work as before; the portable menu offers the pond and About this build.

The default package tries its optional CUDA module first. It queries a usable device, launches a real probe kernel, and initializes the model/world before selecting CUDA. An absent module, driver/device failure or incompatible kernel falls back to CPU. The CPU extension itself imports only `KERNEL32.dll` on this Windows build. A CUDA failure during playback continues on CPU from the last successful snapshot rather than resetting the animal. That recovery path is implemented; actual mid-session hardware loss has not been injected in testing.

The supplied CUDA module follows this repository's current `sm_120` build default, verified on the RTX 5070. Older NVIDIA architectures need a rebuild with appropriate `CMAKE_CUDA_ARCHITECTURES`; this binary falls back to CPU if its kernels cannot execute. Supporting and testing a broader GPU architecture set is a packaging follow-up.

Godot rendering is independent of the AI/physics backend. A machine using CPU inference still uses its supported graphics hardware for the scene. NVIDIA CUDA is the only GPU compute backend implemented here; AMD/Intel/Apple GPU compute is a future decision. Older graphics hardware can use the Compatibility renderer, tested locally through OpenGL. This does not promise support for every computer.

For troubleshooting, run from the extracted folder:

```powershell
.\CudaRLPond.exe --pond-backend=cpu
```

`--pond-backend=cuda` is a strict diagnostic mode and reports an error instead of falling back. Default `auto` is recommended. The separate `CudaRL-Pond-CPU-only.zip` omits the optional CUDA module entirely. The normal ZIP is the appropriate choice for the user's NVIDIA machine.

To select the OpenGL renderer explicitly, run `.\engine.exe --path .\player --rendering-method gl_compatibility` from the extracted folder. This changes rendering only; AI/physics still select their backend automatically.

## What this step delivers

- Saved-controller inference and physical swimming on Windows x64, using the same model file and shared equations on CPU/CUDA.
- Automatic CUDA preference without making NVIDIA libraries mandatory for startup.
- A standalone preview launcher, bundled engine and notices; it is not yet a signed installer or optimized Godot export.
- Existing native CUDA PPO training, CartPole, Racing, full saves and the developer menu remain available in the original project.

This preview does **not** train on CPU, replay a fake learning curve as live learning, grow limbs, or include the other portable demos. Live learning on non-NVIDIA machines, portable CartPole/Racing, macOS packaging and the creature's growth remain subsequent steps. No original training save or packaged neural-network weight is changed by playback.

## Build and package

Visual Studio 2022's C++ toolchain and the project's prebuilt godot-cpp library are required for this Windows player build. CUDA is not required:

```powershell
cmake --preset portable
cmake --build --preset portable --parallel 6
ctest --preset portable
```

For the optional GPU module, retain the existing CUDA developer configuration:

```powershell
cmake --preset windows
cmake --build --preset release --target cudarl --parallel 6
```

`cudarl` also builds `cudarl_pond_cuda.dll`. Stage with the installed Godot executable:

```powershell
.\tools\stage_pond_player.ps1 -GodotPath C:/Users/aiand/Godot/Godot.exe
.\tools\stage_pond_player.ps1 -GodotPath C:/Users/aiand/Godot/Godot.exe -Destination build/portable/cpu-package -CpuOnly
```

Import each staged project's resources using Godot `--headless --editor --path <staged-player-folder> --import` before packaging. Include `.godot/imported`, `.godot/extension_list.cfg` and `.godot/uid_cache.bin`; omit editor and shader caches. The ZIP must include the enclosing launcher's companion files, not just `player`. The new CI job builds/tests the C++ runtime without CUDA or Godot build dependencies; its remote result is pending until these changes are pushed.

The reproducible packaging command performs staging, import and ZIP creation together:

```powershell
.\tools\package_pond_player.ps1 -GodotPath C:/Users/aiand/Godot/Godot.exe
.\tools\package_pond_player.ps1 -GodotPath C:/Users/aiand/Godot/Godot.exe -CpuOnly
```

The pure native benchmark accepts the same optional module:

```powershell
.\build\portable\Release\pond_player_bench.exe godot-project/pond/policies/deep_selector.bin cpu
.\build\portable\Release\pond_player_bench.exe godot-project/pond/policies/deep_selector.bin auto godot-project/addons/cudarl/bin/cudarl_pond_cuda.dll
```

## Evidence and limits

Local Windows/RTX 5070 checks passed:

- All **154 native developer tests**, GPU required, no skips.
- Portable C++ tests build without CUDA. With the optional module: **3,214 checks**, zero failures; repeating with `CUDA_VISIBLE_DEVICES=-1` also passes. The CPU trajectory matches the first 80 recorded pre-portability steps within `2e-6`, and its fixed 60-second run still collects 18 foods. Short CPU/CUDA trajectory checks allow floating-point tolerance; long trajectories need not be bit-identical.
- Final standalone rendered checks: **25 checks each** for CUDA auto-selection, hidden-device CPU fallback, explicit CPU, and a CPU-only package using OpenGL Compatibility. All collect from the three food slots and check camera/menu/help/layout at 1280×720, 768×1024, 390×844 and 844×390. Screenshots were inspected; physical touch and assistive devices were not tested.
- All **374 existing Godot checks** pass: saved controller 27, deep-pond training/resume 67, presentation 138, menu 103 and deep water 39.
- The normal ZIP was extracted to a fresh folder containing spaces. Its actual `CudaRLPond.exe` launched successfully with CUDA available and hidden; the expected optional module was loaded only in the CUDA case. Both windows and launcher processes closed normally.
- The model SHA256 remains `D9730ED4FB2A05904E09ACAB45C43BE47FF7B80B768AA5379C463A11B8D9611D`.

One 1,200-step / 60-simulation-second native timing sample measured CPU **0.072 ms/step**, CUDA **0.327 ms/step**, and automatic hidden-device CPU fallback **0.070 ms/step**. These exclude startup and rendering. Dispatch overhead makes CUDA slower for this single visible creature, though both comfortably fit its 50-ms control interval. CUDA remains the requested default when healthy; these measurements are not evidence about the throughput of 1,024-world training.

No physical AMD/Intel-only PC or Mac was available. Hiding CUDA devices, omitting the module, auditing imports, and building without CUDA demonstrate separation on this machine; actual non-NVIDIA hardware testing and macOS binaries are still required before claiming those platforms work.

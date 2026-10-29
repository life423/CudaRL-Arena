# Windows preview and download website

The user approved Windows first, with the **newest sinking-food experience and genuine CPU learning** when NVIDIA CUDA is unavailable. No GPU server, paid cloud learner, macOS port or browser game is part of this release. The website is a download page.

## Player experience

Download `TadpolePond-Windows-x64.zip`, extract the complete archive, and open `TadpolePond.exe`. Godot's release engine and native runtime are included; players do not install the editor, Python, PyTorch, a CUDA toolkit or a compiler. Keep the PCK, native extension, data and notices together. This early preview is unsigned and Windows may display a publisher/reputation warning.

The menu has one **Open pond** action. Experiments and Help/training were removed from player navigation at the user's request; native developer routes and CartPole/Racing remain. The pond still exposes meaningful feeding, current, camera, pause, save and progressively disclosed AI controls.

The native extension probes a CUDA device **and executes a kernel** before choosing it. This distribution compiles sm80/86/89/120, covering Ampere and newer GPU architectures (including RTX 30/40/50). An unsupported card/driver or no NVIDIA device falls back to CPU. A `--pond-backend=cpu` argument can force the fallback for diagnostics. Rendering uses the available graphics hardware separately from the AI backend. This is a Windows 10/11 x64 preview; older GPU generations and physical AMD/Intel-only computers have not yet been directly tested.

## Genuine CPU learning

`feeding_cpu.h` reuses the articulated-body integration, pellet equations, food refills, selector features, 37 motor observations, Gaussian PPO loss, GAE and Adam reference math. There is no scripted locomotion and no older three-food fallback masquerading as learning. CPU practice uses 16 worlds to limit load; CUDA keeps 1,024. CPU PPO uses the same 64-step rollouts, four epochs/eight minibatches, discount .99, GAE .95, clip .2, entropy .002 and optimizer settings. Lower population changes sample throughput/variance; identical learning speed or outcomes are not promised.

The fixed inherited selector scores all active pellets; the motor learns. Recent user drops/current reach practice at rollout boundaries. The same matched candidate/calm-retention acceptance criteria guard deployment to the live body. Saved physical state is not reset on weight adoption. One successful test or short comparison is not general robustness evidence.

CPU full checkpoints use `training-cpu.bin`, separate from the existing CUDA `training.bin`. They preserve actor, value estimator, exploration, all Adam moments/counters, practice worlds, context, validated motor, and the visible swimmer/pellets. Autosave is every 30 seconds and on normal exit. Checkpoint leases prevent two windows writing the same save; close one before reopening another. Personal evolving training checkpoints are never packaged.

## Rebuild and package

Use Godot **4.4.1**, its official Windows x86_64 release export template, the existing godot-cpp library and CUDA toolkit for a CUDA-enabled build. Static CRT/CUDA runtime linking avoids requiring CUDA runtime DLL installation on player machines.

```powershell
cmake -S . -B build/windows-distribution -G "Visual Studio 17 2022" -A x64 `
  "-DCMAKE_CUDA_ARCHITECTURES=80;86;89;120" -DCUDARL_BUILD_BENCH=OFF `
  "-DCUDARL_ADDON_BIN=$PWD/build/windows-distribution/native"
cmake --build build/windows-distribution --config Release --target cudarl cudarl_tests
./tools/package_windows_release.ps1 -GodotPath <Godot.exe> `
  -TemplatePath <windows_release_x86_64.exe> -NativeDirectory build/windows-distribution/native
```

Release packaging requires a clean commit. `-AllowDirty` is only for local verification. Each package uses a fresh staging/output directory, excluding stale QA outputs and user saves. Immutable policies remain external native-readable files beside the executable because a C++ file reader cannot open PCK resource paths. The ZIP includes redistribution notices, a file-hash manifest with source commit/dirty state, and a separate `SHA256SUMS.txt`. Publish only the clean, checked commit's ZIP.

## Hosting and updates

- Azure resource group: `rg-tadpole-pond`, Central US.
- Static Web App: `tadpole-pond`, **Free** SKU; default hostname `gray-pond-0d7501010.2.azurestaticapps.net`.
- Site source: `website/dist`, buildless HTML/CSS/JS and an actual rendered pond capture.
- Download assets: public GitHub Releases in `life423/CudaRL-Arena`; initial version `v0.1.0-preview.1`.
- No VM, GPU, database, Functions backend, user account system or custom paid domain is created. Free hosting quotas apply; downloads are served by GitHub.

Deploy the static folder locally with Azure's Static Web Apps CLI using an Azure deployment credential only in process environment/memory. Do not print or persist it. The automatic approval review rejected transferring that credential into GitHub Secrets; no such secret or automated deploy workflow was installed. Use the authorized direct-to-Azure path. Check Azure resource SKU and public HTTPS responses after deployment.

GitHub Actions must pass for each exact feature/main commit. Public downloads also require inspecting the uploaded asset/checksum and cold-launching an extracted copy. Hosted no-GPU CI is distinct from the local GPU-required suite.

## Verification and limits

Initial correctness evidence: native CPU feeding test **12,946 assertions**, including physical integration/particles, terminal-observation bootstrap, genuine PPO, bit-identical uninterrupted versus restored continuation, full session resume and candidate validation. A 16-world update took about **93 ms** on the developer machine, not a minimum-hardware guarantee.

The exported executable passes rendered CPU feeding **194** checks and CUDA **172**, both with actual continued PPO after full restore. CUDA update 16 improves the matched feeding count 9.833→10.333/min with calm 12/12 and is applied live; update 32 ties and is rejected. CPU's short candidates do not beat its incumbent and are correctly rejected. Full native suite: **180/180**, GPU required, zero skips. Menu checks: **41**, four window sizes. Website checks: **42**, five widths (320–1440), keyboard FAQ/navigation, reduced motion and zero page errors. Subsequent final validation is recorded in the session checkpoint/release evidence.

These are synthetic Windows/Chrome checks on one RTX 5070 machine, including forced CPU and a separately built C++-only runtime. Physical non-NVIDIA machines, assistive technology, every graphics driver and long dense-pellet play sessions remain follow-up checks. Broad robustness, optimality and guaranteed improvement per drop are not claimed.

Internal UI review heuristics: hierarchy, usability, consistency and feedback 9; accessibility/responsiveness 8 pending real touch/assistive tests; performance 8 pending slower-machine profiling. The priority is broader hardware/player testing, not more decoration. Browser automation's in-app tool could not start because the unrelated D: drive is BitLocker-locked; isolated local Chrome/Playwright provided the actual rendered website checks instead.

Final local follow-up: developer-demo checks **104**, preserved-reference navigation **81**, both with empty final stderr; portable CTest **2/2**. All eleven protected inputs rehashed unchanged. Evidence is under `docs/experiments/swimmer/windows-release/`. Source-format backups are retained only in ignored build output. Exact-commit CI and public endpoint verification follow publication.

Custom-domain follow-up: the user chose **https://cudarl.drewclark.io**. Its CNAME points to the Azure default hostname; public DNS and valid HTTPS were verified, and Azure reports Ready. This is the canonical website address, using the existing domain and managed free TLS. The site remains unpublished until release integration checks/asset publication complete.

The latest site revision removes empty left columns from steps/FAQ, integrates the learning explanation into the shared full-width section, gives the source link a GitHub icon and matched button dimensions, and adds a keyboard-accessible four-stage architecture/CPU-versus-CUDA walkthrough. It is explanatory, not a browser policy or live training. Test with `node tests/web/download_site_smoke.cjs` after installing Playwright/Chromium and serving `website/dist`; `CUDARL_SITE_URL` selects a hosted target and `CUDARL_CHROME` optionally selects an installed Chrome binary.

The user saved a deferred future option of at most $15/month for GPU-backed browser play with genuinely enforced spending limits. This does not authorize a GPU provision now. Complete the Windows/static-download release first.

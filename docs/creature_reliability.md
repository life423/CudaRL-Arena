# Young swimmer reliability and Godot recovery preview

October 28 follow-up: the latest agent now supports a [live Calm/Gentle current switch](experiments/swimmer/live-gentle-current/README.md). New CPU/CUDA response tests cover both steady flow and switching during playback. The original calm-water evidence below remains historical; practice/online promotion is still future work.

October 27, 2026. First implemented step of the [approved creature journey](CREATURE_PLAN.md): a dependable young swimmer, then challenging interactions, measured adaptation and physical growth. The interactive learning/growth loop remains to build.

## What is implemented

The actual single-creature playback runtime now accepts both the original food selector and the progress-aware selector, on CPU or the optional CUDA module. Both use the retained swimming network and shared physical equations. Existing original-policy playback, backend selection and CPU fallback remain available. This does not install the experimental selector as the default pond policy.

`pond_reliability` observes that runtime continuously without training, scripted recovery, food replacement beyond the existing sampler, resets or teleports. It checks finite body state, valid underwater food, sequential simulation steps, food timing and sustained immobility. A three-second window with head positions all within 0.1 model units (0.4 cm) of the current position starts a suspected immobility interval. Five continuous seconds confirms an event and fails the provisional gate. This engineering threshold is not a user study or a guarantee against every undesirable motion. Food gaps are reported separately: moving toward distant food and being motionless are different failures.

Failure snapshots contain the body, food, selector target and host progress-history mirror. They are diagnostic records, not full resumable training saves or certified bit-exact CUDA checkpoints. Device and CPU history arithmetic can differ. Original full learner saves remain intact.

## Calm-water playback baseline

Four seeds (7, 0xC0DE7101, 0xC0DE7102, 0xC0DE7103), environment zero, each run for thirty continuous simulated minutes. No episode resets. Original and bank-trained selectors each run on CPU and CUDA: sixteen sessions, eight simulated hours total. These are calm-water tests; the bank recording below is a separate current challenge.

| Controller / backend | Collected foods across four sessions | Longest food gap | Confirmed immobility events |
| --- | --- | --- | --- |
| Original / CPU | 535, 532, 547, 530 | 12.75 s | 0 |
| Bank-trained / CPU | 540, 535, 545, 530 | 12.30 s | 0 |
| Original / CUDA | 540, 524, 543, 525 | 11.25 s | 0 |
| Bank-trained / CUDA | 539, 520, 546, 532 | 9.55 s | 0 |

All sessions pass this provisional gate, with no qualifying three-second immobility window, invalid body/food state or skipped step. Source policies compare unchanged. CPU and CUDA long trajectories need not be identical; this small suite does not establish a collection ranking between candidates or backends. It also does not supersede the [known current/bank failures](experiments/swimmer/gentle-currents/bank-recovery/README.md). Neither policy is claimed stall-proof or ready for arbitrary water.

Small raw reports are archived in [experiments/swimmer/playback-reliability](experiments/swimmer/playback-reliability). Complete local outputs remain in `build/pond-reliability-study/`.

## Watch the Godot demonstration

**Use the live agent first:** open `godot-project/pond/latest_agent.tscn`, or choose **Latest trained agent** in the deep pond sidebar. This runs the latest selected bank-trained selector (update 320) with the retained update-64,350 swimming actor. Physics, food selection and respawning happen live; it is not a recorded trajectory. The view prefers CUDA when healthy and falls back to CPU. The bottom graph measures cumulative food/minute in this session, with elapsed seconds on its axis. It is not a training curve. **Show untrained policy** restarts the same layout with random weights for both networks; switching back restores the saved candidate. No learning or checkpoint writes happen in this viewer. Water is calm in this view; current challenges remain separate experiments.

The user explicitly corrected the initial recorded-only presentation: seeing the latest saved agent running live is the preferred demonstration. The scene above is delivered for that purpose; the recording below is supplementary diagnostic evidence. The live process was opened and confirmed using CUDA on the RTX 5070, with no startup errors.

Open `godot-project/pond/bank_preview.tscn`, or choose **Recovery comparison** in the deep pond sidebar when the generated recording exists. This development preview uses the existing pond presentation. It starts playing the bank-trained case and repeats the sixty-second clip.

- **Compare from start** switches selectors and restarts the identical physical/food state.
- The policy dropdown also switches cases. **Restart**, **Pause replay** and the seek slider support close inspection; arrow keys seek when the slider has keyboard focus.
- **Closer view** changes the camera. **Back to demos** returns to the existing launcher.
- The display explicitly says **Recorded comparison · no live training**. It does not consume or update the user's training save.

This is a real native CUDA recording from held-out seed 0xBAA7E001, environment 1019, in a 1 cm/s current. Original selector update 250 collects zero food during the next minute; bank-trained selector update 320 collects thirteen, first reaching food at 3.85 seconds. The swimming actor remains update 64,350. Starting position, orientation, food, joints and links match exactly. Recording preserves the original evaluation outcomes for every world in this pool.

The example was selected after evaluation to illustrate recovery; it is not an unbiased estimate of overall improvement. The full fresh-bank result remains **83/86 within twenty seconds**, with residual failures. The default live pond still uses the original saved policy. There are no limbs, growth triggers, player current controls or live candidate promotion in this preview.

## Reproduce locally

Run from the repository root in PowerShell. Outputs must be new paths; source policies and datasets are read-only. The CUDA recording commands require the NVIDIA training build. No Python or PyTorch is used.

```powershell
cmake --build --preset release --target pond_reliability pond_bank_study cudarl --parallel 6

# Actual single-creature calm-water playback, four thirty-minute sessions.
build/Release/pond_reliability.exe --policy docs/experiments/swimmer/gentle-currents/frozen-policy.bin --output build/reliability-repeat/original-cpu --backend cpu
build/Release/pond_reliability.exe --policy docs/experiments/swimmer/gentle-currents/bank-recovery/selected-policy.bin --progress --output build/reliability-repeat/bank-cuda --backend cuda --module godot-project/addons/cudarl/bin/cudarl_pond_cuda.dll

# Frozen recorded comparison, same archived held-out bank pool.
build/Release/pond_bank_study.exe --policy docs/experiments/swimmer/gentle-currents/frozen-policy.bin --pool docs/experiments/swimmer/gentle-currents/bank-recovery/data/0xBAA7E001/pool.bin --output build/pond-reliability-study/preview-original --seconds 60 --record 1019
build/Release/pond_bank_study.exe --policy docs/experiments/swimmer/gentle-currents/frozen-policy.bin --progress docs/experiments/swimmer/gentle-currents/bank-recovery/selected-policy.bin --pool docs/experiments/swimmer/gentle-currents/bank-recovery/data/0xBAA7E001/pool.bin --output build/pond-reliability-study/preview-trained --seconds 60 --record 1019
tools/merge_bank_preview.ps1
& C:/Users/aiand/Godot/Godot.exe --path godot-project res://pond/bank_preview.tscn

# Preferred: run the newest saved candidate in a live simulation; no recording generation needed.
& C:/Users/aiand/Godot/Godot.exe --path godot-project res://pond/latest_agent.tscn
```

The generated replay stays in ignored `build/`; a fresh checkout needs the recording commands before this development viewer is available. The conditional sidebar link is hidden when it is absent. `pond_reliability` exits nonzero on infrastructure failures; quality failures are recorded in `summary.json` as `passes_provisional_gate: false`, so automated users must inspect that value.

## Verification and remaining limits

Release build and **167/167 native tests** pass with GPU required and zero skips. Existing C++-only playback regression also passes. Godot's recovery preview passes 57 headless checks and 61 rendered checks (four screenshot writes included). Original replay, untrained toggle and menu regressions are checked against the rebuilt extension. Source policies/full saves and the physics source are checked separately. See the archived logs and hashes for exact results.

The live candidate integration passes **32 rendered CUDA checks** and **28 headless CPU checks**: correct candidate/motor metadata, an actual minute of food collection, live movement, measured graph values, random-weight/saved-weight round trips, same initial food/body state, four responsive sizes, source hashes and return to menu. It exposes zero background learners and no inherited original-policy validation score. Native optional-module checks also pass (3,298 assertions). Live desktop and portrait screenshots were inspected; all four rendered capture writes and bounds checks pass.

Rendered screenshots were inspected at 1280×720, 768×1024, 390×844 and 844×390. Comparison/restart/pause/seek/back actions and keyboard seeking were exercised through Godot's own test harness; action/dropdown/slider targets are at least 44 px and layout bounds pass. At small sizes the sidebar scrolls and secondary statistics are below the initial view. The desktop inspection helper could not start because drive D: is BitLocker-locked, so physical mouse/touch and assistive-device behavior were not verified. No drive or security setting was changed.

Internal review heuristics, not measured scores: usability, hierarchy, consistency and feedback 9/10; accessibility verification 8/10; performance verification 8/10. Before platform expansion, test real touch/assistive input and scrolling to secondary information. Before concurrent live training is connected, measure sustained rendering and training throughput together. Those follow-ups do not justify more cosmetic work now.

Next AI milestone: retain the residual difficult bank/current cases and evaluate dependable recovery before adding one bounded player interaction. The recorded improvement is evidence toward that milestone; genuine background adaptation, safe promotion of candidates and physical growth remain later stages.

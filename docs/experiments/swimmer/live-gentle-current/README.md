# First live gentle-current interaction — October 28, 2026

The latest saved bank-trained agent now runs with an optional **steady 0.5 cm/s horizontal current** in the live Godot pond. The switch changes the existing water-relative drag input; it preserves body position, velocity, joints, food, counters, decision cadence and session graph. No actor weights, PPO reward, physical equations or training saves change.

The frozen agent handles the tested gentle flow without a new training phase. This delivers the first bounded environmental interaction in the [creature plan](../../../CREATURE_PLAN.md), not the later practice/promotion/growth loop. Existing difficult bank/current cases remain separate and unresolved.

## Watch it

From the launcher, choose **Tadpole Pond → Deep water → Latest trained agent**. The live view starts in Calm. Enable **Gentle current** near the top of the sidebar to apply the flow; disable it to return to Calm. Status text gives the active strength and direction. It is uniform +X flow across the pond, not random direction changes, turbulence or CFD. Food stays at its existing sampled underwater locations.

**Show untrained policy** still compares random initial networks with the saved agent, restarting the same initial layout as before. That comparison preserves the selected water condition. Changing water alone does not reset anything. The bottom graph measures actual cumulative food/minute in the current live session, not learning updates.

To open directly with the current already enabled:

```powershell
& C:/Users/aiand/Godot/Godot.exe --path godot-project res://pond/latest_agent.tscn -- --gentle-current
```

The visible creature uses selector update **320** with retained swimming update **64,350**, on healthy CUDA by preference. The original packaged controller remains separate. Watching this scene does not train either network.

## Frozen response results

The [protocol](PROTOCOL.md) was written before the response trials. Four fresh seeds, five continuous minutes each, three conditions, CPU and CUDA: **24 sessions / two simulated hours**, with no resets or rescue. All cases are archived, including empty event files.

| Backend | Water condition | Food/minute | Longest food gap | Confirmed immobility events |
| --- | --- | ---: | ---: | ---: |
| CPU | Calm | 17.90 | 13.50 s | 0 |
| CPU | Gentle from start | 17.60 | 6.80 s | 0 |
| CPU | Gentle after 60 s | 17.40 | 11.95 s | 0 |
| CUDA | Calm | 18.05 | 8.30 s | 0 |
| CUDA | Gentle from start | 17.25 | 9.10 s | 0 |
| CUDA | Gentle after 60 s | 17.65 | 8.55 s | 0 |

All sessions pass the provisional five-second immobility gate, with no qualifying three-second window, invalid body/food state, skipped step or >20-second food gap. CUDA collection is about 4.4% lower with steady flow than calm, and about 2.2% lower with the later switch; CPU differences are about 1.7% and 2.8%. These are descriptive results from four seeds per condition, not significance claims or evidence of global optimality. CPU/CUDA use separate execution paths and long trajectories can diverge; keep each backend's comparison separate. Both conditions share each seed's initial layout, but later food consumption/respawns can diverge with the behavior.

The current challenges the existing learned behavior without requiring more training in this bounded suite. Do not manufacture a new learning phase merely to make a graph rise. Harder settled-bank cases, changing directions, other current strengths and arbitrary water are not established by these results. Next AI work can diagnose the retained difficult cases and build genuine practice/validation, while preserving the supported gentle interaction.

## Native integration

- `PondPlayback::set_current()` validates a finite horizontal flow up to 1 cm/s. This UI exposes only Calm and +X 0.5 cm/s. CPU and CUDA pass it into the existing shared physical model.
- `PondFrame` now retains the applied current, including during existing CPU failover. Diagnostic snapshot format increments to version 2. These remain diagnostic records: device history is mirrored on the host and an exact CUDA challenge export/restore API is still future work.
- The original optional-module v1 create/step/destroy ABI is preserved. A separate `cudarl_get_pond_current_api` extension supplies the setter. Older modules can still run calm CUDA. If current control is requested with an older module, automatic mode continues on CPU without resetting the body; strict CUDA reports that the module needs rebuilding.
- `DeepPondLab` exposes a boolean gentle-current control and actual current/backend metadata. Switching between trained/random policies applies the same flow to the new playback before replacing it.
- Missing configured playback policies now fail through the loader instead of silently entering legacy motor training. An explicitly empty policy path still selects the preserved training route. This closes the startup issue identified in the audit, with a regression check that no background trainer starts.
- No new motor/selector sensors, parameters, optimizer steps, forces, limbs or scripted recovery are introduced.

The reliability runner supports `--current-cm-s 0.5` and optional `--switch-seconds 60`; its default remains calm. Example repeat, choosing a new output directory:

```powershell
build/Release/pond_reliability.exe --policy docs/experiments/swimmer/gentle-currents/bank-recovery/selected-policy.bin --progress --backend cuda --module godot-project/addons/cudarl/bin/cudarl_pond_cuda.dll --output build/gentle-repeat --minutes 5 --current-cm-s 0.5 --switch-seconds 60 --seed 0xC0DEC801 --seed 0xC0DEC802 --seed 0xC0DEC803 --seed 0xC0DEC804
```

Quality failures remain in `summary.json`; the runner returns nonzero for infrastructure failures, so automation must inspect both. New CSV columns describe imposed flow and switch time. A failed scenario is recorded, never reset to disguise it.

## Verification and review

Full release build, including all CI-built profiling tools, passes. **169/169 native tests pass, zero skipped, GPU required.** Tests verify live-switch continuity, shared-force behavior, invalid-flow rejection, CPU/CUDA agreement and existing zero-current references.

The C++-only portable suite passes. Optional-module integration passes **3,303 assertions** with both the current-capable module (CUDA remains active) and a preserved old module (safe CPU continuation on current selection). Its original saved-policy reference remains eighteen foods/minute in that specific legacy seed/protocol; do not substitute that value for the new table above.

Godot current-control integration passes **51 rendered checks on CPU and 51 on CUDA**, including keyboard/mouse activation, continuity, live collection, missing-policy safety, flow preservation across random/saved comparisons, model hashes, responsive bounds, status visibility and menu return. Final headless control runs pass **47 checks each on CPU and CUDA**; rendered runs additionally verify four screenshot writes. Headless uses keyboard input; pointer routing is exercised in the rendered Windows runs. Existing headless latest-agent (28), untrained comparison (191), menu (103), current replay (32) and bank replay (57) regressions pass.

Rendered desktop 1280×720, tablet 768×1024, portrait 390×844 and landscape 844×390 screenshots were inspected. Current control/feedback remains visible on narrow layouts; secondary statistics scroll. Graph captions now use the available width when the watch button is hidden. Internal review heuristics: usability/hierarchy/consistency/feedback 9/10; accessibility verification 8/10; performance verification 8/10. Follow-ups are real touch/assistive-device checks before platform expansion and sustained frame-time/training-contention measurements when practice is connected. No further cosmetic redesign is needed for this control.

[Seven protected source/full-save hashes](preserved-inputs.json) remain unchanged, as do the selected candidate and `deep_physics.h`. [All four full learner saves](full-save-integrity.json), including both bank-pilot saves, also match their sleep-backup hashes. Full local saves are outside Git; the sleep backups remain intact. No ongoing learner is started. The user requested committing and pushing this work on `codex/pond-gentle-current`. Hosted CI must be checked for that exact commit; the previous main CI success belongs to `7ed0f80`, not these new changes. Main remains unchanged by this feature checkpoint.

# Strong-current stall diagnosis — October 27, 2026

The user requested investigating the recorded stall before training and watching the result. The original trained bundle and full saves remain unchanged. The viewer was opened, the user confirmed seeing the stall, and the replay was closed normally at their request.

## Evidence

The original evaluator's GPU kernels reproduce seed 3228672513, environment 24, current `(−0.007071067579, 0, −0.007071067579)` m/s: eight foods and a 265.6-second food gap over five minutes. The last collection is at 34.4 seconds. By 60 seconds the tadpole is nearly stationary at the bank, its four joints are near their negative limits, and the selected physical food remains Food 2 (zero-based slot 1). Native contact sampling shows small negative bank clearance consistent with the existing compliant contact model. This observation alone does not establish contact as the sole cause.

At 60 seconds, all diagnostic branches restore the **same body state, physical foods, counters and prior target**. Both networks stay frozen. Each branch runs 60 more seconds using the original 64-world GPU execution shape. The unchanged branch's continuation matches the original recording exactly at 120 seconds.

| Diagnostic branch | New foods in next 60 s | First new food | Interpretation |
|---|---:|---:|---|
| Original controller/current | 0 | None | Stall continues; target never changes |
| Remove current only | 0 | None | Removing the disturbance does not release the settled controller |
| Nearest-food choice every second | 0 | None | Nearest chooses the same stuck target |
| Hold Food 1 | 11 | 10.50 s | Existing motor can recover when directed elsewhere |
| Hold Food 2 | 0 | None | Original target retains the trap |
| Hold Food 3 | 13 | 10.65 s | Existing motor can recover under the original current |
| Food 1 for one second, then original selector | 0 | None | A brief alternative is not universally sufficient |
| Food 2 for one second, then original selector | 0 | None | No useful intervention |
| Food 3 for one second, then original selector | 15 | 5.25 s | A brief target change releases this case; learned selection then resumes |

These are deliberately programmed **diagnostic interventions**, not a trained improvement or a deployed rescue mechanism. No production controller, rewards, observations or physical parameters were updated. Removing the current changes that branch's environment only; all target-choice branches preserve the original current and motor.

The evidence points to a coupled target-selection/motor recovery trap rather than an inability of the body to swim in this current. The selector is feedforward and its input contains body state, foods and previous-target membership, but **no time-since-food or progress history** (`deep::observe` and `pond_selector::features`). Once observations settle, waiting longer alone cannot change its deterministic decision. Adding progress information is a plausible experiment, not yet a demonstrated fix. Test recovery across additional cases and retain calm-water performance before promoting a candidate.

## Replay and reproducibility

An initial independent replay implementation did not reproduce the original failure and was rejected. The final recorder calls the original evaluator's initialization, observation, physical stepping and neural execution path; its original-trial and snapshot-continuation gates pass. This prevents presenting an unrelated swim as the observed case. The precise cause of the initial numerical/execution-path difference was not isolated, so these results remain specific to the recorded implementation and conditions.

```powershell
cmake --build --preset release --target pond_current_eval --parallel 6
.\build\Release\pond_current_eval.exe --diagnose --policy docs/experiments/swimmer/gentle-currents/frozen-policy.bin --output build/pond-current-study/diagnosis
```

Open `godot-project/pond/current_replay.tscn`, or choose **Watch current test** in the deep pond after generating the recording. The replay starts at 60 seconds and offers Before stall, Stalled, pause/restart, keyboard-accessible seeking and three selected scenarios: original, current removal, and the one-second Food 3 target change. The transport remains visible at desktop/tablet/phone dimensions. It displays recorded native poses, rather than running PPO or pretending to train. The original calm-water viewer remains available.

Raw evidence: [one-second telemetry](trace.csv), [all branch results](branches.csv), [frozen 60-second state](stall-at-60s.bin), [native validation log](diagnose.log), [verification and recording hash](verification.json). The 20-Hz full recording is generated under ignored `build/` (about 15 MB), rather than duplicated in Git.

Rendered verification passed 32 replay checks, 191 untrained comparison checks, 27 saved-controller checks, 138 presentation checks and 103 menu checks. Screenshots were inspected at desktop, tablet and both phone orientations; physical touch/assistive-device testing remains pending. Internal UI review: usability/hierarchy/feedback/consistency 9, accessibility and small-screen data visibility 8. Prioritize real touch/assistive checks and scrolled-stat access when that platform work resumes; the playback transport itself is visible without scrolling. The final wording change makes clear that Food 3 is **selected**, not moved.

## GPU utilization question

This diagnostic uses CUDA for the original simulated worlds and both neural networks. It is not a full-throughput workload: 64 worlds produce one 64-thread physical-step block, and recording reads data back to the CPU every step and writes JSON. Watching the recording primarily renders the scene. A sampled NVIDIA utilization reading taken after diagnosis had finished is not a measure of training utilization.

A future performance pass should measure simulation steps/second and wall time over increasing batch sizes and reduced readback, with a profiler showing CPU submission, copies and GPU execution. No maximum-hardware-efficiency claim follows from this diagnostic, a percentage meter, memory occupancy or prior large-matrix benchmarks. No stress test or new training was started in response to the question.

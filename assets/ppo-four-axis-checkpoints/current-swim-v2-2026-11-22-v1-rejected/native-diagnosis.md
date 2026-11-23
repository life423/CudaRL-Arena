# Closed V2 current-swim diagnosis — 2026-11-22

This is a read-only analysis of the closed study `C:/Users/aiand/Documents/programming/CudaRL-Arena/.worktrees/minimal-foundation/build/ppo/current-swim-v2-2026-11-22-v1`. It does not run physics, profiling, PPO, or an optimizer and does not authorize another study. The independent full-data audit and its separate closure seal passed with qualified=false (988 SHA-bound inputs). The actor remains rejected: **4/16 baseline → 8/16 final**, not the required 16/16. All 32 updates are spent; no withheld or endurance assessment was run. Final native counters are 89,566,320 controls / 895,663,200 substeps / 344 Adam steps / 0 critic steps.

## Evidence and exact arithmetic

Inputs are the native `baseline-current/summary.csv`, `training-started/assessment-1/summary.csv` through `assessment-32/summary.csv`, `updates.jsonl`, final/baseline `states.csv` and `controls.csv`, and `training-started/current-policy-update-32.bin`, all relative to the absolute run directory above. Numerical reconstruction imports only the offline kinematic functions `metrics`, `motion`, and `body_array` from `C:/Users/aiand/Documents/programming/CudaRL-Arena/.worktrees/minimal-foundation/tools/verify_current_swim.py`; these do not integrate dynamics. The snapshot source inputs are bound by `study-frozen-inputs.json` and `gate-frozen-inputs.json`.

The 16 means are episode-level Gaussian means; deterministic head coefficients are theta[k]=0.5*tanh(mean[k]). For each joint j, latent=base[j]+theta[4j]*flow_forward+theta[4j+1]*flow_side+theta[4j+2]*flow_forward*base[j]+theta[4j+3]*flow_side*base[j]. Flow is in the head's body frame divided by 0.002 m/s. At exact zero flow the explicit branch returns the original 4 outputs without adding zero. Joint axes are proximal yaw, pitch then distal yaw, pitch.

For each of 160 recorded POST states per case, the cost components were reconstructed in double precision from saved binary32 state values: horizontal=||v_xz-v_target_xz||²/(0.008²), masked for controls 1–40; recovery=((v_y-clamp(-signed_depth/2,-0.001,0.001))/0.008)²; lane=0.05*(lateral/0.002)²; depth=0.05*(depth/0.001)²; posture=0.05*(roll/10)²+0.01*(pitch/10)²; effort=0.002*recorded_mean_squared_torque. Quaternion kinematics reconstruct COM and speed. Each component is averaged over 160 controls, then 16 cases. Native interval reward is -float32(0.05)*sum(costs), and episode return is sum(rewards)/8. Reconstructed baseline/final cost totals are 0.111904672527/0.062971690488; native totals 0.111904676400/0.062971698523 differ only by float rounding.

## Outcome progression

Pass counts: updates 1–2:4; 3–11:6; 12–17:8; 18–25:7; 26–32:8. Total mean cost declined 43.727%, but the acceptance count plateaued. Five new cases passed; one original passing case was lost.

| Case / flow (mm/s) / pose | Baseline pass | Final pass | Baseline → final speed (mm/s) | Baseline → final max lane (mm) | Passing updates |
|---|---:|---:|---:|---:|---|
| 0 / (1,0) / 0 | 1 | 1 | 7.997 → 7.558 | 0.676 → 0.626 | 1–32 (32/32) |
| 1 / (-1,0) / 0 | 0 | 1 | 7.103 → 7.259 | 0.917 → 0.833 | 3–32 (30/32) |
| 2 / (0,1) / 0 | 0 | 0 | 7.531 → 7.189 | 3.570 → 2.021 | never |
| 3 / (0,-1) / 0 | 0 | 1 | 7.214 → 7.459 | 2.758 → 1.706 | 12–32 (21/32) |
| 4 / (2,0) / 0 | 1 | 0 | 8.416 → 6.962 | 0.744 → 0.592 | 1–17 (17/32) |
| 5 / (-2,0) / 0 | 0 | 0 | 6.623 → 6.789 | 1.091 → 0.827 | never |
| 6 / (0,2) / 0 | 0 | 0 | 7.111 → 5.953 | 6.790 → 4.677 | never |
| 7 / (0,-2) / 0 | 0 | 0 | 6.524 → 7.008 | 6.529 → 3.232 | never |
| 8 / (1,0) / 3 | 1 | 1 | 8.064 → 7.630 | 0.649 → 0.575 | 1–32 (32/32) |
| 9 / (-1,0) / 3 | 0 | 1 | 7.114 → 7.308 | 0.749 → 0.734 | 3–32 (30/32) |
| 10 / (0,1) / 3 | 0 | 1 | 7.548 → 7.273 | 3.156 → 1.565 | 12–32 (21/32) |
| 11 / (0,-1) / 3 | 0 | 1 | 7.269 → 7.445 | 3.150 → 1.954 | 26–32 (7/32) |
| 12 / (2,0) / 3 | 1 | 1 | 8.511 → 7.271 | 0.894 → 0.412 | 1–32 (32/32) |
| 13 / (-2,0) / 3 | 0 | 0 | 6.610 → 6.856 | 0.840 → 0.766 | never |
| 14 / (0,2) / 3 | 0 | 0 | 7.105 → 6.217 | 6.197 → 3.730 | never |
| 15 / (0,-2) / 3 | 0 | 0 | 6.600 → 6.909 | 7.162 → 3.587 | never |

All 8 final failures have steady speed below 7.2 mm/s. Cases 2, 6, 7, 14, 15 also exceed the 2 mm lane limit. All 16 final cases satisfy RMSE, depth, tilt, path-angle and work-per-distance gates: maxima respectively 2.337627 mm/s, 0.404568 mm, 9.657241 deg, 4.384283 deg and 0.000445442 J/m. Case 2 is close but still fails: 7.189010 mm/s and 2.021357 mm lane. No threshold is changed by this diagnosis.

## What learning changed

| Mean cost rate | Baseline | Final |
|---|---:|---:|
| horizontal velocity | 0.02529755 | 0.02604215 |
| vertical recovery | 0.00031139 | 0.00037348 |
| lane | 0.07899538 | 0.02995909 |
| depth | 0.00068150 | 0.00123768 |
| posture | 0.00661171 | 0.00535207 |
| effort | 0.00000714 | 0.00000721 |

Lane cost decreased 62.1% while horizontal-speed cost increased 2.943%. Thus most reward improvement came from reducing lateral drift, with a remaining propulsion tradeoff. Strong cross-current cases ±Z 2 mm/s never passed any assessment. Their final speeds are5.953–7.008mm/s, lanes3.232–4.677mm. Both -X 2 mm/s cases remain slow, 6.789/6.856 mm/s.

The original +X 2 mm/s pose 0 case loses its pass at update 18 and finishes 6.961912 mm/s (baseline 8.415804). Its measured final proximal-yaw gain is 0.6655 on average over the2–8s saved trace; command standard deviation falls 0.6630→0.4472. The head shares trim/gain coefficients across all states/cases, so stronger steering correction and propulsion are coupled. This is an observed compromise, not proof of an architectural ceiling or insufficient physical actuator authority.

Final coefficients, grouped [forward trim, side trim, forward gain, side gain]:

- Joint 0: `-0.072062181, -0.443833892, -0.330807959, 0.090003933`.
- Joint 1: `-0.124999150, 0.087862101, 0.047475526, -0.094582831`.
- Joint 2: `0.152556696, -0.427490723, -0.046790599, -0.237952013`.
- Joint 3: `0.297407136, -0.278941445, -0.276996927, 0.432565981`.

The final checkpoint envelope checksum was read independently and its means exactly match the JSON ledger. Its frozen 2532 body floats, 2 learned hold parameters and saved gain 0.26211655139923096 are bit-identical to initial policy. Checkpoint selected capacity 16,384/update 32/Adam 344. SHA-256 `39671e78702414a6df146d24e320ae41dd8086c954ffd84e56bfde14642b570c`. The completed independent audit confirms calm retention: 4/4 short swims, 6/6 holds and the exact 28/28 historical long-window keys. Saved summaries are preserved in the rejected archive.

## Concrete next design decision

Do not extend the same 16 affine trim/gain parameters by default. Their trainable correction observes only flow plus one frozen output scalar per joint, although the existing 41 body inputs already contain the requested velocity, measured velocity and lane error. A defensible next version would feed the 3 appended flow features into the existing body MLP through initially zero flow-only input weights, allowing those current inputs to interact with the existing request/state feedback. Keep one motor, the same 4 joint outputs, the original weights/checkpoints, and an exact zero-flow branch. A new preregistered objective should measure simultaneous lane and speed requirements rather than treating reduced aggregate cost as qualification. This is a proposal requiring a fresh bounded protocol and safety/data gates, not evidence that the change will pass. No new physical work is performed here.

## Frozen causal source SHA-256 references

- `tools/current_swim_v2_study.cu` — SHA-256 `e452047d18c0bd17f86c3f44671d24537bded693205e97a923e5e16ee40aefb7`
- `tools/current_swim_math.h` — SHA-256 `f6b791cfaa2b4d561e25c065b9e0a1c32a85141e6b6a27d144ae018cd11edc25`
- `src/nn/motor/current_task_v2.h` — SHA-256 `47da1967616d0233922a173b4850aa988f596066162897b532a461627e6f4627`
- `src/nn/motor/current_model_v2.h` — SHA-256 `3e4eba48a8961a132a0836105fb6c1c013c1ee1956679d25a8b8cb1731552c5f`
- `src/nn/motor/current_motor_v2.h` — SHA-256 `142a678c16bc694a25b64dad3fd07423365325671532d7b6b39647a42ed590e5`
- `src/nn/motor/straight_task.h` — SHA-256 `994f7719f497e237ad4428c74feac2d73ce57e2ea6ea8e12ab8085882e28df22`
- `build/Release/ppo_current_swim_v2.exe` — SHA-256 `359eeed59d6792ff62a7f5ec225e5c607ebb11559189d2a4d9009b4b28ed4d49`

Offline diagnosis helper SHA-256 `8f3fbf37a161e6e1244c864fab9c218ebdabe14b093cf7d50302d7eeeb805d72`. The study and original raw arrays remain preserved under the ignored run directory. The compact rejected archive carries the completed independent proof, closure seal, original-evidence SHA references and per-file SHA manifest.

## GPU timing context

The selected 16,384-world pipeline was the fastest measured end-to-end option: 500,294 controls/s over its two timed 32-control prefixes. Across the 32 full training collections, recorded CUDA physics events total 37.876557 s, observation 0.552853 s, MLP 0.271325 s; CPU readback 10.730119 s, validation 12.502256 s and evidence writing 98.459580 s, within 161.234638 s total collection pipeline time. These categories do not include the remainder of assessment/PPO/runtime overhead. Evidence output is the principal measured pipeline cost; a lower overall GPU busy percentage does not mean smaller batches would train faster. No physics settings were changed for performance.

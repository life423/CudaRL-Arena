# Learned persistent amplitude: calm holding qualified

The November 22 study learned a deterministic holding amplitude that passes the
registered calm-water gates. At update 7 it passed all six original 64-second
holds, all six withheld 64-second holds and both uninterrupted five-minute holds.
All four short swims and the original 28 long-window passes remained unchanged.
Independent verification confirms the result. PPO stopped immediately; there was
no retry, extension or production adoption.

This establishes **calm holding from rest in the tested cases**, using a learned
amplitude in the existing body motor. It does not establish moving-start braking,
disturbance/current recovery, arbitrary request switching or indefinite holding.

## What changed

The [frozen protocol](HOLD_LEARNED_AMPLITUDE_PROTOCOL_2026-11-22.md) made the
persistent amplitude a learned policy decision and matched training to complete
64-second holding episodes. Earlier experiments' short, externally attenuated
segments remain separate failed studies.

The canonical 336 MLP and inherited optimizer stayed frozen. Native CUDA PPO/Adam
learned two fresh parameters: Gaussian mean `mu` and variance coordinate `v`, with
`sigma=0.05+7.95*sigmoid(v)`. Each holding episode samples one Gaussian latent `u`,
uses `g=sigmoid(u)` throughout its 1,280 physical controls, and applies four joint
latents `g*b(x)` through the original tanh/filter/servo. No extra per-control noise,
teacher, neutral label, posture clamp, changed physics or scripted recovery was
used. All four body outputs still come from the same frozen body network.

PPO scores that one sampled latent once, using the whole undiscounted episode
return. A leave-one-out baseline uses other episodes in the same initial case,
without conditioning on the sampled amplitude. One positive global residual
scale comes from independent calibration. Deterministic tests **retain the learned
amplitude** `sigmoid(mu)` and disable sampling.

Swimming is structurally preserved by its original computation. This is genuine
restricted policy-parameter learning, not evidence that a fully trainable shared
network retained swimming without forgetting or learned a general movement motor.

## Before learning and actual progress

The original gain-one baseline passed 4/4 short swims and the same 28 long windows,
and passed 0/6 holds. The new initial deterministic amplitude was
approximately 0.99 and also passed **0/6 holds** before any optimizer update.
Holding was not installed in the initial actor.

Independent calibration collected 256 uninterrupted 64-second episodes. **52/256**
passed every physical hold gate, split 27/25 across poses 0/3. These were sampled
policy actions, not manual amplitude assignments. Calibration's global within-case
residual RMS was 0.48350932; absolute return p99/max were 1.17453468/1.17457247,
within their frozen caps. The report preceded the separately launched PPO phase;
the first fresh training batch repeated coverage and integrity gates.

| Deterministic assessment | Amplitude | Original holds | Short swims | Long windows |
| --- | ---: | ---: | ---: | ---: |
| Initial policy, no updates | 0.99000007 | 0/6 | Original unchanged | Original unchanged |
| Update 5 | 0.63343281 | 0/6 | 4/4 | 28/30 |
| Update 6 | 0.44328344 | 0/6 | 4/4 | 28/30 |
| Update 7 | 0.26211655 | **6/6** | **4/4** | **28/30** |

The first six post-update assessments failed holding. Update 7 was the first full
original pass and stopped PPO. Its learned `mu=-1.03499663`, `v=1.34003341` and
training `sigma=6.35033894` are saved with the complete fresh optimizer state.
Sampling variance is not active in deterministic acceptance or frozen playback.

## Registered physical acceptance

Every case includes its initial sample and every 50 ms endpoint. Limits remain:
3D speed <=0.5 mm/s, fixed-anchor error <=2 mm, absolute depth <=1 mm, lane <=2 mm,
tilt <=10 degrees, and forward excursion <=2 mm throughout the final eight seconds.
The creature's joints may move; no joint freeze is required.

| Worst metric across each suite | Six original 64 s | Six withheld 64 s | Two 300 s |
| --- | ---: | ---: | ---: |
| Peak 3D speed, mm/s | 0.098904 | 0.095227 | 0.091504 |
| Peak fixed-anchor error, mm | 0.838193 | 0.607249 | 0.385946 |
| Peak absolute depth error, mm | 0.829287 | 0.596666 | 0.364766 |
| Peak lane error, mm | 0.183693 | 0.180107 | 0.177336 |
| Peak tilt, degrees | 1.90670 | 1.68714 | 1.49842 |
| Final-eight-second forward excursion, mm | 0.000102 | 0.000055 | 0 at saved precision |
| Full passes | **6/6** | **6/6** | **2/2** |

Original cases use poses 0/3 with depth offsets 0/-0.5/+0.5 mm. Withheld cases
use poses 1/2 with offsets 0/-0.25/+0.25 mm; they never enter calibration or
training. Endurance uses centered poses 0/3 for 6,000 uninterrupted controls each,
without resets or reference changes. Its final window begins at control 5,840;
the 64-second window begins at 1,120. Zero excursion at recorded precision is
not a claim of mathematical immobility or indefinite stability.

All seven swimming assessments retained four short cases and the same 28 long
window keys, with CSV traces byte-identical to baseline. The two known startup
speed-window misses remain failures; 28/30 is not complete formal long-swimming
acceptance. Depth offsets being tolerated inside the registered limits are not
proof of general disturbance rejection or exact depth recovery.

## Exact spend and independent verification

| Consumed work | Controls |
| --- | ---: |
| Original baseline assessment | 10,720 |
| Six initial deterministic holds | 7,680 |
| Independent 256-episode calibration | 327,680 |
| Seven fresh 256-episode PPO batches | 2,293,760 |
| Seven post-update assessments | 75,040 |
| Six withheld 64-second cases | 7,680 |
| Two five-minute holds | 12,000 |
| **Total** | **2,734,560** |

Actual physical substeps: **27,345,600**. PPO updates: **7**. Fresh two-parameter
Adam steps: **112**. Collection/training critic calls and steps: **0**. The
remaining 25-update allowance, 8,460,000 controls, is closed and cannot be reused.
The result consumed the reserved validation once and qualified without a physical,
numerical or integrity fault. No additional study or policy adoption is automatic.

Native accounting records 6.3478226 prior active seconds and 37.0096797 training
seconds, **43.3575023 combined**. Process logs record 6.4462761 calibration and
37.069543 training seconds, **43.5158191 combined**, below the 900-second cap.

`tools/verify_hold_learned_amplitude.py` independently checked all seven updates,
eight head/optimizer checkpoints, full per-control body observations, four joint
latents, complete episode rewards and all deterministic suites. It recomputed
actual scalar Gaussian likelihoods, whole-episode returns, leave-one-out exclusion,
the global calibration scale, physical peaks including reset, final-window
excursions and continuous 300-second counters. Maximum errors: old likelihood
9.537e-7, frozen CPU/GPU body-network output 9.304e-7, LOO advantage 5.960e-8.

All 118 independently read inputs retained their SHA-256 hashes. The compiled
read-only envelope checker validated 30 checksummed binaries containing
589,749,489 payload bytes, with zero physics or optimizer steps. All 81 frozen
execution inputs were unchanged across both phases. A separate preservation
check verified all **5,883 captured prior artifact files** unchanged; its result
is `build/learned-amplitude-preparation-2026-11-22/preservation-after.json`.
Earlier policies, saves and study evidence remain preserved; later viewer work
is separately authorized.

## Evidence and viewing

Full executed source/executable, all eight policy/optimizer states, eight complete
calibration/training batches, body/joint pose traces and independent verification
remain in `build/ppo/learned-amplitude-2026-11-22-v1`. Compact protocol, initial
and final policies, measured summaries and provenance are preserved in
`assets/ppo-four-axis-checkpoints/learned-amplitude-2026-11-22-qualified`.
Ignored build evidence is not preserved by Git alone.

The native viewer may inspect the resulting saved research candidate under the
user's existing request. Recorded poses and live frozen physics are distinct from
ongoing learning. Viewer fidelity controls and rendered checks have separate
accounting; this study does not claim their verification. Canonical checkpoint
336 is unchanged and no merge, publication or production replacement occurred.

The later native viewer-package fidelity check used **80 separate controls /
800 substeps**, with zero optimizer work. Its maximum saved-pose mismatch was
4.903e-12 model units. Evidence is in
`build/ui-native-v2-2026-11-22/verification.json`; these controls are not part of
the closed learning study. Rendered GUI checks are reported separately.

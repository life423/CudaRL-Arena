# Corrected shoreline v4 qualification

CPU qualification passed36/36 cases on its single preregistered launch:
68,000 attempted/accepted controls and688,000 physical substeps in21.779152s.
There were no failed/skipped cases, additional trajectories, CUDA calls or
optimizer updates in this CPU allocation. Original250 asset binaries and both
live policy packages remained byte-identical.

| Gate | Actual result |
| --- | --- |
| Retained original/new behavior |30,400 controls each; body/filter/observation differences zero |
| Swimming |All4 short cases; same28/30 original long-window pass keys retained exactly |
| Resting |All14 original/withheld/endurance holds pass original criteria |
| Real bank contact |8/8,all1,600 controls; maximum observed overlap115.389µm,limit500µm |
| Half-step convergence |4/4,all800 controls; maximum COM position0.212712mm,velocity0.095927mm/s,orientation0.162040° |
| Outward frozen-motor playback |4/4 full60s horizons,all4,800 controls; physical qualification only |
| Passivity |Maximum recorded passive-energy excess0 |

The corrected certificate uses the gradient of gap=y−height, rather than reversing
its horizontal components. Runtime4/query arithmetic2/envelope2 preserve body
dimensions, intended physical force, torque/filter/servo limits, integration and
acceptance criteria. Within a control, an accepted POST query can serve the next
identical-body/terrain PRE; control entry obtains a fresh query. Independent pure
checks passed2,513 targeted plus4,885 existing assertions before dynamics. The
checker preflight used zero integration and400 placement inspections. The physical
run additionally recorded600 placement inspections and59,776 pure product-endpoint
queries. These are not extra physical controls.

The separate CUDA gate passed all8 matched cases×128 controls:1,024 CPU plus1,024
CUDA controls and10,240 substeps each,in16.000674s overall. Maximum position
difference was3.577e−7 model units and linear-velocity difference4.657e−9m/s;
all original tolerances passed. The raw receipt also records13.413s GPU physics
event time and0.712s CPU-reference time. This eight-world agreement check is not
a GPU throughput benchmark or evidence of autonomous navigation.

## Installed Godot version

The corrected original-motor DLL was installed at the existing canonical path
after the passing gates. The previous mapped DLL was renamed into preserved
rollback evidence; the user's existing PID26188 continued running. No computer
input, app close or manual relaunch occurred. The already-open window retains its
older mapped runtime until the next normal Project Manager Run.

The actual registered `live` project was loaded headlessly through its ordinary
main scene. All9 entry checks pass with runtime4/query2/envelope2, original saved
motor, exact CPU/CUDA source bindings and zero physical controls/substeps. Details
shows CPU36/36 and CUDA8/8 independently, plus current certified whole-body
clearance, measured/requested horizontal speed, tilt and depth. It continues to
state that learned pond navigation is unqualified; no candidate was installed.

Private rendered UI verification passed248/248 checks across1280×800,390×844,
768×1024 and844×390 with zero integration/optimizer work. All16 screenshots are
retained; the four new verified Details views were inspected. Existing aesthetics,
camera/zoom, modal focus and controls remain. Short landscape Details scrolls.

## Evidence bindings

- CPU protocol/seal: `build/pw30/sealed-cpu-v4.json`,SHA256
  `2b2bb22b88c8f1a5c0ea1994eaf997d5e214749c7c0fc2a995f744239b9e16d6`.
- CPU receipt: `build/pw30/run-v4/result.json`,SHA256
  `3651799d383aa87cfd85e3f1aa254e214b0afcd567925c77ada7f8e585093920`.
- CUDA receipt: `build/pw31/gpu/result.json`,SHA256
  `5299c8c1b27fa8e41421cdae490a75547b1ebab8bff9e27da17f49cc8ff56aee`.
- Installed DLL SHA256:
  `1b69c205700b2ffbac913e5e088c93eb8f6c2e27467f244a22e4695ad3fbabb9`.
- Original saved motor SHA256:
  `4b6400c4db65873bd346c3e0c4fed384e3a77f9e5174d0dcea5d74073b94616d`.
- Installation/entry/UI: `build/pw30/installation.json`,
  `registered-entry.json`, `ui/verification.json`.

Earlier v3 traces/receipts remain preserved. Their sloped-body certificates used
the discovered sign error, so those reported bounds must not be treated as valid
independent certificates. The earlier60s Godot slowdown is an observed reproduction
under that historical simulator, not a matched before/after baseline for a v4
candidate. A comparison across the simulator correction cannot attribute a change
solely to learning. No new60s default-Godot trajectory was run in this v4 allocation.

## Historical candidate v1 status

Details now explicitly says that original motor336 is playing and that the
separate candidate stopped after8 PPO updates after leaving the2mm lane. The
candidate's first long-swim assessment reached12.281mm lateral error. A small
SHA-bound receipt references its actual preserved checkpoint, training result,
assessment and control trace; missing/changed local evidence shows an unavailable
fallback. No candidate weights were copied into or loaded by the app.

The changed registered entry passed10/10 zero-step checks. The existing rendered
UI suite passed256/256 checks at the same four sizes, with no physical trajectory
or optimizer calls. Candidate-status evidence is in
`build/pw30/candidate-status-ui/verification.json`; the open user app was untouched.

## Portable candidate v2 status

The latest Details status records32 completed GPU PPO updates. All20 retention
cases pass: the original28 passing long-swim windows,14 resting cases and4 short
swims remain retained. Requested turns pass3/16, unchanged from the untrained
baseline; turning and autonomous pond navigation remain unqualified. Original
motor336 remains selected. The separate candidate was not installed.

Small native receipts, case/window tables, a movement summary and a checkpoint
digest are published under `live/research_data/studies/turn32-v2-2026-11-23/`.
The candidate itself is preserved separately at
`assets/research-evidence/pond-turning-v2/candidate-32.bin`,SHA256
`23b48585239dc2b2699a3c0ef1294877f3116412783a87a96e8a44b9beb48cfb`.
The app does not load these candidate weights. Its portable study files total
35,720 bytes including the historical v1 record and README; large rollout traces
remain outside the app. SHA-bound recorded results remain visible in a source
checkout without the local DLL or ignored build directory. A new native build
still needs its own qualification binding; historical results do not qualify it.

Final verification passed10/10 registered-entry checks,8/8 source-only checks
without a DLL or private build files, and128/128 rendered desktop/phone checks.
Eight screenshots were saved and both verified Details views inspected. All
checks used zero physical controls/substeps, optimizer updates or CUDA calls.
The open user app was untouched. Evidence is in
`build/turn32-ui/verification.json`,SHA256
`42550fe0e1035d6909a8786ca72b9d5f778a210e4908ed7d1f8737d4cbf7449f`.

## Course-learning v3 status — November 24

The terminal v3 study is now the recorded result in Details, labelled
“Course learning · v3”. It completed 32 GPU PPO updates and retained all 20
original movement cases, including 14 resting cases, four short swims and all
28 original passing long-swim windows. Requested turns remain 3 / 16, unchanged
from its baseline. This is not qualified turning or autonomous navigation.

Original motor 336 and the verified runtime DLL remain byte-identical and
selected. The separate v3 checkpoint is preserved at
`assets/research-evidence/pond-turning-v3-course/candidate-32.bin`, SHA256
`e6415ff704750f331d7aefff28e06ef5d54db194aaef1057fb4123f8ffe4bcb6`.
Portable native assessment/training outputs, baseline result, case/window tables,
movement summary and checkpoint digest occupy 24,310 bytes under
`live/research_data/studies/turn33-v3-course-2026-11-24/`. Candidate weights are
not loaded by the app. Earlier v1/v2 records remain available.

The registered ordinary main-scene entry passed 10 / 10 checks. A source-only
copy without a native DLL or private build files passed 9 / 9; rendered desktop
and phone views passed 128 / 128. Both latest Details screenshots were inspected,
including the study label and full outcome. Eight screenshots were saved. These
UI checks executed zero physical controls/substeps, optimizer updates or native
CUDA kernels. The user's open app was not controlled or closed. Source changes
take effect on its next normal Project Manager Run. Evidence is preserved under
`build/turn33-ui/`.

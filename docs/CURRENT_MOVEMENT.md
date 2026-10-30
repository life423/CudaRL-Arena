# Current movement handoff

**Latest:** task4/reward2 learner passes4/4 and is selected for the isolated
experiment. Details and checkpoint identity are in the final section below.

Latest local commit remains3e0e4ab (verified numerical repairs and rolled-contact
check). All later movement work is uncommitted in this worktree. Original
checkout/baseline and personal saves are unchanged. No push/merge/deployment.

## Working baseline and view

Experimental model6 has a coherent rod/fin mass/inertia/area description and
four torque-limited internal joint actuators tracking reference position, rate
and acceleration. The unmodeled old passive joint damper was a strong actuator
limitation. Matched prescribed-stroke speed rose .275 to7.968mm/s with the new
actuator alone, then8.228mm/s with the complete geometry/drag model. Energy,
clearance, CPU/CUDA and timestep checks passed; dry COM drift.200mm/8s remains
a numerical limitation. This is approximate free-water physics, not validated
contact or real-world hydrodynamics.

Task4 is fixed4s straight swimming toward8mm/s; scoring window2..4s. Gates:
mean speed7.2..8.8mm/s, velocity RMSE<=2.5mm/s, lateral<=2mm, depth<=1mm,
COM path direction<=5degrees, tilt<=10degrees, positive work/distance<=1mJ/m,
plus native finite/torque/energy/10mm-envelope safety. Head yaw is reported
separately (~27degrees); no head-stabilization, arrival/braking/holding claim.
Earlier task3 arrival/heading acceptance remains0/4, unchanged.

The supplied stroke generator is intentional. A neural policy chooses two
parameters: strength and frequency. The **untrained initialized default passes
4/4task4 cases**,8.326mm/s steady speed. It remains the preserved fallback.
The reward2 learned candidate now also passes4/4 with better speed tracking and
effort. This is learning stroke controls, not discovering the supplied waveform.
Selected weights/traces are in `build/straight-motor/2026-10-30-task4-steady-reward-v2/`;
`selected-controller.json` selects `update-25.bin`, explicitly `runtime_adopted=false`.

From the worktree run `./tools/view-movement.ps1`, or open
`replay/project.godot` and press F6/F5. The standalone Godot viewer defaults to
the selected learned recording and can compare the passing original default. It reads
recorded20Hz poses only: no simulation, inference or learning. Geometry is the
recorded model6 rod/fin, not the old viewer's solid envelope. Main
`godot-project` and its live runtime physics2 remain unchanged.

## Learning results and correction

First task4 trial:64updates,33440controls/334400steps,3.756s native. Default4/4;
candidate0/4,12.735mm/s,return-1.898 versus-1.274baseline. Safety passed.
The one approved follow-up used complete80-control/4s episodes instead of
32-control/1.6s fragments, with GAE lambda1 rather than.95. Returns are the
complete observed discounted reward sum; the critic remains an advantage
baseline but no longer supplies a fragment-end bootstrap. Gamma.99, exploration,
optimizer, physical model and task gates stayed unchanged.25batches consumed
32672controls/326720steps,3.296s native;250Adam steps each versus256previously.

Follow-up return improved-1.274 to-1.210, but mean speed8.871mm/s exceeded8.8:
candidate0/4; default retained. Startup return improved .0774 while the measured
steady-window return worsened .0142. The objective still rewards startup more
than steady-window acceptance; this is a concrete next learning-design decision,
not a reason for another body rewrite. No third trial is authorized automatically.

## Fast workflow and verification

`tools/run-straight.ps1` configures/builds only the opt-in learner target when
needed; arguments control episode length, updates, lambda, seed, speed and wall
limit without source edits. It rejects duplicate output identities and allocations
over32768training/33440total controls, snapshots44source files and executable,
registers numeric settings, publishes `build/straight-motor/latest-status.json`,
and returns `result.json` immediately. Native wall stop<=20s; wrapper stops its
own process after that allowance+5s. Ready-compiled setup was .575s; build time
is separate. It performs no automatic retry, extension or live adoption.

Strict CUDA build passed. Four focused zero-integration tests passed, including
full-return independence from an intentionally wrong terminal bootstrap. Actual
CPU/CUDA physical/learner execution ran; it was not a GPU skip. Last-run source
and1220older evidence files passed preservation checks; original checkout clean.
Physical source, stroke interface and task4gate hashes matched between trials.
All protocols/results/initial weights/optimizer snapshots remain in their unique
ignored build directories. The first stopped model run (136attempts) and its
matrix-operator fix chronology remain preserved, not replaced by the passing run.

## Latest approved correction: steady-window reward2 passed

User approved23:00:47. Only tracking reward timing changed: speed/path costs
apply to intervals41..80 (2..4s), effort applies to all80intervals. Full-episode
returns retain credit from warmup actions to later scored outcomes. No physical
model, stroke bounds, seed, optimizer, exploration or acceptance gate changed.
Seven focused zero-integration tests and strict incremental CUDA build passed.
One fresh run:32672controls/326720native steps,25updates,3.264s native/.586s
ready-compiled setup, no faults/retries. Default4/4, candidate4/4. Speed
8.326->7.958mm/s; RMSE2.147->1.805mm/s; lateral.757->.594mm; work9.071->8.149microJ.
Work/metre.422069->.387931mJ/m. Path angle.867->1.398degrees, still within5degree
gate. Head yaw26.6degrees remains a limitation; no arrival/holding/current claim.

Selected artifact `build/straight-motor/2026-10-30-task4-steady-reward-v2/update-25.bin`.
SHA256 `31C9982778F32A5E0F421CDBA88AF4431ED27431A355C66D9E290D60EDC97092`.
Selection is for this frozen experiment, not live runtime adoption. This is a
checksummed research snapshot including actor/critic, Adam, physical/task/filter
states and episode counters; automatic resume/production load remain untested.
Default `initial-policy.bin`, failed candidates and all previous protocols remain.

The selected policy meets the frozen promotion rule:4/4, return improves>.01,
no worse speed RMSE or positive work/metre. Return-.147603->-.104422 under
reward2; do not compare this numeric return directly to old reward1 totals.
Preservation:44frozen sources unchanged,1220older evidence files unchanged,
original checkout clean. No additional physical/training execution. Movement
code remains uncommitted after3e0e4ab. Standalone `./tools/view-movement.ps1`
now opens recorded winning poses and compares the passing default. No native
world/inference runs in that viewer; `godot-project` still uses runtime physics2.

## Separate live demo completed

`./tools/view-live.ps1 -SkipBuild` opens the verified live model6 project paused.
The selected learned actor runs frozen, controlling stroke strength/frequency;
the world actually integrates native physics. Start/Pause/Reset and speed display
are available; it auto-pauses at4seconds. A small portable actor and provenance
are tracked so the live demo does not depend on ignored research checkpoints.
Native4/4 and Godot4/4 matched the winning recording; rendered live execution
passed at720x1080. Total7,200/9,600registered substeps, no retries or training.
See [live instructions and exact limits](LIVE_DEMO.md). Existing physics2 and
recorded playback remain separate. Local commit is authorized after these checks.

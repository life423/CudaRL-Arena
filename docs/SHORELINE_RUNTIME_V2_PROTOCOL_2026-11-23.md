# Shared shoreline product qualification v2 — November23,2026

Status: preparation; no v2 dynamic launch has started. After v1 closed at its
half-step gate, the user explicitly requested hurrying the next test and getting
to training, with native CPU/CUDA use and no Python training/runtime. This new
finite correction check supersedes the v1 stop ONLY for the phases below. Keep
the v1 sealed sources, executable, raw failures and unused budget closed.
The selected subsequent learning goal is whole-pond exploration.

## Smallest physical correction

V1's brief impact yielded16.48% different damping impulse/energy across5ms and
2.5ms stepping; the remaining velocity offset caused0.252716709mm COM difference
against the unchanged0.25mm gate. Do not alter that gate.

Runtime v2 evaluates the unilateral spring/damper at the solved new normal
velocity. For gap g=y−H, row a=grad(g)·J and velocity v, use predicted gap
g_new=g+h av, elastic force−k g_new while compressed, and approach-only damping
−c av while approaching an ALREADY compressed contact(g_old<0). First-touch
steps receive the implicit spring; the dashpot becomes eligible on the next
step. This onset convention avoids a discontinuous first-touch dashpot law
having no consistent implicit velocity solution. No new gain is introduced.
Add h²k aᵀa and hc aᵀa to the discrete solve, with
−hk g aᵀ on its right side. A finite active-set solve must make compression and
approach classifications consistent with the solved velocity. Use a fixed
16-iteration cap and reject a repeated active-set cycle. Unconverged
classification is an explicit failure. Contact multipliers must be compressive;
there is no pose projection, force replacing a neural decision, or stabilization
force outside the declared contact law.

Keep k0.8N/m and link-mass damping scale2sqrt(k mass), geometry envelope v2,
0.5mm overlap limit,10µm query uncertainty and131072query visits. Separate true
contact damping from implicit elastic work/dissipation. Retain v1's finite
diagnostic, discrete kinetic-work and total K+U passivity checks unchanged.
No-contact inertia, buoyancy/drag, torque±10µNm, joint±1rad, target1Hz,
servo3Hz, filter-rate/acceleration caps and10×5ms production cadence remain.

## Phase1: unchanged CPU product qualification

Use all36 v1 cases and their original limits/scoring, with a separately bound v2
executable/protocol/result identity. At most68000attempted controls and688000
attempted physical substeps,900s, no retry. This includes original/new paired
retention,8bank impacts,4half-step references and4outward saved-motor cases.
The v1 clear-water compatibility fixture and endpoint-only product-prefix
queries remain explicitly distinct from the actual product pond. All attempted
cases/failures/skips and the first PRE/candidate must be recorded. No threshold,
case, horizon or success criterion is changed after viewing results.

## Phase2: native CPU/CUDA fixed-input parity

Run only after phase1 passes. New standalone CUDA translation units use explicit
`--fmad=false` for physical arithmetic and a validated fixed-geometry cbrtf
profile: argument bits889811040/859599123/861293837 map to the original CPU result
bits1006834286/996615495/997278936. Host cbrtf remains unchanged; unknown arguments
fail. Validate current fixed volumes and host results before any integration.
This corrects backend arithmetic; it does not remove a force or change its
equation. Generic optimized GPU NN/PPO kernels are not globally disabled.

Exactly8matched cases×128controls on EACH backend:4straight saved-motor starts
and4bank impacts at headings0/90/180/270. Commands are identical between backends;
recorded CPU motor means/commands may supply the common sequence. Preserve all
body/servo/filter state. Compare every5ms PRE/candidate/accepted body, reference,
requested/applied torque and energy ledger through an observational trace hook.
No independent closed-loop parity claim follows from identical inputs.

Use the ACTUAL preserved CPU/CUDA agreement limits (not a remembered substitute):
body position in model units/quaternion/joint angles2e−5; angular/joint rates
2e−5; linear velocity2e−7m/s; normalized motor/torque values2e−5; energy
1e−12J+1e−4relative. Report the first numerical divergence and do not relax limits.
Unknown certificates, nonfinite state, physical rejection or parity disagreement
stop affected work. One launch,≤2048total attempted controls/20480substeps,
≤180s. No additional profiling trajectories are included in this phase.

Combined cap:70048attempted controls/708480physical substeps,1080s native process
time and2GiB evidence across phases, zero optimizer. Static algebra, geometry
queries, compiler attempts and pure GPU kernels are separately counted rather
than misreported as physics or training. Hash and preserve all250asset binary
saves and the actual live policy before/after every outcome.

## Preparation, adoption and learning boundary

Preserve v1. Complete zero-integration active-set algebra/geometry/provenance/
observation checks and compile each phase before sealing its source/executable/
inputs. Phase1 CPU validation may run while the independent CUDA translation
unit compiles; its shared physics source is frozen at launch. Any required
equation/source correction stops affected work rather than silently changing
the running experiment. Phase2 requires its own completed compile and static
preflight before its seal/launch. Use short preflighted paths. No automatic
new attempt if v2 fails.

CPU product installation requires phase1 success and matching source/DLL/input
bindings. GPU learning requires phase2 success as well. Update and inspect the
actual Godot Project Manager Run path after installing the qualified CPU build.
Keep the user's existing save namespace and the original saved motor intact.
Actual physical bank response and a live frozen swimmer are not learned pond
navigation. A separate frozen-request/repeated-turn qualification must establish
which interface the motor can execute before chooser PPO; if it fails, prepare
a bounded continuation of that SAME motor with swim/rest retention instead.

Training is requested but held until its prerequisites and own finite PPO
protocol are concrete. No optimizer is part of this correction run. Any future
GPU optimization/profile/training allocation must count all simulated worlds,
internal substeps, rejected batches and optimizer steps, preserve failed data,
and select batch size from measured end-to-end throughput/memory. No Python,
PyTorch or device clock/power changes are introduced.

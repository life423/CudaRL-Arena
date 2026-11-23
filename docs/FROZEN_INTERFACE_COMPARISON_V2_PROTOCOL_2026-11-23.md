# Arithmetic repair and frozen interface comparison v2

The user explicitly authorized fixing CPU/GPU mismatch, rerunning the controlled
comparison, preserving the saved motor and updating Godot. Their subsequent
instructions require first-divergence analysis, unchanged physical equations
and tolerances, corrected-runtime skill retention before the 88 cases, complete
accounting, a compact comparison and one conclusion, then no training.
The actual user messages are retained with this study's grant.

This is ONE new bounded attempt, separate from the closed v1 attempt. No
motor/chooser training, optimizer updates, saved-weight changes, physical-model
retuning, new stabilization, shortened prerequisite, automatic retry, push or
merge. All old sources, checkpoints and failed evidence remain immutable.

## Repair hypothesis and zero-time diagnosis

The narrow candidate arithmetic profile is
cpu_compatible_no_implicit_fma_v1. The isolated v2 CUDA mechanics target uses
--fmad=false to prevent implicit contraction of multiply/add/subtract. Host
flags remain unchanged. Model6 equations, geometry, masses, water parameters,
1 Hz reference filtering, 3 Hz physical tracking, +/-10 microN*m caps, 50 ms
control and ten5ms substeps remain unchanged. Existing neural static libraries
and saved 41-input actor parameters are unchanged. The production Godot CPU
runtime already uses the reference host path; no experimental actor is installed.

NVIDIA documents this flag as controlling implicit FMAD/FFMA/DFMA contraction;
the installed CUDA13.2 help confirms that behavior. This profile is a proposed
numerical correction, not a new force law or a guarantee of behavioral parity.
Do not zero buoyancy by special case or erase any real force.

The supplemental trace identified a second arithmetic source: cbrtf of the
same argument rounds to adjacent CPU/CUDA floats for the head and middle
link. The final isolated v2 GPU profile caches the ORIGINAL HOST cbrtf bits
for the three exact fixed-geometry radius arguments; CPU arithmetic is
unchanged. Recompute/check these bindings against the unchanged geometry
before physical entry, and reject unknown arguments rather than adapting
geometry silently. This retains the equal-volume sphere formula and original
CPU force coefficients; no radius parameter, force law or stabilization is
changed. The no-FMA-only candidate and failed proof remain preserved.
Canonical argument bits are889811040/859599123/861293837; original host
result bits are1006834286/996615495/997278936, in head/middle/tail order.
The table is derived from the recorded unchanged formula, not fitted movement.

Before integration, evaluate identical saved PRE states and commands from
the failed v1 baseline. Compare CPU with an original fused CUDA evaluator
and the corrected unfused evaluator, both built from the same diagnostic
source. Cover all107 old control inputs from both recorded backends and
three fluid densities (642 fixed-input rows per evaluator). These are
zero-time algebra/force evaluations, not motion integration; report their
counts and actual kernels separately from physical controls/substeps.

Instrument ordered named stages: filter acceleration/rate/angle, articulated
kinematics and COM, volume/mass/buoyancy, assembled M/R/C/force, discrete system,
servo demands, solves, requested/applied torque and next-velocity algebra.
Stop before body integration. Locate first differences in these traced stages
and show downstream propagation; do not assume the known buoyancy difference
explains the complete failure. Record limitations of stage-level attribution.

Corrected neutral buoyancy must be numerically zero on both backends; density
variants must retain their physical nonzero force. Unfused reference-filter
float results must match host bits on these fixed inputs. Compare body/COM/
velocity under the existing physical tolerances and matrices with fixed
absolute1e-12 plus relative2e-5 limits. Independently verify raw outputs,
source/compiler profile/executable hashes and the absence of time integration.
No adjustment of these limits after observing outputs. A failed pure check
may be repaired before source freeze, preserving all attempts; it does not
authorize a physical trial until a complete readiness proof passes.

The first two zero-time profiles ran once each on642 inputs. The unfused
profile corrected filter/neutral-force rounding but failed six stored
system_R[2,0] comparisons:1.1368683772161603e-12 absolute difference exceeds
the fixed1e-12+2e-5relative limit. Both original traces and the failed reader
are preserved. Before changing that arithmetic, allow exactly one supplemental
zero-time CUDA kernel plus642 host evaluations on the same inputs, tracing
viscous radius/coefficient and ordered resistance addends. No body advance,
actor or optimizer is involved. This diagnostic does not extend any physical
budget. If its cause supports a narrower arithmetic repair, record it and
freeze the final profile before one final642-row corrected arithmetic check;
otherwise close the preparation as blocked. Keep the original fixed limits.

## Phase order and finite physical budget

| Phase | Maximum controls | Maximum attempted5ms substeps |
| --- | ---: | ---: |
| Fresh unchanged CPU/CUDA matched-command prerequisite | 1000 | 10000 |
| Original full retention, CPU | 46240 | 462400 |
| Original full retention, corrected CUDA | 46240 | 462400 |
| Original 88 interface trajectories | 34200 | 342000 |
| TOTAL | 127680 | 1276800 |

One combined native process cap is900 seconds, including loading, inference,
scoring and evidence I/O. Collection stops by890 seconds to reserve10 for
closure; an external watchdog enforces900. At most2 GiB new research evidence,
including causal snapshots and prerequisite/retention/comparison records,
with a separate1 GiB free-space reserve. Verify a schema-derived worst-case
projection before physical entry. Compilation, zero-time diagnostics and
independent offline audits cannot execute hidden time integration. Record
their separate counts; they are outside the physical-process timer.

All actual attempts, partial/rejected substeps and failures count. No early
success, resampling, recollection or automatic restart. Ordinary interface
bank/depth/tilt failures stop that case; hard numerical/solver/contact/energy/
backend faults stop the entire attempt. Finish the predeclared retention
assessment after ordinary behavioral misses if no hard fault, then block
comparison if either backend regresses. A blocked phase does not transfer
unused allocation elsewhere.

## Fresh numerical prerequisite: unchanged length and tolerances

Retain the v1 four paired gate cases: A/B pose0 warm80 zero offset through120
controls; A/B pose3 warm90 +5 through130. Execute CPU and CUDA with identical
actual commands per interval. This is (120+120+130+130)*2=1000 controls.
Do not stop at the old107 failure or shorten the baseline to80.

Retain all v1 per-component tolerances: positions/quaternions/joints and angular/
joint velocities2e-5 in their existing units, linear velocity2e-7, motor state/
normalized torque2e-5, energy absolute1e-12 plus relative1e-4, same-observation
NN/command2e-5. Compare exactly the same fields and record all differences.
The isolated corrected profile must pass the whole prerequisite before any
retention or interface trajectory. Saved motor parameters remain frozen.

## Original skill retention on the corrected runtime

Run the complete old25-fixture suite independently closed-loop on CPU and
corrected CUDA, using the actual frozen41-input network and saved holding
gain, original observations, original fixture initialization/offsets, world
arenas and exact scorers. No teacher outputs or old-policy bypass.

| Original category | Fixtures | Controls per backend | Requirement |
| --- | ---: | ---: | --- |
| Short swimming | 4 | 640 | all4 original short requirements |
| Sixty-second swimming | 2 | 2400 | all28 previously passing window keys |
| Depth recovery | 3 | 3840 | all3 original recovery requirements |
| Five-minute swimming | 2 | 12000 | all148 previously passing window keys |
| Original64s holds | 6 | 7680 | all6 original holds |
| Withheld64s holds | 6 | 7680 | all6 original holds |
| Five-minute holds | 2 | 12000 | both original endurance holds |
| TOTAL | 25 | 46240 | all retained requirements |

Keep known startup-speed misses explicit. Extra passing startup windows are
allowed, but every prior passing key must remain. The old long-straight test
arenas are retained only for these original regression fixtures; the pond
comparison arena and boundaries are not enlarged. Retention mechanics use
the original rectangular-domain/contact/energy guards, not the pond ellipse.
Report the distinction rather than treating long straight swimming as pond
navigation. Keep original endpoint-based retained-skill criteria; recorded5ms
extrema are additional diagnostics, not secretly tightened old thresholds.

Preserve full world/task/control records and all accepted5ms body/energy states
for both backends, plus the exact case catalog, gain, arena, decoder, window
keys and native summaries. Independent audit reconstructs all original
requirements from the records. If either suite regresses, main comparison
remains blocked and no interface qualifies from that attempt.

## Unchanged controlled comparison

Use the exact88 main-case grid and arm semantics of
[v1's frozen protocol](FROZEN_INTERFACE_COMPARISON_PROTOCOL_2026-11-23.md),
not a tuned grid. Four fixtures are (pose0,warm80), (pose3,80), (pose1,90),
(pose2,90), each with both observers and all11 scenarios. Case ID is
((fixture*11+scenario)*2+arm). The saved motor and physical settings match.

| Scenario | Persistent request schedule, relative controls after warmup | Post-warm controls |
| --- | --- | ---: |
| 0 | zero at0, original stored task unchanged | 320 |
| 1/2 | single +/-5 at0 | 320 |
| 3/4 | single +/-15 at0 | 320 |
| 5/6 | +/-5 increments at0,20,...100 to +/-30, then160 final-course controls | 260 |
| 7/8 | +/-5 increments at0,10,...110 to +/-60, then160 | 270 |
| 9/10 | +/-5 at0,20,...100 to +/-30; reverse at160,180,...260 back0, then80 | 340 |

Offsets are absolute relative to the initial persistent world course, not
head direction. At each genuine course change both arms rebase stored lane
world-X/Z to their current COM and preserve original Y. Arm A observes the
original lane error. Arm B removes world-horizontal lane error in a read-only
copy before inverse rotation; all other features and depth remain. B activates
even for the zero scenario. Both warmups use the same original observer.

Do not reset body, servo targets/rates, torque history or elapsed time across
requests. Preserve first unsafe PRE/candidate and every attempted case. The
main ellipse remains active .20/.12m and behavioral inset .195/.115m. All-state
5mm depth/20degree tilt ordinary guards and the original hard limits remain.

## Unchanged movement criteria and declared conclusions

Apply the same functional comparisons, course validity and ranking as v1:
complete horizon; every accepted5ms depth<=1mm and tilt<=10degrees; final4s
mean COM forward projection>=6mm/s, course error<=5degrees and full3D velocity
RMSE<=2.5mm/s; every complete8s post-warm arc-speed window>=3mm/s.
Require nontrivial actual signed COM-course progress, not only head direction:
single/repeated final change>=half requested offset, or return cases first
reach15degrees before return and then settle within5degrees of original course.
Unwrapped1s mean courses need a contiguous speed>=2mm/s chain, as before.

Report COM speed/course, depth/tilt, repeated-turn completion, smoothed
conditional radius and valid-sample counts, first half-request response delay,
minimum conservative whole-body pond clearance at every accepted5ms state,
and failures. Clarify this is not an exact continuous swept-volume guarantee.
Retain the v1 lexicographic relative ranking; incomplete cases get zero in its
median-speed tie breaker. Publish all distributions rather than hiding stops.

An interface qualifies for this declared incremental-request envelope only
when all4 zero, all8 one-second +/-30 repeated and all8 return cases pass,
both retained-skill suites pass and the whole88-case comparison completes
without a hard fault. The faster +/-60 characterization and abrupt single
requests remain reported individually; this qualification does not imply
success at arbitrary large jumps, chooser competence or autonomous navigation.

Emit exactly one conclusion: lane_following_qualifies, heading_only_qualifies,
both_qualify, neither_qualifies, or comparison_blocked. A numerical or retention
gate failure makes it blocked, with exact attempted main-case count and no
winner. A complete comparison with no qualifying arm means neither qualifies.
No retrospective criterion changes or new candidate selection after results.

## Bindings, disclosure and stop

Study identity is frozen_interface_comparison_v2, study_version2, JSON schema1,
arithmetic_profile cpu_compatible_no_implicit_fma_v1; new0x32-prefixed binary
protocol/grant/fixture magics prevent old grants or results being reused.
Freeze the whole native causal closure, arithmetic reports, original parent
files, case catalogs, executable/build profiles, protocol and independent
readers. Independently audit zero-integration preparation before recording
the actual user's grant. All old failed evidence/checkpoints remain unchanged.

Audit the entire physical attempt from raw data, including any stopped prefix,
with explicit unfused binary32 reference semantics and unchanged tolerances.
No offline reader replays force integration. Save all failed preparation/audit
attempts and source versions; offline fixes never authorize recollection.

Update Godot with exact-byte-bound verified result/retention/conclusion, keep
the existing saved motor and visuals, and inspect the actual correct worktree
app. Preserve the v1 stopped result as history. GUI tests execute zero additional
physics/optimizer work. Do not enable an experimental autonomous mode or label
a controlled comparison as indefinite pond swimming or ongoing learning.

Finish with a compact table, numerical-fix explanation, retained-skill results,
one declared conclusion and a recommended next step. Then stop. No motor or
chooser training, optimizer update, further physical retry, deployment, push or
merge follows automatically from this authorization.

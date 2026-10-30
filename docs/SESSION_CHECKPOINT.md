# Minimal foundation checkpoint — October30,2026

## Checkout and preservation

Original checkout: branch `codex/autonomous-pond-practice`, baseline commit
`e5e9bd1db7e442715ac8ba3a5f5030c0ede26430`, annotated tag
`baseline/pre-minimal-foundation-2026-10-30`.
New worktree: `.worktrees/minimal-foundation`, branch `codex/minimal-foundation`.
The original checkout remains available and its editor was not closed.

Original ignored `build/foundation-baseline/2026-10-30/` contains verified
separate backups of25 personal saves (~68.2MB), prior477 protected-file records,
and39 current contact-study artifact hashes. One boulder save had changed since
its older protected snapshot; the earlier study copy and the new personal backup
both remain. Other476 protected artifacts were unchanged. Git history alone is
not a backup of these ignored/user files.

## Active execution path

`main.tscn → main.gd → CreatureLab → simulation::PhysicalWorld → physics`.
One world owner, one scene, no autoload or script-inheritance chain. The new
scene starts paused. Four explicit diagnostic inputs select joint targets or
direct torque; all physical torques stay within±10µN·m. No learner, food,
rewards, rescue, network inference or implicit warmup exists in this runtime.
The future persistent request retains destination, terminal velocity and heading.

Godot displays native links, position/heading, head/COM speeds, applied torques,
clock and clearance. The physical engine is independent of Godot/learning.
The optional CUDA wrapper steps the same native state on the caller's stream.
FP32 MLP/backprop/Adam and PPO/GAE/Gaussian primitives remain for future learning;
old motor policies/curricula/session ownership are retired.

New state format: diagnostic magic/version1 plus physics/actuator contract and
checksum. It stores the body, fluid/current, interface, fixed gains, held/raw
last-applied inputs, applied/requested torque traces, work, clocks, safety flags
and contact telemetry. It rejects old learner files atomically. UI uses only
`user://foundation/diagnostic-state.bin`; personal pond saves are not migrated.
Load pauses explicitly. Unapplied slider edits are not saved; this is stated in
the UI tooltip. This is a physical-state checkpoint, not a future learner save.

## Verified so far

- Strict native physics/Godot/CUDA builds passed; independent CPU-only build passed.
- Focused full native suite:49/49 passed,0 skipped, real GPU required,~1.18s.
- Four independent original-physics oracle cases compare body/link geometry,
  M/R/C/forces and pre-solve RHS without integration. Source is preserved baseline
  `src/envs/swimmer/deep_physics.h`, SHA256
  `b2d6dc10481db9a25d5c700b1d1fce62e39cb5eb17b44ee5585bc8060d2142a9`.
- CPU/CUDA:4 matched CPU states +8 GPU controls (including exact GPU replay),
  max defined-field difference1.90734863e−6. Actual-torque bounds retained.
- Total native physical validation:8CPU controls,8CUDA controls, plus37CPU
  substeps preserving the archived contact failure. No additional research run.
- CPU-only executable:6/6 MLP and1/1 checksum/atomic-replacement checks passed
  without repeating physical cases.
- Godot import/resource compilation passed. Rendered UI:43 checks passed with
  exactly2 explicit controls, no save/load/learning. Desktop1280×720,
  tablet768×1024 and phone390×844 screenshots inspected. Viewport sizing and
  narrow camera framing were corrected; final38/38 rendered checks passed with
  zero extra controls, no warnings and all tank corners framed. Keyboard focus
  scrolls the existing controls into view. Real touch input and a complete
  keyboard traversal were not manually playtested.
- Final zero-integration follow-ups passed: full-envelope tilted tail/fin
  support against an independent closed form on all six planes; format checks
  reject impossible safe/failed clocks, including integer-overflow edge cases,
  without mutating the world. The CPU-only hardened format check also passed.
- The active tree contains72 project files, compared with1,314 at baseline;
  code/resource files fell from362 to51. Each retained file has an explicit role
  in `build/active-file-review.json`. This is not a line-by-line proof of every
  historical implementation; those paths have been retired through checked
  dependency groups rather than carried forward unexamined.

Logs, screenshots, budgets and retention manifests are under this worktree's
ignored `build/`. The default test suite is fast; no full learner training or
historical study is part of it. Further format/static-geometry checks consume
zero integration steps and are recorded separately.

## Concrete physical gate remains unresolved

The new contact support geometry uses the full body/fin envelope, but the
registered dynamic gate is NOT passed. The retained zero-command rolled-head
replay still reaches−0.00058811903m floor gap after37substeps. That exceeds the
unchanged0.5mm allowance. The unit test passes by preserving and explicitly
identifying this known failure; it is not successful contact validation.

Historical contact-v2 study:133 geometry checks and2,400 executed CPU/CUDA
comparisons passed;4,440 total substeps ran. Full support/free-water/final
step-size acceptance did not finish. New motor readiness, PPO and policy
comparison counts all remain zero. The earlier paired-motor study is distinct:
its torque arm stopped after6updates on a distal-fin/contact-proxy discrepancy;
its servo arm had0updates. Detailed source/protocol/results remain at baseline
and in the original ignored studies; neither is resumed here.

## Next decision

A separately bounded contact-response correction is needed before declaring
physical readiness. Then design the motor task around turning, braking, slow
arrival and sustained position/heading holding, with unsupported calm/current
holding and supported bottom rest scored separately. Existing learning design
may change substantially; no replacement learning budget is authorized yet.
Chooser training, production adoption, growth/limbs, land movement, merging and
deployment remain outside this structural work.

## Reverification and targeted repairs — October30,2026

User requested another clean/DRY/working audit before the contact correction.
Audited the reduced local tree at `06c14eed0e024355b1b4928d0ca3750cdbe4a314`;
no new contact response, motor trial or PPO training ran. The original checkout
is still clean at the preserved baseline. Independent final reviews found no
remaining actionable blockers in the changed core, numerical, physics/world,
Godot bridge or resource/build paths. This is bounded verification, not a proof
that every possible input or physical behavior is correct.

Concrete changes:

- Shared native body geometry, tank bounds, settings validation and checksum
  code replace duplicate definitions. Removed unused checkpoint helpers and
  the racing-only episode aggregate; retained general PPO/Gaussian utilities.
- Atomic writes acquire an exclusive lease before opening the shared temporary
  file. Competing writes fail without changing the previous save.
- Guard MLP call order/batch/layout overflow, device allocation overflow and
  categorical kernel dimensions before indexed reads or launches. Valid-input
  arithmetic is unchanged. Adam synchronizes only at construction and save/load
  boundaries, with explicit nonblocking-stream continuation coverage.
- Godot validates its native startup/state contract before enabling controls.
  One metadata source supplies dimensions, limits and timing. Control readback
  never issues new commands. Saved custom currents remain unchanged and visibly
  identified; derived clipped next inputs are disclosed. Failure messages remain
  visible and Reset stays available for invalid physical states.
- Contact force telemetry explicitly reports the spring component; it does not
  claim to include damping or prove load balance. No force equation changed.
- Physical schema1 is byte-for-byte compatible with an independently captured
  624-byte state from the old compiled library; restore/reserialize is exact.
  Field renames did not alter ordering, units, checksum or physics contract.

| Final check | Result | Scope |
| --- | --- | --- |
| Strict CUDA/native/Godot build | Passed | Fresh extension installed into the normal project after user closed its editor |
| GPU-required native suite |58/58,0 skipped,1.16s | Real GPU execution; no creature learning |
| Independent CPU-only build/suite |17/17,0 skipped,0.10s | No CUDA dependency |
| Frozen-backend Godot contract |66/66 | Zero physical steps/files; missing/stale API and error states |
| Rendered real-extension UI |43/43 |1280×720,768×1024,390×844; exactly2 explicit controls, no saves |
| Isolated actual save/load |19/19 | Zero physical steps; prior-file survival, corruption rejection, transactional load |
| Final normal-project layout |35/35 | Installed extension; zero controls or saves |
| Preservation hashes |477+39 records and25 personal saves/backups unchanged | Original ignored results and source saves retained |

Tests and screenshots use a disposable project/user directory; actual personal
saves are never failure-test fixtures. The normal project was freshly imported
after installing the final DLL. Old native/editor bookkeeping changes were
reviewed: matching resource UIDs and standard config formatting are retained.
Visual inspection confirms all three sizes frame the tank and retain reachable
scrolling controls. Real touch and complete keyboard playtesting remain unproved.
The creature is intentionally small at whole-tank scale; close inspection is a
future usability improvement, not a new feature added during this audit.

New checks focus on concrete failures, not duplicate learner tests. The native
suite still reproduces the known rolled-contact failure at37substeps; its PASS
means the failure has not been concealed, not that contact is fixed. No safety
thresholds/body/actuator limits were changed. Unsupported holding, braking and
autonomous predator avoidance remain unproved; no trained replacement exists.

Evidence: ignored `build/reverify/` contains build/test/import logs, three rendered
screenshots, the baseline capture, isolated user-directory metadata and refreshed
preservation verification. `build/active-file-review.json` is refreshed for all
retained project files. No merge, push, deployment or production replacement.
The next substantive work remains the separately bounded contact correction.

The retained-file review now contains75 project files, including three focused
validation additions. This checkpoint describes local work on
`codex/minimal-foundation`; it is saved as a separate local rollback commit
before the contact-response experiment. No push or merge is included.

## Bounded contact comparison — registration before execution

The user authorized the next recommended work after reverification. Local audit
rollback commit: `0eb4cfcab986a41da4ee0036d17602a692ddfbde`.
This is one opt-in physical comparison, not another learner or motor curriculum.
Baseline `contact::substep` and the live world remain unchanged. Candidate
`contact_response::substep` is reachable only from the optional probe executable.
Body/mass/inertia, water/drag/buoyancy, four actuators, soft stops and torque
limits remain unchanged. No runtime/save version migration, projection, rescue,
new geometry or contact-tolerance change is allowed. Candidate contact dynamics
DO change: predicted contact and unilateral implicit spring/damping replace
old-gap explicit spring/conditionally implicit damping in the probe only.

One fixed candidate uses the same per-link spring0.8N/m and
`2*sqrt(0.8*link_mass)` damping, split equally over two tail endpoints.
Inward normal velocity is positive when separating. For gap g and step h,
normal force is `max(0,-k*(g+h*J*v_new)-c*J*v_new)`. With free system
`A=M+h*R`, solve the coupled nonnegative impulse problem using Cholesky
response solves and projected Gauss–Seidel. Fixed maximum64 sweeps per
activation pass, at most4 monotone passes. Activation uses existing overlap or
predicted linear/nonlinear free-drift crossing; no distant damping plane.
Projected velocity residual must satisfy `1e-6 + 1e-5*max(abs(J*u),abs(b))`.
Solver failure is reported, never hidden. No-contact branches retain exact
baseline arithmetic. Independent scalar/two-row reference checks precede any
physical integration; these are zero-integration preflight checks.

The immutable numeric catalog is written by `--prepare` before `--run`; its hash
and exact executed source/executable hashes are recorded in a new ignored study
directory. Preparation consumes zero integration. All cases have fixed numeric
states/actions; no random sampling or seed search occurs. The primary catalog
reuses the eight archived contact fixtures, including the protected fin failure
and rolled-head regression. It does not reuse withheld motor evaluation states.

| Stage | Fixed scope | Both responses × CPU/CUDA maximum substeps |
| --- | --- | --- |
| Primary |8 archived fixture types,1.5s at5ms |9,600 |
| Refinement | Primary IDs1/3/5,1.5s at2.5ms and1.25ms |21,600 |
| Withheld |8 fixed perturbed contact starts,1.5s at5ms |9,600 |
| Free water |8 prescribed calm/current cases,0.25s at5ms |1,600 |
| Passivity |3 zero-gravity/fluid/actuator cases,0.5s at5ms |1,200 |
| Total | No rerun, sweep or automatic extension |43,600 |

Safety is checked at the start and every substep: finite state, no escape and
no signed full-envelope gap below−0.0005m. Finite physical failure stops its
trajectory; remaining independent registered cases may run to diagnose the
response. Any nonfinite, solver, stream or data-integrity failure stops the
entire probe. Any candidate gate failure prevents acceptance. No parameter or
fixture edits between stages and no held-out tuning.

Density990 support cases require actual floor contact at every sample of the
last0.5s, mean actual normal floor force≥80% known buoyancy deficit, and
maximum absolute vertical head speed≤0.0005m/s in that window. Force includes
the actual damping/impulse contribution, separately from spring-only telemetry.
This short support criterion is not the old5s gate or ten-second motor holding.
Refinement5ms-versus1.25ms differences: maximum penetration≤0.0001m,
final head position≤0.0005m. CPU/CUDA defined physical floats use the existing
`2e-4+2e-3*abs(reference)` tolerance; gap/force use`2e-6+2e-3*abs(reference)`;
safety flags and executed counts must match exactly. Free-water defined states
must be bit-exact between responses on each backend and stay>2mm from contact.
Passivity starts cover compressed release, closing impact and separating
contact with zero gravity/fluid/commands. Kinetic+spring+joint-stop energy must
remain≤initial energy+`max(1e-10J,0.001*initial energy)`; normal force≥−1e-9N.
No energy tolerance will be relaxed after observing results.

Store all real substep states, force/impulse/solver diagnostics and first failure
conditions. Report safety, response, support, parity, free-water equivalence,
passivity and timestep sensitivity separately. Even a complete pass would only
support this contact correction on these cases; it would not establish learned
swimming, braking, unsupported station keeping or predator avoidance. No PPO,
production adoption, merge or deployment is included.

## Contact comparison result — stopped, candidate rejected

Exact pre-run source commit: `984b9cdf6474b6b24a1a00d354a648aa47a293c5`.
Strict CUDA build passed. The sole prepared run is preserved at
`build/contact-response/2026-10-30-v1/`, including executable/source snapshots,
SHA manifests, numeric protocol, every-substep body/link trace, contact-point
impulses and independent analyses. No parameter edits or repeat occurred.

Actual dispatches/committed poses:22,649 of the43,600 cap, comprising11,436 CPU
and11,213 CUDA. Stage charges: primary7,284; refinement10,618; withheld4,747;
free-water0; passivity0. There are82 recorded model/backend cases of132 planned;
the last is an interrupted legacy case. No unused budget may be repurposed.

| Evidence | Baseline | Candidate |
| --- | --- | --- |
| Primary contact cases, each backend |5/8 pass |5/8 pass |
| Bent / pitched / rolled primary cases | All breach0.5mm | All breach0.5mm |
| Bent / rolled refinements at2.5 and1.25ms | Still breach | Still breach |
| Completed20mm/s impact,1.5s maximum penetration |0.4501mm |0.3688mm |
| Level loaded support, last0.5s of1.5s | Pass | Pass |
| Withheld cases0–5, complete matched pairs |2/6 pass |3/6 pass |
| Free-water and passivity stages | Not reached | Not reached |

The protected fin fixture passed this bounded contact test for both responses.
The candidate also passed withheld case2 where baseline failed, but these
partial improvements do not satisfy the registered gates. Failed-case minima
are FIRST-BREACH-CENSORED, not complete-trajectory maximum penetration. Do not
claim a rolled peak-penetration improvement from different stopping states.
At5ms the rolled baseline breaches after37steps/0.185s at−0.588119mm;
candidate after38steps/0.190s at−0.549510mm. At1.25ms both breach at0.18625s.
This rules out the proposed time-discretization correction ALONE as sufficient
for the tested contact contract. It does not prove physical uncontrollability.

The rolled recontact is materially faster than the old allowance's20mm/s
assumption. Candidate5ms head-floor activation at step34 starts14.05µm clear
with normal closing speed39.46mm/s, row effective mass2.013g, and13.965µN·s
actual impulse (2.793mN over5ms). At2.5/1.25ms, incident speeds are38.93/39.20
mm/s. The response-adjusted effective mass/damping timescale is about24ms,
not the10ms used for two timestep travels. Thus the solver is applying braking,
but the selected material response does not meet the fixed compression limit
for these coupled rolling impacts. Effective mass here is from the implicit
free response, not an independent inertial/material measurement. This supports
recalibrating contact material against measured impact conditions, not treating
more PPO or another policy network as a fix for this zero-actuation failure.

Zero-integration independent scalar/two-row algebra checks passed10/10 rows on
CPU/CUDA. All11,394 reached candidate substeps passed solver/finite/nonadhesive
and independent equation/momentum checks: maximum independent normal-equation
residual1.32244e−6m/s, maximum residual/tolerance0.994982, maximum scaled
generalized momentum residual0.122127<1. These are checks of the registered
FLOAT-assembled equation with independent double accumulation, not an
independently derived all-double simulator or a proof of physical correctness.

The global stop was withheld case6, BASELINE CUDA, step77/0.385s. Fourth joint
rate: CPU−0.0159115531 versus CUDA−0.0154108349rad/s; difference0.0005007181
exceeds the fixed0.0002318231 tolerance. Geometry flags remained equal, gap
difference1.49e−8m and floor-force difference1.23e−7N passed their limits.
The fin endpoint's approaching-motion damping branch differed: CPU separating
~+2.01µm/s, CUDA closing~−1.71µm/s. This supports accumulated floating-point
sensitivity at the legacy discontinuous damping switch, not candidate
nonconvergence. All reached candidate CPU/CUDA comparisons passed. Candidate
withheld6/7, remaining baseline7, free-water and passivity remain UNPROVED.

No motor/chooser training, reward change, network redesign, scripted rescue,
threshold weakening, saved-state migration or viewer change occurred. Default
probe build option restored OFF; CTest inventory still contains only the small
foundation suite. Post-run hashes confirm all477 protected records,39 original
contact artifacts and25 personal saves plus backups are unchanged. Original
checkout remains clean. The live foundation still uses physics2 and its warning.

Narrow next decision: calibrate normal-contact compliance/damping against the
actual rotating/articulated contact loads and stopping distance under the fixed
0.5mm limit, rather than assume static deflection plus two timestep travels
bounds dynamic compression. Keep body/water/actuator limits fixed and retain
the successful numerical machinery where justified. A revised physical
coefficient/model comparison needs a new preregistration and budget; no automatic
gain sweep, remaining-case replay, PPO run or whole-app rewrite is authorized.

## Approved2,024-advance contact pilot — registration before execution

Baseline checkout is clean `6750500aefb667f0b3e3b08d5fde7cebe0eedece`.
User explicitly approved implementation/execution of one bounded SCREEN,
not a training-readiness claim. Existing studies stay closed. New sources are
isolated opt-in tools; live contact2, geometry, mass/inertia, water forces,
two universal joints/four outputs and±10µN·m limits remain unchanged.

New physical version4 identifies two distinct laws. Both first compute the
unchanged free implicit-water/actuator velocity, then normal impulses through
the bare physical mass matrix M. This common operator split is an explicit
contact-integration change from prior candidate3, which used M+hR for contact.
Neither law adds propulsion, a target controller, tangential force or rescue.

A is unilateral implicit Kelvin–Voigt with ONE fixed full contact-PATCH
calibration: k=320N/m and c=1.9595918N·s/m at every analytic contact row.
Endpoints are no longer half-weighted; geometric points themselves do not
change. Declared reference envelope: m_eff≤3g, normal speed≤50mm/s, load≤0.3mN,
available kinetic+convertible potential energy≤25µJ, design compression0.4mm.
The unchanged safety rejection remains0.5mm. The conservative undamped bound
requires k≥2*25µJ/(0.4mm)^2+2*0.3mN/(0.4mm)=314N/m;320 rounds upward.
c=2sqrt(k*3g) sets the reference scalar critical damping. Scalar restitution
must be≤0.15; critical unilateral release predicts about0.135 in continuous
time, not zero. No claim of critical damping in every coupled contact mode.
Independent mass/available-energy checks gate this calibration; no automatic
coefficient increase is permitted. The25µJ available-energy proof is confined
to its fully submerged, calm, passive domain. Current-driven fresh cases add
fluid work and do not inherit that proof.

B removes compliance entirely: H=J*M^-1*Jᵀ, no spring, damper or diagonal ridge.
Zero restitution, compressive impulses and rank-aware deterministic dual-basis
updates (rank≤10, at most64 basis iterations). Velocity constraints are
Jv≥-max(g,0)/h. Saved starts are clear. Existing negative gaps no worse than2µm
receive zero separation bias, never positive Baumgarte velocity. No position
projection, teleport or stabilization fallback. Actual nonlinear proposed gaps
must remain≥−2µm, an additional hard-model consistency check; the0.5mm safety
threshold is not loosened. Frozen-M contact projection must not increase
physical kinetic energy; there is NO hard-contact spring energy.

A uses≤128 projected sweeps and≤4 monotone activation passes. Both numerical
residual limits are1e−6+1e−5*max normal free speed. B rank detection1e−10 of
the Gram diagonal is rank bookkeeping, not compliance; impulse roundoff1e−14
never changes H. All failure proposals are retained separately from unchanged
caller states, and every attempted advance is charged before dispatch.

Independent CPU-double FK/geometry derivatives and recursive rigid-link kinetic
energy polarization verify J and M before dynamics. Exactly3 saved prestates:
legacy CPU rolled primary5 post33 (+22µm whole-body clearance), and legacy
heldout6 CPU/CUDA post76. Inputs are round-trip FP32 copies of the old trace,
SHA256 `F132FA41908D3E792852FE9F7F999E3881E51A7280B506965579A512FF83297D`.
No new capture, translation or projection. FD translation steps10/5µm;
world rotation/joint steps1e−4/5e−5rad. Gap tolerance2µm; J absolute tolerances
1e−5 for translations and2e−7m/rad for rotations, plus1e−4 relative. Normalized
M error≤1e−4, kinetic/meff relative error≤1e−4, normalized mass pivot>1e−12.
All5 floor rows and their full coupling are checked. Hybrid M+hR checks reuse
native R explicitly; they are not an independent drag/Coriolis oracle.
Separate static caps:256 geometry queries,256 kinetic evaluations,96 matrix/LCP
queries. Expected reference counts129/170/60, plus16 zero-advance algebra
oracles. Actual counts and prerequisite failures must be recorded.

| Stage, all at5ms | Allocation across both relevant responses/backends |
| --- | --- |
| Matched-prestate old-law parity |24:2 source states×3 base-vy offsets(−5/0/+5µm/s)×legacy2/candidate3×CPU/CUDA×1step |
| Scalar loaded support/50mm/s impact/separation |600:3×50×newA/B×CPU/CUDA |
| Calm/current free-water equivalence |400:2×50×A/B×CPU/CUDA |
| Zero-energy touch/passive rolling |400:2×50×A/B×CPU/CUDA |
| Common saved rolling recontact |200:1×50×A/B×CPU/CUDA |
| Two fresh sealed momentum/current conditions |400:2×50×A/B×CPU/CUDA |
| TOTAL CEILING, no reallocation |2,024 |

No retries are planned; any retry would still consume this same total ceiling,
not open another allowance. Numerical/algebra and actual physical passivity
are separate prerequisites. Scalar energy includes signed external-load
potential; A includes its actual320N/m patch springs, B does not. Both include
the common joint-stop potential when applicable. Passive whole-system energy
may not exceed initial+max(1e−10J,0.001*abs(initial)). Frozen-M contact-stage
energy and full actual-pose energy are logged separately. Gravity/buoyancy or
prescribed-current work must not be mislabeled as unexplained passive energy.

CPU/CUDA tolerances remain2e−4+2e−3*abs(reference) for defined physical fields,
and2e−6+2e−3*abs(reference) for gaps/net forces. Verify input hashes, exact
safety flags and attempt counts. B dual point impulses can be nonunique;
compare physical generalized/net reaction, not an arbitrary dual distribution.
Same-backend free-water A/B states must be bit-exact with zero contact impulses.
No uncharged third baseline rollout is allowed.

Shared reference/integrity/stream failures stop shared work. An old reference
law parity failure can quarantine that law only if shared checks remain sound.
A/B-specific cheap or physical failures stop that arm's later work; all attempts
and first rejected poses remain recorded. No hidden retry, extra timestep,
fixture change, energy allowance change or parameter tuning. The0.25s episodes
can reject an approach or justify another decision; they cannot pass the prior
1.5s/5s support gates, timestep convergence or ten-second motor holding.
Old inspected withheld states are regressions; fresh endpoints are not used to
choose coefficients. No automatic follow-up, PPO, new chooser/limbs, engine
migration, merge, deployment or production adoption is authorized.

## 2,024-ceiling pilot result — completed screen, no adoption

Exact executed source: `4488a1cbc332bc25acde58e5f34f524838845ab5`.
Both strict CUDA builds passed; preparation froze all fixtures, laws, coefficients,
tolerances and stop rules before any reference query or physical call. The sole
run is preserved at `build/contact-pilot/2026-10-30-v1/` with executable, all82
source files, source/protocol SHA manifests, immutable numeric registration,
static-reference output, algebra/point/step traces and rejected-pose binaries.
No coefficient changes, tuning, reruns or retries occurred.

Actual1,626 attempted physical advances:1,000 A,602 B and24 old-law parity.
CPU813/CUDA813;1,624 poses committed and2 rejected B proposals retained.
Stage charges: parity24, scalar600, free400, passive302, rolling100, fresh200.
Unused398 advances were NOT reused. Zero PPO, actor adoption or engine changes.
Static counts were129 geometry/FK,170 kinetic,76 matrix/LCP queries, within
their256/256/96 caps. The pilot is closed, not resumable.

| Check | A: calibrated compliant | B: zero-compliance constraint |
| --- | --- | --- |
| Scalar support/impact/separation |6/6 CPU/CUDA cases pass |6/6 pass |
| Scalar50mm/s restitution |0.0305022 (limit≤0.15) |0 |
| Scalar impact maximum compression |20.6776µm | Numerical zero |
| Free water |4/4 pass; per-backend A/B bit-exact |4/4 pass |
| Passive zero-energy touch |2/2 pass |2/2 pass |
| Passive rolling,0.25s |2/2 pass | Both stop after first5ms proposal |
| Recorded loaded rolling,0.25s |2/2 pass; max11.4441µm compression | Not run after prerequisite failure |
| Two fresh conditions,0.25s |4/4 pass; max11.8613/11.0418µm | Not run |
| Overall scheduled short cases |20/20 pass |12 pass,2 fail,6 not run |

A is more promising for the next validation, under its explicitly compliant law.
This is NOT evidence that it has universally superior collision geometry: B's
first failure penetrated LESS than A, but violated its own zero-compliance claim.
B's actual nonlinear gap was−3.7252903µm on both backends, past its preregistered
−2µm consistency bound while still safely inside the unchanged−500µm global
safety limit. Its linear head gap was approximately−1.11e−11m CPU/+7.48e−12m
CUDA, solver residual0/8.07e−19, rank1/one basis iteration. The compressive
impulse69.8193µN·s reduced frozen-M kinetic energy by1.52853µJ. This was
rotating-geometry linearization drift, not solver nonconvergence, adhesion,
energy injection or a hidden relaxation of the0.5mm rule. Caller states were
not committed; first rejected proposals remain visible in binary and CSV.

Independent3-state verification passed. Maximum normalized M error1.479e−6,
maximum effective-mass relative error7.576e−7, maximum native/reference gap
error18.68nm. All finite-difference J gates passed. Maximum inertial normal mass
2.1961g stayed under3g; rolling available calm energy8.7671µJ stayed under25µJ.
Fresh-calm initial energy bound8.8850µJ passed; the fresh-current initial8.6523µJ
was explicitly NOT a calm-work guarantee. Hybrid M+hR checks still reuse native
R; independent M/J agreement does not validate all hydrodynamics or Coriolis.

All24 cross-backend SAME-input legacy/v3 parity steps passed, including the six
frozen origin/velocity-offset inputs. Thus this test found no identical-input
CPU/CUDA defect; it supports the previous diagnosis of sensitivity accumulated
along different trajectories at the legacy damping switch. It is not a proof
of long-trajectory parity in every contact condition.

All1,000 A advances passed the registered finite, nonadhesion, energy, numerical
and reached CPU/CUDA checks. Maximum scaled momentum residual0.19653<1;
maximum independent constitutive residual4.677e−7m/s. Stored energy used320N/m
patch springs for A, no contact spring for B, and joint-stop potential for both.
Scalar total energy included its actual external-load potential; applied-load
work, numerical free-kick work and contact-M substage work are distinct fields.
Pure passive full-pose energy checks passed where run. Complete gravity/current
world-work accuracy remains unproved and was not silently called passivity.

The measured scalar rebound0.0305 is substantially below the continuous-time
critical unilateral value~0.135. It meets the registered low-rebound band, but
shows timestep-dependent numerical damping that needs validation. A also
changes endpoint patch calibration and uses the new common operator split;
this screen is not an ablation isolating stiffness, damping, weighting or split.

No old primary bent/pitched catalog, sustained5s support, timestep convergence,
ten-second unsupported holding, learned movement or predator behavior has been
established for A. These two fresh screen cases are now inspected regressions,
not new final held-out conditions. The original pond/runtime and saved actors
remain untouched. Post-run hashes verify477 protected records,39 old contact
artifacts and25 personal saves/backups unchanged; original checkout is clean.
Pilot option restored OFF; default CTest still only contains the small suite.

Recommended smallest next step, NOT executed or authorized here: keep A fixed
and preregister one longer loaded-support validation over the prior bent,
pitched and rolled failures, with fresh withheld conditions and an explicit
check of its new stiffness/timestep-dependent damping. This would decide
whether contact can support the reusable turn→brake→controlled arrival→hold
motor lesson. Do not extend this screen, tune A/B, promote either law or start
PPO merely because the short A cases pass. No merge, push or deployment.

## Fixed-A validation preregistration — accuracy before longer support

New authorization is one separately bounded validation of the existing contact-A
law, not a continuation of the closed2,024-call pilot. Read
`docs/CONTACT_VALIDATION_PROTOCOL.md`. Fixed contact4 coefficients320N/m and
1.9595918N·s/m, operator split, body, water and actuator limits remain unchanged.
New ceiling18,200 TOTAL physical attempts, including CPU/CUDA and failures;
no retries, reallocation, tuning or automatic timestep adoption. Scalar accuracy
uses three exact-double fixtures at5/2.5/1.25ms against an independent analytic
continuous unilateral law. One saved positive-gap rolling state uses four fixed
steps down to0.625ms for a finite-resolution comparison. Aggregate accuracy
failures complete only this registered diagnostic ladder; physical/numerical
failures stop A. Failed5ms accuracy blocks all later5s original support,
free-water preservation, passive and fresh withheld stages. No PPO, runtime
adoption, merge or deployment. Existing law B remains available.

Build-only preparation fixed a local name collision and constexpr warning;
those compiler attempts made ZERO physical calls. The strict CUDA target now
builds without weakening warnings. Before execution, original hashes verified
477 protected records,39 old contact files and25 personal saves plus25 backups.
The executable/numeric protocol/source manifest are to be frozen under ignored
`build/contact-validation/2026-10-30-v1/` before the sole run. Live physics2 and
the original checkout stay unchanged. A component/build pass is not accuracy.

## Fixed-A validation result — closed, accuracy rejected before longer tests

Preregistered source `7d6b53ff06bf5b5b9cd647db94d9cfabbd26a3e8` and numeric
registration were frozen before the sole run. Read
`docs/CONTACT_VALIDATION_RESULTS.md` for the decision, full tolerances and data.
Ignored evidence: `build/contact-validation/2026-10-30-v1/`, including the frozen
executable, 86-file source snapshot/hashes, protocol.bin/JSON, independent static
and continuous-reference reports, all 2,608 trace rows/contact impulses, first
failed pose and derived offline analyses. No physical rerun or tuning occurred.

The run is CLOSED at 2,608/18,200 attempts: CPU 1,400 / CUDA 1,208, all native
trial proposals committed locally. Stage charges: 2,100 scalar + 508 rotation;
all other stages zero. Independent static counts: 137 geometry (including eight
fixture placements), 168 kinetic, 60 matrix; continuous queries: 2,135. No old
unused budget was reused; this run's 15,592 unused calls cannot automatically
be spent later.

A at 5 ms is REJECTED for physical accuracy. Independent impact truth is
56.3198 um peak compression, restitution 0.135335 and 170.3003 uN*s impulse.
A at 5/2.5/1.25 ms gives 20.6776/27.3219/31.4106 um, restitution
0.030502/0.046323/0.063539 and impulse 154.5753/156.9485/159.5309 uN*s.
Every impact fails all five
accuracy criteria; this is excessive numerical dissipation, not merely low
bounce or a missed sampled peak. Scalar support 6/6 and separation 6/6 pass.

All reached states are finite, nonadhesive and inside 0.5 mm. Rotating 5/2.5 ms
pairs complete 0.25 s; 1.25 ms CPU completes but CUDA step 8 (at 10 ms) hits
normalized momentum certificate 1.0991275 > 1 and stops all further calls.
Gap -7.48038 um, constitutive 7.1843e-10 m/s and energy checks pass. Rounded
FP32 velocity/impulse
storage may explain a tiny-impulse certificate failure, but unrounded/per-DOF
residual data are absent, so that cause is NOT established. No criterion relaxed.
0.625 ms never ran; formal rotating accuracy/convergence is unproved. Original
5 s bent/pitched/rolled support, original-free preservation, additional
passivity and fresh 5 s withheld acceptance are all BLOCKED/UNRUN, not passes
or new failures.

The narrow next decision is contact temporal integration and activation/release
of the SAME law, with auditable momentum arithmetic. Not another coefficient,
PPO/reward/NN/body change or engine rewrite. Finer 1.25 ms is not accurate enough
and was not adopted. Before motor training, accuracy and the blocked contact
acceptance stages still must pass under a new authorized bounded protocol.
No follow-up or new budget is authorized by this recommendation.

Source/physics hashes were unchanged through execution. Post-run preservation
again verified 477 protected records, 39 old contact files and 25 personal saves
plus 25 backups; original checkout remains clean at baseline. B, runtime physics
2, Godot and all saved controllers remain unchanged. Validation option restored
OFF; CTest -N lists only foundation without executing extra physical advances.
No PPO, adoption, merge, push or deployment. Two local compilation defects were
corrected before freeze; failed builds consumed zero physical calls.

## Focused event-integration correction — preregistered before execution

New user authorization is ONE correction, with smallest-case rejection first,
independent saved-state momentum diagnosis, and only a conditional short rotating
check. Previous runs remain CLOSED. No coefficient, body, joint, actuator,
reward, 0.5 mm safety, runtime/save, PPO, merge or deployment changes.

Fixed method: exact constant-load free flight; geometric arrival and clipped
normal-force release split within each outer 5 ms step; implicit midpoint during
engaged contact, maximum internal duration 100 us. Release root uses exactly
24 bounded bisections. Count every entry, accepted/trial integration and event
bisection attempt before evaluation; no retry, gain sweep or budget extension.
Midpoint reduces the backward-Euler damping between events; splitting alone
would not address the measured rebound error. Spring/damper remain the EXACT
existing FP32 values 320 N/m and 1.9595918 N*s/m promoted consistently.

NEW TOTAL CEILING: 7,004 charged entries/physical trial advances, across all
CPU/CUDA, failed trials and event subdivisions. Allocation: scalar 3,000;
saved-state audit 4 (two entries plus two legacy replays); rotating 4,000.
No unallocated reserve or reuse. Scalar reference cap 256 analytic queries;
momentum certification/negative-control algebra cap 64, separately counted.
Compiler attempts and immutable-file/hash reads advance no physical time.

First rejection test: existing 3 g scalar impact, initial gap 25 um, velocity
-50 mm/s, CPU only, 20 ms total (four 5 ms outer steps). Stop on failed accuracy,
finite/0.5 mm/nonadhesion/energy criterion; do not build articulated solver first.
If it passes, complete the same six-case CPU/CUDA slate (first case not repeated):
original support, impact, separation, plus impacts with exact arrival times
0.2, 1.7 and 4.3 ms (gaps velocity*arrival). Total twelve trajectories, no old-law
rerun. Reuse the independent continuous oracle; optional gap parameter preserves
its old API. Record internal accepted compression peak, not just outer samples.
Keep existing accuracy limits: compression max(2 um,5%), restitution error .02,
impulse max(1e-8 N*s,5%), time-aligned gap 10 um and speed 1 mm/s. Physical
energy allowance stays max(1e-10 J,.001*abs(initial)); check all internal maxima.
CPU/CUDA use same starts; compare all stored matched outer rows at existing
state tolerances before any articulated step. No tuning from these cases.

Momentum audit is independent: replay the saved prestate of old validation's
1.25 ms CUDA step 8, once per backend. Capture M/J, raw solved velocities and
impulses, FP32 casts, signed conversion errors, each coordinate's residual,
normalization and arithmetic bound. Compare executing-backend M versus CPU M
used by the old checker. Revised certificate retains the old raw physical
residual gate and adds ONLY analytically propagated half-ULP storage error plus
double gamma bound; verify exact casts and detect deliberately bad raw/stored
updates on disposable copies (40 negatives, no extra physical advances).
No tolerance multiplication. Keep old results/checker source preserved.

ONLY IF all scalar cases and the auditable numerical path pass: same saved
positive-gap rotating fixture, 20 ms CPU/CUDA, zero actuator commands, density
990, no current. Implement the SAME event/midpoint method for frozen per-segment
M/R/J, with A=M+dt*R/2, u=A^-1[(M-dt*R/2)v0+dt(F-C)],
b=c/2+k*dt/4, H=J*A^-1*J^T+diag(1/(dt*b)), q=J(v0+u)+k*g0/b.
Pose uses midpoint velocity; terminal stored velocity is v1. No added forces,
projection or rescue. Refresh nonlinear coefficients only at accepted starts;
this is not a claim of globally second-order nonlinear dynamics. Fixed 100 us
max interval and 24 bisections, actual gap/force event boundaries, strict finite,
geometry, nonadhesion, CPU/CUDA, physical energy and raw/bounded momentum checks.
Stop after this one short fixture; no longer support/free-water validation.

If correction fails: stop extending the custom approach and recommend one
narrow B revision or tiny established native-solver reference, without running
it. Report fixed/failed/untested separately. Keep all previous artifacts and
personal saves. Baseline at this registration: minimal branch 7f4cc19; original
checkout stays at e5e9bd1. No physical calls have occurred during implementation.

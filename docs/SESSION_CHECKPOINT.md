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

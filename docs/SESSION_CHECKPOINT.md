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

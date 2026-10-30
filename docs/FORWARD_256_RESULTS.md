# Approved 256-update trial: closed, movement failed

One run under `FORWARD_256_PROTOCOL.md`; no retry or extension. Artifacts:
`build/forward-demo/2026-10-30-256-v1/`. Fresh initial actor weights are byte-identical
to the first pilot's initial weights; its learned weights were not resumed.
Only the update count and matching declarations/display changed in the tool.
All physical/task source stayed unchanged. Strict CUDA target build passed.

- Preparation to launch-ready:44.277s, including compile/freeze. Native run8.595s.
- Readiness32 controls; training131072; evaluation640; total131744controls and
  1317440 actually executed native5ms substeps, exactly the approved cap.
- Completed256 PPO updates/1024 Adam steps per network. No physical/numerical
  domain failures; all intervals passed zero contact,>=10mm sampled clearance,
  finite state/torque, torque limit and transition checks. Readiness parity passed.
- Success0/4 before and after. Mean forward COM displacement before+.000157mm,
  after-.067067mm; final mean goal error20.093683mm versus20mm initial goal.
- Individual final forward travel:-.064859,-.069274,-.064860,-.069274mm.
  Lateral errors1.005-1.062mm; velocity-vector errors10.041-10.046mm/s.
- Evaluated joints bent substantially (absolute angle up to.538rad; peak-to-peak
  up to.552rad) but each joint had only one meaningful velocity reversal over4s.
  These traces show bending/settling rather than a sustained swimming stroke.
- Mean training rewards for first16/last16 updates:-.00320273/-.00319802;
  deterministic final evaluation is worse than the initial policy. More updates
  alone did not produce useful propulsion; no universal PPO/model failure proved.

Initial weights,256 checksummed research update snapshots and all640 before/after
trace rows are preserved. Resume remains untested. All85 frozen source files
were unchanged through execution;561 previous evidence files remained hash-identical
and original checkout stayed clean. Runtime/contact/saves unchanged. No Godot
adoption, extra execution, push/merge/deploy or new commit. Optional target OFF.
This run is CLOSED; no further budget is implied by these results.

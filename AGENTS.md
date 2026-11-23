# Workspace instructions

Read README.md for current usage, architecture and limitations. Check asset JSON
provenance before changing a checkpoint or interpreting acceptance.

- Work only in this minimal-foundation worktree unless explicitly instructed.
  Preserve the original checkout, its editor, personal saves and ignored evidence.
- Baseline tag: baseline/pre-minimal-foundation-2026-10-30, commit
  e5e9bd1db7e442715ac8ba3a5f5030c0ede26430. Original save backups/hashes are in
  the original checkout's build/foundation-baseline/2026-10-30/; Git does not save
  ignored files. e84db5f and626e313 preserve later source milestones.
- All prior physical/training budgets are closed. Freeze a new finite protocol
  before any explicitly approved run; count backend/internal attempts and failures.
  No retries, tuning, tolerance relaxation or unused-budget reuse implicitly.
- Preserve model6 physics, two universal joints/four axes and +/-10microN*m torque
  limits unless a model change is authorized. No global propulsion or scripted
  rescue. Bounded calm holding from rest qualified in the November 22 learned-
  amplitude study: six original and six withheld 64-second cases, plus two
  uninterrupted 300-second cases. Read docs/HOLD_LEARNED_AMPLITUDE_RESULTS_2026-11-22.md
  for its restricted learned-parameter scope. Contact, moving-start braking,
  arbitrary slowdown/mode switches and disturbance/current holding remain
  unvalidated; no new budget or production adoption follows automatically.
- Keep native physics/learning independent of Godot. One native world owner,
  one bridge and one scene; one chooser and one reusable motor, not per-skill nets.
  Store/replay actual sampled actions consistently; critics are training auxiliaries.
- Keep checkpoint formats explicit/checksummed. Reject incompatible data without
  random fallback, silently resetting state or overwriting old personal saves.
- Delete obsolete active code/docs rather than commenting out or adding archives.
  Preserve unique checkpoint/data/evidence and dependencies; inspect hidden contents
  before removing directories. Do not restore retired code merely for history.
- Use focused nonphysical/CPU/CUDA checks; distinguish actual GPU execution from
  skips. No training/expanded physical studies after cleanup without approval.
- Preserve usable44px UI controls and narrow layouts; inspect rendered UI changes.
  Document exactly what was verified. Keep main merges/deployment separate from
  feature-branch publication; new changes stay local until publication is requested.

## User workflow preference — November 22, 2026

- Latest November 23 camera correction supersedes the whole-pond fit: the user
  rejected that as still too far away. Start in a close aerial creature view,
  centered on actual native body poses; allow shoreline to extend offscreen.
  Keep manual wheel/pinch/keyboard zoom and the separate side angle. The header
  must sit on a stable opaque surface with measured text contrast >=4.5:1;
  terrain-dependent dark-on-grass title/status text is unacceptable.
- The user explicitly likes the manual zoom. Preserve wheel, pinch and keyboard
  zoom during later world/camera changes rather than replacing it with only
  fixed camera presets.

- AESTHETICS IS KEY THROUGHOUT THE WHOLE PROCESS (user's explicit direction).
  Visual quality is an acceptance requirement for every feature. Inspect actual
  renders before calling work finished; preserve a coherent world, creature,
  lighting and GUI. User rejected the box-shaped planter/tank pond. Follow
  assets/art-direction/pond-reference-2026-11-22.png: rounded natural pond/basin,
  grassy banks, clustered rocks/reeds, clear muted water, soft light and restrained
  pale presentation. Do not substitute a rectangular engineering demo or postpone
  the look until after implementation. Preserve honest physical behavior while
  making the world attractive.
- The same request also explicitly authorizes teaching goal-directed turning,
  so the creature can eventually keep swimming within its pond. An eight-second
  viewer completion is a test limit; preserved five-minute straight swimming
  already exists. Teach real physical turning through the same PPO motor and
  verify retained skills. Do not simulate turns with rendered rotations,
  teleports or an untrained navigation script. Keep each new physical study
  finite and preregistered; existing closed budgets remain closed.
- Latest accepted visual direction is the organic pond on pale ground. The user
  accepted that interim look, requests a broader open-world area, and wants the
  creature clearly visible immediately without clicking Close view. Start with
  the creature camera; provide optional wider world views. Turning AND swimming
  in current are now explicitly requested alongside gait correction and skill
  retention. Preserve PPO/GPU use and actual native movement; do not present
  an artificial navigation script as learned turning.

- The user wants the creature living in its own pond world and called the
  present swimming a "crooked V." Treat natural-looking physical movement as
  a real product concern alongside speed, depth and retention. Diagnose the
  actual joint gait before changing geometry or presentation. Do not require
  all joints to stay straight: purposeful tail motion is necessary. This
  feedback does not itself reopen any closed physical/training budget.

- Latest current study is CLOSED: docs/CURRENT_SWIM_V2_RESULTS_2026-11-22.md.
  The user explicitly approved the corrected 2,048–16,384-world GPU study.
  Fresh physics passed; 32 PPO updates/344 Adam steps improved 4/16 to 8/16
  original current cases, not 16/16. No withheld/current endurance evaluation or
  viewer adoption followed. All 89,566,320 controls count, including profiling,
  calibration and assessments. No unused-budget reuse or automatic new trial.
  Keep the qualified calm package active; private V3 preview work is parked.

- The user defines the central mission as an evolving RL learner that users can
  interact with, using its training and new experience to adapt and thrive. Motor
  studies support that experience; they are not the whole product. Preserve
  learned skills and distinguish trained reactions from actual learning updates.
- Swimming in water with a current is a core aquatic requirement, including
  steering, progress relative to the pond and position holding against flow.
  Retain still-water skills. Native flow support does not establish trained
  current competence; keep existing calm-only guards and saved-policy contracts
  until a separately versioned, bounded current study is prepared and authorized.
- Keep actual reinforcement learning through native C++/CUDA PPO. A frozen viewer,
  scripted motion or manually chosen quiet coefficient is not a learning result.
- The user explicitly requested effective use of the GPU they configured. Size
  native parallel practice for the actual hardware and measure end-to-end
  throughput, kernel/transfer/I/O time and GPU telemetry. A CUDA launch or busy
  percentage alone does not establish full hardware use. Correctness gates can
  use small matched workloads; performance profiling and training remain bounded
  and all actual physics/optimizer work must count. Do not change device power,
  clocks or driver settings as a substitute for improving the workload.
- The user explicitly requested rebuilding the whole Godot GUI from scratch
  because it had become a mess. Maintain one coherent primary interface around
  the actual creature. Keep ordinary controls simple; disclose research detail
  and history on demand. Remove obsolete routes/controls after dependency checks
  instead of continually adding parallel viewers or dashboard panels.
- Whenever changing learning, simulation or other meaningful app behavior, check
  the GUI in the same pass: update visible information/actions and the selected
  saved-policy metadata, and remove anything no longer needed. Preserve distinct
  labels for actual learning, live frozen inference and recordings. Never show a
  stale or rejected candidate as a successful current capability.
- Keep the visible native body faithful to the shared physical state and leave
  previous source/evidence available for rollback. Verify the rendered result
  and relevant interactions before saying the GUI is finished.
- Keep automated GUI validation windows hidden. Launch the final interactive
  app normally, without test/capture auto-exit arguments, and verify that Godot
  Project Manager points at the current project rather than an old build copy.

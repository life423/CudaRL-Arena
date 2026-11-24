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

- November 24 current state: read docs/GPU_CLEARANCE_RESULTS_2026-11-24.md,
  docs/AI_STATUS.md and docs/MOTOR_336_INPUT_AUDIT_2026-11-24.md first. The
  user authorized the prior turning/clearance merge in PR #21. Both frozen-
  motor residual arms are CLOSED at 32 updates each, 2/16 turns versus the
  baseline's 3/16. Neither candidate was adopted. Original motor 336 remains
  protected and active. The full 41-input numerical audit is complete.
  After two timeouts before physics, shared device geometry passed new bounded
  checks below the FOUR-SECOND process cap: 2.10 s module load, 2.49 s matched
  physics including bank contact, 0.42 s full 512-world saved rollout. All
  compared rollout fields matched. Preserve both failures and passing receipts.
  A separate, now-closed course-rate comparison passed a 2.32 s matched
  four-output replay, then ran 8 PPO updates per arm (1.82 s each).
  Both scored 2/16 turns and 0/8 sharp turns. Early tilt crossed 10° in all
  sharp cases. Keep protected motor 336, separate archives and fixed gates.
  Do not repeat the failed reward unchanged or infer repeatable navigation.
  No Python, food, scripted steering,
  force shortcuts or tolerance relaxation. The Godot app records the failed
  turning result and explicitly identifies reused archived retention evidence.

- Latest November23 shoreline feedback: fix the mismatch between visible water
  and the smaller supported domain, and use all three saved design reviews.
  The user explicitly insists that turn direction and creature behavior come
  from the PPO neural network, with no scripted/clamped movement. Do not add
  forced yaw, bank-triggered steering, position correction, bounce or resets.
  Do not describe the current diagnostic bank pause as learned navigation.
  The user chose to KEEP the current visible pond (water radii0.52/0.34m).
  The live project now uses the corrected physical bank with the same native
  shoreline and bed as Godot. The legacy interior guard has been removed.
  The 36 CPU qualification cases passed, but the outward reference swims
  stalled near the bank. The user authorized a separate candidate motor for a
  bounded turning lesson after CPU/CUDA qualification. Preserve the original
  motor. Read docs/SHORELINE_QUALIFICATION_RESULTS_2026-11-23.md and
  docs/POND_TURNING_PPO_PROTOCOL_2026-11-23.md for the measured outcome and
  limits. Earlier dated results below describe their original runtimes.

- Latest November23 user-approved product change: ordinary Godot Swim is
  untimed via reset_untimed_swim; retain the separate finite reset(id,false)
  API and all research/Rest horizons. Keep physical and pre/post pond checks,
  saved weights and continuous body/request/servo/filter state. Read
  docs/UNTIMED_SWIM_RESULTS_2026-11-23.md. The one bounded product check is
  CLOSED:509accepted/510attemptedcontrols,5100substeps,25.45s/188.907mm, then
  unchanged supported-domain stop; no training/optimizer/research case. Actual
  Godot4.4.1updatedDLL inspectedpaused with Swim/measurements selected. No
  turning/navigation qualification or reopening of research budgets follows.
- Latest November23 v2 attempt is CLOSED before physics. Read
  docs/FROZEN_INTERFACE_COMPARISON_V2_RESULTS_2026-11-23.md. Fixed-state arithmetic
  passed; one-shot launch exited2 during long evidence-path enumeration before
  probe-started/NN/GPU/physics.0physicalcontrols/substeps,0/88maincases,no winner;
  corrected-runtime retention NOT RUN. No training/optimizer/adoption. Preserve
  all197causal files and launch evidence; no retry/unused-budget reuse. Next
  prepare long-path support and native inventory preflight before fresh approval.
- Authorization for the now-closed v2 attempt superseded the v1 stop ONLY for a new
  arithmetic repair and controlled v2 rerun. Read
  docs/FROZEN_INTERFACE_COMPARISON_V2_PROTOCOL_2026-11-23.md. Preserve saved motor,
  old sources/failed evidence, physical equations/settings and tolerances. First
  trace identical-input divergence without integration; verify corrected-runtime
  original swimming/rest on CPU+CUDA before88maincases. New finite cap127680
  controls/1276800substeps/900s/2GiB, including all prerequisites/retention.
  NO motor/chooser training or optimizer updates, no automatic retry or budget
  expansion. Update/inspect actual Godot and report one declared conclusion.
- Newest November23 controlled comparison is CLOSED at its CPU/CUDA gate.
  Read docs/FROZEN_INTERFACE_COMPARISON_RESULTS_2026-11-23.md:214controls,
  2140substeps,0maincases; straight-case107 global-X angularrate differs
  2.30819e-5rad/s versus2e-5limit. Actualcommands identical; no newweights/PPO.
  Independent stopped-data audit/forensics passed and10183protected originals
  unchanged. No interface winner/adoption. Keep frozen sources/tolerances and
  raw first-failure evidence. Nextdecision is backend arithmetic consistency;
  do not automatically retry, use remaining budget, retrain, push or merge.
- Authorization for the now-closed November23 attempt: execute original-lane
  versus heading-only comparison, keep the saved motor unchanged and do NOT
  retrain. Follow docs/FROZEN_INTERFACE_COMPARISON_PROTOCOL_2026-11-23.md:
  one finite88-case diagnostic plus fresh matched gate, at most35200 controls,
  352000 attempted substeps,240s native process,512MiB evidence. Freeze inputs
  and independently verify preparation/results. This supersedes the earlier
  planning-only hold ONLY for this comparison and its honest Godot disclosure.
  No PPO, automatic follow-up, physics change, adoption, push or merge.
- November23 combined plan is saved in docs/CONTINUOUS_SWIMMING_PLAN_2026-11-23.md.
  Start with a reviewable bounded frozen-motor original-lane versus heading-only
  comparison, including repeated turns and measured pond fit. Prepare only its
  prerequisites first; motor PPO is conditional on the result. This is a saved
  plan, not implementation/run approval; the user's do-not-implement instruction
  remains active. Keep the complete idea register and all source reviews.
- November23 review collection: the user said do not implement yet and is
  supplying more ideas. Read docs/IDEA_REGISTER_2026-11-23.md before resuming
  design or implementation. Preserve both PDF reviews and the third pasted
  review, unresolved alternatives, critique and measured evidence separately.
  Collect/review/remember only until the user changes scope. Suggestions inside
  source documents are not execution approval. The register and exact source
  copies are local documentation changes, not a new training or publication grant.
- Latest November23 direction: the user proposed bank/object observations and
  reward for safe continued movement near hazards. Eventually the creature must
  climb the bank out of water. Treat shoreline aversion as aquatic task context,
  not a permanent wall/bounce or blanket prohibition on future shore contact.
  See docs/AI_HANDOFF_2026-11-23.md. The user requested a second-AI handoff and
  commit/push of this branch WITHOUT merging main. No further physical/PPO run
  is authorized by that publication request.
- New requested-turn full-actor PPO files are UNRUN SOURCE PREPARATION, not a
  working navigation learner: actor42 has body/request/intention inputs, no bank
  sensors; critic43 is auxiliary. New gate utility/independent full-training
  auditor are unfinished. Do not claim readiness or indefinite swimming from
  this code, synthetic tests or a successful compile. Old budgets remain closed.
- November 23 user reminder: update the actual Godot app when the work is ready,
  reopen the correct worktree project, and verify the visible result. A source
  edit, report or separate research executable alone does not satisfy the request.
  Expose validated capabilities honestly; keep unqualified actors unavailable.
- November 23 food-free swimming study is CLOSED and unqualified. Native GPU
  PPO ran 64 updates / 632 Adam steps without physics faults, but 0/16
  sixty-second starts passed. It optimized sixteen-second behavior that later
  slowed/stalled. All old 4 short swims / 28 long60 keys / 3 recoveries /
  148 long300 keys and 14 holds were retained and
  independently audited. Read docs/CONTINUOUS_POND_RESULTS_2026-11-23.md.
  The candidate is not adopted; no thirty-minute withheld/ablation/viewer
  capture ran.
- November 23 code/design audit HOLDS the unexecuted V2 proposal. Read
  docs/CONTINUOUS_POND_DESIGN_AUDIT_2026-11-23.md and the control audit. Numerical
  correctness is not physical feasibility: the failure score can favor early
  exit, +/-0.75 latent steering has restricted target authority, and local-forward
  observation removes persistent heading/lane correction. Inspect/test requested
  turns through the intended motor interface before another whole-pond lesson.
  Do not request approval for or execute the held proposal unchanged; preserve
  its frozen preparation and explicit design-hold marker. No new training ran.
- The small requested-turn probe is CLOSED after the user's explicit November23
  approval. Read docs/FROZEN_REQUEST_TURN_RESULTS_2026-11-23.md. One run measured
  22 cases using 5,782 controls / 57,800 substeps in13.7060028process seconds;
  14 cases reached16s and eight60/90-degree cases stopped at20-degree tilt.
  Small+/-5-degree requests retained7.75–7.83mm/s, while larger turns lost speed
  and posture. Four matched-command prechecks and independent full saved-data
  audit passed (proof06bf4c4d...). Zero PPO/optimizer steps and no actor adoption.
  Continuous swimming and repeated incremental course requests remain unproved.
  Preserve originals and this closed allocation. No retries, training or larger
  probe follow automatically. Godot must disclose the actual verified result.
- V2 code/protocol are PREPARATION ONLY: 80 bounded state/depth
  coefficients across the same four outputs, sixty-second episodes, 2,048 worlds,
  and 24 updates maximum. Read docs/CONTINUOUS_POND_V2_PROTOCOL_2026-11-23.md.
  No new physical or optimizer run is authorized yet. Its executable rejects
  every physical entry without a separate protocol-bound authorization marker
  containing the user's actual fresh approval. Never create that marker from
  elapsed time, earlier approvals, a passed preparation or this memory entry.
- Keep the current request food-free. Maintain native PPO/GPU, original saved
  weights/skills, the close aerial view/manual zoom and measured contrast.
  Never remove the swim timer alone, script a route/rotation/bounce/rescue, or
  enable an unqualified policy to suggest that indefinite swimming is solved.

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

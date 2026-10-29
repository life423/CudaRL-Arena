# Autonomous practice: local implementation and verification

The [predeclared protocol](PROTOCOL.md) is implemented on `codex/autonomous-pond-practice`. No new installer, website deployment, main integration, paid service or long offline pretraining is part of this step.

Background CUDA/CPU PPO now generates timed surface-food arrivals and mixes original calm/static retention, calm sinking food, random steady speeds and smoothly fluctuating speeds. All speeds remain at most 0.75 cm/s; food and the body use the same imposed velocity in the existing physical equations. Shared helpers make seeded event requests/current phases reproducible across backends, evaluation and full-save continuation. Original learned inputs/outputs, PPO settings and binary State/Context layouts are unchanged. Player-context updates no longer restart practice clocks.

The visible pond offers optional **Auto feed**, one pellet per 2–6 simulation seconds, with continued manual feeding and full-capacity waiting. It starts off on normal launch, or on with `--auto-feed`. Visible current uses the existing manual controls; varied training currents are independently sampled behind the scenes. Candidate evaluations explicitly enable practice schedules while keeping world autoresets off. Adoption still happens only at a real physics-step boundary after the matched gate, not merely when weights are published.

## Evidence

- Full native regression: **185/185**, GPU required, zero skips. Five new cases cover timing, coverage, current modes/smoothness/bounds, exact CPU learning continuation, CPU/CUDA physical agreement and exact GPU schedule-state continuation.
- C++-only portable suites: **2/2**, including real CPU PPO and full resume.
- Rendered automatic-feeding checks: **32 CPU / 32 CUDA**. Keyboard toggle; no wall-clock-only arrivals; simulation-time food; manual input; capacity/nonreplacement; four viewport sizes; genuine optimizer updates; two candidate checks; real-step adoption; full physical/training resume. Final stderr files are empty.
- A protected copy of the existing CUDA learner resumed at **4,751 → 4,752**, without stepping/resetting its visible physical state. Its original on-disk save and all protected source-policy/full-save checks remain unchanged after QA. The backup/copy remains under ignored `build/autonomous-practice/`; personal training binaries are not versioned.

The CUDA smoke uses a deliberate full-pellet-pool stress before validation, making recent interaction context dominated by that placement. Its update-16 candidate collects 18.0 versus 16.83/minute but loses calm food (35 versus 37), so it is rejected. Update 32 collects 17.0 versus 16.5/minute with calm 37/37 and passes; the next real live step applies it without resetting the physical clock. CPU's two short candidates are rejected. These are bounded implementation/gating examples, not statistically established overall improvement or separate subgroup guarantees.

One first CUDA test-harness assertion checked application while physical stepping was deliberately stopped; publication correctly awaited the next step. The harness now advances a real step before checking delivery and checks that time advanced by exactly .05 seconds. Only the final passing logs are archived here; prior harness logs remain in ignored build output.

Nominal food requests match between policies; actual accepted arrivals can differ at the practice-only 16-pellet occupancy bound. Calm/static and fixed-stock historical food rates are different tasks and should not be compared directly with this protocol. The 37 motor inputs do not directly expose current velocity. Broader held-out current/feeding testing remains necessary before increasing difficulty or adding anatomy.

Internal interface review: hierarchy/usability/consistency/feedback 9; accessibility/responsiveness 8 pending real touch/screen-reader checks; performance 8 pending slower-hardware rendering/training measurements. The existing scroll layout keeps the 44-pixel Auto feed control reachable at 1280×720, 768×1024, 390×844 and 844×390. Prioritize those device/accessibility checks over more visual decoration.

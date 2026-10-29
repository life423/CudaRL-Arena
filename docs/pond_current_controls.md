# Live pond current strengths — October 28, 2026

Open the pond and enable **Current**. Choose **0.5 cm/s** or **0.75 cm/s** beside the switch. Turn Current off to return to calm water. The chosen strength is remembered for the rest of that scene, so switching it back on restores the same speed. The selector is disabled in calm water. It is not a setting saved across app restarts.

Both choices are steady +X flow, affecting the existing articulated swimmer drag and suspended-food transport. The visual flow phase reads the actual native vector and physical simulation time: at 0.75 cm/s it travels 1.5 times as far as at 0.5 for the same simulated interval. No whirlpool, local rock disturbance, new physical equation or extra PPO update is introduced. Slowly turning flow remains research-only.

Strength/on-off switching preserves body position/orientation, velocity, food, selected target, simulation clock, collection counters and the session graph. It does not restart the creature. The separate **Show untrained policy** comparison retains the chosen flow but keeps its existing deliberate same-layout reset semantics. Both saved controllers and full training saves remain preserved.

The new `DeepPondLab.set_steady_current(speed_cm_s, drifting_food=false)` validates finite input from zero through 0.75 cm/s before mutating anything. UI choices are the two evaluated positive strengths. The existing `set_gentle_current()` API remains a compatible 0/0.5 wrapper. Invalid requests leave the physical state/current untouched; existing capability fallback/strict-CUDA behavior is retained. This bound applies to the live Godot API; research runners retain their separate declared limits.

The strength choices use the [frozen-agent evidence](experiments/swimmer/current-075/README.md), not a new learned policy: selector update 320, retained motor 64,350. Passing that finite suite does not guarantee all possible mid-session states or future interactions. Prior difficult-state failures remain a separate research limitation.

## Verification

The release native Godot extension rebuilt successfully. The dedicated current-strength check passes **41 CPU headless / 45 rendered CUDA assertions**. It covers keyboard popup selection, no-reset switching, native rejection of negative/nonfinite/above-limit requests, exact suspended-food transport within floating-point tolerance, matching visual transport, one-minute food collection, zero background learners, saved/random comparison and four viewport sizes. The original current regression passes **51 rendered CUDA assertions**, and the full pond-navigation check passes **141**. Final stderr logs are empty. No complete native-suite rerun or new robustness benchmark is claimed for this wrapper/UI change.

Rendered states at 1280×720, 768×1024, 390×844 and 844×390 were inspected. The switch and strength choice share one row with targets at least 44 px high; secondary controls/stats remain scrollable on compact screens. Evidence is in `docs/experiments/swimmer/live-current-strength/`; screenshots are under ignored `build/current-strength-checks/`. Internal review: usability, hierarchy, consistency, feedback and responsiveness 9/10; accessibility and measured performance 8/10. Priorities remain real assistive/touch-device checks and measured startup/frame-time performance before platform expansion.

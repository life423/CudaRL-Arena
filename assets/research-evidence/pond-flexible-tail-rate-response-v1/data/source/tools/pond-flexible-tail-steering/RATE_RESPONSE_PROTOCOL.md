# Rate response candidate — November 25, 2026

User authorizes a separate native GPU PPO motor candidate to improve slow,
uneven turns while retaining from-rest swimming, with Godot adoption conditional
on reliable bank avoidance. The seven-option program stays closed and intact.

Fork the selected baseline-material stage5 checkpoint0 and actor0, preserving
actor/critic/Adam/body/servo/history/sampling state. New opt-in Study6 binds a
reward-only mode; old study5 remains unchanged. Same body, physical equations,
actuator constraints, observations, requested speeds/rates, seeds and PPO math.

Replace only rate feedback/scoring: compare causal whole-beat COM courses0.5s
apart rather than20ms apart; normalize signed rate error by8deg/s, clamp its
absolute value to4 and score0.75/s*(1-Huber(error)). Huber is0.5*x*x below1,
otherwise abs(x)-0.5. Grade by actual movement. Matching request earns+0.75/s,
zero rate against8deg/s earns+0.375/s, equal opposite rate earns-0.375/s.
All existing speed/course/depth/tilt/effort/stroke terms remain unchanged.

One exact-resume prerequisite executes3 actual updates (196608 decisions,
12582912 maximum substeps). No automatic retry. Then up to128 new updates,
1024 worlds*64 decisions,4096 minibatches,4 passes,8388608 transitions,
536870912 maximum substeps. Native wall cap1200s, external1800s.
Every8 updates plus baseline: unchanged32-turn+12-swim cases per arm and
22 supplemental speed/rate/continuous/bank probes, maximum1483200 substeps
per assessment/30s. Stop on native/numerical fault or12-swim retention failure.
Learning shortfall alone is not a native failure.

Select among12/12 swims and all8 gentle+all8 30deg main-stroke keys, then most
of12 fixed rate passes, then lower mean course lag. Evaluate the selected motor
and the strongest preserved bank chooser16 on the unchanged12x120s diagnostic
and conditional4x300s endurance suite, maximum2112000 substeps/30s. No new
chooser optimization. Incomplete scores remain failures, not silently retried.
Total physical ceiling579000000 substeps including one possible final bank
assessment of the pretraining motor if needed. Preserve every partial attempt.

Publish only after motor movement/rhythm, both continuous sequences, all bank
diagnostics and endurance qualify. Protect fitted gait, live offset, native DLL,
habitat and all prior checkpoints. Otherwise save measured results and leave
Godot unchanged. This authorization is one session, no indefinite retries.

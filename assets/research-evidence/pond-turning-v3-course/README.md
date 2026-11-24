# Food-free turning with a course-sensitive reward

The 32-update experiment trained a separate candidate from saved motor 336. Each `update-*` directory holds the native skill assessment and its case and long-swim tables. Each `updates-*-*` directory holds the individual PPO updates, training result and native file-write timings. The full local rollouts and every candidate checkpoint are preserved in `build/turn33`.

The [candidate checkpoint](candidate-32.bin) includes the final weights and optimizer state. Its SHA-256 is `e6415ff704750f331d7aefff28e06ef5d54db194aaef1057fb4123f8ffe4bcb6`. Godot continues to play the preserved motor.

The [course measurements](turn-cases.csv) come from the frozen CPU assessment traces. [Timing summary](timing-summary.json), [first invocation](runner-first-invocation.json), [baseline recovery](baseline-recovery.json) and [continuation](runner-continuation.json) distinguish native work, the runner repair and the time between checks. The machine protocol records local paths from the experiment. The original receipt, proposal and outputs remain bound to the original protocol hash.

[Full results and comparison with exercise 2](../../../docs/POND_TURNING_PPO_COURSE_RESULTS_2026-11-24.md)

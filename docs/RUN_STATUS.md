# Run reporting audit

The closed variable-speed trial started23:51:47UTC, status/log explicitly reached
training23:51:48 and finished23:51:50. Native3.347s; compiled setup.582s. The missed
conversation start notice was not a long GPU run: it was delayed assistant/task
communication. The parent conversation must monitor/report actual state; this
launcher cannot deliver voice/chat messages or fix coding-agent silence.

Confirmed code gaps: status overwritten directly rather than atomically; no native
PID/progress in the public record; only running/training/completed native phases;
pre-launch failures could leave stale building/check state. Fixed with one small
run-status helper, atomic same-volume replacement, UTC start/heartbeat/PID fields,
readiness/baseline/training/candidate-evaluation phases, update/control/substep
progress and explicit terminal result/failure with failed phase. Latest state is
build/speed-motor/latest-status.json; immutable per-run evidence remains separate.
Status-write failures now propagate instead of being swallowed with partial native
JSON reads. The wrapper stops only its own bounded native process on failure.

Nine synthetic phase transitions plus identity/progress/error and failed-write
preservation checks passed via tools/test-run-status.ps1, zero native processes.
A Windows empty-backup-path issue in the first atomic helper was found by these
checks and corrected to overwrite-rename. Script parsing and native compilation
passed. No training or physical rerun was performed for instrumentation. The
closed run's archived executable/source/result are unchanged. Full future
native-process status flow has not been re-exercised after instrumentation.

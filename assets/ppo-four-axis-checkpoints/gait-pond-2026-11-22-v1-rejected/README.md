# Closed gait study: stopped before PPO

The initial gait correction was exactly the original swimmer. Compatibility execution stopped at control 32, fixture 0 (pose 0 SWIM), because independently inferred CPU/CUDA closed-loop distal yaw joint rates differed by 0.000020951032638549805 rad/s, exceeding the fixed 0.00002 limit. The corresponding reference-rate discrepancy and other endpoint differences are preserved in the independent report and raw endpoints.

All 10 fixtures remain physically safe. Legacy CPU and identity-head CPU are exactly identical. This is a closed-loop numerical agreement failure, not evidence that the identity head changed the original swimmer, nor a demonstrated physical envelope breach. No baseline, calibration or PPO execution followed.

Actual attempted work: 960 controls, 9,600 substeps, zero PPO updates, zero Adam steps. The independent saved-evidence audit passed and confirms truthful rejection; qualified=false and runtime_adopted=false. Unused controls/time are closed.

Included are the complete 32-tick gate endpoints, exact initial checkpoint/parents, zero-only preparation/preflight proof, frozen source manifest and source snapshot, independent proof/aggregate, GPU telemetry and failed initial zero-FD fixture history. The first zero attempt compared an unscaled loss finite difference with a correctly scaled mean-minibatch gradient; its source/executable/input snapshot is preserved. The corrected numerical checks passed before the one physical attempt.

Compiled library/executable snapshot bytes not duplicated in the source-only snapshot remain preserved locally and are SHA-bound by source-snapshot-reference.json. Equal prepare/verified-zero payloads are stored once with explicit aliases. manifest.json covers every archive file except itself. No native execution or learning occurred while packing this archive.

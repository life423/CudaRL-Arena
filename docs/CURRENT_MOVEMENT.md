# Current movement

Latest result: the narrower supervised constant-speed NN lesson passed10/10
frozen cases, including withheld5/7mm/s, with requested4/5/6/7/8 producing
4.350/5.318/6.259/7.115/7.843mm/s. Run2026-10-31-constant-clone-v1,
600Adam steps,16320physical substeps,.793s native. Selected update-600.bin for
constant requests from rest only; no teacher fallback during evaluation. Existing
live8mm/s actor preserved; switches/deceleration remain unpassed.
See CONSTANT_CLONE_PROTOCOL.md and assets/steady-speed-v1/checkpoint-provenance.json.
The earlier failed trials below remain historical evidence, not the latest result.

The previous rollback is c0c0314. Curated checkpoint, accepted examples and metrics
are tracked under assets/steady-speed-v1; full study evidence remains under build/.
One live model6 scene and one native speed-conditioned path remain active.

The selected reward2 actor still passes straight8mm/s: original four-condition
mean7.958mm/s, RMSE1.805mm/s, positive work8.149microJ, work/metre.387931mJ/m.
It was trained with a supplied oscillator and learns only strength/frequency.
Head yaw26.6degrees remains; this does not establish turning/braking/holding.
`./tools/view-live.ps1 -SkipBuild` opens paused. The selected asset permits only8.

## Variable-speed trial5: closed, candidate rejected

Train4/6/8mm/s, heldout5/7, fresh critic/Adam with the selected actor as warm start.
24updates,16worlds,80controls each. Mean-speed limit10%, RMSE31.25% of request;
path/depth/tilt/work/native integrity gates preserved. Paired frozen evaluation in
cases0/3 plus continuous6->4->8mm/s, preserving world/phase/filter/energy state.
The refactor matched160original8mm/s rows exactly before PPO. CPU/CUDA readiness
passed.32,832controls/328,320steps,3.347s native/.582s setup, no retry or fault.

| Request mm/s | Baseline actual | Candidate actual | Candidate passes |
|---|---:|---:|---:|
|4|7.986|7.790|0/2|
|5 (held out)|7.979|7.790|0/2|
|6|7.972|7.789|0/2|
|7 (held out)|7.965|7.789|0/2|
|8|7.958|7.789|2/2|

Switches passed0/3; candidate segment speeds7.784,11.574,12.237mm/s.
Those longer continuous states expose behavior the4second reset lesson did not
train. The candidate remains almost request-insensitive: steady strength.57435
through.57521rad and frequency.94973 through.94852Hz across the five requests.
This run demonstrates conditioning failure, not a body/physics failure or proof
that additional training cannot work. No additional training is authorized here.

Evidence: build/speed-motor/2026-10-30-variable-v1/ (protocol, frozen sources,
actor/reference/executable hashes, checkpoints, training and paired traces,
per-speed/switch metrics and result.json). The prior passing actor and all old
experiment folders remain intact. The portable actor was rebound to the cleaned
source contract without changing its2498parameters (FNV1699f42d0090c042).

## Clean active path

The retired arrival task has been replaced by shared state math, not wrapped.
There is one trainer, task, native request setter and Godot scene. Removed the
superseded physics2 scene/adapter, replay, contact-study implementations and their
CMake entries/tests. Git provides historical source/rollback; ignored results and
personal data remain. Numerical primitives and physics/persistence tests remain
because they are referenced and useful, not compatibility modes.

Next learning decision should make request dependence explicit and include
continuous speed changes during learning. Do not add per-speed networks, change
body mechanics or blindly rerun the closed24update configuration.

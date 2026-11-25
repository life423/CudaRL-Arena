# Reliable-turn chooser — November 25, 2026

User explicitly authorizes options1,2,7: add usable moderate turns, freeze
candidate motor88, and train a bank-approach curriculum using native GPU PPO.
Original Godot assets and every prior study/checkpoint stay protected.

Opt-in CUDARL_RELIABLE_CHOOSER adds five rates{-8,-4,0,+4,+8}deg/s at existing
speed requests{6,8,12}mm/s,15 categorical actions total,0.8s decisions.
FTCH model version3/request schema2 rejects incompatible nine-action weights.
The behavioral architecture stays one chooser and one frozen motor. The motor
is rate-response-v1 selected88, including its frozen skip and fitted gait.
No motor optimizer, physics change, food, prescribed turns or artificial forces.

Start a separate chooser/critic with the existing orthogonal initialization,
gamma.994/lambda.98, actor LR.0003, critic LR.001, clip.2, entropy.01, value.5,
target/hard KL.015/.05, existing rewards and observations. Up to128 updates,
1024 CUDA worlds,8 macro decisions/world,1024 minibatches,4 passes:
1048576 macro transitions,1342177280 maximum substeps including warmups.
Save complete state and immutable model each update.1200s native/1800s external
wall caps. One exact-resume diagnostic executes3 actual updates/24576 decisions,
31457280 maximum substeps. No failed main restart or automatic extension.

Half starts remain central exploration. Bank starts are balanced by turn sign.
Each lane begins at level0:25–30deg outward slant from tangent and240–270mm
straight runway. Level1:30–36deg,205–240mm. Level2 retains original34–44deg,
170–205mm. Spawn placement must pass whole-body geometry and actual warmup.
Pure geometry preflight checks all1024 initial lanes at all three levels before
GPU work. One explicit zero-physics fixture correction is reserved if that
preflight identifies an invalid placement; record its evidence, never retry
failed optimizer work silently.

Reserved correction used: initial level0 lane4 failed static warmup clearance,
zero physics/optimizer work. Gentle starts now shift their initial placement
inland in1mm increments (maximum200mm) only when needed to meet the unchanged
20mm warmup certificate. Their stated runway is nominal; actual available room
can be greater. Heading and every live episode remain unchanged. Level2 original
placement stays exact. Original failed evidence is preserved at reliable-chooser-v1.

Promote after60 uninterrupted simulated seconds,8 newly credited cells,300mm
actual COM transport and no stall longer than2s. Promotion changes only the
next spawn difficulty; it preserves current body, servo, history and command.
Demote after two scored failures at a nonzero level. Only a scored terminal
respawns training bodies. Curriculum and counters are fully checkpointed.

GPU categorical checks include15 actions, CPU/GPU probabilities, gradients,
sampling and buffer canaries; beyond-capacity17 remains rejected. The passed
frozen motor retention evidence is reused with exact model/input/physics hashes;
no retraining can invalidate it. Record curriculum levels, promotions/demotions,
action counts, losses, value calibration, GPU physics/NN/PPO/transfer/evidence
timing and actual work. Stop numerical/native faults. Bank guard stops are
scored training failures, not silently discarded episodes or physical contacts.

Unchanged evaluation every8 updates plus baseline:12x120s diagnostic cases,
conditional4x300s endurance,2112000 substeps/30s per assessment. All12 plus
coverage>=20% are required before endurance. Keep whole-body COM speed/course,
depth/tilt, certified clearance, visited cells, stalls, perimeter circling,
failure denominator and no-reset rules. Select diagnostic passes, endurance,
then coverage; preserve all checkpoints, including worse final policies.

Full physical ceiling1409538560: main+one diagnostic+17 evaluations.
Only an exact policy passing all pond diagnostics/endurance and motor continuous
qualification may enter Godot. Otherwise archive the measured result and leave
the working project unchanged. This finite session does not prove infinity.

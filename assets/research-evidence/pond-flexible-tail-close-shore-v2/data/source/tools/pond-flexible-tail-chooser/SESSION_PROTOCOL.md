# Sustained native PPO chooser session

## Objective and scope

Train a food-free chooser to keep the flexible-tail creature exploring
the certifiable underwater pond while its fitted gait and update-eight
yaw offset remain frozen. The session owns one shared actor and critic.
The actor chooses keep-course, left 15° or right 15°. The native motor
produces all body motion through the existing actuator and water physics.

The steering interface is selected explicitly before the session.
`measured-course` retains the historical request interpretation.
`held-heading` adds each selected turn to the persistent requested heading.
The matched steering diagnostic must establish the usable course response
and turning room for the selected interface before sustained training.
Neither mode supplies a bank-triggered action or prescribed path.

## Registered first session

- 1,024 CUDA worlds, eight navigation decisions per world per rollout.
- Minibatches of 1,024, four PPO passes, actor learning rate 0.0003 and
  critic learning rate 0.001. The actor has 32 tanh hidden units and three
  logits. The training critic has 32 tanh hidden units and one value output.
- Discount 0.97, GAE lambda 0.90, PPO clipping 0.20, entropy coefficient
  0.01 and value coefficient 0.50. Preserve the native gradient and KL gates.
- At most 64 completed PPO updates and a cooperative target of 1,800 seconds
  of cumulative training-process wall time across this session's invocations.
  Finish an update already started and checkpoint it. An external watchdog
  adds 600 seconds for that final update, startup and checkpoint completion.
  This permits
  at most 524,288 sampled navigation decisions. Both the actual decision
  count and physical substeps must be reported.
- Training starts mix the existing mirrored bank-approach curriculum and
  interior placements. Each world carries body motion, actuator state,
  heading requests, course history and visit state between rollout batches.
  Only a scored training episode terminal initializes another training world.
- The initial actor is imported once from bank-practice update four.
  That legacy artifact contains actor weights only. The new session creates
  its critic and Adam state once, then saves and restores them completely.
- Keep the frozen gait and yaw-offset payload hashes, physics and habitat
  descriptor bound to the session. A mismatched task, model, interface or
  training configuration rejects resume.

The source manifest and a copy of all bound native sources accompany the
first invocation. It preserves relative paths, source hashes, frozen input
hashes, configuration and the selected interface. Each invocation records
the exact executables, arguments, start time, elapsed time and exit status.

## Checkpoint and restart contract

Publish a checksummed atomic full checkpoint after every complete update,
plus an immutable update checkpoint and immutable actor export. The full
checkpoint includes actor, critic, both Adam moments and step counters,
all world state, episode and random-sampling counters, global update index,
training counters and numerical configuration. Cooperative wall and stop-file
checks run at complete-update boundaries. An update already started finishes
and checkpoints before the process exits. Collection is never discarded
merely because the cooperative wall budget elapsed. An external watchdog
or numerical failure preserves the preceding atomic checkpoint, and the
receipt accounts for the interrupted invocation. No incomplete optimizer
update may become a published checkpoint.

Before the first learning session, verify that an uninterrupted two-update
run matches one update followed by save/load and one additional update.
Compare actor, critic, optimizer state, worlds and counters under the same
CUDA executable and math settings. The verification uses its own output
allocation, a 600-second external process limit, and counts its physical work
and optimizer calls separately. The sustained runner binds the passing
verification receipt before admitting a new training session.

The workflow runner holds an exclusive session lease. It never overwrites
a prior invocation or evaluation directory. A numeric, serialization or
contract error stops the workflow and preserves the latest complete state
and failure evidence. Resuming uses the full checkpoint and the remaining
registered training wall budget. A failed process is never silently restarted.

## Evaluation and Godot admission

Evaluate immutable actor exports at update eight and each subsequent
multiple of eight, plus the final completed update. CPU evaluation can run
while CUDA training continues because it reads frozen exports and owns
separate world state. Held-out results do not change the lesson, reward,
optimizer settings or acceptance criteria during this session.

Use the existing 12 fixed 120-second cases, including the exact Godot rest
start. Require all 12 cases to finish without losing positive native
whole-body clearance and with the existing speed, depth, tilt, visit and
turn requirements. Keep the original 20% pooled coverage threshold. Only
after those cases pass, run the four fixed 300-second endurance cases.
The same physics, deterministic argmax chooser and selected steering mode
must be used for evaluation and any future Godot publication.

Each evaluation retains its existing 2,112,000-substep and 20-second
external limits. The script records qualified candidates and leaves Godot
publication to the explicit tested publication step. A failed candidate
does not replace the visible saved motor or activate autonomous mode.

## Measurement scope and remaining limitations

The current 345-cell coverage denominator is a fixed upright-posture
geometry estimate at the exact Godot start's 72 mm depth. The held-out
fixtures span approximately 72–89 mm. Report the pooled fraction as this
specific estimate. It does not certify exploration of every navigable
depth or all possible body configurations.

Earlier zero-tilt and zero-depth-error results came from calm submerged
states. Disturbance recovery, current-plus-bank behavior, actual contact
recovery and swimming throughout a three-dimensional depth volume remain
separate qualifications. The native clearance guard stops on certificate
loss. A guard stop is not a measured body collision.

The learning curves must distinguish sampled navigation decisions,
simulated time and physical substeps. Record transport and coverage,
direction choices and collapse, clearance/cue/posture terminations,
reward, policy loss, value loss, entropy, clipping and KL metrics, and
available value-fit diagnostics. Report CUDA physics/collection,
network/optimization, transfers, evidence writing, CPU evaluation and total
elapsed time with each measurement's actual boundary stated.

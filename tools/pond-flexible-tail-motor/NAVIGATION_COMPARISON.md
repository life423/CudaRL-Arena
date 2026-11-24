# Flexible-tail PPO comparison

One diagnostic pair on the current two-axis flexible-tail creature in clear
water. Both actors start from the same fitted motor and identical native states,
commands and PPO seeds. The split compares the combination of displacement
observations and signed positional progress with heading-only control.

## What changes between arms

Both use a 31 → 32 → 32 → 2 actor with the same original fitted weights. The
three added first-layer columns start at zero. Both use the same initialized PPO
critic, with its new columns also zero. The destination arm sees the held target
minus whole-body centre of mass, expressed in body coordinates and divided by
0.10 m. The heading arm receives three zeros during travel. Both use the same
heading, requested speed, depth, body state and history. When displacement is
below 1.5 mm over the half-second course window, the course components for both
arms are zero. Other directional and velocity inputs remain available.

The heading arm's transport term is projected COM speed divided by the
8 mm/s request, clamped to ±1 with the existing overspeed cost. The destination
arm uses signed 3D distance reduction over the same control and speed scale.
Progress is evaluated before any goal change. The signed distance calculation
has no lower motion threshold. The transport weight remains 0.65 in both
arms. Alignment, turn correction, depth and posture terms are shared. Hold
in both arms observes the full anchor displacement and scores 3D drift and
speed against 2 mm and 2 mm/s, with the same tilt and depth costs. Hold is
requested with zero terminal velocity.

Changing requests preserves the native body, joint velocities, previous action and
course history. The existing 1.25 ms integrator, constraints, water forces and
clearance certificate run 16 times each 20 ms control. The shared CUDA sampler
and PPO gradient use the Gaussian latent action. Physics receives its tanh
transform, and deterministic evaluation applies the same action transform
to the mean. There is no controller, fallback or reset between requests.

## Fixed budget and evaluation

- First arm heading, second destination. Each gets eight updates of 256 worlds
  × 64 steps, with two PPO passes and minibatches of 4,096. That is at most
  131,072 transitions, 2,097,152 physics substeps and 64 optimizer steps for
  each network per arm. The two networks in this count are actor and training
  critic. Shared seed 20261124, Gaussian log standard deviation −3.2,
  gamma 0.998, lambda 0.9797, actor/critic learning rates 0.0001/0.0003,
  clip 0.2 and KL cap 0.05.
- Both arms take the same hashed training fixture bundle and held-out evaluation
  bundle. Use retained native warm-motion captures and distinct stationary
  and poorly oriented starts. The fixed world targets and timing are known to
  the task. The learner chooses all actuator commands.
- Score frozen checkpoints 0, 2, 4, 6 and 8. Cover two straight swims,
  both ±15° and ±30° turns, two alternating sequences, onward swimming,
  full-speed brake and hold, calm hold from rest, and stationary and
  poorly oriented recovery. Report individual leg/skill passes and continuous
  sequences. Use the same held-out fixtures at all five checkpoints.
- For moving legs, require at least 6 mm/s projected COM speed and no more than
  5° of course error in the final two seconds. Moving and holding require
  10° peak tilt, 1 mm depth error and positive certified whole-body clearance.
  Station keeping uses the existing four-second braking and eight-second hold
  windows, 0.5 mm/s hold speed, 2 mm anchor radius and axial excursion, and
  2 mm lateral error. Report actual goal progress, COM turning radius,
  achieved course change, depth, tilt, speed, physical faults and runtime.
- One launch per arm and one evaluation per available matched checkpoint.
  The fixed external cap is 4.000 wall seconds per process. The CUDA trainer
  and evaluator use internal 3.5-second deadlines. Numerical, physical, KL
  and timeout failures close the affected process and retain evidence.
  Only completed matched checkpoints enter the learning curves.
  Exclude compile and reporting time from trainer throughput.
- Nominate the destination motor only if its results improve movement across
  the held-out tasks and it passes straight movement, both signs, full repeated
  sequences, onward swimming, both hold cases and both recoveries. A gain in
  training reward or reduced waypoint distance alone cannot establish
  reliable movement. Report results before any extended study or chooser run.

Existing checkpoints are read from their original files and checked by hash
before and after the comparison. Actors, fixture bundles, raw results and
process receipts must use a new output directory. The runner records the
exact source, executable and evidence bindings used for this pair.

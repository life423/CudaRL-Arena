# Food-free flexible-tail PPO chooser

Train one chooser on the RTX 5070 using the frozen fitted gait and
update-eight yaw offset used by the east-bank comparison. Bind the
same native flexible-tail equations, actuator limits, fully submerged
clearance envelope and Godot habitat descriptor. The CUDA learner
changes only the categorical chooser actor and its training critic.

Choose among keeping course and requesting a 15° left or right turn
from the measured COM course. Decisions require four elapsed seconds
and a valid COM course measured across two complete root-yaw beats.
The cue expires after six seconds. Native simulation integrates each
1.25 ms substep and evaluates motor inference at the existing 80 ms
cadence. Body state, held targets and applied trim carry across
requests. Four seconds is the shortest experimental request interval.
It leaves room for repeated turns inside the certified underwater area
at the measured swimming speed. This faster cueing needs its own
physical qualification in the held-out suite.

There is no scripted direction, fixed route or food. The chooser sees
the native terrain clearance envelope near the body and along candidate
directions, motion and visit recency. The actor and critic each have 32 tanh hidden units, with three and one linear outputs respectively. Every selected action passes to
the same frozen motor. The physics guard stops any trajectory that
loses positive certified whole-body clearance. It does not change a
requested turn or move the swimmer.

## Training allocation

One native CUDA process with 96 worlds, four chooser decisions per
world per rollout, up to 16 PPO updates, 128 samples per minibatch and
four optimization passes. The experiment collects at most 6,144 policy
decisions. Use discount 0.97, GAE lambda 0.90, clip 0.20, entropy 0.01,
critic factor 0.50 and actor/critic learning rates 0.0015/0.003. Use the
archived training bundle and fixed seed 20261124. Training placements sample the inner 40% of normalized radial distance and yaw uniformly. The
held-out bundle is excluded. The process checkpoints every four
completed updates and saves the newest completed actor as the sole
held-out candidate, with a cooperative cutoff at 17.5 s and a hard 20 s launch cap. Including the at-most 300 control
steps per decision and per episode warm-up, the upper limit is
58,982,400 native substeps. Numeric or contract failure rejects the candidate. Course-cue,
posture and bank-envelope stops are learning outcomes in training
worlds. Training restarts a world only at an episode boundary.

Reward actual movement at up to +0.04375 per simulated second, with no
transport reward below 6 mm/s rolling speed. A first 50 mm cell visit
earns +0.60 after at least 1 s and 6 mm of same-cell displacement.
Later visits earn up to +0.15 after a 90–300 s absence. Clearance
shaping subtracts 0.15 times positive increases in squared clearance
risk inside 25 mm. A trajectory failure costs 4. These quantities use
native whole-body COM and clearance.

## Held-out admission

Use the final completed chooser checkpoint and deterministic argmax,
the selection rule intended for Godot. Run 12 fixed independent native
cases for 120 simulated seconds each, including the exact cold Godot
start and unseen placements/headings based on the archived evaluation
bundle. If all 12 pass and aggregate coverage reaches 20% of the
certifiable 50 mm cells at the registered depth, run four fixed
300-second endurance cases. The evaluator binds the exact fixture
placements and seeds. Each trajectory remains continuous.

Every case must finish its declared duration with positive certified
whole-body clearance and no numerical, course-cue, bank or posture
stops. Peak tilt must be at most 10° and depth error at most 1 mm.
Mean horizontal transport must reach at least 6 mm/s after the initial
6 seconds, with at least two learned turns. Require at least eight
credited visit cells per regular case and 14 per endurance case. A
credited visit requires a one-second same-cell dwell and at least
6 mm displacement. Report coverage, revisit behavior, time spent
within 20 mm of the clearance envelope and every failed case.

All evaluation, including conditional endurance, has a 2,112,000
substep cap and a separate 20 s hard process cap. Publish autonomous
mode to the registered project only if the full suite passes with
the same native chooser, selected actor and frozen motor. Each
incomplete or failing study leaves the last verified Godot mode active.


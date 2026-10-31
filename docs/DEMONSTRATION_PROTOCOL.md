# Demonstration initialization1: ready proposal, not launched

Keep physical model6,42observations, two learned outputs and all frozen evaluation
limits unchanged. A memoryless feedback teacher reads only the network observation:
requested forward speed r and measured forward speed v. It requests1Hz and
A=sqrt(clamp(28*r+80*(r-v),.05^2,.65^2))rad. Coefficients have unitsrad^2*s/m;
28 is an initial command-space estimate from the recorded late-stroke response,
not a validated inverse fluid law. Both coefficients must pass physical calibration.
No integral/history hidden from the NN. All commands use the same filtered stroke
generator and torque-limited native joints. No added forces or body kinematics.

One proposed run, build/speed-motor/2026-10-31-demonstration-v1/:
-32CPU/CUDA readiness controls.
-960teacher controls:4/6/8mm/s each in cases0/3 for8s, scored6..8s.
-240teacher controls: continuous8->4->6 requests, each4s scored2..4s.
-All9teacher profiles must pass the existing speed/path/depth/tilt/work/safety
 criteria. This longer constant calibration establishes settled examples; it
 does NOT replace the unchanged4second frozen NN evaluation. Stop before fitting
 on any teacher failure; no retuning, retry or budget reuse.
-Then paired frozen baseline/candidate evaluation,2080controls total: five speeds
 4/5/6/7/8 in cases0/3 plus6->4->8 sequence.5/7 are withheld from all examples.
-Supervised initialization: only the recorded1200real teacher observations and
 applied two-latent-action labels. Warm-start passing actor; fresh Adam, lr.001,
 eps1e-5, grad cap.5. Fixed600minibatches of256=153600sample presentations.
 Frequency target is1Hz; the NN still produces both outputs at evaluation. No
 teacher/feedback blend or fallback remains active during frozen NN evaluation.

Hard ceiling3312physical controls/33120native substeps, failed/backend work included;
600optimizer steps maximum;20second inclusive native wall limit plus25second own
process watchdog. No PPO, sweep, extension or automatic retry. Selection unchanged:
10/10constant AND3/3switches, better normalized speed error, no worse8mm/s RMSE or
work/metre. Otherwise retain passing8mm/s actor. Source/actor/exe/settings freeze
before actual launch. Readiness notification and explicit launch approval required.

Four nonphysical checks pass: feedback bounds/units, identical observable state
produces identical labels and rotation invariance, invalid-input rejection, real
CUDA MSE gradients. No teacher simulation or fitting has run. The teacher may fail
to achieve slow speeds; the next run would measure that, not assume feasibility.
Research snapshot schema2 is actor+Adam only; old PPO source/results stay archived.


## Approved execution outcome

Closed01:20:42UTC on2026-10-31 after.668621native seconds.1152controls/11520steps,
7teacher passes,1failure,1unrun. Constant4/6/8settled speeds pass;8->4deceleration
fails at5.234mm/s (allowed3.6..4.4) and1.315mm/sRMSE (limit1.25). Minimum strength
command for56/80intervals. No fitting/evaluation/adoption/retry. Unused21600steps
closed; existing passing actor remains unchanged. See closure.json in the run folder.

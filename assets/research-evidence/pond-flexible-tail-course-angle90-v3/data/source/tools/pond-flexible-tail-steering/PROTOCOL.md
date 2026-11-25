# Faster flexible-tail steering — registered session 1

November 24, 2026. The user authorized step 1: train a separate faster
steering candidate. Preserve the fitted swimming stroke, original update-eight
yaw-offset checkpoint, habitat, physical equations, integration, actuator limits
and slew. No chooser training, food, route, bank-triggered turn, artificial force
or production publication belongs to this session.

## Learning task and budget

One shared linear yaw-offset actor retains the four historical inputs and
zero intercept. Initialize its coefficients from update eight. Only those
four coefficients train. The fitted gait and pitch output remain frozen.
The critic is a training auxiliary with 36 observations: the 31 existing gait
observations, four steering observations and remaining episode time. It has
32 tanh hidden units and a scalar value. This adds no behavioral controller.

Use 1,024 CUDA worlds, 64 sampled 80 ms actuator decisions per world per
rollout, minibatches of 4,096 and four PPO passes. Each full rollout has
65,536 transitions. Actor learning rate 0.0003, critic 0.001, fixed Gaussian
log standard deviation -1.2, discount 0.995 and GAE lambda 0.97 per 80 ms,
clipping 0.20, value coefficient 0.50, entropy coefficient 0.01, target KL
0.015 and hard KL 0.05. Score the sampled Gaussian latent before tanh and
the existing actuator slew. Partial decisions use elapsed-time discounting.

Train at most 128 updates / 8,388,608 transitions, with a cumulative
cooperative training-process wall budget of 1,200 seconds. Finish an update
already started and save it. The external process deadline is 1,800 seconds
including startup and final checkpoint work. No silent restart or extension.
Save actor, critic, both Adam states, all body/request/filter/history state,
episode and sampling counters after every complete update. Export immutable
actors for checks every eight updates. Source/configuration and frozen input
identities must match on resume. A numeric, native physics, serialization or
contract fault stops the study; retain the prior complete checkpoint and fault
evidence. A scored training clearance/posture failure terminates its episode
and is counted before a new training episode starts.

Training requests are balanced left/right 15°, 30°, 45° and 60° from the
archived measured whole-beat incoming course, using only the training bundle.
Hold each target for 13 seconds, then swim on it for two seconds. Carry body,
servo state, momentum, applied trim and course history through that handoff and
across rollouts. Only an episode terminal starts a new body. Evaluation starts
are withheld. The original half-second COM course observation and decoder
limits stay unchanged to isolate whether sustained PPO can learn a faster gain.

Per simulated second the reward is 0.25 projected transport, 1.5 course
alignment, 0.75 signed course-error reduction, minus squared tilt (10° scale),
0.75 squared depth error (1 mm scale) and 0.01 squared applied-offset effort
(0.25 scale). Transport is clipped to [-1, 1.5] of the existing 8 mm/s request.
Alignment is exp(-(course error / 15°)^2), graded by actual rolling COM speed
up to 6 mm/s. Correction is error reduction divided by 15°/s, clipped to
[-1, 1]. Transport/alignment require the existing depth/posture envelope.
Clearance or posture episode failure adds -2. No motion-free coverage credit.

## Prerequisites and accounting

Reuse the already verified shared physics/habitat. Before training, compare
the frozen sparse gait and original readout against full CPU inference on saved
observations, with zero physical steps. Verify exact inactive bypass. Run the
fixed physical baseline once. Verify full CUDA resume by update one, continuous
update two, then reload update one and repeat update two: three actual updates,
196,608 transitions and at most 12,582,912 physics substeps, recorded separately.
The verification has a 600-second external limit. Do not call it zero training.

The main session has at most 536,870,912 attempted physical substeps. The
fixed baseline is one evaluator invocation. Checks at 8,16,...,128 compare
immutable candidates against the same original. Each evaluator invocation has
at most 883,200 substeps and a 30-second external deadline; no failed invocation
is automatically retried. Retention failure requests a cooperative training stop.
Lack of turning mastery is a measured learning result, not an automatic abort.

## Fixed comparison and decision

Run each of four distinct moving phases (archive rows 0,2,4,6) with both
signs of 15°,30°,60°: 24 cases per policy. Compare the original update-eight
offset and candidate with identical fitted gait, habitat and starts. Give each
13 seconds to turn, two seconds onward and two seconds for complete-beat
closure. Evaluate the final two turn seconds and first two onward seconds.
Require actual mass-weighted COM course error <=5°, projected and rolling
speed >=6 mm/s, whole-case tilt <=10°, depth error <=1 mm and positive native
whole-body clearance. Report actual signed course change, time to one second
continuously aligned, effective radius from COM travel, minimum speed and gap,
all failures and fixed-window reversal counts. Require >=80% of the matched
original complete root/passive beat frequency and root/passive swing in both windows.

Also run all 12 archived straight/from-rest cases for 12 seconds per policy,
with the original two final two-second windows, using the exact inactive
offset bypass. This checks swimming and start-from-rest retention. It does
not establish braking or a new holding skill. The existing frozen hold policy
is untouched; full rest/current/disturbance qualification is outside this task.

Fast-steering qualification requires all 24 turn cases and all 12 retention
cases to pass. Rank experimental candidates by passing 60° cases, then 30°,
then 15°, then lower mean 60° course error, accepting only retained swimming.
Report left and right separately and retain every scheduled checkpoint result.
Do not tune reward, criteria, starts, seeds or budget using held-out results.
Repeated-turn and full pond-navigation qualification precede any Godot adoption.

Record compilation, GPU physics/collection, network/PPO, transfers/CPU waits,
evidence writing, CPU evaluation and end-to-end elapsed time separately.
Overlapping timings must not be added together as exclusive costs.

# Straight-swim feedback and the second food-free turning exercise

The candidate starts from motor 336 with its saved holding gain. For a straight
request, the motor once again receives the sideways displacement from its
requested line in body coordinates. A displaced swimmer can see which side of
the line it is on. The network supplies the joint targets. Rest and heading-only
turns retain their previous inputs and the body moves through the same water
physics. [Protocol](POND_TURNING_PPO_V2_PROTOCOL_2026-11-23.md)

## Before training

With identical inherited weights and matched 60-second straight requests,
restoring the observation reduced peak sideways drift from **11.78 to 0.96 mm**
for the first pose and from **9.43 to 0.85 mm** for the second. The limit is
2 mm. The candidate recovered **all 28** of the saved motor's passing
long-swim windows across the two poses. It also passed all four short swims
and all 14 rests, including two five-minute holds. Three of 16 turns passed.
The optimizer started only after the entire baseline assessment.

## Checkpoints

All 32 updates completed. Each checkpoint passed the same three turns from
pose 0. Mean course error below uses the last available full four seconds of
**horizontal COM velocity** for each of the 16 cases, including the two that
ended early. The delay is the time after a request changes until the end of
the first four-second window meeting the course, speed, depth and tilt
criteria. Its measurement resolution is 50 ms.

| Update | Long windows | Short swims | Rests | Turns | Mean course error | −15° completion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 28/28 | 4/4 | 14/14 | 3/16 | 23.45° | 15.40 s |
| 8 | 28/28 | 4/4 | 14/14 | 3/16 | 23.59° | 15.35 s |
| 16 | 28/28 | 4/4 | 14/14 | 3/16 | 23.57° | 16.00 s |
| 24 | 28/28 | 4/4 | 14/14 | 3/16 | 23.44° | 15.35 s |
| 32 | 28/28 | 4/4 | 14/14 | 3/16 | 23.46° | 16.00 s |

The −5° request took 4.00 s and the +5° request took 4.15 s at every checkpoint.
The other 13 turns never produced a qualifying window. The native evaluator's
fixed late interval starts at 16 s and ends at termination or 20 s. On that
definition the mean error was 23.066° before PPO and 23.067° afterward.
[Per-case courses and times](../assets/research-evidence/pond-turning-v2/turn-cases.csv)

All 13 failures at update 32 turned in the requested direction but fell
short. From pose 0, −30° and +30° requests produced about −19.4° and +14.9°
relative to the original requested course. Across the 16 cases, 12 exceeded
1 mm depth error, eight exceeded 10° tilt and four missed 6 mm/s projected
COM speed. These failures overlap. The minimum measured whole-body terrain
clearance was 64.96 mm. Both +90° trials ended at the 5 mm depth threshold.
On the straight swims, final peak lane errors were 0.93 and 0.82 mm.

## Policy inputs and reward

The first recorded rollout included 248 turn worlds, with 132 positive and
116 negative requests. At the scheduled change, the requested world course
reconstructed from the actual GPU actor inputs agreed with the assigned
course within 0.000010°. Requested velocity pointed in the same direction.
The largest horizontal component of the scaled position-error input on these
turns was below 1e−9. The sideways straight-swim signal therefore did not
pull these turning worlds back toward their original lanes.
[Input audit](../assets/research-evidence/pond-turning-v2/input-audit.json)

The reward favors COM motion toward the new heading. At exactly 8 mm/s and
equal depth, tilt and effort, a 15° course error still earns 97.8% of the
aligned movement reward. That provides a weak incentive to correct a moderate
error. The experiment isolated the observation change and did not test how
much this reward shape contributed to underturning.

The timescales also matter. The successful −15° turn took 15–16 s to
qualify, while one rollout segment spans 3.2 s. Body and filter state carry
across segments and the critic bootstraps at the boundary. With the fixed
discount 0.995 and GAE lambda 0.95, a residual 40 controls (2 s) later
contributes about 10.5% as much to the present advantage within the same
segment. This gives a concrete credit-assignment hypothesis. Its effect was
not isolated in this experiment.

## Time and saved candidate

| Activity | Measured time |
| --- | ---: |
| Initial candidate directory created through final evaluation result written, including orchestration gaps | 626.89 s |
| Four native training commands, including their output capture and checkpoint writes | 292.20 s |
| GPU rollout collection, including physics | 289.58 s |
| GPU physics within rollout collection | 288.82 s |
| GPU PPO optimization intervals | 0.504 s |
| Rollout readback plus CPU reward aggregation | 0.100 s |
| Five native CPU skill assessments | 40.05 s |

The run collected 1,048,576 transitions, about 3,589 per training-command
second, and executed 1,024 actor and 1,024 critic Adam calls. The phase spans
overlap. The raw batches and 32 checkpoints totaled 397,942,272 bytes.
Training included serialization, synchronous flushes and replacements in its
command times. This build lacked a dedicated evidence-writing timer, so those
operations cannot be accurately separated retrospectively. The public
[run summary](../assets/research-evidence/pond-turning-v2/final-summary.json)
includes the individual stage times and the bounds of the end-to-end clock.
The evaluator recorded 98,351 bytes in its ledger while its CSV files were
open. The five assessment directories actually held 408,844,773 bytes after
closure, within the 1 GiB allocation. The next evaluator closes the writers
before charging the files to the ledger. Skill metrics and elapsed times
come from the original assessments.
[Closed-file byte audit](../assets/research-evidence/pond-turning-v2/evidence-byte-audit.json)

The [update-32 research checkpoint](../assets/research-evidence/pond-turning-v2/candidate-32.bin)
has SHA-256 `23b48585239dc2b2699a3c0ef1294877f3116412783a87a96e8a44b9beb48cfb`.
Godot plays the original motor 336. The candidate retained its swimming and
resting skills and did not qualify for reliable turning or pond navigation.

## How the run was measured

The native Release trainer ran 512 CUDA worlds with one candidate motor on the
RTX 5070. Each PPO update used 64 controls per world, four epochs and
minibatches of 4,096. Each 50 ms control advances ten physical substeps.
The CPU evaluated each frozen checkpoint on the same starts and requests.
Turn success uses actual center-of-mass course and speed, depth error and tilt.
Long straight compatibility uses the original clear-water fixture. Turn
assessments use the physical pond. Whole-pond exploration requires a navigation
trial with the chooser.

The [public evidence](../assets/research-evidence/pond-turning-v2) contains
compact native assessments, per-window and per-case results and training
timings. Full rollout and substep traces stay in the preserved local study
directory `build/turn32`.

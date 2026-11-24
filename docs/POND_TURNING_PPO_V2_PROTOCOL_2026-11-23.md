# Food-free turning with straight-swim feedback

Study ID `pond-motor-ppo-v2-2026-11-23`. Start a separate candidate from
the preserved motor 336 and its saved holding gain. Use the same training
seed, actual pond physics, water, joint limits and request mix as the
[first exercise](POND_TURNING_PPO_PROTOCOL_2026-11-23.md). The one behavioral
change is to give a Straight request all 41 original observations, including
horizontal cross-track error. Rest keeps its original inputs. The Turn task
retains the earlier depth-preserving, heading-only inputs throughout its
warmup and requested turn. One motor produces all four actuator targets.

Reuse the verified runtime 4 CPU and CUDA physics results with identical
physical sources. Before any optimizer call, measure the actual update-zero
candidate from the two original long straight starts, all four short swim
starts and all 14 Rest cases, including the two five-minute holds. Require
every one of the parent's 28 passing long-swim windows, all four short swims,
all 14 rests, and no hard fault. The parent missed two of 30 windows near
startup. Also measure all 16 requested turns. Successful turning is a learning
outcome, and is not an entry condition.

One GPU training run is authorized, with at most 32 PPO updates. Each update
uses 512 worlds × 64 controls, four epochs and minibatches of 4,096. Keep the
previous reward, thresholds, hyperparameters and initialization. Retain original
files and previous candidate checkpoints. Save every completed update and
evaluate at 8, 16, 24 and 32. Stop training at any failed retention gate,
invalid batch, hard physics fault, nonfinite result, KL above 0.05 or budget
limit. Complete the remaining planned skill measurements on a frozen checkpoint
after a behavioral miss. A hard fault stops the affected assessment. No
checkpoint is promoted to the viewer by this exercise.

The training cap is 1,048,576 controls, 10,485,760 physical substeps, 1,024
actor and 1,024 critic Adam calls, 1,200 native seconds and 4 GiB of evidence.
The five assessments together allow at most 184,000 controls, 1,840,000
substeps, 900 seconds and 1 GiB of evidence. Bind the candidate, protocol,
evaluator, trainer, parent and prior physics receipts before collection.
Store raw outputs under `build/turn32`; publish compact measured results.
Report turn completion using actual whole-body motion, and compare retained
skills on matched starts. Continuous navigation requires a separate assessment.

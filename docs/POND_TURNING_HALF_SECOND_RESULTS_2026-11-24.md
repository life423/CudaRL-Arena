# Gentle turns with half a second of movement feedback

**Training and verification of the candidate's movement took 10.78 seconds.** Eight GPU PPO updates produced five stabilized right 10° requests during practice. On the original 16 independent turns the candidate passed the same three as motor 336. The visible Godot swimmer still uses that original motor. [Checkpoint](../assets/research-evidence/pond-turning-v5-half-second/candidate-8.bin)

## Movement

The RTX 5070 ran one shared policy across 512 native C++/CUDA worlds. Each update collected 64 control ticks per world. Four passes used minibatches of 4,096. The new reward measures horizontal whole-body center-of-mass displacement over ten 50 ms ticks. The direction and projected travel over that interval feed the turning reward. The practice stabilization check uses that course with the existing instantaneous projected-speed, depth and tilt limits. Request changes preserve the swimmer's body, momentum, actuators and movement history. [Protocol](POND_TURNING_HALF_SECOND_PROTOCOL_2026-11-24.md)

| Independent check | Preserved motor | Candidate at update 8 |
| --- | ---: | ---: |
| Turning | 3/16 | 3/16 |
| Mean absolute course error | 23.45° | 23.50° |
| Mean speed toward the turn request | 6.68 mm/s | 6.69 mm/s |
| Right 15° completion time | 15.40 s | 15.30 s |
| Short swims | 4/4 | 4/4 |
| Required windows in two 60 s swims | 28/28 | 28/28 |
| Rest, including two five-minute tests | 14/14 | 14/14 |

Both checkpoints passed right 5°, left 5° and right 15° from the first pose. Right 5° took 4.00 simulated seconds and left 5° took 4.15 seconds at both checkpoints. The other 13 requests still undershot. The left 15° trial turned 6.94° from its pre-request course and finished 7.97° off the requested heading. Straight-swim speeds in the candidate's post-startup windows ranged from 7.88 to 8.35 mm/s. The previously measured zero-update baseline was reused only after verifying the parent actor, holding gain, fixtures, observations, scoring, physics and receipts. All 36 independent cases ran freshly on the trained checkpoint. [Turn measurements](../assets/research-evidence/pond-turning-v5-half-second/turn-cases.csv) · [Skill summary](../assets/research-evidence/pond-turning-v5-half-second/checkpoint-summary.csv)

During practice the candidate received **254 left and 254 right requests**, matched by magnitude from 10° through 15°. Five right 10° requests met the stabilization rule. There were 249 timeouts, zero episode interruptions and 254 requests still open at the allocation limit. None of the 43 left 10° requests stabilized. This gives the next experiment a specific comparison: measure the change in course and joint commands during matched left and right 10° attempts. The larger fixed turns still require independent improvement. [Practice counts and trace hashes](../assets/research-evidence/pond-turning-v5-half-second/v5-training-summary.json)

## Time on this PC

| Timed work | Wall time |
| --- | ---: |
| Revalidate the archived zero-update baseline | 0.05 s |
| Eight GPU updates, 262,144 transitions | 2.22 s |
| Fresh CPU evaluation of 16 turns and 20 retention cases | 6.59 s |
| Native runner through all skill results, including initialization | **10.78 s** |
| Initial command through its publication error | 11.60 s |
| Separate publication recovery, including Godot validation | 2.21 s |
| Active time across the two commands | **13.81 s** |
| Original launch to published, verified Godot result, including repair | **49.82 s** |

The training processes handled about 118,296 transitions per second for this first eight-update block. CUDA physics took 1.30 s, GPU optimization 0.15 s and transfer plus host reward reduction 0.03 s inside that total. The matched formatter benchmark reproduced every byte of the prior 38.7 MB turn trace, reducing formatting time from 0.92 s to 0.34 s. These numbers apply to this short exercise. [Timing and checkpoint identities](../assets/research-evidence/pond-turning-v5-half-second/time-and-checkpoints.json)

The first wrapper completed training, saved the checkpoint and verified the movement results. Its next assignment used `$run`, which PowerShell treated as the existing switch parameter `$Run`. The repaired wrapper uses a separate variable name. Its publication-only mode checked the saved evidence and displayed the result in Godot without repeating a PPO update or a physical trial. A zero-optimizer fixture also exercised the repaired normal command path. The 15 s goal was met through evaluation. A single uninterrupted invocation through Godot publication still needs a future timed run. [Original error](../assets/research-evidence/pond-turning-v5-half-second/first-process-error.txt) · [Godot verification](../assets/research-evidence/pond-turning-v5-half-second/godot-details-result.json)

The main pond controls now show Play, Restart and the activity. Swim → Details contains the optional left/right 5° checks. It pauses the pond, confirms the request, and keeps the body state intact until play resumes. Rendered checks passed at four window sizes. [UI checks](../assets/research-evidence/pond-turning-v5-half-second/ui-checks.json)

# Recovery from the Godot open-water approach

**One different chooser decision 15.274 seconds before the archived stop sent the same swimmer onto a course it could sustain for the next half-hour.** At 27.2 seconds, its actual horizontal center-of-mass speed was 12.761 mm/s. Frozen +112 requested 12 mm/s and +8°/s. Giving the same speed and −4°/s for 0.8 seconds, then returning control immediately, completed 1,800 more seconds. On release +112 chose −8°/s itself. The saved update-128 parent and motor 88 were preserved.

| Exact fork and treatment | 120-second result | 1,800-second result |
|---|---|---|
| 24.0 s, 13.655 mm/s actual. Change +112’s 12 mm/s, −4°/s to 12 mm/s, −8°/s for 0.8 s | Pass, minimum clearance 30.21 mm | Pass, 12.78 mm/s mean speed, 18.83 mm minimum clearance |
| 27.2 s, 12.761 mm/s actual. Leave +112 in control | Clearance stop 15.27375 s later | Same stop |
| 27.2 s, 12 mm/s and −4°/s for 0.8 s, then +112 | Pass, minimum clearance 24.69 mm | Pass, 12.90 mm/s mean, 18.13 mm minimum clearance, zero stalled controls |
| 27.2 s, 12 mm/s and −8°/s for 0.8 s, then +112 | Pass, minimum clearance 27.83 mm | Clearance stop after 1,003.12 s |
| 28.8 s, 14.431 mm/s actual. Transfer the complete decision sequence to the frozen parent | Pass; all 30 brief interventions followed by +112 failed here | Parent passed with 13.35 mm/s mean, 4.75 mm minimum clearance and a longest stall of 4.08 s |
| 29.6 s through 42.4 s, every 0.8 s | No passing parent or brief intervention | No long follow-up |

There were 24 consecutive decision states from 24.0 to 42.4 s. The latest recoverable **brief handoff** was 27.2 s. A **continuous parent takeover** worked as late as 28.8 s, when the original stop was still 13.674 seconds ahead. These are finite tests of specified requests. Other multi-decision sequences could have different outcomes.

## What changed and what stayed fixed

The native evaluator independently reproduced all 53 archived actions and the stop on exactly the 33,980th attempted substep, after 33,979 accepted. Every +112 reference fork rejoined the original trajectory and stopped at exactly 42.47375 s. The candidates were bitwise copies of the decision state *within each fork*, including physical body, motor and controller history, visitation memory, clock and stall counts. The checksum column identifies shared copies within each executable run. It hashes the C++ object representation, including ABI padding, and is not a portable cross-process state identity. Across runs, the replay checks COM, clearance, logits, actions and the terminal physical state against the [archived Godot trace](godot-start.csv).

Every state was tested with +112, sustained parent control, and all 15 speed/rate choices held for one or three chooser periods and then returned to +112. Each short comparison ran for 120 seconds *after the fork* unless the physical guard stopped it sooner. The diagnostic required positive whole-body clearance, ≥6 mm/s mean horizontal transport, ≤407 consecutive native slow controls, ≤10° tilt and ≤1 mm depth error. No branch reset the creature. A negative certified clearance bound causes a stop and does not establish impact.

At both 24.0 and 27.2 s the whole-body clearance bound under the current pose was about 60 mm. This number measures clearance from the bed, not horizontal distance to the bank. At 42.4 s only 0.726 mm remained and actual speed was 10.447 mm/s. Repeating the same request or switching speed/rate at that late decision led to a stop 0.07375 s later. The sign reversal that worked at 27.2 s applies to that state and is not a shoreline rule.

Success was phase/state dependent. The number of passing brief interventions, out of 30 per fork, was 18, 12, 11, 3 and 14 at 24.0, 24.8, 25.6, 26.4 and 27.2 s respectively, then zero through 42.4 s. Of the 14 that passed short evaluation at 27.2 s, ten also passed 1,800 s: five of eight 0.8-second interventions and five of six 2.4-second interventions. The first ranked intervention by 120-second clearance later failed; all 14 long results are preserved. The persistent parent succeeded in the short check at 28.0 and 28.8 s even though no brief request handed back to +112 did. [Decision grid](grid.csv) · [Full initial matrix](initial/forks.csv) · [Resolution matrix](refinement/forks.csv) · [First 27.2 s long result](refinement/long-confirmation.csv) · [Remaining 27.2 s long results and parent result](long/long-grid.csv)

## Budget and use

The initial native run took 12.208 seconds wall time and the refinement took 16.246 seconds. The 14 additional confirmations took 20.652 seconds. The three runs performed 32,451,276 attempted substeps including their prefixes, with zero optimizer updates. Initial and refinement each tested 384 matched short branches and one long follow-up. The final 14-branch supplement verified each branch's 120-second endpoint against the saved short result while continuing the exact same physics to 1,800 seconds.

The replay supports GPU training on independently generated open-water approaches while preserving near-parallel practice. The tested states, actions and restored physical trajectories are held-out diagnostic evidence; no override belongs in Godot or the PPO dataset. [Original protocol](protocol.md) · [Resolution supplement](refinement.md) · [Long confirmation supplement](confirmation.md)

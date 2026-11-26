# Can motor 88 recover the twelve failed moving approaches?

The +104 replay matched the archived position, control time, clearance, logits and decision at each original chooser cue, and reproduced all 12 physical stops at the same substep. We preserved the first 0.8 seconds of native swimming in every fork. Actual horizontal speed at the decision ranged from 10.30 to 10.95 mm/s. The earlier static geometry audit found just 38.57–62.50 mm of forward horizontal room at spawn.

| Depth (mm) | Approach | +104 stop (s from spawn) | Longest of 87 branches (s) | First measured upstream translation with a qualifying 180-s swim |
|---:|---|---:|---:|---:|
| 72 | A | 4.08 | 4.30 | 120 mm |
| 72 | B | 6.58 | 7.81 | 60 mm |
| 72 | C | 3.95 | 5.02 | 120 mm |
| 72 | D | 4.03 | 4.03 | 120 mm |
| 75 | A | 4.56 | 4.83 | 60 mm |
| 75 | B | 6.12 | 6.77 | 60 mm |
| 75 | C | 4.23 | 4.66 | 120 mm |
| 75 | D | 4.55 | 4.55 | 120 mm |
| 80 | A | 5.04 | 5.39 | 60 mm |
| 80 | B | 5.46 | 5.96 | 60 mm |
| 80 | C | 3.97 | 4.55 | 120 mm |
| 80 | D | 5.20 | 5.39 | 60 mm |

The 1,044 original forks included the three greedy policies, every native speed and turn option at four durations, and two-stage turn/straighten or reverse requests. Every fork reached the native clearance stop before eight seconds. Feasibility under other command sequences is open. [Per-start measurements, including whole-body clearance](results/per-start.csv)

In translated controls, 6/12 starts had a physically qualified three-minute swim with 60 mm more approach room. All 12 had one at 120 mm. The three-depth parent itself completed all 12 at 120 mm. +104 completed six autonomously. At 60 mm, +104 completed three autonomously, and a temporary request followed by +104 succeeded on four starts. Room was measured upstream along the starting COM velocity. The shifted cases provide evidence for learning earlier decisions, with zero credit toward the original 12 starts.

For 80-mm D with 60 mm more room, +104 initially asked for 12 mm/s and +8°/s and eventually stopped at 76.68125 s. Keeping that turn rate and asking for 8 mm/s for 6.4 s before handing back to +104 gave 180 s with 4.615 mm minimum clearance and mean swimming speed 14.859 mm/s. The parent also passed from that shifted start. [All 80-mm alternatives and outcomes](results/depth-80/branches.csv)

Each process completed within 12.08 s wall time. In total, the three CPU evaluations took 31.20 s and attempted 20,070,190 native substeps. The study ran zero optimizer updates. Whole-body gap includes the first rejected sample in clearance-stop cases. The 20-second indicator also requires a 5 mm clearance gain over the path minimum and can stay false during safe travel across a flat bed. The independent 180-second physical and movement gates determine recovery. Each results folder includes the invoked executable and arguments, elapsed wall time, all alternatives and its native receipt. [Frozen protocol](protocol.md) · [Checkpoint and input hashes](results/bindings.json) · [Reconciled results](results/verification.json)

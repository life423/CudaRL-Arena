# Moving approaches and whole-pond retention

Freeze motor 88 and the fitted gait. Use the three-depth observer and the existing native chooser, skill and terrain equations. Preserve parent 128 and +112. Training starts come from other pose IDs, and validation and final poses and trajectories cannot enter training. Use parent 128 and +112 as frozen controls and score every eighth new checkpoint through +128. Only new checkpoints can win.

## Independent selection fixtures

At each depth of 72, 75 and 80 mm use the two archived moving phases 8 and 9 reserved from training, alternating phase with case index. A fixture starts with the archived body and actuator state and fresh chooser memory. The moving and tangent cases last 180 seconds, requiring three visited cells. The two interior cases run uninterrupted for 900 seconds and require 14 cells. Every case must finish with accepted physics, native velocity, posture and depth gates and a maximum of 407 consecutive low-speed controls. Measure contiguous stalls in controls of 20 ms. A fault or reset fails the case.

The six independent moving starts at each depth have measured straight-ahead native horizontal ranges of 45, 60, 75, 95, 125 and 150 mm. Place the whole animal using static translation along its actual COM travel direction. Require agreement to 1 mm and initial speed of at least 6 mm/s. This range describes the initial shape and speed direction. It is a fixture parameter, independent of which turn the animal actually executes.

| Fixture | Normalized bank azimuth | Incoming heading relative to outward radial |
|---|---:|---:|
| Short A | 22° | +27° |
| Short B | −112° | −32° |
| Medium A | 115° | −23° |
| Medium B | −38° | +36° |
| Long A | −164° | +17° |
| Long B | 67° | −41° |

Rotate every azimuth by 0°, 7°, and 19° at 72, 75, and 80 mm respectively. At each depth add two near-parallel starts at azimuths +51° and −147° with that rotation, tangent offsets −6° and +174°, phase 9 and 8, and native gaps 15 and 19 mm. For the rest and moving interior starts respectively use radius-normalized XZ and heading (−.16,+.12,−27°) and (+.31,+.14,121°) at 72 mm, (+.13,−.15,64°) and (−.34,−.12,−82°) at 75 mm, (+.11,+.18,147°) and (+.17,−.32,19°) at 80 mm. The rest and moving interior starts use archived phases 0 and 9.

Rank the 16 candidates lexicographically by highest minimum passing fraction across moving, near-parallel and interior suites, then most moving passes, most total passes, most safe 15-minute swimmers meeting their individual parent-128 validation cell counts, greatest worst long-swim cell ratio to that parent (unsafe swims score zero), most safe near-parallel swims with 15 seconds inside the near-shore band, greatest safe long-swim cell sum, then earliest update. An all-zero moving pass count remains visible. Record range and outcome per swimmer. Allow 60 seconds per independent depth process, with a 55-second internal deadline. At most 7,776,000 native substeps per checkpoint across the three depths. Bind the protocol, inputs, binaries and evaluator source before training and freeze the selected actor hash before the final suites.

## Final tests

Compare the selected actor against the original twelve failed moving approaches, four at each depth. Their archived placements must reproduce the previous evaluation starts file byte-for-byte. Run the original 24 shoreline, 12 ordinary, conditional four five-minute, and four 30-minute tests on that same actor. A deployment requires all twelve moving approaches to pass, all 24 shoreline finishes with at least four sustained near-shore swimmers and a 10.96 s mean near-shore time at each depth, all ordinary/endurance navigation requirements, and four safe 30-minute finishes with zero resets and maximum stall at most 407 controls per swimmer. Preserve the individual 30-minute floors of 185, 195, 218 and 201 credited cells, in the Godot, northeast, west and southwest starts. Preserve the pre-existing coverage and playback requirements. Complete the twelve exact-start probes before training. If none recover, also test approaches with additional runway and distinguish upstream-only recovery from exact-start recovery. Do not select a replacement using final outcomes.

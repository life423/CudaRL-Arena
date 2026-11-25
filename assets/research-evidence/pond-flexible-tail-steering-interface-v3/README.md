# Persistent heading requests improve measured steering

Repeated relative requests previously used the swimmer's measured course
as the next heading origin. The new opt-in interface accumulates a selected
left/right 15-degree request against the existing held heading. It changes
only the request and uses the same frozen gait, yaw trim and native physics.

| Paired diagnostic | Measured-course requests | Persistent-heading requests |
| --- | ---: | ---: |
| Actual course reaches 60 degrees | 0/16 | 16/16 |
| Mean effective turn radius | 426.7 mm | 329.9 mm |
| Minimum rolling speed | 9.31 mm/s | 9.65 mm/s |
| Peak tilt / depth error | 0 degrees / 0 mm | 0 degrees / 0 mm |
| Eventual clearance stops | 8/16 | 8/16 |

The first-cue physical states matched exactly for all 16 pairs.
Persistent requests reached 60 degrees in 33.44–33.76 simulated seconds
after the initial cue. All 32 trajectories ran continuously under their
declared stimuli. The retained trajectories were scored for 48 seconds.
The fixed diagnostic commands supplied no navigation policy or training
labels. No case reached 90 degrees before the end or native clearance stop.

The native process accepted 1,175,056 of 1,175,072 attempted substeps
in 0.616 seconds internal execution. The external receipt includes launch
and writing time. Zero optimizer calls ran. Persistent heading was selected
for the sustained PPO session because it produced greater actual course
change and a smaller radius in both directions. The navigation admission
criteria still require the full independent suite.

[Protocol](data/protocol.md) · [All measured cases](data/assessment/cases.csv) ·
[Bound inputs](data/bindings.json)


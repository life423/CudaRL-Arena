# Active parking: slow-approach research checkpoint

The same four torque-limited joints receive learned joint targets. A Hold-only
readout uses position and speed feedback and slow phase features. It can apply a
small corrective stroke and then let the body settle instead of forcing zero
motion. Swim still follows checkpoint 336 exactly. No current, threat, action
clamp, position snap, production loading or Godot change was used.

The fourth PPO pilot did not park (0/2). Its last batch entered the spatial
zone 709 times but recorded no gentle entries or quiet rows. A neutral-readout
control passed 8/8 short 8-second trials through coasting but failed 0/8 after
the check was extended to 20 seconds. The first untargeted phase search also
selected the passive control (0/12 in 20 seconds).

The native paired stroke diagnostic used ordinary actuation to verify that
reversing the phase lag of the two yaw joints reduced 12-second overshoot from
about 1.232 mm to 0.257–0.267 mm in both development poses. These commands were
not installed. The feedback study used their sign and scale as a starting point
and selected a learned readout via a fixed 26-trial mirrored parameter search on
development cases only. Trial 21 improved the starting nonzero seed's score
(0.512 to 0.594). It passed 12/12 20-second cases, including four new positional
variations. It was then frozen before the longer validation.

## Frozen 64-second validation

A case must stay safe throughout and be continuously within 1 mm of the pad,
at most 0.5 mm/s in translational speed and at most 0.035 rad/s on each measured
body/joint angular velocity component for the final 16 seconds. Duration is the
span between first and last consecutive passing samples at 50 ms intervals.

| Approach speed | Cases passed | Final quiet duration in successes |
| --- | --- | --- |
| 0.50 mm/s | 6/6 | 52.25–52.55 s |
| 0.55 mm/s | 6/6 | 52.30–52.55 s |
| 0.60 mm/s | 5/6 | 52.20–52.25 s |

The one failure was safe and very slow but finished 1.016 mm from center,
outside the unchanged 1 mm limit. No faster trials ran. The result establishes
parking in still water from these slow front approaches, not return after a
disturbance, arbitrary start positions, current, threat or production readiness.
Initial transient quiet streaks are not counted as final continuous hold.

Full Swim regression traces match byte for byte, including raw actor outputs
from four short and two long trials. Original parent checkpoint and protocol
SHA-256 hashes are unchanged. The separately checksummed feedback checkpoint
is research-only. No running viewer or default asset was replaced.

## Evidence

- `response/`: paired native stroke probe.
- `selection/`: fixed development search and 20-second assessment.
- `endurance/`: frozen checkpoint, raw 64-second traces and summary.
- `earlier-screening/`: negative controls and the capped fourth PPO pilot.
- `source/`: exact experiment sources and local build/dependency fingerprints,
  including the otherwise untracked evaluation helper. They are records of
  the Windows worktree build, with line endings normalized. They are not
  integrated production build targets.

Run `python docs/experiments/active-parking-2026-11-21/verify_evidence.py`
to recheck the recorded capstone results without executing any physics.

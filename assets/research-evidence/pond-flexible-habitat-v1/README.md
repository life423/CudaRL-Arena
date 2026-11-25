# Larger flexible-tail pond

The standalone flexible-tail project uses a nominal 1.76 × 1.20 m water
outline and an 180 mm deep central bed. This gives it about three times the
surface area of the previous 1.04 × 0.68 m outline. Godot reads the same
[descriptor](../../../flexible-tail-preview/habitat.json) as the native
fixture, including the irregular shoreline and sloped bed.

## Verified movement

Each of four independent starts swims for six simulated seconds and then
receives a 30° turn request for twelve seconds. Every case keeps its body
and joint state through the request change. The diagnostic measures actual
horizontal COM course, forward speed, depth and tilt. The full head and
tapered tail must remain above the bed and below the surface at every
sampled substep. The conservative vertical clearance bound encloses each
tail span and accounts for the highest possible bed under it.

| Shoal | Actual COM turn | Projected speed | Minimum vertical clearance |
| --- | ---: | ---: | ---: |
| East | +25.88° | 10.38 mm/s | 22.48 mm |
| West | −26.03° | 10.37 mm/s | 21.09 mm |
| North | +25.88° | 10.38 mm/s | 45.22 mm |
| South | −26.03° | 10.37 mm/s | 43.12 mm |

All four cases passed and finished over the submerged slope, outside the
previous water outline. Six geometric controls exercised the center, two
shoals, the surface, floor and dry-bank region. The case runner finished
in 0.419 seconds including startup and evidence writing. [Case measurements](cases.csv) · [Run log](native-log.txt)

An independent comparison of 144 land, slope, and shoreline samples matched
the renderer to the native profile. Maximum horizontal error was 0.000121 mm
and maximum height error was 0.000047 mm. [Geometry comparison](renderer-parity.json)

The [Turn recording](../../../flexible-tail-preview/recordings/turn.json)
in the standalone Godot project shows the east shoal case. The heading
request was supplied to the test fixture, which applied bounded root
targets. This is a movement and clearance measurement. The flexible tail
still needs a compatible PPO motor and chooser, plus verified shoreline
interaction. Continuous learned navigation and bank-contact recovery remain
unqualified.

## Consecutive turns

The second diagnostic used continuous body, actuator, and diagnostic feedback
state. It ran once in 0.680 seconds and performed 97,933 native substeps.
Each cue requested a turn relative to the measured final course of the
previous leg.

| Start | Result |
| --- | --- |
| Interior, alternating ±30° | Four consecutive passes over 54 simulated seconds, 25.76–25.88° each, with at least 68 mm of whole-body vertical clearance |
| Earlier outward start, repeated +30° | Three passes totaling 78.59° of course change. The fourth turn reached the conservative clearance bound. |
| Later outward start, repeated +30° | The first turn reached the conservative clearance bound. |

The fixture attempted nine of eighteen scheduled turns and passed seven.
The two clearance stops establish where the fixture could no longer certify
submergence. Actual bank contact and recovery were outside this diagnostic.
The results show that the body can link moving turns in open water and that
the tested outward paths need a different course or an earlier turn.
[Case measurements](repeated-turns.csv) · [Protocol](repeated-turns-protocol.json)

In **Tadpole Pond | Flexible Tail**, **Zoom to tadpole** switches from the full
pond to the follow camera. The mouse wheel, pinch, keyboard **+ / −**, and
on-screen zoom buttons adjust the follow framing. Press **Pond view** to see
the whole shoreline again.

## Reproduce

Run from the minimal-foundation worktree with a fresh output directory.

```powershell
cmake -S tools/pond-flexible-tail-check -B build/flexible-tail/native
cmake --build build/flexible-tail/native --config Release
& build/flexible-tail/native/Release/pond_flexible_tail_check.exe build/flexible-tail/new-habitat-check --habitat
& build/flexible-tail/native/Release/pond_flexible_tail_check.exe build/flexible-tail/new-repeated-turns-check --repeated-habitat-turns
```

The recorded runner used a four-second external process cap. The raw pose
traces for all four attempted starts remain in the local
`build/flexible-tail/habitat-v1` evidence directory.

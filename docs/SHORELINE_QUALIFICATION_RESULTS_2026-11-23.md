# Full-pond shoreline qualification

Godot and the native simulator use the same shoreline and submerged bed. The
water radii are 0.52 m and 0.34 m. The former 0.20 m by 0.12 m guard has been
removed from the live simulation. Movement comes from the four motor outputs,
the joint servos, water forces and physical bank contact. The runtime supports
a fully submerged body. Emerging from the water is a later capability.

## Runtime 4 qualification

The previous clearance search used the horizontal slope of terrain height in
three places that require the slope of body-to-terrain gap. On a rising bank,
those slopes point in opposite directions. The corrected search uses the gap
gradient for the support point, the local bound and the search direction. The
force solver continues to use the upward terrain normal. Across 24 saved body
poses the corrected clearance intervals agreed with an independent dense
reference. The reference intervals were at most 1.58 µm wide.

| Test | Measured result |
| --- | --- |
| Saved swimming and resting | All four short swims, all 14 rests and the original 28 passing long-swim windows were retained. |
| CPU shoreline and timestep | 36/36 cases, 68,000 controls and 688,000 substeps in 21.78 s. Largest matched-timestep COM displacement difference was 0.213 mm against a 0.25 mm limit. All four outward swims ran for 60 s. |
| CPU versus CUDA | All eight cases completed 128 controls on each backend. All 10,240 substeps per backend met the original tolerances. Maximum linear velocity difference was 0.00000466 mm/s. |
| Validation timing | 16.00 s elapsed, including 13.41 s of GPU physics. This timing measures the eight-case comparison. The 512-world training path has its own timings. |

The saved CPU results are in `build/pw30/run-v4`. The GPU results and sealed
inputs are in `build/pw31/gpu` and `build/pw31/sealed.json`. Earlier attempts
below remain part of the record.

## Runtime 3 timestep and retained behavior

In the first runtime, changing the step from 5 ms to 2.5 ms changed the energy
lost during a short impact by 16.48%. The resulting velocity difference
accumulated until the center-of-mass positions differed by 0.252717 mm. The
limit was 0.25 mm. A second integration rule delayed damping until the next
step after first touch and still missed the limit at 0.250734 mm.

The qualified CPU runtime accounts for contact beginning or ending within
the step. It uses the change in spring potential over the traveled distance
to compute the average spring force and applies the approach damper for the
fraction of the step spent in compression. The original contact constants,
body dimensions, actuator limits, 5 ms timestep and acceptance thresholds
are preserved.

| CPU check | Result |
| --- | --- |
| All planned cases | 36/36 passed across 68,000 controls and 688,000 substeps |
| Reference swimming and resting | All four short swims, 14 rests and the same 28 qualifying long-swim windows. Body, filter and command comparisons were exact over the paired reference runs. |
| Physical bank contact | Eight impacts passed. Worst whole-body gap across qualification was −0.1191 mm against a −0.5 mm limit. |
| Timestep comparison | All four passed. Largest COM position difference was 0.200 mm. |
| Outward swims | All four ran for 60 seconds. Final four-second mean COM speeds were 0.570, 0.030, 0.875 and 0.068 mm/s. |

The saved swimmer reached the bank and slowed considerably. Maximum tilt in
the four outward cases ranged from 53° to 125°. These runs establish the need
for turning practice. They do not establish continuous whole-pond swimming.

At the first control that recorded impact damping, the head was about 8 cm
below the surface. Along the ray from the pond center through the head, the
waterline was still 11–17 cm farther out. A top view can therefore show a
wide strip of water beyond a body already touching the rising floor. The
saved Swim motor receives a persistent heading, speed and depth reference.
Its observations contain no distance to the bank and it has no learned
navigation chooser selecting a new heading in this view.

The original failed runs are in `build/pw26/run-v1` and `build/pw27/run-v2`.
The CPU result and independent audit are in `build/pw28/run-v3/result.json`
and `build/pw28/independent-cpu-audit.json`. The CPU result SHA-256 is
`d679e5ced4ac75e96a708f3c54878d5e17c62c0723d37d735c6966054e8c5aa4`.

## Runtime 3 Godot verification

Godot 4.4.1 Project Manager launched the registered `live` project. Its child
window was responding as **Tadpole Pond (DEBUG)** and loaded the qualified
shoreline DLL. The app uses the original saved motor. Open **Details** for
shoreline status. The Project Manager entry retains its registered name,
**CudaRL Frozen Live Demo**, for save compatibility.

The launch receipt is in `build/pw28/godot-install/manager-run.json` and the
UI inspection receipt is in `build/pw28/ui/verified-ui-receipt.json`.

### Ordinary Swim reproduction

One 60-second run used Godot's normal Swim control, the installed native
extension and the original saved motor. All 1,200 controls and 12,000 physics
substeps completed. No automatic reset, pause or termination occurred. The
8 mm/s request persisted.

| Time | Measured behavior |
| --- | --- |
| 47.40 s | First confirmed contact. The whole-body clearance interval was −0.118 to −0.108 mm. The head center was 68% of the way out to the waterline along its radial direction. |
| 50.50 s | Last recorded increment of contact work or damping. |
| 60.00 s | Certified whole-body clearance 5.968–5.978 mm, horizontal COM speed 0.568 mm/s, tilt 117.3°. The swimmer remained bent and slow after losing contact. |

Native bed heights and Godot's drawn triangle surface differed by at most
0.137 mm along the sampled centers in this run. That comparison samples the
trajectory and does not bound interpolation error over every mesh triangle.
The evidence indicates a failed recovery after a real encounter with the
submerged slope. A screenshot alone cannot identify the contact state.
See `build/pw29/stall/run-01/result.json` and its per-control CSV.

## Runtime 3 CUDA qualification

The unoptimized diagnostic completed 2 of the required 128 controls per case
before its time limit. The optimized Release checker passed its static algebra
check. It then completed 15 controls in each of eight continued worlds per
backend. All 1,200 recorded substeps per backend agreed within tolerance.
The 180-second deadline stopped new work. Final wall time was 192.65 seconds
including the last kernel. The dynamic comparison remains incomplete. The
trainer rejected the stopped receipt and performed zero PPO updates.

The 12-second cycle is the elapsed time for a validation tick with reference
physics, CUDA physics, readback, logging and comparison. A 512-world training
rollout has different work and timing. The original optimized receipt is in
`build/pw29/gpu/result.json`. Its source, executable, parent checkpoint and
CPU proof are bound in `build/pw29/sealed.json`.

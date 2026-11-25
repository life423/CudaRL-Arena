# Matched alternating handoffs

Use archived `moving_a_left` and `moving_a_right`, respectively rows
0 and 1 of the whole-beat evaluation bundle. They use one matched
saved pose and opposite starting requests. The two continuous test
sequences are +15° → −15° → +15° and −15° → +15° → −15°.
Keep the fitted actor, update-eight offset, physical equations,
actuator limits, 80 ms action cadence and 1.25 ms integration fixed.
This test permits zero optimizer updates.

Run each original 600-control turn once. During the onward swim,
measure horizontal whole-body COM travel across two complete root yaw
beats using three positive-going root-rate reversals. Fork at the first
80 ms decision with at least two seconds of onward swim and three
reversals. Issue it by four seconds onward. Copy native body, held targets, offset, COM history,
control time and beat trackers before editing either request.

At the first fork, the control extends the current Swim request.
The test requests a 15° turn opposite the first, from the measured course, followed
by onward swimming on the new heading. Each branch advances 600
comparison controls, 100 onward-scoring controls, and 100 additional
controls to close cycles near the window boundary.

Capture the next fork from that test while it swims onward, under
the same two-to-four-second cue rule. Compare continuing on the
current heading with a 15° request opposite the second turn, followed
by onward swimming. Each branch runs another 600 + 100 + 100 controls.
The second-turn branch also continues under its current heading for
complete-beat scoring. Its closing samples do not feed the third cue.
Reuse expired request slots for the third cue. Preserve the active
Swim and previous Turn so offset activation is continuous. Each
comparison uses an exact native state copy at its fork.

Apply the established motion gates in the two final scored windows.
Require projected and rolling horizontal COM speed of at least
6 mm/s, course error at most 5°, signed movement in the requested
direction, tilt at most 10°, depth error at most 1 mm and positive
certified clearance. Compare complete-beat frequency and root and
passive swing with the matched control at each fork, using the
existing 80% floor. Use the archived zero-offset references from
the tail-beat diagnostic for the first turns. Preserve the fixed-window reversal counts as
separate measurements. Record applied yaw offsets, reversal times,
course scores and envelope metrics.

For each first opposite-turn trace, rotate and translate the recorded
substep geometry to the native east bank. Set the incoming direction
7.5° outward from the local tangent, with the 15° cue pointing inward.
Bracket the shallowest certified setback to 1 mm with the native
head and tail envelope at every sample. Require inward COM travel
over the first two seconds onward. Record the limiting pose and
clearance. This is a geometry replay of clear-water motion.

One CPU process, with a 3.5-second internal deadline, a four-second
external cap and at most 128,000 substeps. Preserve all partial
results if a physical, numerical or preparation failure occurs.

# Matched consecutive-turn handoff

Run one deterministic local evaluation using `moving_a_right`, zero-based
row 1 of the saved whole-beat turn fixtures. Use the archived fitted Gait
actor and the whole-beat pilot's update-eight offset. Preserve the water,
body, servo, 80 ms policy cadence and 1.25 ms integration. This allocation
has zero optimizer updates.

Replay the original 600-control right turn once. During onward swimming,
sample physical root-yaw rate at every substep. Confirm three positive-going
rate reversals with the existing 0.02 rad/s dead band. Interpolate COM at
crossings and use horizontal displacement between the first and third to
measure the approach course over two complete beats. Fork at the next
80 ms decision, expected at control 760. Stop if the cue is unavailable
by control 800.

Copy the exact body, actuator targets, held offset, request state, COM ring,
clock and beat trackers into both branches. The control extends the same
Swim leg and heading, preserving learned offset activation. The test requests
15° left of the measured approach course, then swims onward at that heading.
Each branch runs 600 controls for the comparison, 100 for onward scoring and
100 more to close cycles around the window boundary.

Evaluate each branch against its own heading in the final 100 controls of
the comparison and the next 100 controls. Require projected and rolling
COM speed of at least 6 mm/s and course error no more than 5°. Check the
test's signed course change from the measured approach. Check 10° tilt,
1 mm depth error and positive certified clearance throughout. Record
complete root periods, longest reversal gap, and root and passive swing.
Use at least 80% of the matched control frequency and each swing in
the scored windows. Keep fixed-window reversal counts in the evidence.

Save the applied offset and target at each decision, the continuous
reversal times, joint swing, physical checks and COM metrics for both
branches. Measure the offset's sign change and its average across complete
beats. Limit one process to 3.5 seconds internally and four seconds
externally. Count at most 38,400 attempted physics substeps including the
shared prefix. Preserve a partial or failed result if a stage stops.

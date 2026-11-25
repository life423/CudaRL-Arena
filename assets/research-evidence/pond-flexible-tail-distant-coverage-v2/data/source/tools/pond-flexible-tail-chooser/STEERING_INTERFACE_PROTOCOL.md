# Persistent-heading interface comparison

This diagnostic compares the existing measured-course heading requests with
opt-in persistent-heading requests. Both use the frozen fitted motor and
update-eight offset and the unchanged habitat, equations and limits.

Run every archived moving turn start from the eight-row held-out bundle in
both directions under both request modes, for 32 continuous trajectories.
At each valid 4–6 second whole-beat cue, issue a 15-degree diagnostic turn
for the first six decisions. Then retain the heading. Continue each case
for up to 48 simulated seconds under native clearance, depth and tilt
guards. These fixed diagnostic commands establish control authority and
do not train or deploy navigation behavior. The first-cue physical state
must match between modes for every paired start.

Measure actual horizontal COM course and speed, time to turn 60/90
degrees, effective radius from travelled arc, final target error, tilt,
depth error and every native clearance stop. For sustained learning,
prefer the interface that produces more accepted 60-degree moving turns
in both directions and a smaller measured radius while preserving the
6 mm/s speed, 10-degree tilt and 1 mm depth requirements. Report every
failed trajectory. A geometric failure cannot qualify autonomous movement.

The finite diagnostic has a 1,228,800 attempted-substep allocation, no
optimizer updates, a 55-second internal deadline and a 60-second external
process cap. Preserve the original measured interface as a research
compatibility mode.

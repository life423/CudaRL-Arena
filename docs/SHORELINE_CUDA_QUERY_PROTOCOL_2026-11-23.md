# CUDA shoreline query correction

The default optimized checker completed 15 of 128 controls per case before
its deadline. Its evidence remains in `build/pw29/gpu`. This follow-up
addresses the user's request to finish GPU validation, measure the actual
collector and preserve whole-body contact on curved banks.

## Scope and costs

Use fixed saved states to compare GPU execution strategies. First preserve
the scalar search's sampled contact point, shape, normal, upper/lower
clearance, scope failures and visit accounting. Additional speculative
terrain evaluations belong in separate diagnostic counters. Changing only
kernel scheduling must not alter the physical domain or contact equations.
Report cold and warm GPU event times separately from end-to-end time.

The ordinary production collector timing probe may replay one control in
512 flat-bed worlds and one in 512 bank worlds. Maximum 1,024 world-controls,
10,240 physical substeps, zero optimizer calls. Stop new launches at 120 s
and record the actual finish of an in-flight kernel. Bind native kernels,
source, fixtures and commands. Record host and device timings with scope,
including synchronization. Real-policy inference with a fixed replayed
command is a performance measurement and yields no on-policy learning data.

## Corrected dynamic check

Compare the corrected native CUDA execution against the original CPU scalar
reference on exactly the same four straight and four bank trajectories.
Carry independent state for 128 controls per case on each backend and compare
all ten 5 ms substeps per control. No intermediate CPU pose or certificate
may substitute for CUDA's independently computed state. A successful
previous CUDA POST certificate can serve as the next PRE only when there was
no change to the body or terrain between the two. Recheck entry to each
control.

Use one attempt for this corrected implementation after comparing frozen
query outputs. Bind executable, sources, compiler flags, saved motor and CPU
qualification before execution. Keep the original limits of 180 seconds,
2,048 total world-controls and 20,480 substeps. The checker has a 512 MiB
evidence cap. Retain every attempted case and the first failing trace.
The optimized scalar run and this execution have separate ledgers.

The acceptance criteria are unchanged: positions in model units, quaternion
components, joint angles, angular rates and normalized motor values within
2e−5, linear velocity within 2e−7 m/s and energy within
1e−12 J plus 1e−4 relative. Ordinary swimming/resting retention and all CPU
bank cases already passed. An uncompleted trajectory comparison supplies no
PPO admission. A completed passing check must match the implementation
actually used for candidate collection. The original motor remains in Godot.

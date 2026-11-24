# Live steering with the preserved swimmer

Godot's Swim mode now offers **Turn left 5°** and **Turn right 5°**. These change
the requested heading. Original motor 336 chooses every joint action; native
physics produces the movement. The request preserves the body, velocity, time,
held targets, servo/filter state and previous torques. It does not restart or
rotate the body. Pause remains paused. Default Rest and straight Swim retain
their original observation path.

After an explicit turn request the motor receives the same depth-preserving,
heading-only input mapping used in the turn assessments. The original saved
contract is schema 4; this heading overlay is the schema 5 Turn mapping. No new
actor, chooser, optimizer, candidate weights or learned bank avoidance is added.
Repeated turns and autonomous navigation remain unqualified.

## Measured checks

Four matched cases requested ±5° after 40 controls (2 seconds), from starts 0
and 3, then continued through 400 controls (20 seconds). Their original-lane
warmup is preserved; this differs from historical all-heading-only warmups.
All four passed the existing 1 mm depth, 10° tilt, 6 mm/s COM speed and 5° course
criteria. Final course errors ranged from 0.62° to 3.60°; speed was 8.00–8.05 mm/s.
The default 64-second Rest and 8-second Swim were also compared.

The independent native reference executed 3,040 controls; the actual installed
Godot DLL executed the same 3,040 controls. Total: **6,080 controls and 60,800
substeps**, with no optimizer or CUDA calls. All body, filter, servo, command,
time and clearance values matched within 7.1e−15, against the unchanged CPU
tolerance of 2e−6. Each heading request itself preserved physical state exactly.
The pure adapter comparison matched all 492 scalar observations exactly.

The existing runtime 4 physical sources are byte-identical to the qualified
shoreline sources. The new bridge DLL has its own fingerprint and narrow
comparison; the original qualified DLL and 36-case CPU / 8-case CUDA receipts
remain preserved. This check establishes these requested turns, not a new skill
learned by the frozen actor or reliability for every sequence.

## Run and develop

`./tools/view-live.ps1` uses the shipped verified Windows DLL, checks its selected
descriptor, actor and evidence digests, and opens Godot paused. No build or
simulation study runs as part of that ordinary command.

Developer builds are explicit: `./tools/build-native.ps1 -Target frozen_swimmer`
or the `frozen_swimmer` CMake target. They stage output under `build/`; even an old
cached output inside `live/` is redirected. The target generates
`frozen_swimmer-steering-v1.dll`. A rebuilt file is not automatically given a new
qualification hash. Preserve the old descriptor/binding until the new DLL,
source-equivalence checks and actual-DLL comparisons are ready together.

`tools/pond-steering-check` builds the native reference and its zero-integration
adapter check. `tests/godot/pond_live_steering_check.gd` compares the installed
library against the reference and enforces its finite control budget. Full local
source seals, reference trajectories, installation and UI evidence are preserved
under `build/steer35/`; compact results and case measurements are published here.

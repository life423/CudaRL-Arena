# Model 6 slow-approach parking diagnostic

With a fixed pad 2 mm in front, the same model was tested with starting
forward speeds of 0.5 and 1.0 mm/s. Initial joint angles, stock 1 Hz
target filter, torque limits and physics were kept. The fixed-target
comparison sent the joint angles from the initial pose through the stock
actuator path. It did not use a learned controller and is not a change to
the standing policy.

| Condition | 0.5 mm/s | 1.0 mm/s |
| --- | --- | --- |
| Fixed initial joint targets | 2 of 2 settled | 0 of 2 settled |
| Frozen checkpoint 336 | 0 of 2 settled | 0 of 2 settled |

At 0.5 mm/s the fixed-target cases first entered at 2.30 s, at about
0.382 mm/s, and finished about 0.404 mm from the pad at 0.313 mm/s.
At 1.0 mm/s the comparator finished outside the pad at about 1.167 mm.
The frozen checkpoint began large strokes on the first control, entered
all four at 2.88 to 4.03 mm/s, and overshot. This establishes feasibility
only for the tested slow starts. It does not establish learned parking
or feasibility at arbitrary speeds.

The one locked run consumed 640 controls and 6400 native substeps; there
were no optimizer steps or retries. The original checkpoint and protocol
hashes were verified before and after. Each trace has 81 native samples,
including initial state. The last and peak torque columns are normalized
to the stock joint torque limit; the raw action columns are empty on the
initial sample because no command has yet executed.

The original native diagnostic source and build remain in the local
ignored build directory. This commit archives the exact evidence and
protocol, not unrelated in-progress motor code, executables, or
checkpoints. To verify the archived verdicts without another physical run,
run `python verify_evidence.py` in this folder.

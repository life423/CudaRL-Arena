# Seven-option program: measured status

Read-only aggregation; zero added physics or optimizer updates. Motion / main-stroke / legacy are separate counts, each out of 8 cases per angle.

| Stage | Updates; selected | 15° motion/main/legacy | 30° | 60° | 90° | Swimming |
|---|---|---|---|---|---|---|
|stage-01-rhythm-baseline|128;128|8/8/7|8/8/5|0/0/0|0/0/0|12/12|
|stage-02-joint-baseline|128;56|8/8/1|8/8/5|0/0/0|0/0/0|12/12|
|stage-03-nonlinear-baseline|128;128|8/8/8|8/8/8|8/0/4|0/0/0|12/12|
|stage-04-speed-baseline|115;72|8/8/0|8/8/0|8/0/0|0/0/0|12/12|
|stage-05-rate-baseline|128;0|8/8/0|8/8/0|8/0/0|0/0/0|12/12|
|stage-07-rate-half-yaw|128;120|8/8/0|8/8/0|4/0/0|0/0/0|12/12|

The selected nonlinear motor completed all eight 60° movement cases while the strict matched passive-frequency criterion still failed. Its principal passive rhythm after alignment was about 1 Hz versus the still-underturning reference around 1.7 Hz; this score failure was preserved. This is actual moving-turn progress, not full rhythm qualification. No selected motor completed all 90° movement cases.

| Stage | Speed | Rate | Alternating requests | Bank requests |
|---|---|---|---|---|
|stage-01-rhythm-baseline|2/6|4/12|2/6|1/2|
|stage-02-joint-baseline|0/6|4/12|3/6|1/2|
|stage-03-nonlinear-baseline|0/6|6/12|2/6|1/2|
|stage-04-speed-baseline|4/6|4/12|4/6|1/2|
|stage-05-rate-baseline|4/6|4/12|4/6|1/2|
|stage-07-rate-half-yaw|2/6|4/12|3/6|1/2|

Alternating requests are two uninterrupted 30°/60°/90° sequences. Later guard stops remain in the denominator. Bank probes are motor request tests and do not establish autonomous navigation. Main-stroke comparisons for supplemental requests are archived-reference diagnostics.

| Chooser update | Pond diagnostic | Endurance | Union coverage | Guard stops |
|---|---|---|---|---|
|0|6/12|0/0|53.3%|6|
|8|6/12|0/0|50.4%|6|
|16|8/12|0/0|58.3%|4|
|24|5/12|0/0|61.7%|7|
|32|4/12|0/0|61.2%|8|
|40|5/12|0/0|60.0%|7|
|48|5/12|0/0|60.0%|7|
|56|5/12|0/0|58.6%|7|
|63|4/12|0/0|61.7%|8|

The selected soft-body motor retained 12/12 swimming cases, but the frozen chooser passed only 4/12 pond cases with 8 guard stops, versus 8/12 and four stops on the baseline body. Endurance was not attempted. The body variant is not a navigation improvement under this comparison.

Strongest selected baseline-body 60° movement: 8/8 cases, mean course error 1.08°, projected speed 13.82 mm/s, initial response 1.92 s and effective complete-beat radius 165.3 mm. The soft-body selection passed 8/8 main-stroke 30° cases at 13.31 mm/s, 1.34° error, 1.80 s initial response and 259.0 mm effective radius; it passed only 4/8 movement cases at 60°. Initial response is not settled completion time.

Speed matching remained partial: the selected baseline motor delivered about 8.40 / 9.23 / 10.64 mm/s for 6 / 8 / 12 mm/s requests, passing the 8 and 12 requests on both starts. The soft body delivered 12.75 / 13.06 / 13.52 mm/s, passing only the 12 request. Neither reliably matched the slowest request.

Holding was not physically retested. Endurance remains untested whenever all 12 pond diagnostics have not qualified. Coverage is the union of separate starts, not one animal exploring the full map. Original working Godot assets remain protected until the deployment gates pass.

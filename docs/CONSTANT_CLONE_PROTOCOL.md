# Constant-speed clone1: frozen narrow lesson

Explicit user approval01:42UTC. Reuse only960samples from accepted profiles0..5
of demonstration-v1. Whole source trial remains failed; switch profiles6/7 are
excluded. Source gate SHA B24675B4DE9742D81A3508D038B5809EB5F62EF44139639E5B32AD64D976BC24;
original data SHA79EF33B32865650A7E5B4B37A900CE7574DD84BEFCF033C5BF0414AF61490BC3.
Physics/teacher/observation source identity and interleaved row provenance checked;
accepted slice SHA5638C685E3BB91ACFD6DE94916772FD155F92DA7435088B8508AFF23AD73BFE6.
Native loader validates schema,960rows, goal order and every recorded latent label.

Same model6,42inputs/twooutputs. Warm-start the selected actor; fresh Adam lr.001,
eps1e-5, grad cap.5; seed0x53574D32.600supervised steps, minibatch256. No PPO,
new teacher stepping, feedback blend or fallback during frozen evaluation.

New run build/speed-motor/2026-10-31-constant-clone-v1/.32readiness controls plus
1600paired frozen evaluation controls:4/5/6/7/8mm/s, cases0/3,4s each, score2..4s.
5/7 withheld from all fitting examples. Hardcap1632controls/16320physical substeps,
600Adam steps,20native seconds inclusive; wrapper25s own-process watchdog.
No retries/extensions. Prior unused budgets remain closed and unused.

Acceptance:10/10constant cases; mean within10%, RMSE<=31.25%request, lateral<=2mm,
depth<=1mm, path<=5deg, tilt<=10deg, work/metre<=1mJ, native finite/energy/torque/
10mmclearance throughout. Better normalized RMSE overall and no worse8mm/s RMSE
or work/metre than the preserved actor. Acceptance means only constant requests
from rest in this4s lesson. Switches/slowdown remain UNPASSED, are not rerun or
relabelled successful. No live deployment or baseline overwrite in this trial.


## Execution outcome

Closed2026-10-31T01:49:58.385UTC:600steps,1632controls/16320substeps,.793035s
native,.521774s setup, no retries/faults.10/10constant cases pass including5/7held
out. Requested4/5/6/7/8 produced4.350/5.318/6.259/7.115/7.843mm/s. Selected
update-600.bin for this lesson, SHA256
0A19AF86FBBDCD189A330547A281666B4D8C781922A7743C870E7A58D9FC6411.
Live actor unchanged. Switches unrun and prior slowdown failure unchanged.

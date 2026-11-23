# Food-free swimming: design audit and next decision

The creature cannot yet swim indefinitely. The first study is closed at 0/16
qualifying starts. **Do not execute the prepared V2 proposal.** Its numerical
preparation passed, but the code audit found unresolved control and objective
problems. A correct implementation of an objective does not make that objective
appropriate. No V2 physical controls or optimizer updates have run.

## Highest-impact findings

1. **Early failure can improve the proposed score.** V2 accumulates negative as
   well as positive utility while alive, subtracts four once on failure, and
   contributes zero for unexecuted time. With an equal prefix and zero effort,
   failing now gives -4 while continuing four seconds at rate -2 and failing
   gives -12. These are algebraic objective examples, not measured creatures.
   A larger horizon and a stall terminal do not remove this incentive.
2. **The new head does not have unrestricted joint-target authority.** It adds
   at most +/-0.75 before the target's tanh. In the saved zero-head autonomous
   baseline, the correction cannot reverse yaw-target sign on 50.335%/51.818%
   of controls. This does not prove reversal is necessary or turning impossible;
   it invalidates treating four residual outputs as a demonstrated solution.
3. **The free-swim observer changes the trained motor's task.** Every control
   replaces desired heading with current head azimuth and resets the horizontal
   lane origin. Persistent yaw/lane error disappears. Reusing the exact weights
   therefore does not preserve the old closed-loop behavior in this mode. It
   may need to learn local-forward stability and turning together.
4. **The lesson asks for pond survival before demonstrating a clean turn.**
   The saved search shows some steering, but no clean repeated 90-degree turns
   with the required speed, depth and posture. Episode-level search over 80
   coefficients provides one score per trajectory; extending that trajectory
   alone does not localize which turn action helped. Passing exact bypassed old
   modes establishes their preservation, not free-swim competence.

The [learning audit](CONTINUOUS_POND_LEARNING_AUDIT_2026-11-23.md) also quantifies
the search change: V2 proposes 49,152 independent controller samples across 80
coefficients, versus V1's 262,144 across sixteen. That is 26.7 times fewer
episode samples per coefficient. It is valid parameter-space PPO, but neither
longer trajectories nor more simulated ticks supply extra per-action credit.

The detailed [control audit](CONTINUOUS_POND_CONTROL_AUDIT_2026-11-23.md) records
target intervals, observation semantics and representational limits. It finds
no obvious joint-axis/sign wiring defect. Cancellation by the old motor,
physical impossibility and guaranteed success of an alternative are not proven.

## Concrete change in approach

Teach and measure a **persistent requested turn** through the intended motor
interface before training whole-pond choices. Start with small left/right
heading changes, then larger turns, and retain propulsion, depth and posture.
An explicit requested heading is a task, not a scripted body rotation. It must
stay fixed while the creature catches up; it must not chase the head each tick.

First inspect the existing motor's requested-heading response without advancing
physics. Any later physical probe must determine whether the inherited motor
already tracks those requests. If it does, pond choice learning can use that
interface. If it does not, train the same reusable motor on turn requests from
the inherited weights. Reuse the native per-control PPO actor/critic machinery
instead of automatically adding another coefficient wrapper. Critics remain
training auxiliaries. Do not assume either outcome, add a brain per ability,
or solve navigation by supplying an untrained route. Native CUDA PPO, unchanged
physical limits, no food, preserved saves and the final movement gates remain.

For the next objective, use a bounded nonnegative forward-motion utility with
quality/effort weighting, and account for the remaining failed episode time
with an absorbing negative value. The offline specification in
`tools/audit_continuous_pond_design.py` has utility in [0,1], a -1 value for
each unexecuted failed second, and the separate four-unit failure penalty.
For the same prefix, any valid continuation then improves over failing now.
This is a reward-design property, not proof of learnability, sustained turns,
or final qualification. It is **not wired into an active trainer**.

Do not change the frozen V1/V2 sources or silently resume their budgets. Prepare
the narrow turn lesson and a new finite protocol, then obtain the required
fresh approval before physical execution. No V2 authorization marker exists;
`design-hold.json` explicitly records this decision.

## Verification and presentation

The read-only design audit verified saved native envelopes and all 4,180 active
zero-head baseline rows for target authority. Its proposed objective passed 576
same-prefix return inequalities and 101 bounded/monotonic speed samples, with
zero physical or optimizer steps. The control audit independently reproduces
the authority result and binds source/data hashes. These checks do not execute
or validate a new moving controller.

The existing Godot app was restarted from the correct project and its actual
Details panel inspected on screen: it shows 64 updates, 0/16 starts, the stalled
candidate's rejection, unrun thirty-minute validation and retained old skills.
Godot Manager still targets this worktree's `live` project. The accepted camera,
zoom, contrast and working saved actor remain. The failed candidate is absent
from the player modes. Current UI checks and limits are in the
[presentation review](CONTINUOUS_POND_UI_REVIEW_2026-11-23.md).

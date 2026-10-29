# Passing predator: first shelter skill

User choice: a passing predator shadow, October 29, 2026. This supersedes the
suggested surge as the first shelter event. No limbs or extra habitat objects
are part of this experiment. Lily pads, a visible overhang, roots and an open
log remain habitat options for discussion.

Preserve the full working boulder save. Import its actually applied swimming
actor and shared food scorer into a separate pilot. Food stays present and
physically edible throughout. A threat never scripts a target, body pose,
swimming command, teleport or pickup suppression in the live creature.

The first diagnostic freezes both actors and directly requests geometrically
valid cover goals in separate test worlds. This tests whether the current
motor can operate the requested skill; it does not test a learned safety
decision. Evaluate 24 brief cases, with calm/0.5/0.75 cm/s flow and varied
starting headings. Report approach and a continuous five-second protected
dwell separately. A reached head target is not sufficient evidence of hiding.

Threat model: a bounded oblique pass, four seconds of warning, twenty seconds
of exposure risk, then a calm interval. Native event time supplies bird-eye
position and a separate fixed-sunlight shadow projection. Predator visibility
uses the same sphere and torus solids as contact. The water surface and
decorative vegetation do not provide opaque cover. Sample 216 articulated
body-surface points, including the distal tail; describe this as sampled
visibility, not exact continuous silhouette detection. Count a protected tick
only with at most 5% visible samples and at least 0.08 model units of whole-body
solid/bed/surface clearance. Already covered starts are not successful escapes.

If the frozen motor cannot hold cover, first practice that physical skill with
native continuous PPO, zero-expanding the inherited actor for goal type and
desired heading. Mix old feeding/navigation practice and check retention.
Then the existing shared goal scorer can learn when food or cover is worth
pursuing. This extends the two decision controllers; it does not add a third
decision network. Goal assignment in isolated motor exercises is a task
instruction, not learned high-level prioritization.

Report actual parameter changes, optimizer continuation, held-out exposure,
clearance, sustained dwell and return to feeding. Keep development scenarios,
promotion checks and final evaluation seeds separate. Never replace the live
working actor from reward alone. A trial that cannot hold protection is not a
completed shelter behavior, even if its mean exposure falls. Preserve such
failures and the original actor rather than weakening the gate.

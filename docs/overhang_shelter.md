# Overhanging bank rock

The user chose this shelter object before lily pads on October 29. This is a
geometry step, separate from the unresolved learned holding/control diagnosis.
Do not open an unqualified predator demo or call a static cover pose learned
hiding. The user wants that demo opened after its behavioral test passes.

## Shape and physical meaning

Layout 3 opts into the existing cove and boulder plus an open-sided stone ledge
on the far bank. A broad roof joins a bed-rooted rear support; the opening faces
the existing camera. Both pieces are rounded oriented boxes. Native signed
distance and surface normals drive force-based contact, swimmer sensing, food
clearance and predator sight obstruction. The renderer uses those same native
descriptors; material color/normal detail must not move the solid silhouette.

One model unit is four centimetres. Shelter is evaluated against all 216 sampled
head/tail/fin surface points with positive bed/solid/surface clearance. A low-flow
zone or a dark pixel does not establish cover. The fluid remains the project's
prescribed boundary-aware drag model, not CFD.

## Isolation

- Layouts 0–2, their geometry and existing training/save paths stay unchanged.
- No Habitat/BodyState binary layout is expanded to store this fixed object.
- Geometry getters do not start training, read personal saves or move a creature.
- Existing production/research learners remain on their original habitats. This
  step does not silently migrate a running learner or deploy a failed actor.
- The new geometry fixture is explicitly static QA with no neural decisions or
  learning. It does not replace the normal pond menu or the readiness gate.

## Verification

Pending implementation: analytic distance/normal checks, force/contact tests,
whole-body cover across the moving predator's sight line, camera visibility,
valid food/sensing, direct CPU/CUDA agreement, and four-size rendered Godot QA.
Record exact results here before declaring this object step complete.

## Next AI work

Keep the prior failure diagnosis: demonstrate holding/braking from controlled
initial velocities, audit actions through Gaussian probabilities/gradients and
optimizer updates, and compare safe versus failing behavior's return. Only then
choose one controlled training change. A future learned demonstration must
approach shelter, remain covered, resume feeding and retain existing swimming.

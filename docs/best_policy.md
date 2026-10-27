# Best evaluated pond policy — October 26, 2026

The user requested automatic saving of the best policy, a current-versus-best food/min comparison and a viewable result. The pond preserves the strongest matching comparison result across sessions. It now also [resumes full training state](training_resume.md) from a separate file. Start new run archives the full prior save and resets current learning/curve while retaining the strongest playback actor.

## What the display means

- **Current**, amber solid line: the latest evaluated learner. Evaluation runs every 25 updates over the existing 64 fixed one-minute ponds. Between evaluations this value remains unchanged even as optimizer updates continue.
- **Best**, mint dashed step line: the highest comparison rate seen, including the best matching record loaded when this run started. It never decreases. A loaded record may have a higher update number than the fresh learner.
- **Best saved · update N** confirms a successful disk write or a validated load. Save failures are displayed separately with details in the graph tooltip; they do not become fatal simulation errors.
- **Watch best** applies that actor to the existing visible animal. Its body, food and clock stay in place. Background learning and both metrics continue. The button becomes **Watch current** to return to the latest actor. Best view follows a new best when one is published.

This is a comparison-suite record, not proof of peak ability or optimal routes. Selecting many candidates on the same fixed suite can produce selection bias; independent layouts and long-run evaluation still matter. The visible animal's rolling food/min remains a separate measurement from the population comparison.

## Capturing the right weights

`BasicSwimmerTrainer` captures an `EvaluatedPolicy` immediately after a canonical evaluation, before the next optimizer update. The record pairs its exact actor parameters with update, total food, trial count, horizon and seeds. Strict improvements replace the record; ties keep the earlier actor. Evaluations with a separate holdout seed cannot replace the canonical best. No gradients, optimizer state or environment state are modified by selection.

`TrainingSession` publishes that record under its existing mutex alongside the latest actor, with a separate revision. Optional C++17 detection leaves CartPole/Racing trainers without best records unchanged. A later UI poll can never attach a score to newer, unevaluated weights. `stop()` joins the worker while leaving publications readable, allowing the binding to collect and save an in-flight winner before releasing the session.

## Persistence

The default file is:

`user://pond-policies/regional-v1/best.json`

This is in Godot's per-user application data, outside the repository. The graph tooltip shows the resolved absolute path. Files survive closing the application and starting over. An exported test scene can set `best_policy_path` to an isolated location or an empty string for in-memory-only operation; tests never overwrite the user's real record.

The Godot binding retains the best matching record loaded from disk and compares each worker winner against it. It writes only a new record, or retries an unsaved record when another evaluation arrives. JSON is written with full float precision to a sibling temporary file, flushed and closed, then renamed over the destination. The prior destination remains if replacement fails. All disk work happens at publication/save boundaries, not on every rendered frame.

Loading validates the format/protocol, 13→32→32→2 architecture, inertial body ID, canonical evaluation seed, 64 trials, 1,200-step horizon, deterministic mean-action mode, integer score/update ranges, weight count and finite numeric weights. The score is derived from counts rather than trusting the displayed rate. Incompatible data is preserved and reported instead of silently overwritten; learning can continue. Change the protocol identifier and storage namespace when changing evaluation physics, placement rules or scoring duration. Concurrent writers to the same application-data path have not been validated.

This JSON file is a **deterministic playback actor**. The separate `.training` binary restores critic, Adam state, exploration, background worlds and training history. Watch best leaves the live optimizer untouched. For the explicitly approved one-time upgrade from the previous build, a matching best actor can initialize learning when no full save exists; this is labelled as actor import rather than exact training resume. See [full persistence and validation](training_resume.md).

## Verification

- Six PPO demo tests pass, including a synthetic publication test whose strongest actor precedes the latest weights; publication and shutdown retain the evaluated actor.
- Nine foraging tests and thirteen legacy swimmer tests pass. The real trainer keeps the initialized actor on tied zero scores, and holdout evaluations cannot change the selected canonical best. Existing evaluation-isolation checks still show identical optimizer results.
- The live Godot smoke test passes **187 checks**, covering disk persistence and metadata, a nondecreasing best curve, retention across fresh sessions/Start over, actual saved-actor playback, no teleport on selection, continued learning during best view and keyboard switching.
- The checkpoint failure test passes **14 checks**: incompatible data remains untouched, failed writes are reported, learning continues after save errors and in-memory mode works.
- Menu regression passes **101 checks** and archived one-food playback passes **100 checks**. Tests use separate application-data paths.
- Rendered results were inspected at 1280×720, 768×1024, 390×844 and 844×390. Both curves/numeric values and the 44 px Watch best control fit; the pond stays separate from the chart.

Internal UI review: hierarchy 9 (pond remains distinct and both series have numeric labels), consistency 9 (existing palette/controls), usability 8 (short landscape requires sidebar scrolling), accessibility 8 (keyboard, labels, dashed line and target sizes checked; native assistive/device testing remains), responsiveness 8 (four window sizes; physical devices untested), feedback 9 (saved/failure state and view mode are explicit), performance 8 (winner-only readback/write; frame-time impact unprofiled). Priorities are assistive/device testing, shorter-window usability and measured save-time/frame-time contention. No redesign or additional gameplay is implied.

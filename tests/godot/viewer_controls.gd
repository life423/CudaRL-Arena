extends SceneTree
# Opt-in UI contract check. Disable automatic processing before the viewer enters
# the tree; never call its native step. This cannot become a physical trial.
var view: Node
var checks := 0

func _initialize() -> void:
	view = load("res://main.tscn").instantiate()
	view.process_mode = Node.PROCESS_MODE_DISABLED
	root.add_child.call_deferred(view)
	_check_controls.call_deferred()

func _check(ok: bool, reason: String) -> bool:
	checks += 1
	if not ok:
		if view.swimmer != null: view.swimmer.call("pause")
		print(JSON.stringify({"error":reason,"checks":checks,"physical_substeps":view.total_substeps}))
		quit(3)
	return ok

func _check_controls() -> void:
	if not _check(view.swimmer != null and bool(view.state.get("loaded",false)),"Actor did not load"): return
	if not _check(bool(view.state.paused) and int(view.state.substeps) == 0,"Startup must be paused"): return
	view._start()
	if not _check(not bool(view.state.paused) and view.status.text.begins_with("RUNNING"),"Start disagrees with native pause state"): return
	view._process(0.0)
	view._pause()
	if not _check(bool(view.state.paused) and view.pause_button.disabled and not view.start_button.disabled,"Pause/button state disagrees"): return
	view._reset()
	if not _check(bool(view.state.paused) and int(view.state.controls) == 0,"Reset did not restore paused state"): return
	view._start()
	view._fail("Expected viewer audit error")
	if not _check(bool(view.state.paused) and view.start_button.disabled and view.pause_button.disabled,"UI failure did not pause native controller"): return
	view._reset()
	if not _check(view.status.text.begins_with("PAUSED") and not view.start_button.disabled,"Reset did not recover UI state"): return
	if not _check(int(view.state.substeps) == 0 and view.total_substeps == 0,"Zero-step check advanced physics"): return
	print(JSON.stringify({"viewer_checks_passed":checks,"physical_substeps":0}))
	quit()

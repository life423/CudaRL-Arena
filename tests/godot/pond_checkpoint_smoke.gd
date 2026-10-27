extends SceneTree

var checks := 0
var failures := 0

func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)

func _initialize() -> void:
	_run.call_deferred()

func wait_for_best(lab: Object) -> void:
	var started := Time.get_ticks_msec()
	while not lab.get_stats().best_ready and Time.get_ticks_msec() - started < 10000:
		await create_timer(0.025).timeout
	check(lab.get_stats().best_ready, "Baseline checkpoint was not published")

func _run() -> void:
	var directory := "user://tests/checkpoint-errors-%d-%d" % [OS.get_process_id(), Time.get_ticks_usec()]
	DirAccess.make_dir_recursive_absolute(directory)
	var invalid := directory + "/incompatible.json"
	var original := '{"version":999,"protocol":"other-task"}'
	var file := FileAccess.open(invalid, FileAccess.WRITE)
	file.store_string(original)
	file.close()
	var lab := TadpoleLab.new()
	check(lab.start_training(128, 1, 0, invalid), "Incompatible saved data prevented fresh training")
	await wait_for_best(lab)
	var stats := lab.get_stats()
	check(not stats.best_saved and not stats.checkpoint_error.is_empty(), "Incompatible checkpoint did not report a save error")
	check(FileAccess.get_file_as_string(invalid) == original, "Incompatible saved file was overwritten")
	lab.set_paused(false)
	var started := Time.get_ticks_msec()
	while lab.get_stats().update < 2 and Time.get_ticks_msec() - started < 10000:
		await create_timer(0.025).timeout
	check(lab.get_stats().update >= 2 and lab.get_error().is_empty(), "Checkpoint error stopped learning")
	lab.stop()
	check(FileAccess.get_file_as_string(invalid) == original, "Shutdown overwrote incompatible saved data")
	var blocker := directory + "/blocked"
	file = FileAccess.open(blocker, FileAccess.WRITE)
	file.store_string("keep this file")
	file.close()
	check(lab.start_training(128, 1, 0, blocker + "/best.json"), "Write failure prevented fresh training")
	await wait_for_best(lab)
	stats = lab.get_stats()
	check(not stats.best_saved and not stats.checkpoint_error.is_empty(), "Failed write was presented as saved")
	check(FileAccess.get_file_as_string(blocker) == "keep this file", "Failed save damaged the existing file")
	check(lab.get_error().is_empty(), "Save failure became a fatal simulation error")
	lab.stop()
	check(lab.start_training(128, 1, 0, ""), "In-memory checkpoint mode failed")
	await wait_for_best(lab)
	check(lab.get_stats().best_ready and not lab.get_stats().best_saved and lab.get_stats().checkpoint_error.is_empty(), "Disabled disk saving reported the wrong state")
	lab.stop()
	print("POND_CHECKPOINT_SMOKE %d checks, %d failures" % [checks, failures])
	quit(0 if failures == 0 else 1)

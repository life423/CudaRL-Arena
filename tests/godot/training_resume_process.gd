extends SceneTree

var lab := TadpoleLab.new()
var phase := ""
var path := ""
var report := ""
var failures := 0
func check(condition: bool, message: String) -> void:
	if not condition:
		failures += 1
		push_error(message)
func _initialize() -> void:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--phase="):
			phase = argument.trim_prefix("--phase=")
		elif argument.begins_with("--checkpoint="):
			path = argument.trim_prefix("--checkpoint=")
		elif argument.begins_with("--report="):
			report = argument.trim_prefix("--report=")
	_run.call_deferred()
func _run() -> void:
	create_timer(45).timeout.connect(func(): quit(1))
	check(not path.is_empty() and not report.is_empty(), "Resume-process paths missing")
	check(lab.start_training(128, 1, 0, path, true, true), "Process trainer failed to start")
	var deadline := Time.get_ticks_msec() + 15000
	while not lab.get_stats().training_ready and lab.get_error().is_empty() and Time.get_ticks_msec() < deadline:
		await create_timer(0.025).timeout
	check(lab.get_error().is_empty(), "Process could not load training state: " + lab.get_error())
	if failures:
		quit(1)
		return
	if phase == "close":
		check(not lab.get_stats().resumed_training, "First process accidentally resumed an old run")
		lab.set_paused(false)
		deadline = Time.get_ticks_msec() + 15000
		while lab.get_stats().update < 35 and Time.get_ticks_msec() < deadline:
			await create_timer(0.025).timeout
		lab.set_paused(true)
		await create_timer(0.25).timeout
		var stats := lab.get_stats()
		check(stats.update >= 35 and stats.training_saved_update == 0, "Close test must contain unsaved optimizer updates")
		var file := FileAccess.open(report, FileAccess.WRITE)
		file.store_string(JSON.stringify({"update": stats.update, "steps": stats.env_steps}))
		file.close()
		print("RESUME_PROCESS_READY_FOR_NORMAL_CLOSE update=%d" % int(stats.update))
		# Quit this real engine process normally; a second process must restore
		# the shutdown save, rather than sharing an in-memory trainer.
		quit(0 if failures == 0 else 1)
	else:
		var expected: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(report))
		var stats := lab.get_stats()
		check(stats.resumed_training and stats.update == expected.update and stats.env_steps == expected.steps, "Second process did not restore the exact closed update")
		check(stats.training_saved_update == stats.update and stats.paused, "Restored process secretly advanced")
		lab.step()
		check(lab.get_stats().policy_update == expected.update, "Visible actor metadata restarted from zero")
		lab.set_paused(false)
		deadline = Time.get_ticks_msec() + 10000
		while lab.get_stats().update <= int(expected.update) and Time.get_ticks_msec() < deadline:
			await create_timer(0.025).timeout
		check(lab.get_stats().update > int(expected.update), "Restored process could not continue learning")
		lab.stop()
		print("TRAINING_RESUME_PROCESS failures=%d restored_update=%d" % [failures, int(expected.update)])
		quit(0 if failures == 0 else 1)
func _finalize() -> void:
	lab.stop()

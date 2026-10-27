extends SceneTree

# Opt-in local view comparison. Uses the real resumed shuffled run, keeps it
# learning, and saves normally on exit. Do not run beside another owner of that
# training file. Reports frame intervals, not GPU utilization or GPU kernel time.
func _initialize() -> void:
	Engine.max_fps = 60
	_run.call_deferred()
func _run() -> void:
	root.size = Vector2i(1280, 720)
	change_scene_to_file("res://pond/shuffle_compare.tscn")
	for i in 8:
		await process_frame
	var scene = current_scene
	var limit := Time.get_ticks_msec() + 15000
	while not scene.learner_pond.lab.get_stats().training_ready and scene.learner_pond.lab.get_error().is_empty() and Time.get_ticks_msec() < limit:
		await create_timer(0.025).timeout
	if not scene.learner_pond.lab.get_error().is_empty() or not scene.learner_pond.lab.get_stats().training_ready:
		push_error("Camera benchmark could not resume the existing learner")
		quit(1)
		return
	scene.learner_pond.warmup_timer.stop()
	scene.learner_pond.lab.set_paused(false)
	await create_timer(2.0).timeout
	var rows: Array[Dictionary] = []
	for index in [1, 0, 0, 1]:
		scene.camera_selector.select(index)
		scene.camera_selector.item_selected.emit(index)
		await create_timer(1.0).timeout
		var start_stats: Dictionary = scene.learner_pond.lab.get_stats()
		var start := Time.get_ticks_usec()
		var last := start
		var intervals: Array[float] = []
		while Time.get_ticks_usec() - start < 6000000:
			await process_frame
			var now := Time.get_ticks_usec()
			intervals.append(float(now - last) / 1000.0)
			last = now
		var end_stats: Dictionary = scene.learner_pond.lab.get_stats()
		var seconds := float(Time.get_ticks_usec() - start) / 1000000.0
		intervals.sort()
		var row := {"view": "overview" if index == 1 else "follow", "seconds": seconds,
			"steps_per_second": float(int(end_stats.env_steps) - int(start_stats.env_steps)) / seconds,
			"frames": intervals.size(), "mean_frame_interval_ms": seconds * 1000.0 / intervals.size(),
			"p95_frame_interval_ms": intervals[mini(intervals.size() - 1, int(ceil(intervals.size() * 0.95)) - 1)],
			"start_update": int(start_stats.update), "end_update": int(end_stats.update)}
		rows.append(row)
		print("POND_CAMERA_BENCH " + JSON.stringify(row))
	var file := FileAccess.open("res://../build/pond-learning-checks/camera-benchmark.json", FileAccess.WRITE)
	file.store_string(JSON.stringify({"viewport": [1280, 720], "frame_cap": 60, "sequence": "overview,follow,follow,overview", "rows": rows}, "  "))
	file.close()
	quit(0)

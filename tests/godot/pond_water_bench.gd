extends SceneTree

# Opt-in local comparison using the existing resumed learner. Close its other
# game window first. Both cameras follow at a 60 FPS cap; only disturbances vary.
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
		push_error("Water benchmark could not resume the saved learner")
		quit(1)
		return
	scene.learner_pond.warmup_timer.stop()
	scene.learner_pond.lab.set_paused(false)
	await create_timer(2.0).timeout
	var rows: Array[Dictionary] = []
	for enabled in [false, true, true, false]:
		for pond in [scene.reference_pond, scene.learner_pond]:
			pond.set_water_disturbances(enabled)
		await create_timer(1.5).timeout
		var first: Dictionary = scene.learner_pond.lab.get_stats()
		var start := Time.get_ticks_usec()
		var previous := start
		var intervals: Array[float] = []
		while Time.get_ticks_usec() - start < 6000000:
			await process_frame
			var now := Time.get_ticks_usec()
			intervals.append(float(now - previous) / 1000.0)
			previous = now
		var last: Dictionary = scene.learner_pond.lab.get_stats()
		var seconds := float(Time.get_ticks_usec() - start) / 1000000.0
		intervals.sort()
		var row := {"disturbances": enabled, "seconds": seconds,
			"steps_per_second": float(int(last.env_steps) - int(first.env_steps)) / seconds,
			"frames": intervals.size(), "mean_frame_interval_ms": seconds * 1000.0 / intervals.size(),
			"p95_frame_interval_ms": intervals[mini(intervals.size() - 1, int(ceil(intervals.size() * 0.95)) - 1)],
			"start_update": int(first.update), "end_update": int(last.update),
			"reference_active": scene.reference_pond.ripples.active_count(float(scene.reference_pond.current.elapsed)),
			"learner_active": scene.learner_pond.ripples.active_count(float(scene.learner_pond.current.elapsed))}
		rows.append(row)
		print("POND_WATER_BENCH " + JSON.stringify(row))
	var file := FileAccess.open("res://../build/pond-learning-checks/water-benchmark.json", FileAccess.WRITE)
	file.store_string(JSON.stringify({"viewport": [1280, 720], "camera": "follow", "frame_cap": 60, "sequence": "off,on,on,off", "rows": rows}, "  "))
	file.close()
	quit(0)

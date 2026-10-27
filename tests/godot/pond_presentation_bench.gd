extends SceneTree
func _initialize() -> void:
	Engine.max_fps = 60
	_run.call_deferred()
func _run() -> void:
	create_timer(70).timeout.connect(func(): quit(1))
	var source := ProjectSettings.globalize_path("user://pond-policies/deep-pond-v1/training.bin")
	var checkpoint := ProjectSettings.globalize_path("user://tests/pond-presentation-bench-%d.bin" % OS.get_process_id())
	DirAccess.make_dir_recursive_absolute(checkpoint.get_base_dir())
	if DirAccess.copy_absolute(source, checkpoint) != OK:
		quit(1)
		return
	var scene = load("res://pond/deep_pond.tscn").instantiate()
	if scene.get_script() == null:
		quit(1)
		return
	scene.checkpoint_path = checkpoint
	root.size = Vector2i(1280, 720)
	root.add_child(scene)
	await create_timer(3).timeout
	var windows: Array = []
	for enabled in [false, true, true, false]:
		for item in [scene.water_surface]:
			item.visible = enabled
		await create_timer(1).timeout
		var first: Dictionary = scene.lab.get_stats()
		var start := Time.get_ticks_usec()
		var previous := start
		var durations: Array[float] = []
		while Time.get_ticks_usec() - start < 4000000:
			await process_frame
			var now := Time.get_ticks_usec()
			durations.append((now - previous) / 1000.0)
			previous = now
		var seconds := (Time.get_ticks_usec() - start) / 1000000.0
		var last: Dictionary = scene.lab.get_stats()
		durations.sort()
		windows.append({"water_layers": enabled, "steps_per_second": (last.env_steps - first.env_steps) / seconds, "mean_frame_ms": seconds * 1000 / durations.size(), "p95_frame_ms": durations[mini(durations.size() - 1, int(durations.size() * 0.95))]})
		print("POND_PRESENTATION_BENCH ", JSON.stringify(windows[-1]))
	var file := FileAccess.open("res://../build/pond-learning-checks/pond-presentation-bench.json", FileAccess.WRITE)
	file.store_string(JSON.stringify({"viewport": "1280x720", "fps_cap": 60, "population": 1024, "mode": "follow", "comparison": "new water surface/section/waterline off/on/on/off; same new camera/terrain/lighting", "windows": windows}, "\t"))
	scene.queue_free()
	await process_frame
	quit()

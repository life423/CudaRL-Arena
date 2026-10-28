extends SceneTree
var checks := 0
var failures := 0
func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)
func settle() -> void:
	for tick in 8:
		await process_frame
func _initialize() -> void:
	Engine.max_fps = 60
	_run.call_deferred()
func _run() -> void:
	create_timer(90).timeout.connect(func(): quit(1))
	var policy := "res://pond/policies/deep_selector.bin"
	var policy_hash := FileAccess.get_sha256(policy)
	var scene = load("res://pond/bank_preview.tscn").instantiate()
	root.add_child(scene)
	current_scene = scene
	scene.set_physics_process(false)
	await settle()
	check(scene.lab.get_error().is_empty(), "Preview did not load: " + scene.lab.get_error())
	check(scene.lab.case_index == 1 and scene.scenario.selected == 1, "Preview should start with the trained selector")
	check(scene.lab.data.cases.size() == 2 and scene.lab.frames().size() == 1201, "Unexpected recording structure")
	var original: Dictionary = scene.lab.data.cases[0].frames[0]
	var trained: Dictionary = scene.lab.data.cases[1].frames[0]
	for field in ["position", "orientation", "foods", "links", "angles"]:
		check(original[field] == trained[field], "Comparison starts differ: " + field)
	check(scene.title_status.text.contains("Recorded") and scene.title_status.text.contains("no live training"), "Preview implies live learning")
	check(not scene.untrained_button.visible and not scene.curve.visible, "Preview shows misleading training controls")
	check(scene.lab.get_stats().background_swimmers == 0, "Preview pretends to run background training")
	scene._seek_time(5)
	check(scene.state.eaten >= 1, "Trained replay does not show the recorded food arrival")
	scene.compare_button.pressed.emit()
	check(scene.lab.case_index == 0 and scene.state.elapsed == 0 and scene.scenario.selected == 0, "Compare does not restart the original from the same state")
	scene.lab.paused = true
	scene._seek_time(60)
	check(scene.state.eaten == 0 and scene.state.speed_cm_s < 0.1, "Original replay does not match the recorded stall")
	scene._select_scenario(1)
	scene.lab.paused = true
	scene._seek_time(60)
	check(scene.state.eaten == 13, "Trained replay does not match the held-out outcome")
	scene.before_button.pressed.emit()
	check(scene.state.elapsed == 0 and not scene.lab.paused, "Restart does not resume from the start")
	scene.replay_pause.pressed.emit()
	var time: float = scene.lab.get_state().elapsed
	scene.lab.step()
	check(scene.lab.get_state().elapsed == time, "Pause did not stop playback")
	scene.replay_pause.pressed.emit()
	scene.lab.step()
	check(scene.lab.get_state().elapsed > time, "Play did not resume playback")
	scene.lab.paused = true
	scene._seek_time(5)
	scene.timeline.grab_focus()
	var key := InputEventKey.new()
	key.keycode = KEY_RIGHT
	key.pressed = true
	root.push_input(key)
	key = InputEventKey.new()
	key.keycode = KEY_RIGHT
	root.push_input(key)
	check(scene.state.elapsed > 5, "Keyboard seeking did not move the timeline")
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		scene._select_scenario(1)
		scene.lab.paused = true
		scene._seek_time(5)
		await settle()
		check(scene.layout.get_global_rect().end.x <= dimensions.x + 1, "Preview layout overflows horizontally")
		check(scene.viewport_container.size.y >= 100, "Preview viewport collapsed")
		for button in [scene.replay_pause, scene.before_button, scene.compare_button]:
			check(button.size.y >= 44, "Preview action is too small")
			check(button.get_global_rect().end.x <= dimensions.x + 1 and button.get_global_rect().end.y <= dimensions.y + 1, "Preview action is outside the viewport")
		check(scene.timeline.size.y >= 44 and scene.scenario.size.y >= 44, "Timeline or policy selector is too small")
		if DisplayServer.get_name() != "headless":
			await RenderingServer.frame_post_draw
			var capture_path := ProjectSettings.globalize_path("res://../build/pond-reliability-study/preview-%dx%d.png" % [dimensions.x, dimensions.y])
			DirAccess.make_dir_recursive_absolute(capture_path.get_base_dir())
			var saved := root.get_texture().get_image().save_png(capture_path)
			check(saved == OK, "Rendered capture could not be saved: " + capture_path)
			print("BANK_PREVIEW_CAPTURE display=%s path=%s saved=%d" % [DisplayServer.get_name(), capture_path, saved])
	check(FileAccess.get_sha256(policy) == policy_hash, "Preview modified the active policy")
	scene.find_child("BackToDemos", true, false).pressed.emit()
	await settle()
	check(current_scene.name == "DemoMenu", "Back to demos failed")
	print("BANK_PREVIEW checks=%d failures=%d" % [checks, failures])
	quit(1 if failures else 0)

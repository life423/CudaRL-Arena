extends SceneTree
var failures := 0
var checks := 0
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
	var expected := "CPU"
	var snapshot_prefix := ""
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--expected-backend="):
			expected = argument.get_slice("=", 1)
		elif argument.begins_with("--snapshot-prefix="):
			snapshot_prefix = argument.get_slice("=", 1)
	if snapshot_prefix.is_empty():
		snapshot_prefix = expected
	check(ProjectSettings.get_setting("cudarl/portable_player", false), "Not the standalone player project")
	check(not ClassDB.class_exists("CudaDevice") and not ClassDB.class_exists("PpoLab"), "Portable player loaded the CUDA training extension")
	var policy_hash := FileAccess.get_sha256("res://pond/policies/deep_selector.bin")
	var scene = load("res://pond/deep_pond.tscn").instantiate()
	root.add_child(scene)
	current_scene = scene
	await create_timer(2.5).timeout
	check(scene.lab.get_error().is_empty(), "Portable backend error: " + scene.lab.get_error())
	var stats: Dictionary = scene.lab.get_stats()
	check(stats.compute_backend == expected, "Wrong compute backend: " + str(stats))
	check(scene.training_stats.text.contains(expected), "Selected backend is not shown")
	check(stats.background_swimmers == 0 and stats.validated_controller, "Playback falsely started training")
	check(scene.state.elapsed > 0, "The portable tadpole is frozen")
	check(not scene.pause_button.visible and not scene.speed_button.visible, "Portable playback offers false training controls")
	scene.set_physics_process(false)
	var slots := {}
	for tick in 1200:
		scene.previous = scene.state.duplicate(true)
		scene.state = scene.lab.step()
		if scene.state.eaten_slot >= 0:
			slots[scene.state.eaten_slot] = true
	check(scene.state.eaten > 10 and slots.size() == 3, "Retained AI did not collect from all three food slots")
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		await settle()
		check(scene.layout.get_global_rect().end.x <= dimensions.x + 1, "Portable layout overflows")
		check(scene.camera_button.size.y >= 44 and scene.viewport_container.size.y >= 100, "Camera or view is unusable")
		var overview: bool = scene.overview
		scene.camera_button.pressed.emit()
		check(scene.overview != overview, "Camera control did not switch views")
		scene.camera_button.pressed.emit()
		await settle()
		if DisplayServer.get_name() != "headless":
			await RenderingServer.frame_post_draw
			root.get_texture().get_image().save_png(ProjectSettings.globalize_path("res://../../../../build/portable-checks/%s-%dx%d.png" % [snapshot_prefix, dimensions.x, dimensions.y]))
	check(FileAccess.get_sha256("res://pond/policies/deep_selector.bin") == policy_hash, "Portable playback modified the AI")
	scene.find_child("BackToDemos", true, false).pressed.emit()
	await settle()
	check(current_scene.rows.size() == 1 and current_scene.buttons.size() == 1, "Portable menu offers unavailable demos")
	current_scene._show_guide()
	await settle()
	check(current_scene.guide.visible and current_scene.guide.dialog_text.contains("CPU"), "Portable help is missing")
	check(current_scene.guide.size.x <= root.size.x and current_scene.guide.size.y <= root.size.y, "Portable help overflows the window")
	print("PORTABLE_GODOT backend=%s checks=%d failures=%d" % [expected, checks, failures])
	quit(1 if failures else 0)

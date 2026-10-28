extends SceneTree
var checks := 0
var failures := 0
func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)
func settle(frames := 8) -> void:
	for i in frames:
		await process_frame
func _initialize() -> void:
	Engine.max_fps = 60
	_run.call_deferred()
func _run() -> void:
	create_timer(90).timeout.connect(func(): quit(1))
	var policy := "res://../build/pond-choice-study/chosen.bin"
	var policy_hash := FileAccess.get_sha256(policy)
	var training := "user://pond-policies/deep-pond-v1/training.bin"
	var training_hash := FileAccess.get_sha256(training)
	var scene = load("res://pond/deep_pond.tscn").instantiate()
	scene.controller_policy_path = policy
	root.add_child(scene)
	current_scene = scene
	await create_timer(2.5).timeout
	check(scene.lab.get_error().is_empty(), "Saved selector failed to load: " + scene.lab.get_error())
	var stats: Dictionary = scene.lab.get_stats()
	check(stats.validated_controller and stats.training_ready, "Scene did not load the selected controller")
	check(stats.background_swimmers == 0 and stats.paused, "Saved-controller playback unexpectedly started training")
	check(scene.state.elapsed > 0, "The saved controller did not begin swimming")
	check(not scene.pause_button.visible and not scene.speed_button.visible, "Playback offers misleading training controls")
	check(not scene.find_child("SaveTrainingNow", true, false).visible and not scene.curve.watch_button.visible, "Playback offers misleading save/latest controls")
	check(scene.title_status.text == "Live · original saved agent" and scene.training_heading.text == "Saved controller", "Playback status is misleading")
	check(scene.curve.plot_title == "Live collection rate" and not scene.curve.show_best_score and scene.curve.status_caption.contains("no learning"), "Playback graph implies learning or uses a validation score")
	scene.set_physics_process(false)
	var slots := {}
	var targets := {}
	var start_depth: float = scene.state.position.y
	var minimum := start_depth
	var maximum := start_depth
	for tick in 1200:
		scene.previous = scene.state.duplicate(true)
		scene.state = scene.lab.step()
		targets[scene.state.selected_food] = true
		if scene.state.eaten_slot >= 0:
			slots[scene.state.eaten_slot] = true
		minimum = minf(minimum, scene.state.position.y)
		maximum = maxf(maximum, scene.state.position.y)
	check(scene.state.eaten > 0, "The loaded neural controller did not collect food")
	check(slots.size() >= 2 and targets.size() >= 2, "The loaded controller did not use multiple physical food slots")
	check(maximum - minimum > 0.2, "The loaded controller did not swim at different depths")
	check(scene.lab.get_stats().selector_update == stats.selector_update, "Playback changed the saved policy")
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		await settle()
		check(scene.layout.get_global_rect().end.x <= dimensions.x + 1, "Playback layout overflows horizontally")
		check(scene.camera_button.size.y >= 44, "Playback camera control is too small")
		check(scene.viewport_container.size.y >= 100, "Playback viewport collapsed")
		if DisplayServer.get_name() != "headless":
			await RenderingServer.frame_post_draw
			root.get_texture().get_image().save_png(ProjectSettings.globalize_path("res://../build/pond-choice-study/selector-%dx%d.png" % [dimensions.x, dimensions.y]))
	check(FileAccess.get_sha256(policy) == policy_hash, "Playback modified its controller artifact")
	check(FileAccess.get_sha256(training) == training_hash, "Playback modified the original training checkpoint")
	var eaten: int = scene.state.eaten
	scene.find_child("BackToDemos", true, false).pressed.emit()
	await settle()
	check(current_scene.name == "DemoMenu", "Playback Back to demos failed")
	print("POND_SELECTOR_SMOKE checks=%d failures=%d eaten=%d slots=%d" % [checks, failures, eaten, slots.size()])
	quit(1 if failures else 0)

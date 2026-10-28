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
func press_space() -> void:
	var event := InputEventKey.new()
	event.keycode = KEY_SPACE
	event.pressed = true
	root.push_input(event)
	event = InputEventKey.new()
	event.keycode = KEY_SPACE
	event.pressed = false
	root.push_input(event)
func _initialize() -> void:
	Engine.max_fps = 60
	_run.call_deferred()
func _run() -> void:
	create_timer(90).timeout.connect(func(): quit(1))
	var snapshot_dir := ""
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--snapshot-dir="):
			snapshot_dir = argument.trim_prefix("--snapshot-dir=")
	var policy := "res://pond/policies/deep_selector.bin"
	var model_hash := FileAccess.get_sha256(policy)
	var training := "user://pond-policies/deep-pond-v1/training.bin"
	var training_hash := FileAccess.get_sha256(training)
	var scene = load("res://pond/deep_pond.tscn").instantiate()
	root.add_child(scene)
	current_scene = scene
	scene.set_physics_process(false)
	await settle()
	check(scene.lab.get_error().is_empty(), "Controller failed: " + scene.lab.get_error())
	check(scene.untrained_button.visible and not scene.untrained_button.button_pressed, "Untrained toggle is missing or on by default")
	var initial: Dictionary = scene.lab.get_state().duplicate(true)
	var trained: Array[Dictionary] = []
	for tick in 80:
		trained.append(scene.lab.step().duplicate(true))
	scene.untrained_button.grab_focus()
	press_space()
	await settle()
	check(scene.untrained_button.button_pressed, "Keyboard did not enable the untrained policy")
	check(scene.lab.get_stats().showcase_untrained and not scene.lab.get_stats().validated_controller, "Random weights are presented as a validated policy")
	check(scene.title_status.text == "Untrained · random weights", "Untrained state is not clearly labeled")
	check(scene.training_heading.text == "Untrained controller", "Untrained details still describe trained weights")
	check(scene.state.elapsed == 0 and scene.state.eaten == 0, "Switch inherited time or food counters")
	check(scene.state.position == initial.position and scene.state.foods == initial.foods, "Untrained comparison changed the starting layout")
	check(scene.previous.position == scene.state.position and scene.accumulator == 0, "Switch interpolates from a stale trained pose")
	check(scene.curve.plot_title == "Trained controller reference" and scene.curve.plot_caption.contains("not the untrained"), "Trained validation is attributed to the untrained swimmer")
	check(scene.lab.get_stats().background_swimmers == 0 and scene.lab.get_stats().env_steps == 0, "Untrained viewing started training")
	var random: Array[Dictionary] = []
	for tick in 80:
		random.append(scene.lab.step().duplicate(true))
	check(random[-1].position.distance_to(trained[-1].position) > 0.001, "Random weights did not change behavior")
	check(is_finite(random[-1].speed_cm_s), "Untrained physical state is invalid")
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		scene.state = random[-1].duplicate(true)
		scene.previous = scene.state.duplicate(true)
		await settle()
		check(scene.layout.get_global_rect().end.x <= dimensions.x + 1, "Untrained controls overflow")
		check(scene.untrained_button.size.y >= 44, "Untrained toggle is too small")
		check(scene.untrained_button.size.x <= scene.sidebar.size.x, "Untrained text exceeds sidebar")
		if not snapshot_dir.is_empty() and DisplayServer.get_name() != "headless":
			await RenderingServer.frame_post_draw
			root.get_texture().get_image().save_png(snapshot_dir.path_join("untrained-%dx%d.png" % [dimensions.x, dimensions.y]))
	scene.untrained_button.button_pressed = false
	await settle()
	check(not scene.lab.get_stats().showcase_untrained and scene.lab.get_stats().validated_controller, "Trained policy was not restored")
	check(scene.title_status.text == "Trained PPO food selector", "Trained label was not restored")
	check(scene.curve.plot_title == "Saved controller validation", "Trained validation label was not restored")
	for tick in 80:
		var actual: Dictionary = scene.lab.step()
		check(actual.position.distance_to(trained[tick].position) < 0.000001 and actual.selected_food == trained[tick].selected_food, "Restoring trained weights changed the original trajectory")
	scene.untrained_button.button_pressed = true
	await settle()
	for tick in 80:
		var actual: Dictionary = scene.lab.step()
		check(actual.position.distance_to(random[tick].position) < 0.000001, "Untrained weights change on repeated switching")
	check(FileAccess.get_sha256(policy) == model_hash, "Comparison modified trained model file")
	check(FileAccess.get_sha256(training) == training_hash, "Comparison modified original training save")
	scene.find_child("BackToDemos", true, false).pressed.emit()
	await settle()
	check(current_scene.name == "DemoMenu", "Comparison Back to demos failed")
	print("POND_UNTRAINED checks=%d failures=%d" % [checks, failures])
	quit(1 if failures else 0)

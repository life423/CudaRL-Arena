extends SceneTree

var checks := 0
var failures := 0
var output := "res://../build/live-feeding-checks"
var production_check := false

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

func open_pond(path: String):
	var scene = load("res://pond/feeding_pond.tscn").instantiate()
	scene.population = 16 if "--pond-backend=cpu" in OS.get_cmdline_user_args() else (1024 if production_check else 128)
	scene.feeding_checkpoint_path = path
	root.add_child(scene)
	current_scene = scene
	scene.set_physics_process(false)
	await settle()
	return scene

func _run() -> void:
	production_check = "--production-learning-check" in OS.get_cmdline_user_args()
	if ProjectSettings.get_setting("cudarl/release_build", false):
		output = OS.get_executable_path().get_base_dir().path_join("qa-output")
	create_timer(120).timeout.connect(func(): quit(1))
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(output))
	var source := "res://../docs/experiments/swimmer/gentle-currents/bank-recovery/selected-policy.bin"
	if ProjectSettings.get_setting("cudarl/release_build", false):
		source = OS.get_executable_path().get_base_dir().path_join("data/selected-policy.bin")
	var source_hash := FileAccess.get_sha256(source)
	var path := "user://tests/live-feeding-%d-%d.bin" % [OS.get_process_id(), Time.get_ticks_usec()]
	var scene = await open_pond(path)
	check(scene.lab.get_error().is_empty(), "Feeding scene could not start: " + scene.lab.get_error())
	if not scene.lab.get_error().is_empty():
		quit(1)
		return
	check(scene.stats.feeding_mode and scene.stats.policy_inputs == 37, "Wrong feeding controller mode")
	var before: Dictionary = scene.lab.get_state().duplicate(true)
	for i in 16:
		scene._drop_at(Vector3(-1.5 + (i % 4), 0, -1.0 + (i / 4) * 0.5))
	check(scene.state.foods.size() >= 16, "Drops still replace food after three pellets")
	check(scene.state.position == before.position and scene.state.elapsed == before.elapsed, "Dropping food moved/reset the creature")
	var count: int = scene.state.foods.size()
	scene._drop_at(Vector3(100, 0, 0))
	check(scene.state.foods.size() == count, "Shore/outside drops mutated food")
	# Route a real pointer event through the input overlay and its surface ray.
	var pixel: Vector2 = scene.camera.unproject_position(Vector3(0.7, 0, 0.4))
	var point: Vector2 = scene.feed_surface.get_global_rect().position + pixel * scene.feed_surface.size / Vector2(scene.viewport.size)
	for pressed in [true, false]:
		var event := InputEventMouseButton.new()
		event.position = point
		event.global_position = point
		event.button_index = MOUSE_BUTTON_LEFT
		event.pressed = pressed
		root.push_input(event, true)
	await settle()
	check(scene.state.foods.size() == count + 1, "Pointer drop did not reach native physics")
	var button: Button = scene.find_child("DropFoodAtCenter", true, false)
	button.grab_focus()
	for pressed in [true, false]:
		var event := InputEventKey.new()
		event.keycode = KEY_SPACE
		event.pressed = pressed
		root.push_input(event)
	await settle()
	check(scene.state.foods.size() == count + 2, "Keyboard feeding alternative failed")
	for pressed in [true, false]:
		var event := InputEventScreenTouch.new()
		event.position = point
		event.index = 0
		event.pressed = pressed
		root.push_input(event, true)
	await settle()
	check(scene.state.foods.size() == count + 3, "Tap did not create exactly one pellet")
	check(not scene.untrained_button.visible, "Learning mode exposes a disconnected random-policy switch")
	scene._set_current(true)
	var initial_positions: PackedVector3Array = scene.state.foods
	for tick in 12:
		scene.state = scene.lab.step()
	check(scene.state.foods[0].y < initial_positions[0].y, "Dropped food did not sink")
	check(scene.state.foods[0].x > initial_positions[0].x, "Dropped food did not drift with current")
	for tick in 400:
		scene.state = scene.lab.step()
		if tick % 20 == 0:
			await process_frame
	scene.previous = scene.state.duplicate(true)
	scene._poll()
	var started := Time.get_ticks_msec()
	var requested_updates := 32 if production_check else 4
	while (scene.stats.update < requested_updates or (production_check and scene.stats.evaluation_count < 2)) and Time.get_ticks_msec() - started < 70000 and str(scene.stats.learning_error).is_empty():
		var time_before: float = scene.state.elapsed
		scene.state = scene.lab.step()
		check(absf(scene.state.elapsed - time_before - 0.05) < 0.0001, "Policy adoption reset the live physical clock")
		await create_timer(0.1).timeout
		scene._poll()
	check(str(scene.stats.learning_error).is_empty(), "Native PPO failed: " + str(scene.stats.learning_error))
	check(scene.stats.update >= 4 and scene.stats.learner_weights_changed, "No genuine PPO swimming-weight updates occurred")
	check(scene.stats.trained_context_version == scene.stats.context_version, "User-created conditions did not reach practice worlds")
	check(scene.stats.selector_update == 320, "The stage silently replaced the inherited selector")
	check(scene.stats.background_swimmers == scene.population, "Background practice population is not reported honestly")
	if production_check:
		check(scene.stats.update >= 32 and scene.stats.evaluation_count >= 2, "Periodic candidate validation did not execute")
		scene.state = scene.lab.step()
		scene._poll()
		check(scene.stats.adopted_update == scene.stats.approved_update, "A validated candidate was not applied to the live actor")
		print("FEEDING_CANDIDATE_CHECKS update=%d evaluated=%d approved=%d applied=%d score=%.2f original=%.2f" % [scene.stats.update, scene.stats.evaluation_count, scene.stats.approved_update, scene.stats.adopted_update, scene.stats.best_food_per_minute, scene.stats.original_food_per_minute])
	check(scene.curve.status_caption.contains("PPO"), "Learning status is missing from the visible graph")
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		await settle()
		scene.sidebar_scroll.ensure_control_visible(button)
		await settle()
		check(scene.viewport_container.size.y >= 100 and button.size.y >= 44, "Feeding layout is unusable")
		check(button.get_global_rect().end.x <= dimensions.x, "Feeding action is clipped")
		if DisplayServer.get_name() != "headless":
			await RenderingServer.frame_post_draw
			check(root.get_texture().get_image().save_png(ProjectSettings.globalize_path(output.path_join("feeding-%dx%d.png" % [dimensions.x, dimensions.y]))) == OK, "Screenshot failed")
	scene.lab.set_paused(true)
	await create_timer(1).timeout
	scene._poll()
	var saved_update: int = scene.stats.update
	before = scene.lab.get_state().duplicate(true)
	scene.set_process(false)
	scene.lab.stop()
	scene.queue_free()
	await settle()
	check(FileAccess.file_exists(path.get_basename() + "-cpu.bin" if "--pond-backend=cpu" in OS.get_cmdline_user_args() else path), "Closing did not save full learning state")
	scene = await open_pond(path)
	started = Time.get_ticks_msec()
	while scene.stats.update <= scene.stats.loaded_update and Time.get_ticks_msec() - started < 15000 and str(scene.stats.learning_error).is_empty():
		await create_timer(0.1).timeout
		scene._poll()
	check(scene.stats.learning_ready and str(scene.stats.learning_error).is_empty() and scene.stats.update > scene.stats.loaded_update, "The restored optimizer/worlds did not continue PPO")
	check(scene.stats.resumed and scene.stats.loaded_update >= saved_update, "Optimizer/world training did not resume")
	check(scene.state.position == before.position and scene.state.foods == before.foods and scene.state.elapsed == before.elapsed, "Resume lost the physical pond or user pellets")
	check(FileAccess.get_sha256(source) == source_hash, "Learning overwrote the preserved source policy")
	print("LIVE_FEEDING checks=%d failures=%d resumed_update=%d" % [checks, failures, scene.stats.loaded_update])
	scene.set_process(false)
	scene.lab.stop()
	quit(1 if failures else 0)

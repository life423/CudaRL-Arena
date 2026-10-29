extends SceneTree

var checks := 0
var failures := 0
var backend := "cuda"
var output := "res://../build/autonomous-practice/rendered"

func check(ok: bool, message: String) -> void:
	checks += 1
	if not ok:
		failures += 1
		push_error(message)

func settle() -> void:
	for tick in 8:
		await process_frame

func _initialize() -> void:
	Engine.max_fps = 60
	run.call_deferred()

func run() -> void:
	create_timer(150).timeout.connect(func(): quit(1))
	if "--pond-backend=cpu" in OS.get_cmdline_user_args():
		backend = "cpu"
	output += "-" + backend
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(output))
	var path := ProjectSettings.globalize_path(output.path_join("training-%d.bin" % Time.get_ticks_usec()))
	var scene = load("res://pond/feeding_pond.tscn").instantiate()
	scene.population = 16 if backend == "cpu" else 1024
	scene.feeding_checkpoint_path = path
	root.add_child(scene)
	current_scene = scene
	scene.set_physics_process(false)
	await settle()
	check(scene.lab.get_error().is_empty(), "Autonomous feeding scene failed to start")
	if not scene.lab.get_error().is_empty():
		quit(1)
		return
	var button: CheckButton = scene.find_child("AutoFeed", true, false)
	check(button != null and not button.button_pressed, "Auto feed should be an optional explicit control")
	var before: Dictionary = scene.lab.get_state().duplicate(true)
	button.grab_focus()
	for pressed in [true, false]:
		var key := InputEventKey.new()
		key.keycode = KEY_SPACE
		key.pressed = pressed
		root.push_input(key)
	await settle()
	check(scene.auto_feed_enabled and button.button_pressed, "Keyboard did not enable auto feed")
	check(scene.state.position == before.position and scene.state.elapsed == before.elapsed, "Enabling auto feed reset/moved the body")
	var first_due: float = scene.auto_feed_next_time
	check(first_due >= before.elapsed + 2 and first_due <= before.elapsed + 6, "Automatic feeding interval is invalid")
	scene._advance_auto_feed()
	await create_timer(0.15).timeout
	scene._advance_auto_feed()
	check(scene.auto_feed_count == 0, "Auto feeding used wall time without physical time advancing")
	for tick in 440:
		scene._physics_process(0.05)
		if tick % 20 == 0:
			await process_frame
	check(scene.auto_feed_count >= 3 and scene.auto_feed_count <= 11, "Timed automatic food arrivals are missing")
	scene._poll()
	check(scene.stats.context_version >= scene.auto_feed_count, "Automatic food did not reach practice context")
	before = scene.lab.get_state().duplicate(true)
	scene._drop_at(Vector3(0.7, 0, -0.6))
	check(scene.state.elapsed == before.elapsed and scene.state.position == before.position, "Manual feeding changed/reset the live body")
	scene._set_current(true)
	check(scene.stats.water_current_cm_s > 0, "Visible manual current stopped working")
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		await settle()
		scene.sidebar_scroll.ensure_control_visible(button)
		await settle()
		check(button.size.y >= 44 and button.get_global_rect().end.x <= dimensions.x, "Auto feed is clipped or too small")
		check(scene.viewport_container.size.y >= 100, "Pond viewport collapsed")
		await RenderingServer.frame_post_draw
		check(root.get_texture().get_image().save_png(ProjectSettings.globalize_path(output.path_join("auto-%dx%d.png" % [dimensions.x, dimensions.y]))) == OK, "Automatic pond screenshot failed")
	button.button_pressed = false
	var automatic_count: int = scene.auto_feed_count
	for tick in 140:
		scene._physics_process(0.05)
		if tick % 20 == 0:
			await process_frame
	check(scene.auto_feed_count == automatic_count, "Disabling auto feed did not stop automatic arrivals")
	# Match a due event with a full pond. Never replace food or enqueue an interaction.
	while scene.state.foods.size() < 128:
		if scene.lab.drop_food(Vector3(2.5, 0, 0)) < 0:
			break
		scene.state = scene.lab.get_state()
	scene._poll()
	button.button_pressed = true
	scene._physics_process(0.05)
	while scene.state.foods.size() < 128:
		scene.lab.drop_food(Vector3(2.5, 0, 0))
		scene.state = scene.lab.get_state()
	scene._poll()
	before = scene.lab.get_state().duplicate(true)
	var context_before: int = scene.stats.context_version
	scene.auto_feed_last_time = scene.state.elapsed - 0.05
	scene.auto_feed_next_time = scene.state.elapsed
	scene._advance_auto_feed()
	scene._poll()
	check(scene.state.foods == before.foods and scene.stats.context_version == context_before, "Auto feeding replaced full-capacity food or sent rejected practice context")
	check(scene.feed_feedback.text.contains("waiting for space"), "Full automatic-feeding state lacks feedback")
	button.button_pressed = false
	var started := Time.get_ticks_msec()
	while scene.stats.evaluation_count < 2 and Time.get_ticks_msec() - started < 85000 and str(scene.stats.learning_error).is_empty():
		await create_timer(0.1).timeout
		scene._poll()
	check(str(scene.stats.learning_error).is_empty() and scene.stats.learner_weights_changed, "Autonomous practice failed to perform real PPO")
	check(scene.stats.evaluation_count >= 2, "Scheduled candidate evaluation did not execute")
	# Publication is deliberately applied at the next real physical step.
	var clock_before: float = scene.state.elapsed
	scene.state = scene.lab.step()
	scene._poll()
	check(absf(scene.state.elapsed - clock_before - 0.05) < 0.0001, "Policy adoption reset the physical clock")
	check(scene.stats.adopted_update == scene.stats.approved_update, "Validated update was not delivered")
	check(scene.training_stats.text.contains("varying"), "Practice conditions are not explained honestly")
	scene.lab.set_paused(true)
	scene.set_process(false)
	before = scene.lab.get_state().duplicate(true)
	scene.lab.stop()
	scene.queue_free()
	await settle()
	var resumed = load("res://pond/feeding_pond.tscn").instantiate()
	resumed.population = 16 if backend == "cpu" else 1024
	resumed.feeding_checkpoint_path = path
	root.add_child(resumed)
	current_scene = resumed
	resumed.set_physics_process(false)
	await settle()
	resumed._poll()
	check(resumed.stats.resumed and resumed.stats.loaded_update >= 32, "Autonomous practice full save did not resume")
	check(resumed.state.foods == before.foods and resumed.state.position == before.position and resumed.state.elapsed == before.elapsed, "Practice resume changed visible physical state")
	resumed.set_process(false)
	resumed.lab.stop()
	print("AUTONOMOUS_FEEDING backend=%s checks=%d failures=%d loaded=%d" % [backend, checks, failures, resumed.stats.loaded_update])
	quit(1 if failures else 0)

extends SceneTree
var failures := 0
var checks := 0
func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)
func settle() -> void:
	for i in 8:
		await process_frame
func _initialize() -> void:
	Engine.max_fps = 60
	_run.call_deferred()
func _run() -> void:
	create_timer(90).timeout.connect(func(): quit(1))
	var checkpoint := "user://tests/deep-pond-smoke-%d.bin" % OS.get_process_id()
	var scene = load("res://pond/deep_pond.tscn").instantiate()
	scene.checkpoint_path = checkpoint
	root.add_child(scene)
	await settle()
	var start := Time.get_ticks_msec()
	while not scene.lab.get_stats().training_ready and Time.get_ticks_msec() - start < 12000:
		await create_timer(0.025).timeout
	var baseline: Dictionary = scene.lab.get_stats()
	check(scene.lab.get_error().is_empty(), "Pond training error: " + scene.lab.get_error())
	check(baseline.transferred and baseline.loaded_update > 0, "The learned 3D swimmer was not transferred")
	check(baseline.policy_inputs == 33 and baseline.food_count == 3, "Wrong neural interface or food count")
	check(scene.foods.size() == 3 and scene.state.foods.size() == 3, "Three foods are not present")
	for i in 3:
		for j in i:
			check(absf(scene.state.foods[i].y - scene.state.foods[j].y) >= 0.5, "Food depths are not distinct")
	start = Time.get_ticks_msec()
	while scene.lab.get_stats().update < baseline.loaded_update + 3 and Time.get_ticks_msec() - start < 12000:
		await create_timer(0.05).timeout
	check(scene.lab.get_stats().update >= baseline.loaded_update + 3, "Transferred learner did not continue")
	scene.pause_button.pressed.emit()
	await create_timer(0.2).timeout
	var paused_update: int = scene.lab.get_stats().update
	await create_timer(0.2).timeout
	check(scene.lab.get_stats().update == paused_update, "Pause did not stop optimization")
	check(scene.state.elapsed > 0, "Visible swimmer did not simulate")
	check(scene.lab.get_stats().watching_best, "The strongest evaluated controller should be shown by default")
	scene.curve.watch_button.toggled.emit(false)
	check(not scene.lab.get_stats().watching_best, "Latest-controller viewing failed")
	scene.curve.watch_button.toggled.emit(true)
	check(scene.lab.get_stats().watching_best, "Best-controller viewing failed")
	# Keep the native snapshot fixed while checking rendered positions/layout.
	# SceneTree.process_frame resumes before node _process, so a collection on
	# the latest physics tick could otherwise leave a respawn awaiting rendering.
	scene.set_physics_process(false)
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		await settle()
		check(scene.layout.get_global_rect().end.x <= dimensions.x + 1, "Layout overflows horizontally")
		check(scene.viewport_container.size.y >= 100, "Pond view collapsed")
		check(scene.pause_button.size.y >= 44, "Pause target is too small")
		for overview in [true, false]:
			scene.overview = overview
			for i in 50:
				await process_frame
			for i in 3:
				check(scene.foods[i].position.distance_to(scene.state.foods[i]) < 0.001, "Food rendering changed the physical position")
			if overview:
				for p in scene.state.foods:
					var projected: Vector2 = scene.camera.unproject_position(p)
					check(Rect2(Vector2.ZERO, scene.viewport.size).has_point(projected), "Overview clips a food piece")
			if DisplayServer.get_name() != "headless":
				await RenderingServer.frame_post_draw
				root.get_texture().get_image().save_png(ProjectSettings.globalize_path("res://../build/pond-learning-checks/deep-pond-%s-%dx%d.png" % ["overview" if overview else "follow", dimensions.x, dimensions.y]))
	scene.lab.save_training_now()
	await create_timer(0.1).timeout
	check(scene.lab.get_stats().saved_update == paused_update, "Save failed while paused")
	# Render the real inherited controller over simulated time; no scripted gait.
	var food_before: int = scene.state.eaten
	for i in 500:
		scene.previous = scene.state.duplicate(true)
		scene.state = scene.lab.step()
	check(scene.state.eaten > food_before, "Transferred swimmer did not collect food in the pond")
	scene.queue_free()
	await settle()
	var resumed := DeepPondLab.new()
	check(resumed.start_training(1024, 7, checkpoint), "Resume failed to start")
	await create_timer(0.35).timeout
	check(resumed.get_stats().resumed and resumed.get_stats().loaded_update == paused_update, "Full three-food save was not restored")
	resumed.stop()
	var router := root.get_node("DemoLaunch")
	check(router.SCENES.deep_water == "res://pond/deep_pond.tscn", "Menu still opens the tank")
	var menu = load("res://launcher/launcher.tscn").instantiate()
	root.add_child(menu)
	await settle()
	check(menu._headless_command().contains("--pond") and menu._headless_command().contains("deep-pond-v1"), "Headless menu command opens the wrong task")
	menu.queue_free()
	print("DEEP_POND_SMOKE checks=%d failures=%d inherited_update=%d" % [checks, failures, baseline.loaded_update])
	quit(1 if failures else 0)

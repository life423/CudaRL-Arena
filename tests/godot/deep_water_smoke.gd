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
	var scene = load("res://pond/deep_water.tscn").instantiate()
	scene.checkpoint_path = "user://tests/deep-smoke-%d.bin" % OS.get_process_id()
	scene.population = 16
	root.add_child(scene)
	await settle()
	var start := Time.get_ticks_msec()
	while scene.lab.get_stats().update < 3 and Time.get_ticks_msec() - start < 12000:
		await create_timer(0.05).timeout
	check(scene.lab.get_error().is_empty(), "Deep-water backend error: " + scene.lab.get_error())
	check(scene.lab.get_stats().update >= 3, "Deep water did not learn automatically")
	check(scene.state.links.size() == 3, "The articulated 3D body is missing")
	check(scene.state.food.y < 0 and scene.state.food.y > -4, "Food is outside the water")
	scene.pause_button.pressed.emit()
	await create_timer(0.25).timeout
	var paused_update: int = scene.lab.get_stats().update
	await create_timer(0.3).timeout
	check(scene.lab.get_stats().update == paused_update and scene.lab.get_stats().paused, "Pause failed")
	check(scene.state.elapsed > 0, "Visible controller did not simulate")
	scene.lab.save_training_now()
	await create_timer(0.1).timeout
	check(scene.lab.get_stats().saved_update == paused_update, "Manual save did not work while paused")
	var blocker := DeepWaterLab.new()
	blocker.start_training(16, 7, scene.checkpoint_path)
	await create_timer(0.2).timeout
	check(not blocker.get_error().is_empty(), "A second trainer was allowed to own the same save")
	blocker.stop()
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		await settle()
		check(scene.layout.get_global_rect().end.x <= dimensions.x + 1, "Deep-water layout overflows horizontally")
		check(scene.viewport_container.size.x > 100 and scene.viewport_container.size.y >= 100, "Swimming view collapsed")
		check(scene.pause_button.size.y >= 44 and scene.camera_button.size.y >= 44, "Controls too small")
		check(scene.sidebar.get_global_rect().end.x <= dimensions.x + 1, "Sidebar overflows")
		scene.sidebar_scroll.ensure_control_visible(scene.camera_button)
		await settle()
		check(scene.camera_button.get_global_rect().end.y <= dimensions.y, "Camera control is unreachable")
		scene.pause_button.grab_focus()
		check(root.gui_get_focus_owner() == scene.pause_button, "Keyboard focus failed")
		for overview in [false, true]:
			scene.overview = overview
			for i in 50:
				await process_frame
			if DisplayServer.get_name() != "headless":
				await RenderingServer.frame_post_draw
				root.get_texture().get_image().save_png(ProjectSettings.globalize_path("res://../build/pond-learning-checks/deep-%s-%dx%d.png" % ["overview" if overview else "follow", dimensions.x, dimensions.y]))
	var path: String = scene.checkpoint_path
	scene.queue_free()
	await settle()
	var resumed := DeepWaterLab.new()
	check(resumed.start_training(16, 7, path), "Resume failed to start")
	await create_timer(0.3).timeout
	check(resumed.get_stats().resumed and resumed.get_stats().loaded_update == paused_update, "Full deep-water training was not restored")
	resumed.stop()
	var menu = load("res://launcher/launcher.tscn").instantiate()
	root.add_child(menu)
	await settle()
	check(menu.find_child("pond_feeding_watch", true, false) != null, "Menu is missing the main pond")
	check(menu.find_child("guide_train", true, false) == null, "Removed help section remains visible")
	print("DEEP_WATER_SMOKE checks=%d failures=%d" % [checks, failures])
	quit(1 if failures else 0)

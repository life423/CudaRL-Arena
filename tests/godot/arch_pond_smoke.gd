extends SceneTree

var checks := 0
var failures := 0
var output := "res://../build/arch-navigation/rendered"
var source := "user://pond-policies/feeding-learning-v1/training.bin"
var protected_source := ""
var checkpoint := ""
var production := false

func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)

func settle() -> void:
	for tick in 8:
		await process_frame

func key_activate(control: Control) -> void:
	control.grab_focus()
	for pressed in [true, false]:
		var event := InputEventKey.new()
		event.keycode = KEY_SPACE
		event.pressed = pressed
		root.push_input(event)
	await settle()

func _initialize() -> void:
	Engine.max_fps = 60
	run.call_deferred()

func open_pond():
	var scene = load("res://pond/arch_pond.tscn").instantiate()
	scene.population = 1024 if production else 128
	scene.source_feeding_checkpoint = protected_source
	scene.arch_checkpoint = checkpoint
	root.add_child(scene)
	current_scene = scene
	scene.set_physics_process(false)
	await settle()
	return scene

func arch_distance(position: Vector3, geometry: Dictionary) -> float:
	var point := Basis(Vector3.UP, -float(geometry.yaw)) * (position - (geometry.center as Vector3))
	var ring := Vector2(point.x, point.y).length() - float(geometry.major_radius)
	return Vector2(ring, point.z).length() - float(geometry.tube_radius)

func run() -> void:
	create_timer(180).timeout.connect(func(): push_error("Arch smoke test timed out"); quit(1))
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--arch-source="):
			source = argument.trim_prefix("--arch-source=")
		elif argument.begins_with("--arch-output="):
			output = argument.trim_prefix("--arch-output=")
	production = "--production-learning-check" in OS.get_cmdline_user_args()
	output = ProjectSettings.globalize_path(output)
	check(DirAccess.make_dir_recursive_absolute(output) == OK, "Could not create arch QA output")
	check(FileAccess.file_exists(source), "Source feeding full save is missing")
	if not FileAccess.file_exists(source):
		quit(1)
		return
	var source_hash := FileAccess.get_sha256(source)
	var unique := "%d-%d" % [OS.get_process_id(), Time.get_ticks_usec()]
	protected_source = output.path_join("source-feeding-%s.bin" % unique)
	checkpoint = output.path_join("arch-training-%s.bin" % unique)
	check(DirAccess.copy_absolute(ProjectSettings.globalize_path(source), protected_source) == OK, "Could not protect source feeding save")
	check(FileAccess.get_sha256(protected_source) == source_hash, "Source snapshot does not match the feeding save")
	var unsupported = DeepPondLab.new()
	check(not unsupported.start_arch_learning(protected_source, checkpoint + ".unsupported", 128, "cpu"), "Arch prototype silently claims CPU learning support")
	check(unsupported.get_error().contains("requires NVIDIA CUDA") and unsupported.get_error().contains("feeding pond supports CPU"), "Unsupported backend does not explain the available feeding alternative")
	unsupported.stop()
	var scene = await open_pond()
	check(scene.lab.get_error().is_empty(), "Arch scene failed to start: " + scene.lab.get_error())
	if not scene.lab.get_error().is_empty():
		scene.lab.stop()
		quit(1)
		return
	scene._poll()
	var geometry: Dictionary = scene.lab.get_arch_geometry()
	check(geometry.radii.is_equal_approx(Vector2(6.9, 5.25)), "Arch basin does not use the larger native dimensions")
	check(scene._pond_radii().is_equal_approx(geometry.radii), "Rendered basin differs from the native habitat")
	check(scene.rock_arch.position.is_equal_approx(geometry.center), "Rendered arch center differs from native geometry")
	var torus: TorusMesh = scene.rock_arch.mesh
	check(is_equal_approx(torus.inner_radius, geometry.major_radius - geometry.tube_radius) and is_equal_approx(torus.outer_radius, geometry.major_radius + geometry.tube_radius), "Torus mesh does not match the native solid")
	check(absf((scene.rock_arch.basis * Vector3.UP).dot(Basis(Vector3.UP, geometry.yaw) * Vector3.FORWARD)) > 0.999, "Rendered arch axis differs from native Z axis")
	check(scene.stats.get("arch_mode", false) and scene.stats.policy_inputs == 55, "Arch controller has the wrong mode or input count")
	check(scene.stats.parent_update > 0, "Inherited validated swimming provenance is absent")
	check(scene.stats.water_current_cm_s == 0, "First arch experiment is not calm water")
	check(scene.state.foods.size() == 2, "The automatic arch task should begin with two foods")
	if scene.state.foods.size() == 2:
		var a: Vector3 = scene.state.foods[0] - (geometry.center as Vector3)
		var b: Vector3 = scene.state.foods[1] - (geometry.center as Vector3)
		check(a.z * b.z < 0, "Initial foods are not on opposite sides of the arch")
	var safe_foods := true
	for food: Vector3 in scene.state.foods:
		safe_foods = safe_foods and arch_distance(food, geometry) > 0.1 and food.y < -0.1
	check(safe_foods, "Native automatic food intersects solid rock or leaves the water")
	check(not scene.find_child("WaterControls", true, false).visible and not scene.auto_feed_button.visible and not scene.feed_surface.visible, "Controlled arch task exposes unsupported current/manual-food controls")
	check(not scene.curve.watch_button.visible and not scene.speed_button.visible and not scene.untrained_button.visible, "Arch task exposes disconnected policy/speed controls")
	check(scene.curve.plot_title == "Live collection rate" and scene.curve.tooltip_text.contains("session"), "Live collection plot is mislabeled as candidate evaluation")
	var before: Dictionary = scene.lab.get_state().duplicate(true)
	var camera_before: Vector3 = scene.camera.position
	await key_activate(scene.camera_button)
	check(not scene.overview and scene.state.position == before.position and scene.state.elapsed == before.elapsed, "Keyboard camera control reset the physical world or did not activate")
	await key_activate(scene.camera_button)
	check(scene.overview and scene.camera.position.is_equal_approx(camera_before), "Pond overview did not restore its stable camera")
	await key_activate(scene.pause_button)
	scene._poll()
	check(scene.stats.paused, "Pause learning does not work by keyboard")
	await key_activate(scene.pause_button)
	scene._poll()
	check(not scene.stats.paused, "Resume learning does not work by keyboard")
	var moved := 0.0
	var finite := true
	var clocks := true
	for tick in 240:
		var old_time: float = scene.state.elapsed
		scene.previous = scene.state.duplicate(true)
		scene.state = scene.lab.step()
		clocks = clocks and absf(scene.state.elapsed - old_time - 0.05) < 0.0001
		finite = finite and (scene.state.position as Vector3).is_finite()
		moved = maxf(moved, (scene.state.position as Vector3).distance_to(before.position))
		if tick % 20 == 0:
			await process_frame
	scene.previous = scene.state.duplicate(true)
	scene._poll()
	check(finite and clocks, "Live arch steps have invalid state or reset the clock")
	check(moved > 0.2, "The inherited live swimmer did not move in the larger habitat")
	check(scene.state.foods.size() == 2, "Automatic food replenishment stopped during live simulation")
	var started := Time.get_ticks_msec()
	while (scene.stats.update < 32 or scene.stats.evaluation_count < 2) and Time.get_ticks_msec() - started < 115000 and str(scene.stats.learning_error).is_empty():
		await create_timer(0.1).timeout
		scene._poll()
	check(str(scene.stats.learning_error).is_empty(), "Background arch PPO failed: " + str(scene.stats.learning_error))
	check(scene.stats.learning_ready and scene.stats.learner_weights_changed and scene.stats.update >= 32, "The arch motor did not perform genuine PPO updates")
	check(scene.stats.evaluation_count >= 2, "Matched candidate/retention evaluation did not execute")
	check(scene.stats.background_swimmers == scene.population, "Reported practice population differs from the requested population")
	var clock_before: float = scene.state.elapsed
	scene.previous = scene.state.duplicate(true)
	scene.state = scene.lab.step()
	scene._poll()
	check(absf(scene.state.elapsed - clock_before - 0.05) < 0.0001, "Applying an approved actor reset the body clock")
	check(scene.stats.adopted_update == scene.stats.approved_update, "Approved actor was not applied at the next physical step")
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		await settle()
		scene.sidebar_scroll.scroll_vertical = 0
		await settle()
		check(scene.viewport_container.size.y >= 100 and scene.viewport_container.get_global_rect().end.x <= dimensions.x + 1, "Arch viewport collapsed or overflows")
		check(scene.camera_button.size.y >= 44 and scene.pause_button.size.y >= 44, "Arch actions are smaller than 44 pixels")
		check(scene.camera_button.get_global_rect().end.x <= dimensions.x + 1, "Camera control is clipped horizontally")
		var pixel: Vector2 = scene.camera.unproject_position(geometry.center)
		check(Rect2(Vector2.ZERO, Vector2(scene.viewport.size)).has_point(pixel) and not scene.camera.is_position_behind(geometry.center), "Arch opening is outside the overview camera")
		if DisplayServer.get_name() != "headless":
			await RenderingServer.frame_post_draw
			check(root.get_texture().get_image().save_png(output.path_join("arch-%dx%d.png" % [dimensions.x, dimensions.y])) == OK, "Arch screenshot failed")
	root.size = Vector2i(1280, 720)
	await settle()
	await key_activate(scene.details_button)
	check(scene.details_group.visible and scene.training_stats.text.contains("inherited"), "AI details do not explain retained skills and new learning")
	check(scene.save_status.text.contains("Original feeding save preserved"), "Save provenance is not explained")
	if DisplayServer.get_name() != "headless":
		await RenderingServer.frame_post_draw
		check(root.get_texture().get_image().save_png(output.path_join("arch-details.png")) == OK, "Arch details screenshot failed")
	scene.lab.set_paused(true)
	await create_timer(0.6).timeout
	scene._poll()
	var saved_update: int = scene.stats.update
	before = scene.lab.get_state().duplicate(true)
	scene.set_process(false)
	scene.lab.stop()
	scene.queue_free()
	await settle()
	check(FileAccess.file_exists(checkpoint), "Stopping arch learning did not save its full checkpoint")
	scene = await open_pond()
	scene._poll()
	check(scene.stats.resumed and scene.stats.loaded_update >= saved_update, "Arch learner did not restore its own full training checkpoint")
	check(scene.state.position == before.position and scene.state.orientation == before.orientation and scene.state.foods == before.foods and scene.state.elapsed == before.elapsed, "Arch resume changed the live physical scene")
	started = Time.get_ticks_msec()
	while scene.stats.update <= scene.stats.loaded_update and Time.get_ticks_msec() - started < 15000 and str(scene.stats.learning_error).is_empty():
		await create_timer(0.1).timeout
		scene._poll()
	check(scene.stats.learning_ready and scene.stats.update > scene.stats.loaded_update and str(scene.stats.learning_error).is_empty(), "Restored arch optimizer/worlds did not continue PPO")
	check(FileAccess.get_sha256(source) == source_hash and FileAccess.get_sha256(protected_source) == source_hash, "Arch practice overwrote the original or copied feeding save")
	scene.set_process(false)
	scene.lab.stop()
	var summary := {"checks": checks, "failures": failures, "population": scene.population, "update": scene.stats.update, "resumed_update": scene.stats.loaded_update, "source_sha256": source_hash, "checkpoint": checkpoint, "max_live_displacement": moved, "arch_passages": scene.stats.get("arch_passages", 0), "arch_contacts": scene.stats.get("arch_contacts", 0)}
	var report := FileAccess.open(output.path_join("summary.json"), FileAccess.WRITE)
	if report:
		report.store_string(JSON.stringify(summary, "\t"))
	print("ARCH_POND_SMOKE " + JSON.stringify(summary))
	quit(1 if failures else 0)

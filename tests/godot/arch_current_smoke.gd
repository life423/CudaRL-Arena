extends SceneTree

var checks := 0
var failures := 0
var source := "res://../build/arch-current/preserved-20261029104613/arch-v1-training.bin"
var baseline_path := "res://../build/arch-current/v1-baseline.json"
var output := "res://../build/arch-current/rendered"
var checkpoint := ""
var population := 1024

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
	scene.population = population
	# A resumed arch save contains its source; this file deliberately does not exist.
	scene.source_feeding_checkpoint = "res://__missing_feeding_source_for_arch_resume__.bin"
	scene.arch_checkpoint = checkpoint
	root.add_child(scene)
	current_scene = scene
	scene.set_physics_process(false)
	await settle()
	scene._poll()
	return scene

func point(values: Array) -> Vector3:
	return Vector3(values[0], values[1], values[2])

func arch_distance(position: Vector3, geometry: Dictionary) -> float:
	var local := Basis(Vector3.UP, -float(geometry.yaw)) * (position - (geometry.center as Vector3))
	var ring := Vector2(local.x, local.y).length() - float(geometry.major_radius)
	return Vector2(ring, local.z).length() - float(geometry.tube_radius)

func run() -> void:
	create_timer(210).timeout.connect(func(): push_error("Arch current smoke timed out"); quit(1))
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--arch-v1-source="):
			source = argument.trim_prefix("--arch-v1-source=")
		elif argument.begins_with("--arch-v1-baseline="):
			baseline_path = argument.trim_prefix("--arch-v1-baseline=")
		elif argument.begins_with("--arch-output="):
			output = argument.trim_prefix("--arch-output=")
	output = ProjectSettings.globalize_path(output)
	source = ProjectSettings.globalize_path(source)
	baseline_path = ProjectSettings.globalize_path(baseline_path)
	DirAccess.make_dir_recursive_absolute(output)
	check(FileAccess.file_exists(source) and FileAccess.file_exists(baseline_path), "Missing preserved v1 checkpoint or exact baseline")
	if not FileAccess.file_exists(source) or not FileAccess.file_exists(baseline_path):
		quit(1)
		return
	var source_hash := FileAccess.get_sha256(source)
	var baseline: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(baseline_path))
	check(source_hash == str(baseline.source_sha256), "The v1 baseline belongs to a different checkpoint")
	checkpoint = output.path_join("arch-current-%d-%d.bin" % [OS.get_process_id(), Time.get_ticks_usec()])
	check(DirAccess.copy_absolute(source, checkpoint) == OK, "Could not copy preserved v1 checkpoint")
	var scene = await open_pond()
	check(scene.lab.get_error().is_empty(), "Current arch failed to resume: " + scene.lab.get_error())
	if not scene.lab.get_error().is_empty():
		scene.lab.stop()
		quit(1)
		return
	check(scene.stats.resumed and scene.stats.loaded_update == int(baseline.loaded_update), "The full v1 learner was restarted")
	check(scene.stats.adopted_update == int(baseline.adopted_update) and scene.stats.policy_inputs == 55, "The inherited validated actor changed during migration")
	check(scene.state.position == point(baseline.position), "Migration changed the saved body position")
	var orientation: Array = baseline.orientation
	check(scene.state.orientation == Quaternion(orientation[0], orientation[1], orientation[2], orientation[3]), "Migration changed the saved orientation")
	check(absf(scene.state.elapsed - float(baseline.elapsed)) < 0.00001 and scene.state.eaten == int(baseline.eaten), "Migration changed physical time or collection history")
	check(scene.state.foods.size() == baseline.foods.size() and scene.state.pellet_ids == PackedInt64Array(baseline.pellet_ids), "Migration changed the active food identities")
	for index in baseline.foods.size():
		check(scene.state.foods[index] == point(baseline.foods[index]), "Migration changed a saved food position")
	check(scene.stats.water_current_cm_s == 0 and not scene.current_switch.button_pressed, "A v1 calm save was silently given a current")
	check(scene.find_child("WaterControls", true, false).visible and scene.current_switch.size.y >= 44, "Current controls are unavailable")
	check(not scene.auto_feed_button.visible and not scene.feed_surface.visible, "The controlled arch task unexpectedly enables manual food")
	var before: Dictionary = scene.lab.get_state().duplicate(true)
	root.size = Vector2i(1280, 720)
	await settle()
	if DisplayServer.get_name() != "headless":
		await RenderingServer.frame_post_draw
		check(root.get_texture().get_image().save_png(output.path_join("current-calm.png")) == OK, "Calm water screenshot failed")
	await key_activate(scene.current_switch)
	scene._poll()
	check(is_equal_approx(scene.stats.water_current_cm_s, 0.5) and scene.current_switch.button_pressed, "Keyboard did not enable the gentle current")
	check(scene.state.position == before.position and scene.state.foods == before.foods and scene.state.elapsed == before.elapsed, "Changing the current moved or reset the body or food")
	check(scene.stats.context_version > 0, "Current interaction did not reach the learner context")
	var geometry: Dictionary = scene.lab.get_arch_geometry()
	var sample_points := PackedVector3Array([geometry.center + Vector3(geometry.major_radius, 0, 0), Vector3(0, -1, 2.5)])
	var velocities: PackedVector3Array = scene.lab.get_arch_flow_vectors(sample_points)
	check(velocities.size() == 2 and velocities[0] == Vector3.ZERO and velocities[1].length() > 0.00001, "Visual field does not respect the solid arch or active current")
	var flow_before: PackedVector3Array = scene.flow.tracer_positions.duplicate()
	var clock_ok := true
	var drifted := false
	for tick in 40:
		var old_time: float = scene.state.elapsed
		var step_foods: PackedVector3Array = scene.state.foods.duplicate()
		var step_ids: PackedInt64Array = scene.state.pellet_ids.duplicate()
		scene._physics_process(0.05)
		clock_ok = clock_ok and absf(scene.state.elapsed - old_time - 0.05) < 0.0001
		for index in step_ids.size():
			var current_index: int = scene.state.pellet_ids.find(step_ids[index])
			if current_index >= 0:
				drifted = drifted or step_foods[index].distance_to(scene.state.foods[current_index]) > 0.001
		if tick % 10 == 0:
			await process_frame
	scene._poll()
	check(clock_ok and (scene.state.position as Vector3).is_finite(), "Live current simulation reset its clock or became invalid")
	check(drifted, "Suspended food did not drift in the actual current")
	check(scene.flow.tracer_positions != flow_before and scene.flow.displacement.length() > 0, "Native tracers and water phase did not move")
	await create_timer(0.3).timeout
	check(scene.flow.strength > 0.8 and float(scene.flow.surface.get_shader_parameter("flow_strength")) > 0.8, "The water does not visibly respond to current activation")
	var tracers_clear := true
	for tracer: Vector3 in scene.flow.tracer_positions:
		tracers_clear = tracers_clear and tracer.is_finite() and arch_distance(tracer, geometry) >= -0.00001
	check(tracers_clear, "Native visual tracers entered solid rock")
	if DisplayServer.get_name() != "headless":
		await RenderingServer.frame_post_draw
		check(root.get_texture().get_image().save_png(output.path_join("current-gentle.png")) == OK, "Gentle current screenshot failed")
	before = scene.lab.get_state().duplicate(true)
	scene.current_strength.select(1)
	scene._select_strength(1)
	check(is_equal_approx(scene.stats.water_current_cm_s, 0.75), "The 0.75 cm/s control does not reach native physics")
	check(scene.state.position == before.position and scene.state.elapsed == before.elapsed, "Changing current strength reset the body")
	await key_activate(scene.current_switch)
	scene._poll()
	check(scene.stats.water_current_cm_s == 0 and scene.flow.current_m_s == Vector3.ZERO, "Turning current off did not restore calm water")
	await key_activate(scene.current_switch)
	scene._poll()
	check(is_equal_approx(scene.stats.water_current_cm_s, 0.75), "Current strength was lost when switching back on")
	var started := Time.get_ticks_msec()
	while (scene.stats.trained_context_version != scene.stats.context_version or int(scene.stats.get("evaluated_context_version", -1)) != scene.stats.context_version or scene.stats.update <= scene.stats.loaded_update) and Time.get_ticks_msec() - started < 130000 and str(scene.stats.learning_error).is_empty():
		await create_timer(0.1).timeout
		scene._poll()
	check(str(scene.stats.learning_error).is_empty() and scene.stats.update > scene.stats.loaded_update, "Migrated PPO optimizer did not continue")
	check(scene.stats.trained_context_version == scene.stats.context_version, "The new current did not reach background PPO at a rollout boundary")
	check(int(scene.stats.get("evaluated_context_version", -1)) == scene.stats.context_version, "Current-specific retention/candidate checks did not execute")
	check(scene.training_stats.text.contains("practiced") and scene.training_stats.text.contains("calm"), "Practice and retained skills are not explained")
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		await settle()
		scene.sidebar_scroll.scroll_vertical = 0
		await settle()
		check(scene.current_switch.size.y >= 44 and scene.current_strength.size.y >= 44, "Current actions are too small")
		check(scene.current_strength.get_global_rect().end.x <= dimensions.x + 1 and scene.viewport_container.size.y >= 100, "Current controls clip or pond collapses")
		if DisplayServer.get_name() != "headless":
			await RenderingServer.frame_post_draw
			check(root.get_texture().get_image().save_png(output.path_join("current-%dx%d.png" % [dimensions.x, dimensions.y])) == OK, "Current screenshot failed")
	scene.lab.set_paused(true)
	await create_timer(0.6).timeout
	scene._poll()
	before = scene.lab.get_state().duplicate(true)
	var saved_update: int = scene.stats.update
	var context_before: int = scene.stats.context_version
	scene.set_process(false)
	scene.lab.stop()
	scene.queue_free()
	await settle()
	scene = await open_pond()
	check(scene.stats.resumed and scene.stats.loaded_update >= saved_update, "Current-aware v2 full save did not resume")
	check(scene.state.position == before.position and scene.state.orientation == before.orientation and scene.state.foods == before.foods and scene.state.elapsed == before.elapsed, "Current-aware resume changed the physical world")
	check(is_equal_approx(scene.stats.water_current_cm_s, 0.75) and scene.current_switch.button_pressed and scene.current_strength.selected == 1, "Current control state did not persist across reopening")
	check(scene.stats.context_version == context_before, "The saved current context was replaced during startup")
	started = Time.get_ticks_msec()
	while (scene.stats.update <= scene.stats.loaded_update or scene.stats.trained_context_version != context_before) and Time.get_ticks_msec() - started < 15000 and str(scene.stats.learning_error).is_empty():
		await create_timer(0.1).timeout
		scene._poll()
	check(scene.stats.update > scene.stats.loaded_update and scene.stats.trained_context_version == context_before and str(scene.stats.learning_error).is_empty(), "Restored current-aware optimizer/worlds did not continue PPO")
	check(FileAccess.get_sha256(source) == source_hash, "The preserved original v1 save was overwritten")
	scene.set_process(false)
	scene.lab.stop()
	var summary := {"checks": checks, "failures": failures, "source_sha256": source_hash, "loaded_v1": baseline.loaded_update, "resumed_v2": scene.stats.loaded_update, "current_cm_s": scene.stats.water_current_cm_s, "context_version": scene.stats.context_version, "trained_context_version": scene.stats.trained_context_version, "evaluated_context_version": scene.stats.get("evaluated_context_version", -1), "population": population, "checkpoint": checkpoint}
	var report := FileAccess.open(output.path_join("summary.json"), FileAccess.WRITE)
	if report:
		report.store_string(JSON.stringify(summary, "\t"))
	print("ARCH_CURRENT_SMOKE " + JSON.stringify(summary))
	quit(1 if failures else 0)

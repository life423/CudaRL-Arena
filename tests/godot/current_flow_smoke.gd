extends SceneTree
var checks := 0
var failures := 0
var folder := "res://../build/drifting-food-checks/visual"
func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)
func settle() -> void:
	for tick in 8:
		await process_frame
func capture(label: String) -> void:
	if DisplayServer.get_name() != "headless":
		await RenderingServer.frame_post_draw
		check(root.get_texture().get_image().save_png(ProjectSettings.globalize_path(folder.path_join(label + ".png"))) == OK, "Capture failed")
func _initialize() -> void:
	Engine.max_fps = 60
	_run.call_deferred()
func _run() -> void:
	create_timer(90).timeout.connect(func(): quit(1))
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(folder))
	var scene = load("res://pond/latest_agent.tscn").instantiate()
	root.add_child(scene)
	current_scene = scene
	scene.set_physics_process(false)
	await settle()
	check(scene.lab.get_error().is_empty(), "Live pond failed: " + scene.lab.get_error())
	check(scene.flow.tracers.multimesh.instance_count == 96, "Tracer pool is unbounded or missing")
	scene.flow.set_process(false)
	# Read the actual backend state before testing bit-exact depth retention;
	# the host's initial-layout mirror can differ from CUDA by a few ULPs.
	scene.state = scene.lab.step()
	scene.previous = scene.state.duplicate(true)
	var before: Dictionary = scene.lab.get_state().duplicate(true)
	scene._set_current(true)
	check(scene.lab.get_state() == before, "Changing the water moved native state immediately")
	check(scene.stats.food_drifting, "The live current control leaves food stationary")
	scene.flow._process(1)
	var first: Vector3 = scene.flow.displacement
	scene.flow.advance(2)
	check((scene.flow.displacement - first).is_equal_approx(Vector3(0.25, 0, 0)), "Flow visuals do not use the physical speed and model scale")
	check(scene.lab.get_state() == before, "Rendering current changed physics")
	check(scene.flow.surface.get_shader_parameter("flow_strength") > 0.99, "Flow surface did not visibly enable")
	check(scene.flow.surface.get_shader_parameter("flow_offset").is_equal_approx(Vector2(scene.flow.displacement.x, scene.flow.displacement.z)), "Surface and tracer phases disagree")
	check(scene.lab.set_gentle_current(true, true), "Native drifting-food mode failed")
	scene._poll()
	for tick in 10:
		scene.state = scene.lab.step()
	for slot in 3:
		check(scene.state.foods[slot].distance_to(before.foods[slot]) > 0, "Suspended food did not drift")
		check(scene.state.foods[slot].y == before.foods[slot].y, "Horizontal current changed food depth")
	scene.previous = scene.state.duplicate(true)
	scene._poll()
	check(scene.stats.food_drifting and scene.stats.background_swimmers == 0, "Drifting food started training or was not reported")
	first = scene.flow.displacement
	scene._set_current(false)
	scene.flow.advance(2)
	scene.flow._process(1)
	check(scene.flow.displacement == first, "Current-off visuals kept drifting")
	check(scene.flow.surface.get_shader_parameter("flow_strength") < 0.01, "Current-off visual layer did not settle")
	check(not scene.stats.food_drifting and scene.stats.water_current_cm_s == 0, "Current-off left food advection enabled")
	scene._set_current(true)
	scene.released = true
	first = scene.flow.displacement
	var physical_time: float = scene.state.elapsed
	scene._physics_process(0.05)
	check(absf(scene.state.elapsed - physical_time - 0.05) < 0.0001, "Live physical clock did not advance")
	check((scene.flow.displacement - first).is_equal_approx(Vector3(0.00625, 0, 0)), "Visible flow is not coupled to actual simulation steps")
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		await settle()
		for close_up in [false, true]:
			scene.overview = not close_up
			scene._update_camera(5)
			var camera_name := "close" if close_up else "overview"
			for enabled in [false, true]:
				scene._set_current(enabled)
				scene.flow._process(2)
				scene.flow.advance(1 if enabled else 0)
				var native: Dictionary = scene.lab.get_state().duplicate(true)
				await settle()
				await capture("%s-%s-%dx%d" % [camera_name, "flow" if enabled else "calm", dimensions.x, dimensions.y])
				check(scene.lab.get_state() == native, "Camera or visual flow advanced the native world")
				check(scene.viewport_container.size.y >= 100 and scene.current_switch.size.y >= 44, "Flow view is unusable")
	print("CURRENT_FLOW backend=%s checks=%d failures=%d" % [scene.stats.compute_backend, checks, failures])
	quit(1 if failures else 0)

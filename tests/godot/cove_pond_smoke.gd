extends SceneTree

var checks := 0
var failures := 0
var source := "res://../build/cove-navigation/preserved-20261029113418/arch-v2-training.bin"
var baseline_path := "res://../build/cove-navigation/parent-baseline.json"
var output := "res://../build/cove-navigation/rendered"
var protected_source := ""
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

func point(values: Array) -> Vector3:
	return Vector3(values[0], values[1], values[2])

func open_pond():
	var scene = load("res://pond/cove_pond.tscn").instantiate()
	scene.population = population
	scene.source_arch_checkpoint = protected_source
	scene.cove_checkpoint = checkpoint
	root.add_child(scene)
	current_scene = scene
	scene.set_physics_process(false)
	await settle()
	scene._poll()
	return scene

func screenshot(name: String) -> void:
	if DisplayServer.get_name() == "headless":
		return
	await RenderingServer.frame_post_draw
	check(root.get_texture().get_image().save_png(output.path_join(name)) == OK, "Cove screenshot failed: " + name)

func run() -> void:
	create_timer(240).timeout.connect(func(): push_error("Cove smoke test timed out"); quit(1))
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--cove-source="):
			source = argument.trim_prefix("--cove-source=")
		elif argument.begins_with("--cove-baseline="):
			baseline_path = argument.trim_prefix("--cove-baseline=")
		elif argument.begins_with("--cove-output="):
			output = argument.trim_prefix("--cove-output=")
	source = ProjectSettings.globalize_path(source)
	baseline_path = ProjectSettings.globalize_path(baseline_path)
	output = ProjectSettings.globalize_path(output)
	DirAccess.make_dir_recursive_absolute(output)
	check(FileAccess.file_exists(source) and FileAccess.file_exists(baseline_path), "Missing real arch checkpoint or old-build baseline")
	if not FileAccess.file_exists(source) or not FileAccess.file_exists(baseline_path):
		quit(1)
		return
	var baseline: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(baseline_path))
	var source_hash := FileAccess.get_sha256(source)
	check(source_hash == str(baseline.source_sha256), "Cove baseline belongs to another arch save")
	var unique := "%d-%d" % [OS.get_process_id(), Time.get_ticks_usec()]
	protected_source = output.path_join("arch-source-%s.bin" % unique)
	checkpoint = output.path_join("cove-training-%s.bin" % unique)
	check(DirAccess.copy_absolute(source, protected_source) == OK, "Could not protect the source arch save")
	root.size = Vector2i(1280, 720)
	var scene = await open_pond()
	check(scene.lab.get_error().is_empty(), "Cove scene failed: " + scene.lab.get_error())
	if not scene.lab.get_error().is_empty():
		scene.lab.stop()
		quit(1)
		return
	scene.lab.set_paused(true)
	scene._poll()
	check(scene.stats.layout_id == 1 and scene.stats.policy_inputs == 55, "The cove habitat or inherited motor dimensions are wrong")
	check(scene.stats.loaded_update == int(baseline.loaded_update), "The full arch learner was restarted for the cove")
	check(scene.stats.adopted_update == int(baseline.adopted_update), "The source approved actor was replaced")
	check(scene.state.position == point(baseline.position), "Import moved the live body")
	var orientation: Array = baseline.orientation
	check(scene.state.orientation == Quaternion(orientation[0], orientation[1], orientation[2], orientation[3]), "Import changed body orientation")
	check(absf(scene.state.elapsed - float(baseline.elapsed)) < 0.00001 and scene.state.eaten == int(baseline.eaten), "Import reset time or food history")
	check(scene.state.pellet_ids == PackedInt64Array(baseline.pellet_ids), "Import replaced active pellets")
	for index in baseline.foods.size():
		check(scene.state.foods[index] == point(baseline.foods[index]), "Import relocated existing food")
	var geometry: Dictionary = scene.lab.get_cove_geometry()
	check(geometry.layout_id == 1 and scene.native_outline.size() >= 128, "The native cove outline is missing")
	var outline_samples: PackedVector3Array = scene.lab.sample_cove_terrain(scene.native_outline)
	var contour_matches := true
	var polygon := PackedVector2Array()
	var max_x := -100.0
	for index in scene.native_outline.size():
		contour_matches = contour_matches and absf(outline_samples[index].x) < 0.0005
		var p: Vector3 = scene.native_outline[index]
		polygon.append(Vector2(p.x, p.z))
		max_x = maxf(max_x, p.x)
	check(contour_matches and max_x > 7.5, "Rendered water is not the native outward cove contour")
	var cove_center: Vector3 = geometry.cove_center
	check(Geometry2D.is_point_in_polygon(Vector2(cove_center.x, cove_center.z), polygon), "Water mesh does not include the cove")
	check(scene.water_surface.mesh is ArrayMesh and scene.water_surface.scale == Vector3.ONE, "Cove still uses the old scaled circular water")
	check(scene.shore.scale == Vector3.ONE and scene.shore.native_outline.size() == scene.native_outline.size(), "Dry shore decoration still uses the old ellipse")
	# The explicit cached arrays are the native source of every rendered vertex.
	var positions: PackedVector3Array = scene.native_terrain_positions
	var samples: PackedVector3Array = scene.native_terrain_samples
	check(positions.size() == samples.size() and samples.size() > 10000, "Native basin mesh samples are incomplete")
	var terrain_mesh: ArrayMesh = null
	for node in scene.world.get_children():
		if node is MeshInstance3D and node.material_override == scene.terrain_material:
			terrain_mesh = node.mesh
	check(terrain_mesh != null, "Cove terrain was not rendered")
	if terrain_mesh:
		var vertices: PackedVector3Array = terrain_mesh.surface_get_arrays(0)[Mesh.ARRAY_VERTEX]
		var heights_match := vertices.size() == samples.size()
		for index in range(0, mini(vertices.size(), samples.size()), 53):
			heights_match = heights_match and absf(vertices[index].y - samples[index].x) < 0.00001
		check(heights_match, "Rendered bed heights differ from the native samples")
	var before: Dictionary = scene.lab.get_state().duplicate(true)
	if scene.current_switch.button_pressed:
		await key_activate(scene.current_switch)
	await key_activate(scene.current_switch)
	scene.current_strength.select(0)
	scene._select_strength(0)
	scene._poll()
	check(is_equal_approx(scene.stats.water_current_cm_s, 0.5), "Cove current controls do not apply gentle flow")
	check(scene.state.position == before.position and scene.state.elapsed == before.elapsed and scene.state.foods == before.foods, "Current controls reset the imported world")
	var probe := PackedVector3Array([Vector3(cove_center.x, -1.3, cove_center.z)])
	var velocity: PackedVector3Array = scene.lab.get_arch_flow_vectors(probe)
	var advected: PackedVector3Array = scene.lab.advect_arch_tracers(probe, 0.05)
	check(velocity[0].length() > 0.001 and velocity[0].length() <= 0.005001 and advected[0].distance_to(probe[0]) > 0.001, "Native current/tracer APIs still clip the cove to the old pond")
	var has_cove_tracer := false
	for p: Vector3 in scene.flow.tracer_positions:
		has_cove_tracer = has_cove_tracer or p.x > 6.9
	check(has_cove_tracer, "Native visual tracers never occupy the outward cove")
	var first_foods: PackedVector3Array = scene.state.foods.duplicate()
	var clock_ok := true
	var finite := true
	var cove_food_seen := false
	var cove_visit := false
	var anchor: Vector3 = scene.state.position
	var still := 0
	var max_still := 0
	for tick in 1200:
		var old_time: float = scene.state.elapsed
		scene._physics_process(0.05)
		clock_ok = clock_ok and absf(scene.state.elapsed - old_time - 0.05) < 0.00025
		finite = finite and (scene.state.position as Vector3).is_finite()
		for p: Vector3 in scene.state.foods:
			cove_food_seen = cove_food_seen or p.x > 4.8
		cove_visit = cove_visit or (scene.state.position as Vector3).x > 4.8
		if (scene.state.position as Vector3).distance_to(anchor) > 0.1:
			anchor = scene.state.position
			still = 0
		else:
			still += 1
			max_still = maxi(max_still, still)
		if tick % 40 == 0:
			await process_frame
	scene._poll()
	check(clock_ok and finite, "Cove physical continuation reset its clock or became invalid")
	check(scene.state.foods != first_foods, "Food is static despite active current")
	check(cove_food_seen, "Automatic targets never use the cove")
	check(scene.flow.strength > 0.8 and scene.flow.tracer_positions.size() == 96, "Cove water/current visual failed")
	# Actual initial controller outcomes are recorded; skill gaps are not hidden
	# by teleporting it or substituting a trained QA actor.
	var initial_outcome := {"cove_food_seen":cove_food_seen,"cove_visit":cove_visit,"max_immobility_seconds":max_still*0.05,"foods_collected":int(scene.state.eaten)-int(before.eaten)}
	scene.lab.set_paused(false)
	var started := Time.get_ticks_msec()
	while (scene.stats.update <= scene.stats.loaded_update or int(scene.stats.get("evaluated_context_version", -1)) != scene.stats.context_version) and Time.get_ticks_msec() - started < 150000 and str(scene.stats.learning_error).is_empty():
		await create_timer(0.1).timeout
		scene._poll()
	check(str(scene.stats.learning_error).is_empty() and scene.stats.update > scene.stats.loaded_update, "Full inherited optimizer did not continue PPO")
	check(scene.stats.trained_context_version == scene.stats.context_version and int(scene.stats.get("evaluated_context_version", -1)) == scene.stats.context_version, "Cove context did not reach practice and candidate evaluation")
	check(scene.stats.learner_weights_changed, "Cove PPO does not update its inherited actor")
	var dimensions_list := [Vector2i(1280,720),Vector2i(768,1024),Vector2i(390,844),Vector2i(844,390)]
	for dimensions in dimensions_list:
		root.size = dimensions
		await settle()
		scene.sidebar_scroll.scroll_vertical = 0
		check(scene.viewport_container.size.y >= 100 and scene.current_strength.get_global_rect().end.x <= dimensions.x+1, "Cove viewport or current controls clip")
		check(scene.camera_button.size.y >= 44 and scene.pause_button.size.y >= 44, "Cove actions are too small")
		await screenshot("cove-%dx%d.png" % [dimensions.x,dimensions.y])
	await key_activate(scene.camera_button)
	check(not scene.overview, "Keyboard closer view failed")
	await key_activate(scene.camera_button)
	await key_activate(scene.details_button)
	check(scene.details_group.visible and scene.training_stats.text.contains("cove"), "Cove AI details are missing")
	root.size = Vector2i(1280,720)
	await settle()
	await screenshot("cove-details.png")
	scene.lab.set_paused(true)
	await create_timer(0.6).timeout
	scene._poll()
	before = scene.lab.get_state().duplicate(true)
	var saved_update: int = scene.stats.update
	var context: int = scene.stats.context_version
	scene.set_process(false)
	scene.lab.stop()
	scene.queue_free()
	await settle()
	scene = await open_pond()
	check(scene.stats.resumed and scene.stats.loaded_update >= saved_update and scene.stats.layout_id == 1, "Cove full save did not resume")
	check(scene.state.position == before.position and scene.state.orientation == before.orientation and scene.state.foods == before.foods and scene.state.elapsed == before.elapsed, "Cove resume changed the visible world")
	check(is_equal_approx(scene.stats.water_current_cm_s,0.5) and scene.stats.context_version == context, "Cove current/layout context did not persist")
	started = Time.get_ticks_msec()
	while scene.stats.update <= scene.stats.loaded_update and Time.get_ticks_msec()-started < 20000 and str(scene.stats.learning_error).is_empty():
		await create_timer(0.1).timeout
		scene._poll()
	check(scene.stats.update > scene.stats.loaded_update and str(scene.stats.learning_error).is_empty(), "Restored cove PPO did not continue")
	check(FileAccess.get_sha256(source) == source_hash and FileAccess.get_sha256(protected_source) == source_hash, "Source arch save was modified")
	scene.set_process(false)
	scene.lab.stop()
	var summary := {"checks":checks,"failures":failures,"source_sha256":source_hash,"loaded_parent":baseline.loaded_update,"resumed_cove":scene.stats.loaded_update,"update":scene.stats.update,"population":population,"initial_controller":initial_outcome,"checkpoint":checkpoint}
	var report := FileAccess.open(output.path_join("summary.json"),FileAccess.WRITE)
	report.store_string(JSON.stringify(summary,"\t"))
	report.close()
	print("COVE_POND_SMOKE "+JSON.stringify(summary))
	quit(0 if failures==0 else 1)

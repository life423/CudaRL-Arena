extends SceneTree

# Run with --overhang-source=<real saved boulder learner>. Only a unique private
# copy is opened by the viewer; neither the caller's source nor a personal save
# is ever used as a destination. A real renderer adds four viewport captures.
var source := ""
var output := "res://../build/predator-shelter/overhang-live"
var scene_path := "res://pond/overhang_pond.tscn"
var private_source := ""
var checks := 0
var failures := 0
var measurements: Array[Dictionary] = []
const QA_TITLE := "QA · Rock ledge checks · closes automatically"

func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)

func _initialize() -> void:
	Engine.max_fps = 60
	root.title = QA_TITLE
	run.call_deferred()

func freeze_isolated_scene(node: Node) -> void:
	node.set_physics_process(false)
	node.set_process(false)
	root.title = QA_TITLE

func isolate_menu_scene(node: Node) -> void:
	# SceneTree.node_added runs before the PackedScene root's _ready. The
	# actual menu/router must receive the private source before opening CUDA.
	if node.name == "OverhangPond":
		node.source_boulder_checkpoint = private_source
		node.set_physics_process(false)
		node.set_process(false)
		node.ready.connect(freeze_isolated_scene.bind(node), CONNECT_ONE_SHOT)
	elif node.name == "DemoMenu":
		node.ready.connect(func(): root.title = QA_TITLE, CONNECT_ONE_SHOT)
	elif node.name == "BoulderPond":
		# A route regression must fail in isolation rather than resume a personal
		# learner before the scene-name assertion can detect the wrong destination.
		node.source_cove_checkpoint = private_source + ".missing-cove"
		node.boulder_checkpoint = private_source + ".wrong-route"
		node.population = 0
	elif node.name == "FeedingPond":
		node.feeding_checkpoint_path = private_source + ".wrong-feeding-route"
		node.population = 0

func settle(frames := 6) -> void:
	for tick in frames:
		await process_frame

func same_world(left: Dictionary, right: Dictionary) -> bool:
	for key in ["position", "orientation", "links", "foods", "pellet_velocities", "pellet_ids", "elapsed", "eaten", "seconds_since_food"]:
		if not left.has(key) or not right.has(key) or left[key] != right[key]:
			return false
	return true

func frozen_stats(stats: Dictionary) -> bool:
	return int(stats.get("layout_id", -1)) == 3 and int(stats.get("policy_inputs", 0)) == 55 \
		and int(stats.get("background_swimmers", -1)) == 0 and not stats.get("learning_active", true) \
		and not stats.get("learner_weights_changed", true) and int(stats.get("evaluation_count", -1)) == 0

func food_is_clear(pond) -> bool:
	var foods: PackedVector3Array = pond.state.get("foods", PackedVector3Array())
	if foods.is_empty():
		return false
	var solids: PackedFloat32Array = pond.lab.sample_overhang_solid(foods)
	var terrain: PackedVector3Array = pond.lab.sample_cove_terrain(foods)
	if solids.size() != foods.size() or terrain.size() != foods.size():
		return false
	var geometry: Dictionary = pond.arch_geometry
	var center: Vector3 = geometry.center
	var yaw := float(geometry.yaw)
	var cosine := cos(yaw)
	var sine := sin(yaw)
	# Native pellets have physical radius .0008 m / .04 m per model unit.
	# The larger rendered sphere is presentation, never collision clearance.
	const radius := 0.02
	for index in foods.size():
		var position := foods[index]
		if not position.is_finite() or position.y > -radius + 0.001 or position.y < terrain[index].x + radius - 0.001:
			return false
		if solids[index] < radius - 0.001 or position.distance_to(geometry.boulder_center) < float(geometry.boulder_radius) + radius - 0.001:
			return false
		var relative := position - center
		var local := Vector3(cosine * relative.x - sine * relative.z, relative.y, sine * relative.x + cosine * relative.z)
		var radial := Vector2(local.x, local.y).length() - float(geometry.major_radius)
		if Vector2(radial, local.z).length() - float(geometry.tube_radius) < radius - 0.001:
			return false
	return true

func open_view(packed: PackedScene, path: String):
	var pond = packed.instantiate()
	pond.source_boulder_checkpoint = path
	pond.set_physics_process(false)
	root.add_child(pond)
	current_scene = pond
	pond.set_physics_process(false)
	pond.set_process(false)
	root.title = QA_TITLE
	await settle()
	pond._poll()
	return pond

func close_view(pond) -> void:
	pond.lab.stop()
	pond.queue_free()
	await process_frame

func automatic_loop(pond, label: String) -> void:
	var before: Dictionary = pond.lab.get_state()
	pond.set_process(true)
	pond.set_physics_process(true)
	# Exercise the real inherited _process / _physics_process chain, including
	# interpolation, flow rendering, polling and native stepping. No manual step.
	await create_timer(2.0).timeout
	pond.set_physics_process(false)
	pond.set_process(false)
	pond._poll()
	var after: Dictionary = pond.lab.get_state()
	var elapsed := float(after.get("elapsed", 0)) - float(before.get("elapsed", 0))
	check(elapsed >= 1.5, label + ": automatic scene physics did not advance for two wall-clock seconds")
	check((after.get("position", Vector3(INF, INF, INF)) as Vector3).is_finite() and not same_world(before, after), label + ": automatic scene state froze or became invalid")
	check(pond.lab.get_error().is_empty() and frozen_stats(pond.stats), label + ": automatic scene loop failed or started learning")
	check(food_is_clear(pond), label + ": automatic scene food is not physically valid")
	measurements.append({"automatic_scene_loop":label, "wall_seconds":2.0, "native_seconds":elapsed, "no_manual_steps":true})

func menu_roundtrip(pond, initial: Dictionary) -> void:
	var router = root.get_node("DemoLaunch")
	node_added.connect(isolate_menu_scene)
	var back: Button = pond.find_child("BackToDemos", true, false)
	check(back != null, "The live ledge has no Pond menu return action")
	if back == null:
		await close_view(pond)
		return
	back.pressed.emit()
	await settle()
	check(current_scene != null and current_scene.name == "DemoMenu", "Pond menu did not return from the live ledge")
	if current_scene == null or current_scene.name != "DemoMenu":
		return
	var menu = current_scene
	var action: Button = menu.find_child("pond_overhang_watch", true, false)
	var local_route_available: bool = router.primary_pond() == "pond_boulder" and FileAccess.file_exists("user://pond-policies/boulder-navigation-v1/training.bin")
	check((action != null) == local_route_available, "The menu rock ledge action disagrees with local source availability")
	if action:
		check(action.text == "View rock ledge" and action.size.y >= 44, "The menu ledge action is ambiguous or too small")
		action.pressed.emit()
	else:
		# Hosts without a personal learner still exercise the actual router and
		# private checkpoint; no test fabricates or writes a user:// learner.
		check(router.open_demo("pond_overhang", true) == OK, "The developer ledge route could not reopen")
	await settle()
	var reopened = current_scene
	check(reopened != null and reopened.name == "OverhangPond", "Menu route reopened the old boulder learner instead of the ledge")
	if reopened == null or reopened.name != "OverhangPond":
		return
	check(reopened.source_boulder_checkpoint == private_source, "Menu reopening bypassed the private source override")
	check(reopened.lab.get_error().is_empty() and same_world(reopened.state, initial), "Menu reopening failed or altered the immutable source frame")
	await automatic_loop(reopened, "menu reopen")
	await close_view(reopened)
	node_added.disconnect(isolate_menu_scene)

func capture_viewports(pond) -> void:
	for dimensions in [Vector2i(1280,720), Vector2i(768,1024), Vector2i(390,844), Vector2i(844,390)]:
		root.size = dimensions
		await settle()
		pond._fit()
		pond.overview = true
		pond._update_camera(1.0)
		pond._apply_pose(1.0)
		await settle()
		check(pond.layout.get_global_rect().end.x <= dimensions.x + 1, "Rock ledge layout overflows horizontally at " + str(dimensions))
		check(pond.viewport_container.size.x >= 100 and pond.viewport_container.size.y >= 100, "Rock ledge viewport collapsed at " + str(dimensions))
		check(pond.pause_button.size.y >= 44 and pond.camera_button.size.y >= 44, "Rock ledge actions are too small at " + str(dimensions))
		check(pond.rock_overhang != null and pond.rock_overhang.visible and pond.rock_overhang.get_child_count() == 2, "The live pond does not show both native ledge solids")
		if DisplayServer.get_name() != "headless":
			await RenderingServer.frame_post_draw
			var image: Image = root.get_texture().get_image()
			check(not image.is_empty() and image.get_size() == dimensions, "Cannot capture the rendered rock ledge view")
			check(image.save_png(output.path_join("live-%dx%d.png" % [dimensions.x, dimensions.y])) == OK, "Rock ledge screenshot failed")
		measurements.append({"viewport":str(dimensions), "rendered":DisplayServer.get_name() != "headless", "live_scene":true})

func run() -> void:
	create_timer(150).timeout.connect(func(): push_error("Overhang live smoke timed out"); quit(1))
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--overhang-source="):
			source = argument.trim_prefix("--overhang-source=")
		elif argument.begins_with("--overhang-output="):
			output = argument.trim_prefix("--overhang-output=")
		elif argument.begins_with("--overhang-scene="):
			scene_path = argument.trim_prefix("--overhang-scene=")
	check(not source.is_empty() and FileAccess.file_exists(source), "Pass --overhang-source=<saved boulder learner>; no personal checkpoint is opened by default")
	check(ClassDB.class_has_method("DeepPondLab", "watch_overhang") and ClassDB.class_has_method("DeepPondLab", "sample_overhang_solid"), "The live overhang native API is unavailable")
	if failures:
		quit(1)
		return
	source = ProjectSettings.globalize_path(source)
	output = ProjectSettings.globalize_path(output)
	var source_bytes := FileAccess.get_file_as_bytes(source)
	var source_hash := FileAccess.get_sha256(source)
	check(not source_bytes.is_empty(), "The protected boulder source is empty")
	var unique := "%d-%d" % [OS.get_process_id(), Time.get_ticks_usec()]
	var private_directory := output.path_join("case-" + unique)
	check(DirAccess.make_dir_recursive_absolute(private_directory) == OK, "Cannot create an isolated overhang test directory")
	private_source = private_directory.path_join("source-boulder.bin")
	check(DirAccess.copy_absolute(source, private_source) == OK, "Cannot copy the boulder source for the overhang test")
	check(FileAccess.get_file_as_bytes(private_source) == source_bytes, "The private source copy differs before opening")
	if failures:
		quit(1)
		return
	var original_files := DirAccess.get_files_at(private_directory)
	# Establish the native initial state independently of scene presentation.
	# Loading the same immutable source twice must reproduce the physical frame.
	var native = ClassDB.instantiate("DeepPondLab")
	check(native.watch_overhang(private_source, "cuda"), "Native overhang could not open the copied learner: " + native.get_error())
	if failures:
		quit(1)
		return
	var initial: Dictionary = native.get_state()
	var initial_stats: Dictionary = native.get_stats()
	check(initial.get("ready", false) and float(initial.get("elapsed", 0)) > 0, "A real saved physical frame was not retained")
	check(frozen_stats(initial_stats), "The new native habitat must be frozen 55-input inference with no practice worlds")
	check(int(initial_stats.get("loaded_update", 0)) > 0 and int(initial_stats.get("adopted_update", 0)) > 0 and int(initial_stats.get("policy_update", 0)) > 0 and int(initial_stats.get("selector_update", 0)) > 0, "Trained motor/selector provenance is missing")
	native.stop()
	# Do not preload or statically extend the pond: its scripts refer to the
	# DemoLaunch autoload, which must already be available before parsing them.
	await settle(1)
	check(root.get_node_or_null("DemoLaunch") != null, "The project router autoload did not initialize")
	var packed := load(scene_path) as PackedScene
	check(packed != null, "Cannot load the latest rock ledge PackedScene")
	if packed == null or failures:
		quit(1)
		return
	var pond = await open_view(packed, private_source)
	check(pond.lab.get_error().is_empty(), "The live rock ledge scene failed: " + pond.lab.get_error())
	check(same_world(pond.state, initial), "Opening the scene changed the source physical frame or food")
	check(pond.title_status.text.contains("Live") and pond.curve.status_caption.contains("no learning"), "Live physics and frozen neural weights are not explained honestly")
	check(not pond.curve.show_best_score and not pond.curve.watch_button.visible and not pond.find_child("SaveTrainingNow", true, false).visible, "Frozen overhang view exposes training/publication controls")
	check(not bool(pond.arch_geometry.get("hiding_validated", true)), "The new object must not claim learned hiding")
	check(pond.lab.get_evaluation_history(0).is_empty(), "The geometry viewer inherited a misleading learning curve")
	if not pond.lab.get_error().is_empty() or pond.state.is_empty():
		await close_view(pond)
		quit(1)
		return
	var before: Dictionary = pond.lab.get_state()
	pond.current_strength.select(1)
	pond.current_switch.set_pressed_no_signal(true)
	pond._set_current(true)
	check(same_world(before, pond.lab.get_state()), "Current switching reset or advanced the physical world")
	check(absf(float(pond.stats.get("water_current_cm_s", 0)) - 0.75) < 0.00001, "The visible current control did not change the real water")
	pond.state = pond.lab.get_state()
	pond._toggle_pause()
	var paused: Dictionary = pond.lab.get_state()
	for tick in 5:
		check(same_world(paused, pond.lab.step()), "Paused native body, food or physical clock changed")
	check(pond.paused and pond.stats.get("paused", false) and pond.pause_button.text == "Resume pond", "Pause control does not describe the actual simulation")
	pond._toggle_pause()
	check(same_world(paused, pond.lab.get_state()), "Resuming changed the world before a physical step")
	var distance := 0.0
	var maximum_displacement := 0.0
	var clock_ok := true
	var finite := true
	var food_valid := true
	var last_position: Vector3 = pond.state.position
	var start_time := float(pond.state.elapsed)
	# Native elapsed is exported from steps * float(dt); allow its float ULP
	# at the inherited clock without permitting a missing or reset tick.
	var time_tolerance := maxf(0.0002, (absf(start_time) + 30.0) * 0.0000002)
	for tick in 600:
		pond.previous = pond.state.duplicate(true)
		pond.state = pond.lab.step()
		var position: Vector3 = pond.state.get("position", Vector3(INF, INF, INF))
		distance += position.distance_to(last_position)
		maximum_displacement = maxf(maximum_displacement, position.distance_to(initial.position))
		last_position = position
		clock_ok = clock_ok and absf(float(pond.state.elapsed) - start_time - (tick + 1) * 0.05) < time_tolerance
		finite = finite and position.is_finite() and is_finite(float(pond.state.get("speed_cm_s", INF)))
		if tick % 20 == 0:
			food_valid = food_valid and food_is_clear(pond)
			pond._poll()
			await process_frame
	pond.previous = pond.state.duplicate(true)
	pond._poll()
	check(clock_ok and finite, "The live simulation reset its clock or produced an invalid body")
	check(distance > 0.1 and maximum_displacement > 0.05, "The inherited neural controller did not move the live body")
	check(food_valid, "Live food entered the new native ledge, old rocks, terrain or air")
	check(frozen_stats(pond.stats) and int(pond.stats.update) == int(initial_stats.update) and int(pond.stats.adopted_update) == int(initial_stats.adopted_update), "The geometry viewer trained or replaced either inherited controller")
	check(not str(pond.stats.get("compute_detail", "")).contains("concurrent PPO"), "Frozen runtime metadata falsely reports concurrent training")
	pond.lab.save_training_now()
	await capture_viewports(pond)
	measurements.append({"simulated_seconds":float(pond.state.elapsed) - start_time, "path_length_model":distance, "maximum_displacement_model":maximum_displacement, "food_delta":int(pond.state.eaten) - int(initial.eaten), "learner_update":int(pond.stats.update), "visible_actor_update":int(pond.stats.adopted_update)})
	await automatic_loop(pond, "direct scene")
	await menu_roundtrip(pond, initial)
	# An unavailable source must show a recoverable error, never a frozen pond.
	var unavailable = await open_view(packed, private_directory.path_join("missing-source.bin"))
	check(unavailable.title_status.text == "Rock ledge view could not start" and not unavailable.feed_feedback.text.is_empty(), "Unavailable overhang does not show a clear startup error")
	check(not unavailable.viewport_container.visible and not unavailable.curve.visible and not unavailable.camera_button.visible, "Unavailable overhang displays a frozen game or misleading graph")
	check(unavailable.pause_button.disabled and unavailable.current_switch.disabled and unavailable.current_strength.disabled, "Unavailable overhang leaves unusable playback controls enabled")
	check(unavailable.readiness_message != null and unavailable.readiness_message.visible and unavailable.readiness_message.text.contains("Pond menu"), "Unavailable overhang lacks an obvious return route")
	await close_view(unavailable)
	var source_unchanged := FileAccess.get_file_as_bytes(source) == source_bytes and FileAccess.get_sha256(source) == source_hash
	var private_unchanged := FileAccess.get_file_as_bytes(private_source) == source_bytes and FileAccess.get_sha256(private_source) == source_hash
	check(source_unchanged, "The caller's boulder save changed")
	check(private_unchanged, "The read-only viewer changed its copied source bytes")
	check(DirAccess.get_files_at(private_directory) == original_files, "The frozen viewer created a checkpoint, lease or sidecar beside its source")
	var report := FileAccess.open(output.path_join("summary-" + unique + ".json"), FileAccess.WRITE)
	check(report != null, "Cannot write the isolated smoke report")
	var summary := {"checks":checks, "failures":failures, "source_sha256":source_hash, "source_bytes_preserved":source_unchanged, "private_bytes_preserved":private_unchanged, "scene":scene_path, "live_frozen_inference":true, "learning_updates":false, "hiding_validated":false, "measurements":measurements}
	if report:
		report.store_string(JSON.stringify(summary, "\t"))
		report.close()
	print("OVERHANG_LIVE_SMOKE " + JSON.stringify(summary))
	quit(1 if failures else 0)

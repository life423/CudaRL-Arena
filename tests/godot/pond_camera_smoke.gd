extends SceneTree

var checks := 0
var failures := 0
func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)
func settle() -> void:
	for i in 6:
		await process_frame
func action(window: Window, name: String) -> void:
	var event := InputEventAction.new()
	event.action = name
	event.pressed = true
	window.push_input(event)
	event = InputEventAction.new()
	event.action = name
	event.pressed = false
	window.push_input(event)
	await process_frame
func key(window: Window, code: Key) -> void:
	var event := InputEventKey.new()
	event.keycode = code
	event.window_id = window.get_window_id() if window.get_window_id() >= 0 else root.get_window_id()
	event.pressed = true
	Input.parse_input_event(event)
	event = InputEventKey.new()
	event.keycode = code
	event.window_id = window.get_window_id() if window.get_window_id() >= 0 else root.get_window_id()
	event.pressed = false
	Input.parse_input_event(event)
	await process_frame
func select_camera(scene: Node, index: int) -> void:
	scene.camera_selector.get_popup().hide()
	scene.camera_selector.select(index)
	scene.camera_selector.item_selected.emit(index)
	await settle()
func capture(name: String) -> void:
	await RenderingServer.frame_post_draw
	root.get_texture().get_image().save_png(ProjectSettings.globalize_path("res://../build/pond-learning-checks/" + name + ".png"))
func body_in_frame(pond: Node, bounds: Rect2) -> bool:
	var local_points: Array[Vector3] = [Vector3(0.26, 0, 0), Vector3(0, 0.18, 0.19), Vector3(0, -0.17, -0.19)]
	for i in 17:
		var t := float(i) / 16.0
		var point: Vector3 = pond._tail_point(t, float(pond.current.joint0), float(pond.current.joint1))
		local_points.append(point + Vector3.UP * 0.22)
		local_points.append(point - Vector3.UP * 0.22)
	for point in local_points:
		var world: Vector3 = pond.creature.to_global(point)
		if pond.camera.is_position_behind(world) or not bounds.has_point(pond.camera.unproject_position(world)):
			return false
	return true
func _initialize() -> void:
	Engine.max_fps = 60
	_run.call_deferred()
func _run() -> void:
	create_timer(90).timeout.connect(func(): quit(1))
	var checkpoint := "user://tests/camera-%d-%d/best.json" % [OS.get_process_id(), Time.get_ticks_usec()]
	DirAccess.make_dir_recursive_absolute(checkpoint.get_base_dir())
	var file := FileAccess.open(checkpoint, FileAccess.WRITE)
	file.store_string(FileAccess.get_file_as_string("res://../docs/experiments/swimmer/shuffled-input/fresh-layouts-update-17425/shuffled.json"))
	file.close()
	node_added.connect(func(node: Node):
		if node.name == "TadpolePond" and node.shuffle_food_inputs:
			node.best_policy_path = checkpoint)
	change_scene_to_file("res://pond/shuffle_compare.tscn")
	await settle()
	var scene = current_scene
	var learner = scene.learner_pond
	var limit := Time.get_ticks_msec() + 15000
	while not learner.lab.get_stats().training_ready and Time.get_ticks_msec() < limit:
		await create_timer(0.025).timeout
	learner.warmup_timer.stop()
	scene._pause()
	await create_timer(0.2).timeout
	check(learner.lab.get_error().is_empty() and learner.lab.get_stats().training_ready, "Camera fixture failed to initialize")
	var ponds: Array = [scene.reference_pond, learner]
	for pond in ponds:
		pond.set_physics_process(false)
		pond.set_process(false)
	check(scene.camera_selector.selected == 0 and learner.close_camera, "The first view must introduce close follow")
	var original_state: Dictionary = learner.lab.get_state()
	var original_stats: Dictionary = learner.lab.get_stats()
	# Exercise the real OptionButton/PopupMenu with keyboard input.
	scene.camera_selector.grab_focus()
	await action(root, "ui_accept")
	check(scene.camera_selector.get_popup().visible, "Keyboard did not open camera choices")
	await key(scene.camera_selector.get_popup(), KEY_DOWN)
	await key(scene.camera_selector.get_popup(), KEY_ENTER)
	await settle()
	check(scene.camera_selector.selected == 1 and not learner.close_camera, "Keyboard did not choose pond overview")
	await select_camera(scene, 0)
	var follow_span := 0.0
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		await settle()
		if scene.tabs.visible:
			scene.tabs.current_tab = 1
			await settle()
		check(scene.camera_selector.is_visible_in_tree() and scene.camera_selector.size.y >= 44, "Camera choice is hidden or too small")
		check(scene.camera_selector.get_global_rect().end.x <= dimensions.x, "Camera choice overflows")
		await select_camera(scene, 0)
		for index in 2:
			if scene.tabs.visible:
				scene.tabs.current_tab = index
				await settle()
			var pond = ponds[index]
			var frame := Rect2(Vector2(8, 8), Vector2(scene.viewports[index].size) - Vector2(16, 16))
			var camera_basis: Basis = pond.camera.basis
			# Pond extremes, all headings, straight tails and fully folded joints.
			for i in 24:
				var bearing := float(i) * TAU / 24.0
				var pose: Dictionary = pond.current.duplicate()
				pose.x = cos(bearing) * 3.9
				pose.y = sin(bearing) * 2.8
				pose.heading = bearing
				pose.joint0 = 1.2 if i % 3 == 0 else (-1.2 if i % 3 == 1 else 0.0)
				pose.joint1 = -1.2 if i % 2 == 0 else 1.2
				pond.current = pose
				pond._apply_pose(pose)
				pond.camera.reset_subject()
				pond._update_camera(0, pose)
				check(body_in_frame(pond, frame), "Close camera clipped anatomy at %s, pose %d" % [str(dimensions), i])
				check(pond.camera.basis.is_equal_approx(camera_basis), "Turning the animal rotated the camera")
			# Restore a representative physical pose for image review.
			var hero: Dictionary = pond.lab.get_state()
			hero.x = 0.25 if index == 0 else 1.0
			hero.y = -0.15
			hero.heading = 0.35
			hero.joint0 = -0.55
			hero.joint1 = 0.85
			pond.current = hero
			pond._apply_pose(hero)
			pond.camera.reset_subject()
			pond._update_camera(0, hero)
			var head: Vector2 = pond.camera.unproject_position(pond.creature.to_global(Vector3(0.25, 0, 0)))
			var tip: Vector2 = pond.camera.unproject_position(pond.creature.to_global(pond._tail_point(1, hero.joint0, hero.joint1)))
			if dimensions == Vector2i(1280, 720) and index == 1:
				follow_span = head.distance_to(tip)
		if scene.tabs.visible:
			scene.tabs.current_tab = 1
			await settle()
		await capture("camera-follow-%dx%d" % [dimensions.x, dimensions.y])
		await select_camera(scene, 1)
		for index in 2:
			if scene.tabs.visible:
				scene.tabs.current_tab = index
				await settle()
			var pond = ponds[index]
			check(pond.camera.projection == Camera3D.PROJECTION_ORTHOGONAL, "Overview lost the orthographic whole-pond view")
			var fixed: Transform3D = pond.camera.transform
			pond.camera.follow(Vector3(-3.8, 0.25, 2.2), 0.1)
			check(pond.camera.transform == fixed, "Overview moved with the animal")
			var frame := Rect2(Vector2.ZERO, Vector2(scene.viewports[index].size))
			for piece in pond.foods:
				check(frame.has_point(pond.camera.unproject_position(piece.global_position)), "Overview hid food")
			if dimensions == Vector2i(1280, 720) and index == 1:
				var head: Vector2 = pond.camera.unproject_position(pond.creature.to_global(Vector3(0.25, 0, 0)))
				var tip: Vector2 = pond.camera.unproject_position(pond.creature.to_global(pond._tail_point(1, pond.current.joint0, pond.current.joint1)))
				check(follow_span > head.distance_to(tip) * 2.0, "Close view did not make the animal meaningfully larger")
		if scene.tabs.visible:
			scene.tabs.current_tab = 1
			await settle()
		await capture("camera-overview-%dx%d" % [dimensions.x, dimensions.y])
	check(learner.lab.get_state() == original_state, "Camera changed native body/food state")
	check(learner.lab.get_stats().update == original_stats.update and learner.lab.get_stats().env_steps == original_stats.env_steps, "Camera changed the training counters")
	check(learner.lab.get_stats().watching_best == original_stats.watching_best, "Camera changed policy selection")
	scene.queue_free()
	await settle()
	print("POND_CAMERA_SMOKE %d checks, %d failures" % [checks, failures])
	quit(0 if failures == 0 else 1)

extends SceneTree
var failures := 0
var checks := 0
func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)
func settle() -> void:
	for i in 6:
		await process_frame
func _initialize() -> void:
	Engine.max_fps = 60
	_run.call_deferred()
func _run() -> void:
	var checkpoint := "user://tests/shuffle-%d.json" % OS.get_process_id()
	node_added.connect(func(node: Node):
		if node.name == "TadpolePond" and node.shuffle_food_inputs:
			node.best_policy_path = checkpoint)
	create_timer(70).timeout.connect(func(): quit(1))
	var source_hash := FileAccess.get_sha256("res://pond/policies/shuffle_reference.json")
	var existing_hash := FileAccess.get_sha256("user://pond-policies/regional-v1/best.json")
	change_scene_to_file("res://pond/shuffle_compare.tscn")
	await settle()
	var scene = current_scene
	check(scene.reference_pond.ready_policy, "Frozen checkpoint failed to start")
	check(scene.learner_pond.ready_policy, "Shuffled learner failed to start")
	check(scene.reference_pond.foods.size() == 3 and scene.learner_pond.foods.size() == 3, "Both ponds must contain three foods")
	var reference: Object = scene.reference_pond.lab
	var learner: Object = scene.learner_pond.lab
	check(not reference.get_stats().training and reference.get_stats().background_swimmers == 0, "Reference must have no trainer")
	check(reference.get_stats().best_update == 29125, "Reference was replaced")
	check(learner.get_stats().shuffled_inputs and learner.get_stats().background_swimmers == 1024, "Challenger needs shuffled CUDA training")
	check(str(learner.get_stats().checkpoint_path).contains("tests/shuffle-"), "Test checkpoint isolation failed")
	var started := Time.get_ticks_msec()
	while learner.get_stats().update < 50 and Time.get_ticks_msec() - started < 30000:
		await create_timer(0.05).timeout
	check(learner.get_stats().update >= 50 and not learner.get_stats().finished, "Continuous shuffled training did not advance")
	var saved: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(checkpoint))
	check(saved.get("training_variant", "") == "shuffled", "Saved challenger lost its training-method metadata")
	check(learner.get_error().is_empty() and reference.get_error().is_empty(), "Native error in comparison")
	check(scene.curve.points.size() >= 2, "Comparison curve has no evaluations")
	check(absf(scene.curve.reference - 13.828125) < 0.00001, "Graph baseline differs from the frozen checkpoint")
	scene.pause_button.pressed.emit()
	await create_timer(0.25).timeout
	var update: int = learner.get_stats().update
	var elapsed: float = reference.get_state().elapsed
	await create_timer(0.25).timeout
	check(learner.get_stats().update == update, "Pause did not stop updates")
	check(reference.get_state().elapsed > elapsed, "Pausing training incorrectly paused reference playback")
	var before: Dictionary = learner.get_state()
	scene.watch_button.button_pressed = true
	check(learner.get_stats().watching_best, "Candidate-best playback did not activate")
	check(learner.get_state().x == before.x and learner.get_state().foods == before.foods, "Policy switch teleported the pond")
	scene.watch_button.button_pressed = false
	check(not learner.get_stats().watching_best, "Latest learner playback did not restore")
	scene.camera_selector.select(1)
	scene.camera_selector.item_selected.emit(1)
	check(not scene.learner_pond.close_camera and not scene.reference_pond.close_camera, "Overview selection did not apply to both ponds")
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		await settle()
		check(scene.tabs.visible == (dimensions.x < 900), "Compact view selector is incorrect")
		check(scene.controls.get_global_rect().end.x <= dimensions.x, "Comparison controls overflow horizontally")
		check(scene.pause_button.size.y >= 44 and scene.watch_button.size.y >= 44, "Control targets are too small")
		for index in 2:
			if scene.tabs.visible:
				scene.tabs.current_tab = index
				await settle()
			check(scene.panels[index].is_visible_in_tree(), "Pond cannot be selected")
			check(scene.viewports[index].size.x > 200 and scene.viewports[index].size.y > 100, "Pond viewport collapsed")
			check(scene.panels[index].get_global_rect().end.x <= dimensions.x, "Pond panel overflows horizontally")
			var pond = scene.reference_pond if index == 0 else scene.learner_pond
			var bounds := Rect2(Vector2.ZERO, Vector2(scene.viewports[index].size))
			check(bounds.has_point(pond.camera.unproject_position(pond.creature.global_position)), "Comparison camera lost the tadpole")
			for piece in pond.foods:
				check(bounds.has_point(pond.camera.unproject_position(piece.global_position)), "Comparison camera lost food")
		if DisplayServer.get_name() != "headless":
			await RenderingServer.frame_post_draw
			root.get_texture().get_image().save_png(ProjectSettings.globalize_path("res://../build/pond-learning-checks/shuffle-%dx%d.png" % [dimensions.x, dimensions.y]))
	check(FileAccess.get_sha256("res://pond/policies/shuffle_reference.json") == source_hash, "Frozen reference file was modified")
	check(FileAccess.get_sha256("user://pond-policies/regional-v1/best.json") == existing_hash, "Existing champion checkpoint was modified")
	scene.pause_button.pressed.emit()
	await create_timer(0.25).timeout
	check(learner.get_stats().update > update, "Resume did not restart learning")
	var escape := InputEventKey.new()
	escape.keycode = KEY_ESCAPE
	escape.pressed = true
	root.push_input(escape)
	await settle()
	check(current_scene.name == "DemoMenu" and not learner.get_stats().training, "Exit did not stop the training worker")
	check(Engine.physics_ticks_per_second == 60, "Comparison did not restore physics tick rate")
	current_scene.find_child("pond_shuffle_train", true, false).pressed.emit()
	await settle()
	check(current_scene.name == "ShuffleComparison", "Menu cannot reopen shuffled comparison")
	current_scene.find_child("BackToDemos", true, false).pressed.emit()
	await settle()
	check(current_scene.name == "DemoMenu", "Back to demos failed")
	print("SHUFFLE_COMPARE_SMOKE %d checks, %d failures" % [checks, failures])
	quit(0 if failures == 0 else 1)

extends SceneTree

var checks := 0
var failures := 0

func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)

func _initialize() -> void:
	Engine.max_fps = 120
	_run.call_deferred()

func _run() -> void:
	create_timer(120.0).timeout.connect(func():
		push_error("Live pond learning checks timed out")
		quit(1))
	var lab := TadpoleLab.new()
	var checkpoint := "user://tests/pond-best-%d-%d/best.json" % [OS.get_process_id(), Time.get_ticks_usec()]
	check(not lab.start_training(0, 1, 0), "Invalid swimmer count was accepted")
	check(lab.start_training(1024, 1, 0, checkpoint), "Fresh CUDA training did not start")
	var initial := lab.get_stats()
	check(initial.training and initial.background_swimmers == 1024, "Background population is not 1,024")
	check(initial.food_count == 3 and initial.policy_inputs == 13, "The learner does not see all three foods")
	check(lab.get_state().foods.size() == 3, "Three stable food positions are required")
	check(initial.update == 0 and initial.policy_update == 0 and initial.paused, "Training must start paused and untrained")
	for tick in 100:
		lab.step()
	check(lab.get_state().eaten == 0, "Initial policy unexpectedly behaves like the saved trained controller")
	await create_timer(0.1).timeout
	check(lab.get_stats().update == 0, "Paused initialization performed optimizer steps")
	lab.set_paused(false)
	var start := Time.get_ticks_msec()
	while lab.get_stats().update < 350 and Time.get_ticks_msec() - start < 60000:
		lab.step()
		await create_timer(0.025).timeout
	var before := lab.get_state()
	var after := lab.step()
	var stats := lab.get_stats()
	check(stats.update >= 350 and stats.policy_update >= 350, "Learning did not continue past the old 300-update cap")
	check(not stats.finished, "Continuous learning incorrectly reported a finished run")
	var evaluations := lab.get_evaluation_history()
	check(stats.best_saved and FileAccess.file_exists(checkpoint), "Best evaluated policy was not saved to disk")
	var saved: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(checkpoint))
	check(saved.update == stats.best_update and saved.parameters.size() == 1570, "Saved weights and best-update metadata disagree")
	check(is_equal_approx(saved.food_per_minute, stats.best_food_per_minute), "Saved score disagrees with displayed best")
	check(evaluations.size() >= 15 and evaluations[0].update == 0, "Graph omitted its untrained baseline or periodic measurements")
	for i in evaluations.size():
		var sample: Dictionary = evaluations[i]
		check(sample.count == 64 and is_equal_approx(sample.seconds, 60.0), "Evaluation protocol changed between graph samples")
		check(sample.update == i * 25 and sample.successes >= 0 and sample.successes <= sample.count, "Evaluation point has an invalid update or success count")
		check(is_equal_approx(sample.food_per_minute, float(sample.foods) / 64.0), "Graph score is not actual food collected per minute")
		check(sample.best_food_per_minute >= sample.food_per_minute - 0.00001, "Best score fell below the current score")
		if i > 0:
			check(sample.best_food_per_minute >= evaluations[i - 1].best_food_per_minute, "Best graph decreased")
	check(evaluations[0].foods == 0 and evaluations[-1].food_per_minute > evaluations[0].food_per_minute, "Measured foraging did not improve over the untrained baseline")
	check(lab.get_evaluation_history(evaluations.size()).is_empty(), "Graph history cursor repeated old samples")
	start = Time.get_ticks_msec()
	while lab.get_stats().update <= stats.update and Time.get_ticks_msec() - start < 2000:
		await create_timer(0.025).timeout
	lab.step()
	check(lab.get_stats().policy_update > stats.policy_update, "Visible policy stopped receiving new updates")
	check(stats.env_steps == int(stats.update) * 1024 * 64, "GPU experience count is incorrect")
	check(Vector2(after.x - before.x, after.y - before.y).length() < 0.4, "Applying a policy teleported the tadpole")
	var stable_foods := true
	var distant_replacements := true
	for tick in 1200:
		var prior := lab.get_state()
		var result := lab.step()
		if result.ate:
			var consumed: int = result.eaten_slot
			var old: Vector2 = prior.foods[consumed]
			var replacement: Vector2 = result.foods[consumed]
			var old_region := (1 if old.x >= 0 else 0) + (2 if old.y >= 0 else 0)
			var new_region := (1 if replacement.x >= 0 else 0) + (2 if replacement.y >= 0 else 0)
			distant_replacements = distant_replacements and old.distance_to(replacement) >= 2.6 - 0.0001 and old_region != new_region
		for slot in 3:
			if not result.ate or slot != int(result.eaten_slot):
				stable_foods = stable_foods and result.foods[slot] == prior.foods[slot]
	check(stable_foods, "An uneaten food piece moved")
	check(distant_replacements, "Eaten food respawned nearby or in the same region")
	check(lab.get_state().eaten > 0, "The learning policy did not collect food")
	check(lab.get_error().is_empty(), "Background CUDA training failed")
	print("POND_LEARNING update=%d applied=%d food=%d" % [stats.update, stats.policy_update, lab.get_state().eaten])
	lab.stop()
	var record: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(checkpoint))
	var record_score: float = record.food_per_minute
	check(not lab.get_stats().training and not lab.get_state().ready, "Stopping did not release the training session")
	check(lab.start_training(1024, 1, 0, checkpoint, false, false), "Restart did not create a fresh session")
	check(lab.get_stats().update == 0 and lab.get_state().eaten == 0, "Restart reused learned state")
	check(lab.get_stats().best_saved and is_equal_approx(lab.get_stats().best_food_per_minute, record_score), "Restart lost the saved best")
	check(lab.set_watch_best(true), "Saved best could not be selected")
	var watch_before := lab.get_state()
	var watch_after := lab.step()
	check(Vector2(watch_after.x - watch_before.x, watch_after.y - watch_before.y).length() < 0.4, "Selecting best teleported the tadpole")
	check(lab.get_stats().watching_best and lab.get_stats().policy_update == record.update, "Wrong saved policy was applied")
	for tick in 1200:
		lab.step()
	check(lab.get_state().eaten > 0 and lab.get_stats().update == 0, "Saved best playback failed or secretly trained")
	check(lab.set_watch_best(false), "Could not return to the current learner")
	lab.step()
	check(lab.get_stats().policy_update == 0, "Returning to current did not restore its untrained policy")
	lab.stop()
	var viewport := SubViewport.new()
	viewport.own_world_3d = true
	viewport.size = Vector2i(1280, 720)
	viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	root.add_child(viewport)
	var scene: Node3D = load("res://pond/pond.tscn").instantiate()
	scene.training_updates_per_second = 0.0
	scene.best_policy_path = checkpoint
	viewport.add_child(scene)
	for frame in 8:
		await process_frame
	check(scene.live_learning and scene.ready_policy, "The 3D pond did not default to live learning")
	check(scene.lab.get_stats().background_swimmers == 1024, "The scene started the wrong training population")
	check(scene.lab.get_stats().update == 0, "Scene skipped the initial untrained view")
	scene.lab.set_paused(false)
	start = Time.get_ticks_msec()
	while scene.lab.get_stats().policy_update < 100 and Time.get_ticks_msec() - start < 30000:
		await create_timer(0.025).timeout
	check(scene.lab.get_stats().policy_update >= 100, "Live 3D policy did not advance")
	check(scene.background_value.text.contains("1,024 swimmers") and scene.status.text.contains("update"), "Learning status is unclear")
	check(scene.foods.size() == 3, "The scene does not render all three food pieces")
	for piece in scene.foods:
		check(piece.visible, "A food piece is hidden")
	check(scene.curve_view.points.size() >= 5 and scene.curve_view.points[0].x == 0, "Scene graph is not displaying actual evaluation history")
	check(scene.curve_view.best_points.size() == scene.curve_view.points.size(), "Best and current curves are not aligned")
	check(scene.curve_view.best_saved and scene.curve_view.best_score >= scene.curve_view.current_score, "Current-versus-best UI is incorrect")
	scene.curve_view.watch_button.button_pressed = true
	await create_timer(0.1).timeout
	check(scene.lab.get_stats().watching_best and scene.status.text.begins_with("Best view"), "Watch best control failed")
	var training_before: int = scene.lab.get_stats().update
	start = Time.get_ticks_msec()
	while scene.lab.get_stats().update <= training_before and Time.get_ticks_msec() - start < 5000:
		await create_timer(0.025).timeout
	check(scene.lab.get_stats().update > training_before, "Watching best stopped background learning")
	scene.curve_view.watch_button.grab_focus()
	var watch_key := InputEventKey.new()
	watch_key.keycode = KEY_SPACE
	watch_key.pressed = true
	viewport.push_input(watch_key)
	await process_frame
	watch_key = InputEventKey.new()
	watch_key.keycode = KEY_SPACE
	watch_key.pressed = false
	viewport.push_input(watch_key)
	await process_frame
	check(not scene.lab.get_stats().watching_best, "Keyboard could not switch back to the current learner")
	scene.pause_button.pressed.emit()
	await create_timer(0.2).timeout
	var paused_at: int = scene.lab.get_stats().update
	await create_timer(0.2).timeout
	check(scene.lab.get_stats().update == paused_at and scene.status.text.begins_with("Paused"), "Pause learning failed")
	scene.pause_button.grab_focus()
	var space := InputEventKey.new()
	space.keycode = KEY_SPACE
	space.pressed = true
	viewport.push_input(space)
	await process_frame
	space = InputEventKey.new()
	space.keycode = KEY_SPACE
	space.pressed = false
	viewport.push_input(space)
	start = Time.get_ticks_msec()
	while scene.lab.get_stats().update <= paused_at and Time.get_ticks_msec() - start < 5000:
		await create_timer(0.025).timeout
	check(not scene.user_paused and scene.lab.get_stats().update > paused_at, "Keyboard activation did not resume learning")
	scene.maximum_speed.button_pressed = false
	scene.speed_slider.value = 5
	check(scene.training_updates_per_second == 5.0 and scene.speed_slider.editable, "Paced speed control failed")
	scene.maximum_speed.button_pressed = true
	check(scene.training_updates_per_second == 0.0 and not scene.speed_slider.editable, "Maximum speed control failed")
	scene.set_physics_process(false)
	for tick in 1200:
		scene._physics_process(0.05)
	check(float(scene.current.elapsed) >= 60.0, "Visible rate window did not run for a full minute")
	check(scene.food_rate_value.text == "Food / min  %d" % scene.food_times.size(), "Visible food rate is not the actual rolling-minute count")
	check(not scene.move_button.visible, "Three-food learning exposes an automatic target-moving shortcut")
	scene.set_physics_process(true)
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		viewport.size = dimensions
		for frame in 8:
			await process_frame
		await RenderingServer.frame_post_draw
		viewport.get_texture().get_image().save_png(ProjectSettings.globalize_path("res://../build/pond-learning-checks/learning-%dx%d.png" % [dimensions.x, dimensions.y]))
		check(scene.restart_button.get_global_rect().end.x <= dimensions.x, "Start over button overflows")
		check(scene.side_panel.get_global_rect().end.x <= dimensions.x, "Stats panel overflows the viewport")
		check(scene.curve_view.get_global_rect().end.y <= dimensions.y, "Graph overflows the viewport")
		check(not scene.pond_container.get_global_rect().intersects(scene.curve_view.get_global_rect()), "Graph overlaps the pond")
		var pond_bounds := Rect2(Vector2.ZERO, Vector2(scene.pond_viewport.size))
		check(pond_bounds.has_point(scene.camera.unproject_position(scene.creature.global_position)), "Pond view lost the tadpole")
		for piece in scene.foods:
			check(pond_bounds.has_point(scene.camera.unproject_position(piece.global_position)), "Pond view lost a food piece")
		for button in [scene.pause_button, scene.restart_button, scene.maximum_speed]:
			check(button.size.y >= 44, "Training control is too small for touch")
		check(scene.curve_view.watch_button.size.y >= 44 and scene.curve_view.watch_button.get_global_rect().end.x <= dimensions.x, "Watch best control is clipped or too small")
		if scene.ui_layout.vertical:
			check(scene.restart_button.get_global_rect().end.y <= scene.side_panel.get_global_rect().end.y, "Compact stats clip the restart control")
			check(scene.settings_button.get_global_rect().end.y <= scene.side_panel.get_global_rect().end.y, "Compact stats clip the training settings control")
			scene.settings_button.pressed.emit()
			await process_frame
			check(scene.settings_panel.visible, "Compact training settings did not open")
			scene.settings_button.pressed.emit()
		print("POND_LEARNING_RENDER %dx%d" % [dimensions.x, dimensions.y])
	scene.restart_button.pressed.emit()
	check(scene.lab.get_stats().update == 0 and scene.current.eaten == 0, "Start over did not reset learning and the pond")
	check(scene.curve_view.points.is_empty(), "Start over retained the previous learning graph")
	check(scene.lab.get_stats().best_saved, "Start over deleted the saved best")
	var scene_lab: Object = scene.lab
	scene.queue_free()
	for frame in 4:
		await process_frame
	check(not scene_lab.get_stats().training, "Closing the scene left CUDA training running")
	print("POND_LEARNING_SMOKE %d checks, %d failures" % [checks, failures])
	quit(0 if failures == 0 else 1)

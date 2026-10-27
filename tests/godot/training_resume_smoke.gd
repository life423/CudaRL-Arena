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
func ready_lab(lab: Object) -> void:
	var limit := Time.get_ticks_msec() + 15000
	while not lab.get_stats().training_ready and lab.get_error().is_empty() and Time.get_ticks_msec() < limit:
		await create_timer(0.025).timeout
	check(lab.get_error().is_empty() and lab.get_stats().training_ready, "Training initialization/resume failed: " + lab.get_error())
func make_scene(path: String):
	var scene = load("res://pond/pond.tscn").instantiate()
	scene.best_policy_path = path
	scene.shuffle_food_inputs = true
	scene.training_updates_per_second = 0
	root.add_child(scene)
	return scene
func _run() -> void:
	create_timer(90).timeout.connect(func(): quit(1))
	var directory := "user://tests/resume-scene-%d-%d" % [OS.get_process_id(), Time.get_ticks_usec()]
	DirAccess.make_dir_recursive_absolute(directory)
	var path := directory + "/best.json"
	var fixture := "res://../docs/experiments/swimmer/shuffled-input/fresh-layouts-update-17425/shuffled.json"
	var file := FileAccess.open(path, FileAccess.WRITE)
	file.store_string(FileAccess.get_file_as_string(fixture))
	file.close()
	var scene = make_scene(path)
	await settle()
	await ready_lab(scene.lab)
	scene.warmup_timer.stop()
	check(scene.lab.get_stats().imported_best and scene.lab.get_stats().update == 17425, "One-time upgrade did not retain the learned actor/update")
	check(scene.lab.get_stats().best_food_per_minute >= 15.34375, "Import lost the saved best score")
	scene.lab.step()
	check(scene.lab.get_stats().policy_update == 17425, "Imported actor was labelled as untrained")
	scene.lab.set_paused(false)
	var limit := Time.get_ticks_msec() + 15000
	while scene.lab.get_stats().update < 17455 and Time.get_ticks_msec() < limit:
		await create_timer(0.025).timeout
	scene.pause_button.pressed.emit()
	await create_timer(0.25).timeout
	var saved_update: int = scene.lab.get_stats().update
	scene.lab.save_training_now()
	limit = Time.get_ticks_msec() + 10000
	while scene.lab.get_stats().training_saved_update < saved_update and Time.get_ticks_msec() < limit:
		await create_timer(0.025).timeout
	check(saved_update >= 17455 and scene.lab.get_stats().training_saved_update == saved_update, "Full save missed the latest optimizer boundary")
	check(FileAccess.file_exists(path + ".training"), "Full training file was not created")
	var samples: int = scene.curve_view.points.size()
	scene.queue_free()
	await settle()
	scene = make_scene(path)
	await settle()
	await ready_lab(scene.lab)
	scene.warmup_timer.stop()
	var stats: Dictionary = scene.lab.get_stats()
	check(stats.resumed_training and not stats.imported_best, "Reopen repeated actor-only migration")
	check(stats.update == saved_update and stats.paused, "Reopen restarted the counter or trained before resuming")
	check(scene.curve_view.points.size() >= samples, "Reopen lost the learning graph")
	check(scene.training_save.text.contains("Training saved"), "Saved-state feedback is missing")
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		await settle()
		check(scene.training_save.get_global_rect().end.x <= dimensions.x, "Training-save label overflows horizontally")
		check(scene.curve_view.get_global_rect().end.y <= dimensions.y, "Resumed curve overflows vertically")
		check(not scene.pond_container.get_global_rect().intersects(scene.curve_view.get_global_rect()), "Resumed curve covers the pond")
		check(scene.restart_button.size.y >= 44 and scene.pause_button.size.y >= 44, "Resume controls are too small")
		if DisplayServer.get_name() != "headless":
			await RenderingServer.frame_post_draw
			root.get_texture().get_image().save_png(ProjectSettings.globalize_path("res://../build/pond-learning-checks/resume-%dx%d.png" % [dimensions.x, dimensions.y]))
	# A deliberate reset archives the full prior run while preserving best.json.
	scene.restart_button.pressed.emit()
	await ready_lab(scene.lab)
	scene.warmup_timer.stop()
	check(scene.lab.get_stats().update == 0 and not scene.lab.get_stats().resumed_training, "Explicit new run did not reset learning")
	check(scene.lab.get_stats().best_saved and scene.lab.get_stats().best_food_per_minute >= 15.34375, "New run deleted the strongest playback actor")
	var archived := false
	for name in DirAccess.get_files_at(directory):
		archived = archived or name.begins_with("best.json.training.before-reset-")
	check(archived, "New run overwrote the old full training state without archiving it")
	scene.queue_free()
	await settle()
	# A damaged full save must stop resuming and retain the bytes.
	file = FileAccess.open(path + ".training", FileAccess.WRITE)
	file.store_string("damaged training save")
	file.close()
	var lab := TadpoleLab.new()
	check(lab.start_training(1024, 1, 0, path, true), "Resume request could not be created")
	limit = Time.get_ticks_msec() + 10000
	while lab.get_error().is_empty() and Time.get_ticks_msec() < limit:
		await create_timer(0.025).timeout
	check(not lab.get_error().is_empty() and not lab.get_stats().training_ready, "Damaged save silently became a fresh learner")
	lab.stop()
	check(FileAccess.get_file_as_string(path + ".training") == "damaged training save", "Damaged save was overwritten on failed resume")
	print("TRAINING_RESUME_SMOKE %d checks, %d failures" % [checks, failures])
	quit(0 if failures == 0 else 1)

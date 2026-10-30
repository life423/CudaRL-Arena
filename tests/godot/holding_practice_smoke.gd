extends SceneTree

var checks := 0
var failures := 0
var output := ""
var rendered := false

func check(ok: bool, message: String) -> void:
	checks += 1
	if not ok:
		failures += 1
		push_error(message)

func _initialize() -> void:
	Engine.max_fps = 60
	run.call_deferred()

func settle(frames := 4) -> void:
	for tick in frames:
		await process_frame

func wait_ready(pond) -> bool:
	for tick in 250:
		if not pond.lab.get_error().is_empty():
			return false
		if pond.lab.get_stats().get("learning_ready", false):
			return true
		await create_timer(0.02).timeout
	return false

func scene_at(source: String, checkpoint: String):
	var pond = load("res://pond/holding_pond.tscn").instantiate()
	pond.source_motor_run = source
	pond.holding_checkpoint = checkpoint
	pond.population = 1024
	pond.ready.connect(func():
		pond.set_physics_process(false)
		pond.paused = true
		pond.lab.set_paused(true), CONNECT_ONE_SHOT)
	root.add_child(pond)
	current_scene = pond
	return pond

func sync_view(pond) -> void:
	pond.state = pond.lab.get_state().duplicate(true)
	pond.previous = pond.state.duplicate(true)
	pond._poll()
	pond._apply_pose(1.0)
	pond._update_camera(1.0)

func same_pose(a: Dictionary, b: Dictionary, label: String) -> void:
	for field in ["position", "orientation", "elapsed", "eaten", "foods", "pellet_velocities", "pellet_ids", "goal_position", "safe_hold_seconds", "longest_safe_hold_seconds", "body_clearance", "goal_distance_cm"]:
		check(a.has(field) and b.has(field) and a[field] == b[field], label + ": " + field + " changed")
	check(a.get("links", []) == b.get("links", []), label + ": articulated links changed")

func capture(pond, dimensions: Vector2i) -> void:
	root.size = dimensions
	await settle(6)
	pond._fit()
	pond._update_camera(1.0)
	await settle(4)
	var back := pond.find_child("BackToDemos", true, false) as Button
	check(back != null and not back.disabled, "Working pond menu remains reachable")
	if back:
		pond.sidebar_scroll.ensure_control_visible(back)
		back.grab_focus()
		await settle(2)
		var bounds := back.get_global_rect()
		check(back.size.y >= 44 and bounds.position.x >= -1 and bounds.end.x <= dimensions.x + 1, "Menu target clips horizontally or is too small")
		check(bounds.position.y >= -1 and bounds.end.y <= dimensions.y + 1, "Menu target is outside the window")
		check(root.gui_get_focus_owner() == back, "Menu keyboard focus is missing")
	check(pond.current_switch.disabled and pond.current_strength.disabled, "First holding lesson must remain explicitly calm")
	check(not pond.find_child("WaterControls", true, false).visible, "Unsupported current controls should not compete with the lesson")
	check(pond.goal_ring.visible and pond.goal_ring.position == pond.state.goal_position, "Goal marker must follow the actual native task")
	check("Assigned task" in pond.feed_feedback.text and "Body PPO" in pond.training_heading.text, "Assigned body practice must not imply learned predator avoidance")
	check(pond.curve.get_script().resource_path.ends_with("holding_curve.gd") and "seconds" in pond.curve.plot_caption, "Holding data must use seconds, not food/minute")
	if rendered:
		await RenderingServer.frame_post_draw
		var name := "holding-%dx%d" % [dimensions.x, dimensions.y]
		check(root.get_texture().get_image().save_png(output.path_join(name + ".png")) == OK, "Window capture failed")
		check(pond.viewport.get_texture().get_image().save_png(output.path_join(name + "-3d.png")) == OK, "Physical pond capture failed")

func run() -> void:
	create_timer(150).timeout.connect(func(): push_error("Holding smoke test timed out"); quit(1))
	rendered = DisplayServer.get_name() != "headless"
	var base := ProjectSettings.globalize_path("res://../build/predator-shelter/")
	output = base.path_join("holding-visual")
	check(DirAccess.make_dir_recursive_absolute(output) == OK, "Could not create holding QA folder")
	var source := base.path_join("motor-v4-training.bin")
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--holding-source="):
			source = argument.substr("--holding-source=".length())
	var source_hash := FileAccess.get_sha256(source)
	check(not source_hash.is_empty(), "Qualified full motor source is required")
	if failures:
		quit(1)
		return
	var private_dir := output.path_join("private-%d-%d" % [OS.get_process_id(), Time.get_ticks_usec()])
	check(DirAccess.make_dir_recursive_absolute(private_dir) == OK, "Could not isolate holding saves")
	var private_source := private_dir.path_join("motor-source.bin")
	check(DirAccess.copy_absolute(source, private_source) == OK, "Could not create a protected source copy")
	var checkpoint := private_dir.path_join("training.bin")
	var pond = scene_at(private_source, checkpoint)
	check(await wait_ready(pond), "Actual qualified holding session must start: " + pond.lab.get_error())
	if failures:
		pond.queue_free()
		await process_frame
		quit(1)
		return
	var initial_stats: Dictionary = pond.lab.get_stats()
	check(initial_stats.get("hold_practice_mode", false), "Native holding mode must be active")
	check(initial_stats.get("background_swimmers", 0) == 1024, "The full source population must be retained")
	check(pond.lab.get_error().is_empty(), "Native holding session has an error")
	var before: Dictionary = pond.lab.get_state().duplicate(true)
	check(before.get("goal_kind", "") == "cover", "The motor lesson requires an explicit assigned cover goal")
	for tick in 120:
		pond.lab.step()
	sync_view(pond)
	var after: Dictionary = pond.lab.get_state().duplicate(true)
	check(absf(float(after.elapsed) - float(before.elapsed) - 6.0) < 0.002, "Physical time must advance by fixed simulation steps")
	check((after.position as Vector3).is_finite(), "Visible body must remain physically finite")
	check(after.safe_hold_seconds >= 0 and after.longest_safe_hold_seconds >= after.safe_hold_seconds, "Native hold counters are inconsistent")
	check(float(after.longest_safe_hold_seconds) >= 5.0, "The actual visible lesson must demonstrate a five-second hold before being opened for the user")
	check(after.goal_position == before.goal_position, "The task point must not chase the creature")
	check(pond.goal_ring.position == after.goal_position, "Marker does not match the native target")
	# Resume genuine PPO, then stop at a rollout boundary before measuring save
	# continuation. No script writes a body transform or learned joint command.
	var starting_update := int(initial_stats.get("update", 0))
	pond._toggle_pause()
	var trained := false
	for tick in 500:
		if int(pond.lab.get_stats().get("update", 0)) >= starting_update + 2:
			trained = true
			break
		await create_timer(0.02).timeout
	check(trained, "Background PPO did not advance at least two genuine updates")
	check(pond.lab.get_stats().get("learner_weights_changed", false), "PPO update counters alone must not substitute for actual actor changes")
	pond._toggle_pause()
	await create_timer(0.6).timeout
	sync_view(pond)
	var current_stats: Dictionary = pond.lab.get_stats()
	check(str(current_stats.get("learning_error", "")).is_empty(), "Background holding learner failed")
	pond.lab.save_training_now()
	var saved := false
	for tick in 250:
		var saved_stats: Dictionary = pond.lab.get_stats()
		if FileAccess.file_exists(checkpoint) and int(saved_stats.get("saved_update", -1)) >= int(saved_stats.get("update", 0)):
			saved = true
			break
		await create_timer(0.02).timeout
	check(saved, "Full holding checkpoint did not save while paused")
	var resume_dir := private_dir.path_join("resume")
	check(DirAccess.make_dir_recursive_absolute(resume_dir) == OK, "Could not create isolated resume copy")
	var resume_checkpoint := resume_dir.path_join("training.bin")
	check(DirAccess.copy_absolute(checkpoint, resume_checkpoint) == OK, "Could not copy complete holding checkpoint")
	var saved_pose: Dictionary = pond.lab.get_state().duplicate(true)
	for tick in 20:
		pond.lab.step()
	var expected: Dictionary = pond.lab.get_state().duplicate(true)
	pond.lab.stop()
	pond.queue_free()
	await process_frame
	# Resume is standalone: the original motor source is deliberately unavailable.
	pond = scene_at(private_dir.path_join("missing-source.bin"), resume_checkpoint)
	check(await wait_ready(pond), "Saved holding session must resume without its external source: " + pond.lab.get_error())
	if pond.lab.get_error().is_empty():
		var resumed_stats: Dictionary = pond.lab.get_stats()
		check(resumed_stats.get("resumed", false), "Full resume must be reported honestly")
		check(int(resumed_stats.get("loaded_update", 0)) >= starting_update + 2, "Resumed optimizer counter was lost")
		same_pose(saved_pose, pond.lab.get_state(), "Saved visible state")
		for tick in 20:
			pond.lab.step()
		same_pose(expected, pond.lab.get_state(), "Normalized physical continuation")
		sync_view(pond)
		for dimensions in [Vector2i(1280,720), Vector2i(768,1024), Vector2i(390,844), Vector2i(844,390)]:
			await capture(pond, dimensions)
	pond.lab.stop()
	pond.queue_free()
	await process_frame
	check(FileAccess.get_sha256(source) == source_hash, "Source training run must remain unchanged")
	check(FileAccess.get_sha256(private_source) == source_hash, "Private read-only source was modified")
	var summary := {"checks":checks, "failures":failures, "rendered":rendered, "actual_learning":trained, "source_sha256":source_hash, "private_directory":private_dir}
	var report := FileAccess.open(output.path_join("summary.json"), FileAccess.WRITE)
	if report:
		report.store_string(JSON.stringify(summary, "\t"))
		report.close()
	print("HOLDING_PRACTICE_SMOKE " + JSON.stringify(summary))
	quit(1 if failures else 0)

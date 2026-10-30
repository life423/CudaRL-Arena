extends SceneTree

# UI/runtime verification only. These repeats of registered reference fixtures
# never change the reference study, preflight gate or any saved controller.
var checks := 0
var failures := 0
var output := ""
var require_paired := false
var protected_hashes: Dictionary = {}

func check(ok: bool, message: String) -> void:
	checks += 1
	if not ok:
		failures += 1
		push_error(message)

func _initialize() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--output="):
			output = arg.trim_prefix("--output=")
		elif arg == "--require-paired":
			require_paired = true
	run.call_deferred()

func frames(count := 6) -> void:
	for tick in count:
		await process_frame

func run() -> void:
	create_timer(40.0).timeout.connect(func(): push_error("Motor bench smoke timed out"); quit(1))
	check(ClassDB.class_exists("MotorControlLab"), "Isolated native motor bench class must load")
	check(ClassDB.class_has_method("MotorControlLab", "get_controller_info"), "Paired policy metadata API must load")
	if failures:
		quit(1)
		return
	var a = ClassDB.instantiate("MotorControlLab")
	var b = ClassDB.instantiate("MotorControlLab")
	for path in ["res://../build/predator-shelter/source-boulder.bin", "res://../build/predator-shelter/motor-v5-training.bin"]:
		protected_hashes[path] = FileAccess.get_sha256(path)
	for lab in [a, b]:
		check(lab.configure("res://../build/predator-shelter/source-boulder.bin", "res://../build/predator-shelter/motor-v5-training.bin", ""), "Frozen sources must load")
		check(not lab.has_candidate(), "Prepared untrained motor is not a completed candidate")
		check(lab.get_cases(0).size() == 16, "Registered reference cases must be available")
		check(lab.start_trial(0, 0, 2), "Frozen original actor must start")
	check(a.get_state().links == b.get_state().links, "Matched initial complete poses must agree")
	for tick in range(20):
		var first: Dictionary = a.step()
		var second: Dictionary = b.step()
		check(first == second, "Identical frozen actors must reproduce identical live states")
	check(a.get_state().tick == 20 and a.get_state().learning_updates == 0, "Native physical clock advances without learning")
	check(not a.start_trial(0, 0, 4), "Absent experimental candidate must fail explicitly")
	check(not a.get_error().is_empty(), "Unavailable controller failure must explain itself")
	check(b.start_trial(0, 4, 0), "Actual-motion braking fixture must load")
	var braking: Dictionary = b.get_state()
	check(braking.warmup_ticks > 0 and braking.entry_speed_mm_s > 8.0, "Braking begins with captured real swimming motion")
	var scene: Control = load("res://motor_bench/motor_bench.tscn").instantiate()
	root.add_child(scene)
	root.title = "Motor bench render verification · closes automatically"
	await frames()
	check(scene.labs.size() == 2, "Two live view controllers must be configured")
	var pair_available: bool = scene.labs[0].get_controller_info(5).available and scene.labs[1].get_controller_info(6).available
	var available_arms: Array[bool] = [bool(scene.labs[0].get_controller_info(5).available), bool(scene.labs[1].get_controller_info(6).available)]
	var any_available := available_arms[0] or available_arms[1]
	check(not require_paired or pair_available, "Actual trained files for BOTH paired arms are required by this run")
	check(scene.selectors[0].selected == 5 and scene.selectors[1].selected == 6, "Missing arms must not automatically substitute legacy controller selections")
	if pair_available:
		check(scene.running, "Two actual saved PPO arms must run as the default live comparison")
	elif any_available:
		check(not scene.running and not scene.pause_button.disabled, "A one-arm partial result must open paused, ready to resume only its saved actor")
		check(scene.notice.text.contains("Partial paired experiment"), "A trained partial arm must not be labelled as wholly untrained")
	else:
		check(not scene.running and not scene.views[0].visible and not scene.views[1].visible, "Missing paired files must not silently play old actors as a new result")
		check(scene.notice.text.to_lower().contains("not trained") or scene.notice.text.to_lower().contains("no matched pair"), "Missing or gated training must be stated explicitly")
	for index in range(2):
		var controller_id := 5 + index
		var info: Dictionary = scene.labs[index].get_controller_info(controller_id)
		if not available_arms[index]:
			check(scene.selectors[index].is_item_disabled(controller_id) and scene.states[index].is_empty() and not scene.views[index].visible, "The unavailable arm must stay empty, disabled and un-substituted")
			continue
		check(info.paired and int(info.update) > 0 and int(info.update) <= 64 and int(info.budget) == 64, "Paired arm must report an actual saved training counter")
		check(info.interface == ("torque" if index == 0 else "joint_target"), "Saved policy interface does not match its arm")
		check(not str(info.source_fingerprint).is_empty() and not str(info.environment_fingerprint).is_empty(), "Saved policy binding hashes must be visible")
		protected_hashes[info.path] = FileAccess.get_sha256(info.path)
		check(scene.labs[index].start_trial(0, 0, controller_id), "Paired actor must start with native interface adapter")
		scene.states[index] = scene.labs[index].get_state()
	if pair_available:
		check(scene.states[0].links == scene.states[1].links and scene.states[0].goal == scene.states[1].goal, "Paired arms must start with identical complete physical poses and requests")
	for index in range(2):
		if not available_arms[index]:
			continue
		scene.states[index] = scene.labs[index].step()
		var paired_state: Dictionary = scene.states[index]
		check(paired_state.learning_updates == 0 and paired_state.paired and paired_state.controller_id == 5 + index, "Paired viewer must run frozen native inference, with no learning")
		check(paired_state.raw_actions.size() == 4 and paired_state.commands.size() == 4 and paired_state.actual_torque_uNm.size() == 4, "Raw action and actual torque must have separate complete units")
		check(paired_state.joint_targets_rad.size() == (4 if index == 1 else 0), "Desired angles belong only to the joint-target interface")
		for joint in range(4):
			check(is_finite(float(paired_state.raw_actions[joint])) and absf(float(paired_state.actual_torque_uNm[joint])) <= 10.00001, "Live applied actuator torque exceeds the unchanged physical limit")
			check(is_equal_approx(float(paired_state.actual_torque_uNm[joint]), float(paired_state.commands[joint]) * 10.0), "Applied normalized command and micronewton-meter display disagree")
			if index == 1:
				check(is_equal_approx(float(paired_state.joint_targets_rad[joint]), float(paired_state.joint_limit_rad) * clampf(float(paired_state.raw_actions[joint]), -1.0, 1.0)), "Displayed joint target differs from the real clipped-action interface")
		check(is_finite(float(paired_state.torque_rms_uNm)) and float(paired_state.torque_rms_uNm) <= 10.00001, "Physical effort must be finite and measured in actual torque units")
		scene._apply_state(index)
		check(scene.actuator_labels[index].text.contains("Actual torque") and scene.messages[index].text.contains("PPO weights"), "Paired UI must expose physical action units and saved update counters")
		if int(paired_state.policy_update) < 64:
			check(scene.messages[index].text.contains("partial checkpoint") and scene.messages[index].text.contains("final evaluation not run"), "Partial saved weights must not imply a completed final evaluation")
	if any_available and not pair_available:
		var active_index := 0 if available_arms[0] else 1
		var missing_index := 1 - active_index
		var partial_tick: int = scene.states[active_index].tick
		await frames(4)
		check(scene.states[active_index].tick == partial_tick, "Partial arm must remain paused until requested")
		scene._pause()
		await create_timer(0.15).timeout
		check(scene.states[active_index].tick > partial_tick and scene.states[missing_index].is_empty(), "Resume must run only the actual saved partial arm")
		scene._pause()
		check(not scene.running and scene.states[active_index].learning_updates == 0, "Partial live playback must stay frozen inference")
	# Legacy references remain available by an explicit selection; their labels
	# are never reused for the newer paired actors.
	scene.selectors[0].select(2)
	scene.selectors[1].select(1)
	scene._run_trial()
	check(scene.selectors[1].is_item_disabled(4) == (not scene.labs[1].has_candidate()), "Legacy candidate availability must reflect its actual file")
	scene._pause()
	var paused: int = scene.states[0].tick
	await frames(12)
	check(scene.states[0].tick == paused, "Pause must freeze both physical clocks")
	scene._pause()
	await create_timer(0.2).timeout
	check(scene.states[0].tick > paused, "Resume must advance the native simulation")
	scene.running = false
	scene.run_button.pressed.emit()
	scene.running = false
	check(scene.states[0].tick == 0 and scene.states[1].tick == 0, "Explicit restart returns both controllers to matched starts")
	scene._next_trial()
	scene.running = false
	check(scene.trial.selected == 1 and scene.states[0].case_id == 1, "Next trial changes the declared fixture")
	scene.selectors[0].select(0)
	scene.selectors[0].item_selected.emit(0)
	scene.running = false
	check(scene.states[0].controller_id == 0 and scene.states[0].tick == 0, "Controller selection explicitly restarts the comparison")
	check(not scene.labs[0].configure("res://../build/motor-control/absent-source.bin", "", ""), "Missing source must be rejected")
	scene._run_trial()
	scene.running = false
	check(scene.states[0].is_empty() and not scene.views[0].visible, "Failed start must hide stale physical playback")
	check(scene.summaries[0].text == "Trial unavailable" and not scene.messages[0].text.is_empty(), "Failed start must clear prior metrics and explain the error")
	check(scene.labs[0].configure(scene.ORIGINAL, scene.CALIBRATED, ""), "Original source must be recoverable after failed start")
	# Render actual paired checkpoints when available, otherwise clearly named
	# explicitly selected legacy reference simulations. These are not clips.
	scene.trial.select(0)
	scene.selectors[0].select(5 if any_available else 2)
	scene.selectors[1].select(6 if any_available else 1)
	if any_available:
		check(scene.labs[0].configure(scene.ORIGINAL, scene.CALIBRATED, "", scene.paired_root), "Paired source must recover after failure test")
	scene._run_trial()
	scene._pause()
	check(scene.views[0].visible == (available_arms[0] or not any_available), "Successful restart restores only available physical views")
	for index in range(2):
		if scene.states[index].is_empty():
			continue
		for tick in range(20):
			scene.states[index] = scene.labs[index].step()
		scene._apply_state(index)
	for extent in [Vector2i(1280, 720), Vector2i(1024, 768), Vector2i(780, 1000)]:
		root.size = extent
		await frames(10)
		check(scene.size.is_equal_approx(Vector2(extent)), "Scene must fit the window")
		check(scene.comparison.get_global_rect().end.y <= extent.y, "Comparison must stay inside the viewport")
		check(scene.notice.get_global_rect().end.y <= extent.y, "Status text must remain visible")
		for selector in scene.selectors:
			check(selector.get_global_rect().end.x <= extent.x and selector.size.y >= 44, "Controller controls must fit with usable targets")
		if not output.is_empty() and DisplayServer.get_name() != "headless":
			await RenderingServer.frame_post_draw
			root.get_texture().get_image().save_png(output.path_join("motor-bench-%dx%d.png" % [extent.x, extent.y]))
	for path in protected_hashes:
		check(FileAccess.get_sha256(path) == protected_hashes[path], "Live motor bench modified a protected source/policy: " + str(path))
	print("MOTOR_BENCH_UI_SMOKE checks=%d failures=%d physical_clock=live learning_updates=0" % [checks, failures])
	root.remove_child(scene)
	scene.queue_free()
	await frames(2)
	quit(1 if failures else 0)

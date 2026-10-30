extends SceneTree

# UI/runtime verification only. No training or final-evaluation score is written.
var checks := 0
var failures := 0
var output := ""
var require_pair := false
var protected: Dictionary = {}

func check(ok: bool, message: String) -> void:
	checks += 1
	if not ok:
		failures += 1
		push_error(message)

func frames(count := 6) -> void:
	for index in range(count):
		await process_frame

func _initialize() -> void:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--output="):
			output = argument.trim_prefix("--output=")
		elif argument == "--require-trained-pair":
			require_pair = true
	run.call_deferred()

func run() -> void:
	create_timer(40).timeout.connect(func(): push_error("Motor-v2 UI smoke timed out"); quit(1))
	check(ClassDB.class_exists("MotorV2Lab"), "New isolated native motor-v2 class must load")
	if failures:
		quit(1)
		return
	var scene: Control = load("res://motor_v2_bench/motor_v2_bench.tscn").instantiate()
	root.add_child(scene)
	root.title = "Motor v2 render verification · closes automatically"
	await frames()
	check(scene.labs.size() == 2, "Both comparison panes must have independent native state")
	check(not scene.running, "The new comparison must open paused")
	var available := [false, false]
	for arm in range(2):
		var info: Dictionary = scene.infos[arm]
		available[arm] = bool(info.get("available", false))
		check(info.physics_version == 2 and info.protocol_version == 2, "Viewer must identify new contact/task versions")
		check(info.learning_updates == 0, "Viewer metadata must explicitly report no weight updates")
		check(scene.views[arm].visible == available[arm], "An untrained arm must remain empty without an old-policy substitute")
		if available[arm]:
			check(int(info.update) > 0 and int(info.update) <= 64 and int(info.budget) == 64, "Saved policy must have genuine bounded PPO updates")
			protected[info.path] = FileAccess.get_sha256(info.path)
			check(scene.labs[arm].get_cases().size() == 64, "Actual actor must use the new fixed64 catalog")
			check(scene.states[arm].tick == 0 and scene.states[arm].policy_update == info.update, "Opened actor counter and physical clock must be accurate")
			var other = ClassDB.instantiate("MotorV2Lab")
			check(other.configure(scene.study_root) and other.start_trial(0, arm), "Actual saved actor must start independently")
			check(other.get_state() == scene.states[arm], "Identical complete starts must reproduce metadata and physical pose")
			var actual: Dictionary = scene.labs[arm].step()
			check(actual == other.step(), "New physical interface must reproduce identical one-control playback")
			check(actual.tick == 1 and actual.learning_updates == 0 and actual.physics_version == 2, "Live simulation advances without learning")
			check(actual.actual_torque_uNm.size() == 4 and actual.raw_actions.size() == 4, "Two universal joints must expose all four actuator axes")
			check(actual.joint_targets_rad.size() == (4 if arm == 1 else 0), "Only the servo interface exposes angle targets")
			for axis in range(4):
				check(absf(float(actual.actual_torque_uNm[axis])) <= 10.00001, "Viewer actuator torque exceeds unchanged physical limit")
				if arm == 1:
					check(is_equal_approx(float(actual.joint_targets_rad[axis]), clampf(float(actual.raw_actions[axis]), -1, 1)), "Displayed angle target differs from the actual actuator transform")
			check(is_equal_approx(float(actual.overlap_limit_mm), 0.5), "Viewer must display the registered contact tolerance")
			check(actual.has("floor_contact") and actual.has("floor_force_N") and actual.has("positive_work_J"), "Contact and effort must be separately observable")
			scene.states[arm] = actual
			scene._apply_state(arm)
			check(scene.actuator_labels[arm].text.contains("Two universal joints") and scene.actuator_labels[arm].text.contains("Actual last applied torque"), "Actuator explanation must preserve axis count and physical units")
		else:
			check(scene.states[arm].is_empty() and scene.titles[arm].text.contains("not trained"), "Missing learned state must be labelled untrained")
	check(not require_pair or (available[0] and available[1]), "This invocation requires two actual trained motor-v2 policies")
	var any: bool = available[0] or available[1]
	check(scene.resume_button.disabled == (not any), "Resume availability must reflect real saved actors")
	if any:
		scene._restart()
		check(not scene.running, "Restart must remain paused")
		scene._toggle()
		check(scene.running and scene.resume_button.text == "Pause", "Resume must visibly change playback state")
		scene._toggle()
		check(not scene.running and scene.resume_button.text == "Resume", "Pause must visibly change playback state")
		scene._next()
		check(scene.trial.selected == 1 and not scene.running, "Next case selects a frozen fixture without autoplay")
		scene.trial.select(0)
		scene._restart()
	else:
		check(not scene.running and not scene.views[0].visible and not scene.views[1].visible, "No trained policies must mean no live actor playback")
		check(scene.notice.text.contains("not trained"), "The unavailable experiment must be explained")
	for extent in [Vector2i(1280, 720), Vector2i(1024, 768), Vector2i(780, 1000)]:
		root.size = extent
		await frames(10)
		check(scene.size.is_equal_approx(Vector2(extent)), "Motor-v2 scene must fit window bounds")
		check(scene.notice.get_global_rect().end.y <= extent.y, "Motor-v2 failure/status text must remain visible")
		check(scene.comparison.get_global_rect().end.y <= extent.y, "Motor-v2 panes must remain in viewport")
		check(scene.resume_button.size.y >= 44 and scene.restart_button.size.y >= 44, "Playback actions must have usable targets")
		if not output.is_empty() and DisplayServer.get_name() != "headless":
			await RenderingServer.frame_post_draw
			root.get_texture().get_image().save_png(output.path_join("motor-v2-%dx%d.png" % [extent.x, extent.y]))
	for path in protected:
		check(FileAccess.get_sha256(path) == protected[path], "Viewer changed an actual saved policy: " + str(path))
	print("MOTOR_V2_UI_SMOKE checks=%d failures=%d scene=motor_v2_bench/motor_v2_bench.tscn frozen_weights=true learning_updates=0" % [checks, failures])
	root.remove_child(scene)
	scene.queue_free()
	await frames(2)
	quit(1 if failures else 0)

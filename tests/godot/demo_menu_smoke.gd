extends SceneTree

var failures := 0
var checks := 0

func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)

func _initialize() -> void:
	Engine.max_fps = 120
	_run.call_deferred()

func settle() -> void:
	for i in 4:
		await process_frame

func input_action(action: String) -> void:
	var event := InputEventAction.new()
	event.action = action
	event.pressed = true
	root.push_input(event)
	event = InputEventAction.new()
	event.action = action
	event.pressed = false
	root.push_input(event)

func _run() -> void:
	create_timer(30.0).timeout.connect(func():
		push_error("Demo menu checks timed out")
		quit(1))
	var router := root.get_node("DemoLaunch")
	check(ProjectSettings.get_setting("application/run/main_scene") == "res://launcher/launcher.tscn", "Menu must be the startup scene")
	change_scene_to_file("res://launcher/launcher.tscn")
	await settle()
	check(current_scene.rows.size() == 2 and current_scene.buttons.size() == 4, "Menu must contain exactly two demos and four actions")
	for expected in ["cartpole_watch", "cartpole_train", "racing_watch", "racing_train"]:
		check(root.gui_get_focus_owner().name == expected, "Keyboard focus order is incorrect")
		input_action("ui_focus_next")
		await settle()
	input_action("ui_accept")
	await settle()
	check(current_scene.watch_mode, "Keyboard activation did not open playback")
	var escape := InputEventKey.new()
	escape.keycode = KEY_ESCAPE
	escape.pressed = true
	root.push_input(escape)
	await settle()
	check(current_scene.name == "DemoMenu", "Escape did not return to menu")
	var cp := PpoLab.new()
	check(cp.watch_policy(router.read_policy("cartpole")), "Bundled CartPole policy could not load")
	var dropped := false
	for tick in 500:
		var states := cp.step_showcase(true)
		for car in 12:
			dropped = dropped or states[car * 4 + 3] != 0.0
	check(not dropped, "Bundled CartPole policy dropped a pole in 500 steps")
	cp.stop()
	var racing := RacingLab.new()
	check(racing.watch_policy(router.read_policy("racing")), "Bundled racing policy could not load")
	for tick in 1200:
		racing.step_showcase()
	var race_stats := racing.get_stats()
	check(race_stats.showcase_clean_laps >= 8 and race_stats.showcase_contacts == 0 and race_stats.showcase_walls == 0, "Bundled racing playback failed its 60-second evaluation")
	racing.stop()
	for spec in [["cartpole", true], ["racing", true], ["cartpole", false], ["racing", false], ["racing", true]]:
		var demo: String = spec[0]
		var watch: bool = spec[1]
		var action: Button = current_scene.find_child("%s_%s" % [demo, "watch" if watch else "train"], true, false)
		check(action != null, "Menu action missing")
		if action == null:
			quit(1)
			return
		action.pressed.emit()
		await settle()
		var scene := current_scene
		check(scene != null and scene.watch_mode == watch, "Demo launch mode mismatch")
		var lab: Object = scene.lab
		var started := Time.get_ticks_msec()
		while not lab.get_stats().get("policy_ready", false) and Time.get_ticks_msec() - started < 10000:
			await create_timer(0.025).timeout
		check(lab.get_stats().get("policy_ready", false), "Policy did not become ready")
		check(lab.get_error().is_empty(), "Demo reported a backend error")
		check(lab.get_stats().get("playback", false) == watch, "Watch must have no training session")
		check(lab.get_stats().get("update", -1) == 0, "Fresh or saved demo must not secretly train")
		if watch:
			await create_timer(0.15).timeout
			check(lab.get_stats().get("update", -1) == 0 and lab.get_learning_curve().is_empty(), "Playback started training")
			check(not scene.curve_view.is_visible_in_tree() and not scene.pause_button.is_visible_in_tree(), "Playback must hide training controls")
		else:
			check(scene.paused and scene.pause_button.text == "Start learning", "Fresh training must show update zero")
			scene.pause_button.pressed.emit()
			started = Time.get_ticks_msec()
			while lab.get_stats().get("update", 0) < 2 and Time.get_ticks_msec() - started < 10000:
				await create_timer(0.05).timeout
			check(lab.get_stats().get("update", 0) >= 2, "Start learning did not train")
		var back: Button = scene.find_child("BackToDemos", true, false)
		check(back != null, "Back to demos is missing")
		back.pressed.emit()
		await settle()
		check(current_scene.name == "DemoMenu", "Back did not return to menu")
		check(not lab.get_stats().get("policy_ready", false), "Leaving a demo must stop its session")
	var invalid := PackedFloat32Array([1.0])
	for name in ["PpoLab", "RacingLab"]:
		var lab: Object = ClassDB.instantiate(name)
		check(not lab.watch_policy(invalid), "Mismatched saved weights were accepted")
		check(not lab.get_error().is_empty(), "Invalid policy must report a useful error")
		lab.stop()
	print("DEMO_MENU_SMOKE %d checks, %d failures" % [checks, failures])
	quit(0 if failures == 0 else 1)

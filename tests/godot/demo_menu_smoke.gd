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
	var test_checkpoint := "user://tests/menu-best-%d-%d.json" % [OS.get_process_id(), Time.get_ticks_usec()]
	node_added.connect(func(node: Node):
		if node.name == "TadpolePond":
			node.best_policy_path = test_checkpoint)
	create_timer(30.0).timeout.connect(func():
		push_error("Demo menu checks timed out")
		quit(1))
	var router := root.get_node("DemoLaunch")
	check(ProjectSettings.get_setting("application/run/main_scene") == "res://launcher/launcher.tscn", "Menu must be the startup scene")
	change_scene_to_file("res://launcher/launcher.tscn")
	await settle()
	check(current_scene.rows.size() == 3 and current_scene.buttons.size() == 6, "Menu must contain three demos and six actions")
	for expected in ["cartpole_watch", "cartpole_train", "racing_watch", "racing_train", "pond_live_watch", "pond_experiments_train", "guide_train"]:
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
	var pond_action: Button = current_scene.find_child("pond_live_watch", true, false)
	check(pond_action != null and pond_action.text == "Open pond", "The main pond action is missing")
	pond_action.pressed.emit()
	await settle()
	check(current_scene.stats.get("progress_controller", false), "The main pond action did not open the latest saved agent")
	check(current_scene.stats.selector_update == 320 and current_scene.stats.background_swimmers == 0, "Opening the pond started training or loaded the wrong agent")
	check(current_scene.current_switch.visible, "The live pond has no current control")
	current_scene.find_child("BackToDemos", true, false).pressed.emit()
	await settle()
	current_scene.find_child("pond_experiments_train", true, false).pressed.emit()
	await settle()
	check(current_scene.experiments.visible, "Pond experiments did not open")
	current_scene.experiments.find_child("Open_pond", true, false).pressed.emit()
	await settle()
	check(current_scene.name == "TadpolePond" and current_scene.live_learning, "The explicit learning action did not open shallow-water training")
	var pond_lab: Object = current_scene.lab
	check(current_scene.foods.size() == 3 and pond_lab.get_stats().policy_inputs == 13, "Menu opened the wrong pond task")
	var started := Time.get_ticks_msec()
	while pond_lab.get_stats().update < 2 and Time.get_ticks_msec() - started < 10000:
		await create_timer(0.025).timeout
	check(pond_lab.get_stats().update >= 2 and pond_lab.get_error().is_empty(), "Menu pond did not train automatically")
	var pond_back: Button = current_scene.find_child("BackToDemos", true, false)
	check(pond_back != null and pond_back.size.y >= 44, "Pond is missing its usable Back to demos control")
	pond_back.pressed.emit()
	await settle()
	check(current_scene.name == "DemoMenu" and not pond_lab.get_stats().training, "Back from pond did not stop training and return to menu")
	current_scene.find_child("pond_experiments_train", true, false).pressed.emit()
	await settle()
	current_scene.experiments.find_child("Open_pond", true, false).pressed.emit()
	await settle()
	pond_lab = current_scene.lab
	escape = InputEventKey.new()
	escape.keycode = KEY_ESCAPE
	escape.pressed = true
	root.push_input(escape)
	await settle()
	check(current_scene.name == "DemoMenu" and not pond_lab.get_stats().training, "Escape from pond failed to stop training")
	check(router.open_demo("pond", true) == ERR_INVALID_PARAMETER, "Pond falsely offers a saved-policy playback mode")
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		await settle()
		var scroll: ScrollContainer = current_scene.find_child("DemoScroll", true, false)
		pond_action = current_scene.find_child("pond_live_watch", true, false)
		scroll.ensure_control_visible(pond_action)
		await settle()
		check(pond_action.get_global_rect().end.x <= dimensions.x and pond_action.get_global_rect().end.y <= dimensions.y, "Pond action is not reachable within the menu viewport")
		check(pond_action.size.y >= 44, "Pond menu action is too small")
		for data in current_scene.rows:
			check(data.row.get_global_rect().end.x <= dimensions.x, "Menu row overflows horizontally")
		if dimensions.x >= 1280:
			check(scroll.scroll_vertical == 0, "All three demo rows should fit on the desktop menu")
		if DisplayServer.get_name() != "headless":
			await RenderingServer.frame_post_draw
			root.get_texture().get_image().save_png(ProjectSettings.globalize_path("res://../build/pond-learning-checks/menu-%dx%d.png" % [dimensions.x, dimensions.y]))
	var invalid := PackedFloat32Array([1.0])
	for name in ["PpoLab", "RacingLab"]:
		var lab: Object = ClassDB.instantiate(name)
		check(not lab.watch_policy(invalid), "Mismatched saved weights were accepted")
		check(not lab.get_error().is_empty(), "Invalid policy must report a useful error")
		lab.stop()
	print("DEMO_MENU_SMOKE %d checks, %d failures" % [checks, failures])
	quit(0 if failures == 0 else 1)

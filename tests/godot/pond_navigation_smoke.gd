extends SceneTree

var checks := 0
var failures := 0
var output := "res://../build/pond-ui-checks"

func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)

func settle() -> void:
	for tick in 8:
		await process_frame

func press_space(button: BaseButton) -> void:
	button.grab_focus()
	for pressed in [true, false]:
		var event := InputEventKey.new()
		event.keycode = KEY_SPACE
		event.pressed = pressed
		root.push_input(event)

func capture(label: String, dimensions: Vector2i) -> void:
	if DisplayServer.get_name() != "headless":
		await RenderingServer.frame_post_draw
		var path := ProjectSettings.globalize_path(output.path_join("%s-%dx%d.png" % [label, dimensions.x, dimensions.y]))
		check(root.get_texture().get_image().save_png(path) == OK, "Screenshot failed: " + label)

func _initialize() -> void:
	Engine.max_fps = 60
	_run.call_deferred()

func _run() -> void:
	create_timer(90).timeout.connect(func(): quit(1))
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(output))
	var source := "res://../docs/experiments/swimmer/gentle-currents/bank-recovery/selected-policy.bin"
	var policy_hash := FileAccess.get_sha256(source)
	change_scene_to_file("res://launcher/launcher.tscn")
	await settle()
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		await settle()
		var menu = current_scene
		var open: Button = menu.find_child("pond_feeding_watch", true, false)
		check(open != null and open.text == "Open pond", "Main pond route is not the live agent")
		var menu_scroll: ScrollContainer = menu.find_child("DemoScroll", true, false)
		menu_scroll.ensure_control_visible(open)
		await settle()
		check(open.size.y >= 44 and open.get_global_rect().end.x <= dimensions.x, "Pond entry is clipped or too small")
		await capture("menu", dimensions)
		press_space(menu.find_child("pond_experiments_train", true, false))
		await settle()
		var experiments = menu.experiments
		check(experiments.visible, "Keyboard could not open experiments")
		check(menu.get_node("ModalBackdrop").visible, "Modal does not distinguish itself from the background")
		check(experiments.size.x <= dimensions.x and experiments.size.y <= dimensions.y, "Experiments dialog overflows")
		check(experiments.get_ok_button().size.y >= 44, "Dialog close target is too small")
		check(experiments.find_child("Open_pond_live", true, false) != null, "Experiments lost the live route")
		for action: Button in experiments.actions:
			experiments.scroll.ensure_control_visible(action)
			await settle()
			check(action.size.y >= 44 and action.get_global_rect().end.x <= experiments.size.x, "Experiment action is clipped or too small")
			if action.name.contains("replay") or action.name.contains("recovery"):
				check(action.text == "Play recording", "Recording is mislabeled as live")
		experiments.scroll.scroll_vertical = 0
		await settle()
		await capture("experiments", dimensions)
		press_space(experiments.get_ok_button())
		await settle()
		check(not experiments.visible, "Dialog close control failed")
		check(not menu.get_node("ModalBackdrop").visible, "Closing experiments leaves a blocked screen")
		menu._show_guide()
		await settle()
		check(menu.guide.visible and menu.guide.size.x <= dimensions.x and menu.guide.size.y <= dimensions.y, "Help dialog is clipped")
		await capture("help", dimensions)
		menu.guide.hide()
		# The frozen reference remains a separate experiment route; its controls
		# below intentionally test fixed-policy playback rather than new learning.
		menu._show_experiments()
		await settle()
		press_space(menu.experiments.find_child("Open_pond_live", true, false))
		await settle()
		var pond = current_scene
		pond.set_physics_process(false)
		check(pond.lab.get_error().is_empty() and pond.stats.selector_update == 320, "Pond launched the wrong agent or failed")
		check(pond.stats.background_swimmers == 0, "Live pond started training")
		check(not pond.details_group.visible, "Technical details clutter the default view")
		check(pond.find_child("WatchCurrentTest", true, false) == null, "Ambiguous replay button remains in live controls")
		check(not pond.pause_button.visible and not pond.speed_button.visible and not pond.find_child("SaveTrainingNow", true, false).visible, "Live view offers false training controls")
		check(pond.title_status.text.contains("Live") and pond.curve.status_caption.contains("no learning"), "Live mode is ambiguous")
		var controls_scroll: ScrollContainer = pond.sidebar_scroll
		controls_scroll.ensure_control_visible(pond.current_switch)
		await settle()
		press_space(pond.current_switch)
		await settle()
		check(absf(pond.stats.water_current_cm_s - 0.5) < 0.0001, "UI did not enable the physical current")
		for tick in 240:
			pond.state = pond.lab.step()
		pond.previous = pond.state.duplicate(true)
		pond._poll()
		check(pond.state.elapsed >= 12 and pond.curve.current_score == pond.state.eaten * 60.0 / pond.state.elapsed, "Graph does not measure actual live collection")
		controls_scroll.scroll_vertical = 0
		await settle()
		await capture("live", dimensions)
		controls_scroll.ensure_control_visible(pond.details_button)
		await settle()
		press_space(pond.details_button)
		await settle()
		check(pond.details_group.visible and pond.training_stats.text.contains(pond.stats.compute_backend), "AI details do not expand or show the actual backend")
		press_space(pond.details_button)
		await settle()
		check(not pond.details_group.visible, "AI details do not collapse")
		pond.find_child("PondExperiments", true, false).pressed.emit()
		await settle()
		check(pond.experiments.visible, "Pond experiments control is disconnected")
		for pressed in [true, false]:
			var cancel := InputEventKey.new()
			cancel.keycode = KEY_ESCAPE
			cancel.pressed = pressed
			root.push_input(cancel)
		await settle()
		check(not pond.experiments.visible and current_scene == pond, "Escape from experiments unexpectedly leaves the live pond")
		var playback: RefCounted = pond.lab
		pond.find_child("BackToDemos", true, false).pressed.emit()
		await settle()
		check(current_scene.name == "DemoMenu" and not playback.get_stats().get("training_ready", false), "Leaving pond did not stop its session")
		check(root.title == "Tadpole Pond · CudaRL", "Menu keeps the previous scene's misleading window title")
	check(FileAccess.get_sha256(source) == policy_hash, "UI modified saved weights")
	print("POND_NAVIGATION checks=%d failures=%d" % [checks, failures])
	quit(1 if failures else 0)

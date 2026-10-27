extends SceneTree
var checks := 0
var failures := 0
func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)
func settle() -> void:
	for tick in 8:
		await process_frame
func right_arrow() -> void:
	var event := InputEventKey.new()
	event.keycode = KEY_RIGHT
	event.pressed = true
	root.push_input(event)
	event = InputEventKey.new()
	event.keycode = KEY_RIGHT
	root.push_input(event)
func _initialize() -> void:
	Engine.max_fps = 60
	_run.call_deferred()
func _run() -> void:
	create_timer(90).timeout.connect(func(): quit(1))
	var model := "res://pond/policies/deep_selector.bin"
	var model_hash := FileAccess.get_sha256(model)
	var scene = load("res://pond/current_replay.tscn").instantiate()
	root.add_child(scene)
	current_scene = scene
	scene.set_physics_process(false)
	await settle()
	check(scene.lab.get_error().is_empty(), "Replay failed: " + scene.lab.get_error())
	check(scene.state.elapsed == 60 and scene.state.eaten == 8, "Replay did not start at the recorded stall")
	check(scene.state.speed_cm_s < 0.1 and scene.state.selected_food == 1, "Replay does not show the recorded slow body and selected target")
	check(not scene.untrained_button.visible and not scene.curve.visible, "Replay offers false training/comparison controls")
	check(scene.title_status.text.contains("Recorded") and scene.training_stats.text.contains("Original GPU"), "Recording is presented as live inference")
	scene.replay_pause.pressed.emit()
	var elapsed: float = scene.lab.get_state().elapsed
	scene.lab.step()
	check(scene.lab.get_state().elapsed == elapsed, "Pause did not stop the recording")
	scene.replay_pause.pressed.emit()
	scene.lab.step()
	check(scene.lab.get_state().elapsed > elapsed, "Play did not resume the recording")
	scene._seek_time(25)
	check(scene.state.elapsed == 25, "Before-stall seeking failed")
	scene.timeline.grab_focus()
	right_arrow()
	check(scene.state.elapsed > 25, "Keyboard seeking failed")
	scene.timeline.value = 300
	check(scene.state.elapsed == 300 and scene.state.eaten == 8 and scene.state.seconds_since_food > 265, "Final recording does not match the original failure")
	scene.replay_pause.pressed.emit()
	check(scene.state.elapsed == 0 and not scene.lab.paused, "Replay did not restart the finished recording")
	scene._select_scenario(1)
	check(scene.state.elapsed == 60 and scene.timeline.min_value == 60, "Current-removal branch does not start from the common snapshot")
	scene._seek_time(120)
	check(scene.state.eaten == 8, "Current-removal branch differs from diagnostic output")
	scene._select_scenario(2)
	scene._seek_time(120)
	check(scene.state.eaten == 23, "One-second food change does not show recorded recovery")
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		scene._select_scenario(0)
		scene._seek_time(60)
		await settle()
		check(scene.layout.get_global_rect().end.x <= dimensions.x + 1, "Replay layout overflows")
		check(scene.replay_pause.size.y >= 44 and scene.timeline.size.y >= 44, "Replay controls are too small")
		check(scene.replay_pause.get_global_rect().end.y <= dimensions.y and scene.before_button.get_global_rect().end.y <= dimensions.y, "Replay transport is outside the visible screen")
		check(scene.viewport_container.size.y >= 100, "Replay viewport collapsed")
		if DisplayServer.get_name() != "headless":
			await RenderingServer.frame_post_draw
			root.get_texture().get_image().save_png(ProjectSettings.globalize_path("res://../build/pond-current-study/replay-%dx%d.png" % [dimensions.x, dimensions.y]))
	check(FileAccess.get_sha256(model) == model_hash, "Replay modified trained weights")
	scene.find_child("BackToDemos", true, false).pressed.emit()
	await settle()
	check(current_scene.name == "DemoMenu", "Replay Back to demos failed")
	print("CURRENT_REPLAY checks=%d failures=%d" % [checks, failures])
	quit(1 if failures else 0)

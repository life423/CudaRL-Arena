extends SceneTree

func _initialize() -> void:
	_run.call_deferred()

func _run() -> void:
	create_timer(20.0).timeout.connect(func():
		push_error("Swimmer render checks timed out")
		quit(1))
	var viewport := SubViewport.new()
	viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	root.add_child(viewport)
	var scene_file := "res://experiments/swimmer_compare.tscn" if "--comparison-scene" in OS.get_cmdline_user_args() else "res://experiments/swimmer.tscn"
	var scene: Control = load(scene_file).instantiate()
	viewport.add_child(scene)
	scene.replay_time = 2.0
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844)]:
		viewport.size = dimensions
		for frame in 8:
			await process_frame
		await RenderingServer.frame_post_draw
		var path := "res://../build/swimmer-study/viewer-%dx%d.png" % [dimensions.x, dimensions.y]
		viewport.get_texture().get_image().save_png(ProjectSettings.globalize_path(path))
		assert(scene.worker == -1, "Replay rendering must not secretly start training")
		assert(not scene.latest.is_empty() and scene._states().size() == 6, "Viewer did not load six native trajectories")
		assert(scene.restart.size.y >= 44 and scene.compare.size.y >= 44, "Viewer targets too small")
		assert(scene.restart.get_global_rect().end.x <= dimensions.x and scene.compare.get_global_rect().end.x <= dimensions.x, "Viewer controls overflow")
		if not scene.reference.is_empty():
			var entries: Array = scene._display_entries()
			assert(entries.size() == 2, "Comparison must show both creatures")
			var a: Array = entries[0].state
			var b: Array = entries[1].state
			assert(a[5] == b[5] and a[6] == b[6], "Comparison food placement differs")
			assert(int(scene.reference.seed) == int(scene.latest.seed), "Comparison training seeds differ")
			assert(int(a[12]) == 1 and int(b[13]) == 1, "Comparison did not load tadpole and eel bodies")
		print("SWIMMER_RENDER %dx%d passed" % [dimensions.x, dimensions.y])
	scene.compare.button_pressed = true
	assert(int(scene.baseline.update) == 0 and scene._states().size() == 6, "Untrained comparison did not load update zero")
	scene.compare.button_pressed = false
	if not scene.reference.is_empty():
		scene.focused = -1
		scene._fit()
	viewport.size = Vector2i(1280, 720)
	for frame in 6:
		await process_frame
	var click := InputEventMouseButton.new()
	click.button_index = MOUSE_BUTTON_LEFT
	click.pressed = true
	click.position = Vector2(80, 80)
	scene.world.gui_input.emit(click)
	assert(scene.focused == 0, "Selecting a pond did not enlarge the swimmer")
	for frame in 8:
		await process_frame
	await RenderingServer.frame_post_draw
	viewport.get_texture().get_image().save_png(ProjectSettings.globalize_path("res://../build/swimmer-study/viewer-focused.png"))
	var touch := InputEventScreenTouch.new()
	touch.pressed = true
	touch.position = Vector2(80, 80)
	scene.world.gui_input.emit(touch)
	assert(scene.focused == -1, "Touching the enlarged pond did not return to the overview")
	var key := InputEventKey.new()
	key.pressed = true
	key.keycode = KEY_ENTER
	scene.world.gui_input.emit(key)
	assert(scene.focused == 0, "Keyboard could not enlarge a pond")
	key.keycode = KEY_RIGHT
	scene.world.gui_input.emit(key)
	assert(scene.focused == 1, "Keyboard could not switch swimmers")
	key.keycode = KEY_ENTER
	scene.world.gui_input.emit(key)
	assert(scene.focused == -1, "Keyboard could not restore the overview")
	print("SWIMMER_RENDER_DONE")
	quit()

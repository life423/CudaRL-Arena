extends SceneTree
var checks := 0
var failures := 0

func check(ok: bool, message: String) -> void:
	checks += 1
	if not ok:
		failures += 1
		push_error(message)

func settle() -> void:
	for i in 8:
		await process_frame

func _initialize() -> void:
	Engine.max_fps = 60
	run.call_deferred()

func run() -> void:
	create_timer(90).timeout.connect(func(): quit(1))
	var checkpoint := "user://tests/release-menu-%d.bin" % OS.get_process_id()
	node_added.connect(func(node: Node):
		if node.name == "FeedingPond":
			node.feeding_checkpoint_path = checkpoint
			node.population = 128)
	var folder := OS.get_executable_path().get_base_dir().path_join("qa-output")
	DirAccess.make_dir_recursive_absolute(folder)
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		change_scene_to_file("res://launcher/launcher.tscn")
		await settle()
		var menu = current_scene
		check(menu.buttons.size() == 1, "Player menu has competing actions")
		check(menu.find_child("guide_train", true, false) == null and menu.find_child("pond_experiments_train", true, false) == null, "Removed menu section remains")
		var button: Button = menu.find_child("pond_feeding_watch", true, false)
		menu.find_child("DemoScroll", true, false).ensure_control_visible(button)
		await settle()
		check(button.size.y >= 44 and button.get_global_rect().end.x <= dimensions.x, "Menu action clipped")
		check(root.gui_get_focus_owner() == button, "Main action lacks keyboard focus")
		await RenderingServer.frame_post_draw
		check(root.get_texture().get_image().save_png(folder.path_join("menu-%dx%d.png" % [dimensions.x, dimensions.y])) == OK, "Menu screenshot failed")
		for pressed in [true, false]:
			var event := InputEventKey.new()
			event.keycode = KEY_SPACE
			event.pressed = pressed
			root.push_input(event)
		await settle()
		var pond = current_scene
		check(pond.name == "FeedingPond" and pond.lab.get_error().is_empty(), "Keyboard launch failed")
		check(pond.stats.feeding_mode and pond.stats.policy_inputs == 37, "Release launched an older pond")
		check(pond.find_child("PondExperiments", true, false) == null, "Experiments remain in the pond")
		pond._drop_at(Vector3(0.7, 0, 0.6))
		check(pond.state.foods.size() > 0, "Release feeding failed")
		await settle()
		await RenderingServer.frame_post_draw
		root.get_texture().get_image().save_png(folder.path_join("pond-%dx%d.png" % [dimensions.x, dimensions.y]))
		if dimensions == Vector2i(1280, 720):
			for argument in OS.get_cmdline_user_args():
				if argument.begins_with("--website-screenshot="):
					root.size = Vector2i(1600, 1100)
					await settle()
					pond._drop_at(Vector3(-0.8, 0, -0.4))
					pond._drop_at(Vector3(0.2, 0, -1.2))
					await settle()
					await RenderingServer.frame_post_draw
					check(pond.viewport.get_texture().get_image().save_png(argument.get_slice("=", 1)) == OK, "Website capture failed")
		pond.find_child("BackToDemos", true, false).pressed.emit()
		await settle()
		check(current_scene.name == "DemoMenu", "Return to menu failed")
	print("RELEASE_MENU checks=%d failures=%d" % [checks, failures])
	quit(1 if failures else 0)

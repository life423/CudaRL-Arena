extends SceneTree

func _initialize() -> void:
	Engine.max_fps = 60
	_run.call_deferred()

func _run() -> void:
	var viewport := SubViewport.new()
	viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	root.add_child(viewport)
	var scene: Control = load("res://launcher/launcher.tscn").instantiate()
	viewport.add_child(scene)
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844)]:
		viewport.size = dimensions
		for frame in 6:
			await process_frame
		await RenderingServer.frame_post_draw
		var image := viewport.get_texture().get_image()
		var path := "res://../build/menu-checks/menu-%dx%d.png" % [dimensions.x, dimensions.y]
		image.save_png(ProjectSettings.globalize_path(path))
		for button: Button in scene.buttons:
			var rect := button.get_global_rect()
			assert(rect.size.y >= 44 and rect.position.x >= 0 and rect.end.x <= dimensions.x + 1, "Menu button overflows or has a small target")
		print("MENU_RENDER %dx%d horizontal layout and touch targets checked" % [dimensions.x, dimensions.y])
		if dimensions.x == 390:
			var scroll: ScrollContainer = scene.margin.get_child(0)
			scroll.scroll_vertical = 10000
			await process_frame
			await RenderingServer.frame_post_draw
			viewport.get_texture().get_image().save_png(ProjectSettings.globalize_path("res://../build/menu-checks/menu-phone-bottom.png"))
			assert(scene.buttons[3].get_global_rect().end.y <= dimensions.y, "Last action cannot be reached by scrolling")
	scene.queue_free()
	await process_frame
	var router := root.get_node("DemoLaunch")
	for demo in ["cartpole", "racing"]:
		router.active_demo = demo
		router.mode = "watch"
		var showcase: Control = load(router.SCENES[demo]).instantiate()
		viewport.add_child(showcase)
		for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844)]:
			viewport.size = dimensions
			for frame in 12:
				await process_frame
			await RenderingServer.frame_post_draw
			viewport.get_texture().get_image().save_png(ProjectSettings.globalize_path("res://../build/menu-checks/%s-%dx%d.png" % [demo, dimensions.x, dimensions.y]))
			print("DEMO_RENDER %s %dx%d" % [demo, dimensions.x, dimensions.y])
		showcase.queue_free()
		await process_frame
	print("MENU_RENDER_DONE")
	quit()

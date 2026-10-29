extends SceneTree

func _initialize() -> void:
	Engine.max_fps = 60
	_run.call_deferred()

func _run() -> void:
	var scene = load("res://pond/latest_agent.tscn").instantiate()
	root.add_child(scene)
	current_scene = scene
	scene.set_physics_process(false)
	scene.set_process(false)
	for frame in 8:
		await process_frame
	if not scene.lab.get_error().is_empty():
		push_error(scene.lab.get_error())
		quit(1)
		return
	scene.viewport_container.stretch = false
	scene.viewport.size = Vector2i(960, 600)
	scene.overview = true
	scene._update_camera(5)
	for frame in 8:
		await process_frame
	await RenderingServer.frame_post_draw
	var result: int = scene.viewport.get_texture().get_image().save_png("res://launcher/assets/tadpole-pond.png")
	print("DEEP_POND_THUMBNAIL saved=%s training=%s" % [result == OK, scene.stats.background_swimmers > 0])
	scene.lab.stop()
	quit(0 if result == OK else 1)

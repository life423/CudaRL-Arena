extends SceneTree

func _initialize() -> void:
	_run.call_deferred()

func _run() -> void:
	var viewport := SubViewport.new()
	viewport.own_world_3d = true
	viewport.size = Vector2i(1280, 720)
	viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	root.add_child(viewport)
	var scene: Node3D = load("res://pond/pond.tscn").instantiate()
	viewport.add_child(scene)
	for frame in 8:
		await process_frame
	scene.warmup_timer.stop()
	scene.lab.set_paused(true)
	scene.set_physics_process(false)
	scene.pond_container.stretch = false
	scene.pond_viewport.size = Vector2i(960, 280)
	scene.camera.position = Vector3(0, 5, 11)
	scene.camera.look_at(Vector3.ZERO)
	scene.camera.size = 4.2
	for frame in 8:
		await process_frame
	await RenderingServer.frame_post_draw
	var result: int = scene.pond_viewport.get_texture().get_image().save_png("res://launcher/assets/tadpole-pond.png")
	print("POND_THUMBNAIL result=%d" % result)
	scene.queue_free()
	for frame in 4:
		await process_frame
	quit(0 if result == OK else 1)

extends SceneTree

var checks := 0
var failures := 0

func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)

func _initialize() -> void:
	_run.call_deferred()

func _run() -> void:
	create_timer(30.0).timeout.connect(func(): quit(1))
	var file := FileAccess.open("res://pond/policies/tadpole.json", FileAccess.READ)
	var data: Dictionary = JSON.parse_string(file.get_as_text())
	var weights := PackedFloat32Array(data.parameters)
	var lab := TadpoleLab.new()
	check(not lab.watch_policy(PackedFloat32Array([1.0])), "Invalid policy shape accepted")
	check(not lab.get_error().is_empty(), "Invalid weights did not report an error")
	var zero := PackedFloat32Array()
	zero.resize(weights.size())
	check(lab.watch_policy(zero, 1), "Zero controller did not load")
	for tick in 100:
		lab.step()
	check(lab.get_state().x == 0.0 and lab.get_state().y == 0.0, "The scene supplies hidden propulsion")
	check(lab.watch_policy(weights, 1), "Saved trained policy did not load")
	var ate := 0
	var max_jump := 0.0
	var before := lab.get_state()
	for tick in 2400:
		var state := lab.step()
		max_jump = maxf(max_jump, Vector2(state.x - before.x, state.y - before.y).length())
		if state.ate:
			ate += 1
			check(int(state.eaten) == ate, "Collection count is incorrect")
			check(Vector2(state.food_x - before.food_x, state.food_y - before.food_y).length() > 0.2, "Eating did not move the food")
		before = state
	check(ate >= 20, "Continuous feeding failed to reach enough food in 120 simulated seconds")
	check(max_jump < 0.4, "Tadpole jumped or teleported during feeding")
	print("POND_FEEDING eaten=%d max_step=%.4f" % [ate, max_jump])
	var viewport := SubViewport.new()
	viewport.own_world_3d = true
	viewport.size = Vector2i(1280, 720)
	viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	root.add_child(viewport)
	var scene: Node3D = load("res://pond/pond.tscn").instantiate()
	scene.live_learning = false
	scene.close_camera = false # Whole-pond visibility belongs to overview mode.
	viewport.add_child(scene)
	for frame in 4:
		await process_frame
	scene.set_physics_process(false)
	for tick in 20:
		scene._physics_process(0.05)
	scene._process(0.05)
	check(scene.ready_policy and scene.food.visible, "3D scene did not load its policy and food")
	check(scene.tail.mesh.get_surface_count() == 1 and scene.membrane.mesh.get_surface_count() == 1, "Articulated meshes are missing")
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844)]:
		viewport.size = dimensions
		for frame in 8:
			await process_frame
		await RenderingServer.frame_post_draw
		viewport.get_texture().get_image().save_png(ProjectSettings.globalize_path("res://../build/pond-checks/pond-%dx%d.png" % [dimensions.x, dimensions.y]))
		var head: Vector2 = scene.camera.unproject_position(scene.creature.global_position)
		var food: Vector2 = scene.camera.unproject_position(scene.food.global_position)
		check(Rect2(Vector2.ZERO, Vector2(dimensions)).has_point(head), "Camera lost the tadpole")
		check(Rect2(Vector2.ZERO, Vector2(dimensions)).has_point(food), "Camera lost the food")
		check(scene.restart_button.size.y >= 44, "Restart control is too small")
		print("POND_RENDER %dx%d checked" % [dimensions.x, dimensions.y])
	scene.restart_button.pressed.emit()
	check(scene.current.eaten == 0 and scene.current.x == 0.0 and scene.current.y == 0.0, "Restart failed")
	var position_before := Vector2(scene.current.x, scene.current.y)
	scene._move_food()
	check(position_before == Vector2(scene.current.x, scene.current.y), "Moving food also moved the tadpole")
	print("POND_SMOKE %d checks, %d failures" % [checks, failures])
	quit(0 if failures == 0 else 1)

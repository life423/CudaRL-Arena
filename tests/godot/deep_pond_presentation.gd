extends SceneTree

var checks := 0
var failures := 0
func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)
func settle(frames := 8) -> void:
	for i in frames:
		await process_frame
func pixel_changes(a: Image, b: Image, point: Vector2, radius: int, threshold := 0.015) -> int:
	var changed := 0
	for y in range(maxi(0, int(point.y) - radius), mini(a.get_height(), int(point.y) + radius)):
		for x in range(maxi(0, int(point.x) - radius), mini(a.get_width(), int(point.x) + radius)):
			var p := a.get_pixel(x, y)
			var q := b.get_pixel(x, y)
			if Vector3(p.r - q.r, p.g - q.g, p.b - q.b).length() > threshold:
				changed += 1
	return changed
func _initialize() -> void:
	Engine.max_fps = 60
	_run.call_deferred()
func _run() -> void:
	create_timer(90).timeout.connect(func(): quit(1))
	var source := ProjectSettings.globalize_path("user://pond-policies/deep-pond-v1/training.bin")
	var first := ProjectSettings.globalize_path("user://tests/pond-presentation-%d.bin" % OS.get_process_id())
	var second := ProjectSettings.globalize_path("user://tests/pond-presentation-reference-%d.bin" % OS.get_process_id())
	DirAccess.make_dir_recursive_absolute(first.get_base_dir())
	check(DirAccess.copy_absolute(source, first) == OK, "Could not copy the training snapshot")
	check(DirAccess.copy_absolute(first, second) == OK, "Could not copy the reference snapshot")
	var reference := DeepPondLab.new()
	check(reference.start_training(1024, 7, second), "Reference could not start")
	var scene = load("res://pond/deep_pond.tscn").instantiate()
	if scene.get_script() == null or not scene.get_script().can_instantiate():
		push_error("Deep pond script did not load")
		quit(1)
		return
	scene.checkpoint_path = first
	scene.paused = true
	root.add_child(scene)
	scene.set_physics_process(false)
	await settle()
	var start := Time.get_ticks_msec()
	while not scene.lab.get_stats().training_ready and Time.get_ticks_msec() - start < 10000:
		await create_timer(0.025).timeout
	while not reference.get_stats().training_ready and Time.get_ticks_msec() - start < 10000:
		await create_timer(0.025).timeout
	check(scene.lab.get_error().is_empty() and reference.get_error().is_empty(), "The snapshot did not restore")
	check(scene.lab.get_state() == reference.get_state(), "Presentation changed the initial physical state")
	for tick in 360:
		scene.previous = scene.state.duplicate(true)
		scene.state = scene.lab.step()
		reference.step()
		if tick % 12 == 0:
			scene._apply_pose(0.5)
			scene._update_camera(0.05)
			check(scene.lab.get_state() == reference.get_state(), "Visual effects changed native simulation or food placement")
	check(scene.overview, "The pond should start in its calm stationary overview")
	check(scene.camera.get_parent() == scene.world and not scene.creature.is_ancestor_of(scene.camera), "Camera must not inherit the creature transform")
	for blend in [0.0, 0.5, 1.0]:
		scene._apply_pose(blend)
		check(scene.rendered_swimmer_position.is_equal_approx((scene.previous.position as Vector3).lerp(scene.state.position, blend)), "Camera target does not use the rendered simulation interpolation")
	check(scene.water_surface.visible and scene.shore.rock_count > 10, "Natural shoreline or water is missing")
	check(scene.shore.vegetation.multimesh.instance_count > 200, "Shore vegetation is missing")
	var bounds := [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]
	for dimensions in bounds:
		root.size = dimensions
		await settle()
		check(scene.layout.get_global_rect().end.x <= dimensions.x + 1, "The interface overflows horizontally")
		check(scene.camera_button.size.y >= 44 and scene.pause_button.size.y >= 44, "An important control is too small")
		for overview in [false, true]:
			scene.overview = overview
			await settle(50)
			if overview:
				for point in scene.state.foods:
					check(scene.camera.is_position_in_frustum(point), "Overview clips a food piece")
				var original: Dictionary = scene.state.duplicate(true)
				var transform: Transform3D = scene.camera.transform
				for point in [Vector3(2, -1, 1), Vector3(-2, -3, -1)]:
					scene.state.position = point
					scene._update_camera(0.05)
					check(scene.camera.transform.is_equal_approx(transform), "Overview follows the creature instead of remaining stationary")
				scene.state = original
			else:
				for item: Dictionary in scene.state.links:
					check(scene.camera.is_position_in_frustum(item.position), "Follow view clips a body link")
				check(scene.camera.global_position.y > scene.state.position.y, "Closer view should retain an elevated three-quarter angle")
				check(scene.camera.position.distance_to(scene.state.position) >= 6, "Closer view is uncomfortably close")
				check(scene.camera.position.distance_to(scene.state.position) < 10, "Closer view is too distant")
				var original: Dictionary = scene.state.duplicate(true)
				var transform: Transform3D = scene.camera.transform
				for angle in [0.0, PI * 0.5, PI, PI * 1.5]:
					scene.state.orientation = Quaternion(Vector3.UP, angle)
					scene._update_camera(0.05)
					check(scene.camera.transform.is_equal_approx(transform), "Turning the creature moved the camera")
				scene.state = original
			if DisplayServer.get_name() != "headless":
				await RenderingServer.frame_post_draw
				root.get_texture().get_image().save_png(ProjectSettings.globalize_path("res://../build/pond-learning-checks/presentation-%s-%dx%d.png" % ["overview" if overview else "follow", dimensions.x, dimensions.y]))
	# Continue the paired physical trace through more real food collections.
	root.size = Vector2i(1280, 720)
	scene.overview = false
	for tick in 600:
		scene.previous = scene.state.duplicate(true)
		scene.state = scene.lab.step()
		reference.step()
		if tick % 30 == 0:
			scene._apply_pose(1)
			scene._update_camera(0.05)
			await process_frame
			check(scene.lab.get_state() == reference.get_state(), "The longer rendered trace changed physical state")
	check(scene.lab.get_stats().paused, "Presentation fixture unexpectedly trained")
	# Render-only probes at three physically valid depths. Compare visible and
	# hidden creature pixels, rather than only checking that it is in the frustum.
	var native_state: Dictionary = scene.lab.get_state()
	var saved_state: Dictionary = scene.state.duplicate(true)
	var saved_previous: Dictionary = scene.previous.duplicate(true)
	var old_scale := Engine.time_scale
	Engine.time_scale = 0
	scene.overview = true
	for depth in [0.8, 2.0, 3.2]:
		var p := Vector3(0, -depth, 1.1)
		scene.state.position = p
		scene.state.orientation = Quaternion.IDENTITY
		scene.state.links = [{"position": p, "orientation": Quaternion.IDENTITY}, {"position": p + Vector3(-0.575, 0, 0), "orientation": Quaternion.IDENTITY}, {"position": p + Vector3(-1.225, 0, 0), "orientation": Quaternion.IDENTITY}]
		scene.previous = scene.state.duplicate(true)
		scene._apply_pose(1)
		scene._update_camera(0.05)
		await settle(4)
		if DisplayServer.get_name() != "headless":
			await RenderingServer.frame_post_draw
			var shown: Image = scene.viewport.get_texture().get_image()
			root.get_texture().get_image().save_png(ProjectSettings.globalize_path("res://../build/pond-learning-checks/natural-depth-%.1f.png" % depth))
			scene.creature.visible = false
			await settle(3)
			await RenderingServer.frame_post_draw
			var hidden: Image = scene.viewport.get_texture().get_image()
			var pixel: Vector2 = scene.camera.unproject_position(p)
			var changed := 0
			for y in range(maxi(0, int(pixel.y) - 24), mini(shown.get_height(), int(pixel.y) + 24)):
				for x in range(maxi(0, int(pixel.x) - 24), mini(shown.get_width(), int(pixel.x) + 24)):
					var a := shown.get_pixel(x, y)
					var b := hidden.get_pixel(x, y)
					if Vector3(a.r - b.r, a.g - b.g, a.b - b.b).length() > 0.025:
						changed += 1
			check(changed > 12, "Creature is obscured at depth %.1f (changed pixels %d)" % [depth, changed])
			scene.creature.visible = true
	if DisplayServer.get_name() != "headless":
		# Detail must be visible through water, rather than only present in code.
		scene.terrain_material.set_shader_parameter("bed_detail", false)
		scene.terrain_material.set_shader_parameter("sunlight_patterns", false)
		await settle(3)
		await RenderingServer.frame_post_draw
		var plain: Image = scene.viewport.get_texture().get_image()
		plain.save_png(ProjectSettings.globalize_path("res://../build/pond-learning-checks/pond-bed-plain.png"))
		scene.terrain_material.set_shader_parameter("bed_detail", true)
		scene.terrain_material.set_shader_parameter("sunlight_patterns", true)
		await settle(3)
		await RenderingServer.frame_post_draw
		var detailed: Image = scene.viewport.get_texture().get_image()
		detailed.save_png(ProjectSettings.globalize_path("res://../build/pond-learning-checks/pond-bed-detailed.png"))
		var floor_pixel: Vector2 = scene.camera.unproject_position(Vector3(1.4, -3.9, -1))
		check(pixel_changes(plain, detailed, floor_pixel, 35) > 40, "Pond bed detail is not visible through the water")
		# Cast-shadow ablation at the floor hit along the actual sunlight direction.
		var p := Vector3(0, -2.3, 0)
		scene.state.position = p
		scene.state.links = [{"position": p, "orientation": Quaternion.IDENTITY}, {"position": p + Vector3(-0.575, 0, 0), "orientation": Quaternion.IDENTITY}, {"position": p + Vector3(-1.225, 0, 0), "orientation": Quaternion.IDENTITY}]
		scene.previous = scene.state.duplicate(true)
		scene._apply_pose(1)
		scene._update_camera(0.05)
		var ray: Vector3 = -scene.pond_sun.global_basis.z
		var low := 0.0
		var high := 12.0
		for i in 24:
			var mid := (low + high) * 0.5
			var hit := p + ray * mid
			if hit.y > scene._floor_y(sqrt(hit.x * hit.x / (scene.RX * scene.RX) + hit.z * hit.z / (scene.RZ * scene.RZ))):
				low = mid
			else:
				high = mid
		var shadow_pixel: Vector2 = scene.camera.unproject_position(p + ray * high)
		var opacity: float = scene.pond_sun.get_param(Light3D.PARAM_SHADOW_OPACITY)
		scene.pond_sun.set_param(Light3D.PARAM_SHADOW_OPACITY, 0)
		await settle(3)
		await RenderingServer.frame_post_draw
		var no_shadow: Image = scene.viewport.get_texture().get_image()
		scene.pond_sun.set_param(Light3D.PARAM_SHADOW_OPACITY, opacity)
		await settle(3)
		await RenderingServer.frame_post_draw
		var shadow: Image = scene.viewport.get_texture().get_image()
		check(pixel_changes(no_shadow, shadow, shadow_pixel, 20, 0.008) > 8, "The creature's cast shadow is not visible on the pond bed")
		root.get_texture().get_image().save_png(ProjectSettings.globalize_path("res://../build/pond-learning-checks/pond-bed-shadow.png"))
		var near_point := Vector3(0, -0.8, 1.1)
		var deep_point := Vector3(0, -3.2, 1.1)
		var near_size: float = scene.camera.unproject_position(near_point + scene.camera.basis.x * 0.25).distance_to(scene.camera.unproject_position(near_point))
		var deep_size: float = scene.camera.unproject_position(deep_point + scene.camera.basis.x * 0.25).distance_to(scene.camera.unproject_position(deep_point))
		check(deep_size < near_size, "Perspective should make the same-sized animal appear smaller farther down")
	Engine.time_scale = old_scale
	scene.state = saved_state
	scene.previous = saved_previous
	check(scene.lab.get_state() == native_state, "Render-only depth probes changed native state")
	scene.queue_free()
	reference.stop()
	await settle()
	print("POND_PRESENTATION checks=%d failures=%d" % [checks, failures])
	quit(1 if failures else 0)

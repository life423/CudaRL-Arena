extends SceneTree
var checks := 0
var failures := 0
func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)
func settle() -> void:
	for i in 8:
		await process_frame
func capture(name: String) -> void:
	await RenderingServer.frame_post_draw
	root.get_texture().get_image().save_png(ProjectSettings.globalize_path("res://../build/pond-learning-checks/" + name + ".png"))
func _initialize() -> void:
	Engine.max_fps = 60
	_run.call_deferred()
func _run() -> void:
	create_timer(70).timeout.connect(func(): quit(1))
	var Water = load("res://pond/pond_ripples.gd")
	var probe = Water.new()
	probe.position.x = 100 # Keep the emission probes out of the rendered pond.
	root.add_child(probe)
	for i in 61:
		probe.observe(Vector3.ZERO, Vector3(-1, 0, 0), Vector2.ZERO, 0, i * 0.05)
	check(probe.emitted == 0 and probe.active_count(3) == 0, "A stationary creature fabricated a wake")
	probe.clear()
	for i in 31:
		var head := Vector3(i * 0.025, 0, 0)
		probe.observe(head, head + Vector3(-1, 0, 0), Vector2.ZERO, 0, i * 0.05)
	check(probe.wake_emitted > 0 and probe.tail_emitted == 0, "Coasting must create body wakes without invented tail strokes")
	probe.clear()
	for i in 101:
		var a := sin(i * 0.35)
		probe.observe(Vector3.ZERO, Vector3(-1, 0, a * 0.4), Vector2(a * 0.8, -a), 0, i * 0.05)
	check(probe.tail_emitted > 0 and probe.wake_emitted == 0, "Tail motion was not distinguished from body travel")
	check(probe.records.size() == Water.CAPACITY and probe.active_count(5) <= Water.CAPACITY, "The ripple pool grew without a bound")
	for record in probe.records:
		if not record.is_empty():
			check((record.position / probe.water_radii).length_squared() < 1, "Ripple source escaped the water")
	check(probe.active_count(7) == 0, "Disturbances did not fade after motion stopped")
	probe.set_enabled(false)
	probe.observe(Vector3.ZERO, Vector3.ONE, Vector2.ONE, 0, 9)
	check(not probe.visible and probe.emitted == 0, "Disabling effects still emitted disturbances")
	probe.queue_free()
	var fixture := "res://../docs/experiments/swimmer/shuffled-input/fresh-layouts-update-17425/shuffled.json"
	var reference := TadpoleLab.new()
	check(reference.watch_foraging_checkpoint(fixture, 1), "Reference CPU controller failed to load")
	var scene = load("res://pond/pond.tscn").instantiate()
	scene.live_learning = false
	scene.playback_checkpoint = fixture
	root.add_child(scene)
	await settle()
	scene.set_physics_process(false)
	scene.set_process(false)
	scene._restart()
	reference.reset(1)
	check(scene.ready_policy and scene.foods.size() == 3, "Water fixture is not the three-food pond")
	for i in 320:
		scene._physics_process(0.05)
		scene._process(0.05)
		reference.step()
		if i % 20 == 0:
			check(scene.lab.get_state() == reference.get_state(), "Visual water changed the native simulation or food sequence")
	check(scene.ripples.wake_emitted > 0 and scene.ripples.tail_emitted > 0, "Real trained movement did not generate body and tail effects")
	check(scene.ripples.active_count(float(scene.current.elapsed)) > 0, "No physical-motion effects remain visible")
	print("WATER_MOTION wake=%d tail=%d active=%d food=%d" % [scene.ripples.wake_emitted, scene.ripples.tail_emitted, scene.ripples.active_count(float(scene.current.elapsed)), int(scene.current.eaten)])
	check(not scene.lab.get_stats().training, "Water playback secretly started training")
	await create_timer(0.3).timeout # Let existing food appearance tweens settle.
	var native_before: Dictionary = scene.lab.get_state()
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		await settle()
		for close_up in [true, false]:
			scene.set_camera_close(close_up)
			scene._process(0.05)
			await settle()
			var view := "follow" if close_up else "overview"
			scene.ripples.visible = true
			await capture("water-%s-on-%dx%d" % [view, dimensions.x, dimensions.y])
			scene.ripples.visible = false
			await capture("water-%s-off-%dx%d" % [view, dimensions.x, dimensions.y])
			check(scene.lab.get_state() == native_before, "Camera/effect visibility changed policy or physics state")
			check(scene.camera_selector.size.y >= 44 and scene.camera_selector.is_visible_in_tree(), "Water view lost its usable camera control")
			if not close_up:
				var bounds := Rect2(Vector2.ZERO, Vector2(scene.pond_viewport.size))
				for food in scene.foods:
					check(bounds.has_point(scene.camera.unproject_position(food.global_position)), "Water overview lost food")
	var body_before: Dictionary = scene.lab.get_state()
	scene.set_water_disturbances(false)
	check(scene.lab.get_state() == body_before and scene.ripples.emitted == 0, "Disabling water reset or moved the creature")
	scene.set_water_disturbances(true)
	scene._restart()
	check(scene.ripples.emitted == 0 and scene.ripples.active_count(0) == 0, "Restart retained old wakes")
	scene.queue_free()
	reference.stop()
	await settle()
	print("POND_WATER_SMOKE %d checks, %d failures" % [checks, failures])
	quit(0 if failures == 0 else 1)

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
func key_toggle(button: BaseButton) -> void:
	button.grab_focus()
	for pressed in [true, false]:
		var event := InputEventKey.new()
		event.keycode = KEY_SPACE
		event.pressed = pressed
		root.push_input(event)
func pointer_toggle(button: BaseButton) -> void:
	var motion := InputEventMouseMotion.new()
	motion.position = button.get_global_rect().get_center()
	motion.global_position = motion.position
	root.push_input(motion, true)
	await process_frame
	for pressed in [true, false]:
		var event := InputEventMouseButton.new()
		event.position = button.get_global_rect().get_center()
		event.global_position = event.position
		event.button_index = MOUSE_BUTTON_LEFT
		event.pressed = pressed
		event.button_mask = MOUSE_BUTTON_MASK_LEFT if pressed else 0
		root.push_input(event, true)
		await process_frame
func same_physical_state(before: Dictionary, after: Dictionary) -> void:
	for field in ["position", "orientation", "foods", "eaten", "elapsed", "selected_food", "seconds_since_food"]:
		check(before[field] == after[field], "Changing current reset or moved state: " + field)
func _initialize() -> void:
	Engine.max_fps = 60
	_run.call_deferred()
func _run() -> void:
	create_timer(90).timeout.connect(func(): quit(1))
	var policy := "res://../docs/experiments/swimmer/gentle-currents/bank-recovery/selected-policy.bin"
	var original := "res://pond/policies/deep_selector.bin"
	var policy_hash := FileAccess.get_sha256(policy)
	var original_hash := FileAccess.get_sha256(original)
	var missing_scene = load("res://pond/deep_pond.tscn").instantiate()
	missing_scene.controller_policy_path = "res://pond/policies/missing-current-test.bin"
	missing_scene.lab = ClassDB.instantiate("DeepPondLab")
	check(not missing_scene._start_lab(), "A missing playback policy silently started training")
	check(missing_scene.lab.get_stats().background_swimmers == 0 and not missing_scene.lab.get_error().is_empty(), "Missing policy did not fail safely")
	missing_scene.lab.stop()
	missing_scene.free()
	var scene = load("res://pond/latest_agent.tscn").instantiate()
	root.add_child(scene)
	current_scene = scene
	scene.set_physics_process(false)
	await settle()
	check(scene.lab.get_error().is_empty(), "Live current scene did not start: " + scene.lab.get_error())
	check(scene.stats.water_current_cm_s == 0 and not scene.current_switch.button_pressed, "Initial water should be calm")
	for tick in 27:
		scene.state = scene.lab.step()
	scene.previous = scene.state.duplicate(true)
	scene._poll()
	var before: Dictionary = scene.lab.get_state().duplicate(true)
	key_toggle(scene.current_switch)
	await settle()
	check(scene.current_switch.button_pressed and absf(scene.stats.water_current_cm_s - 0.5) < 0.0001, "Keyboard did not enable the actual gentle current")
	check(scene.water_status.text.contains("0.5"), "Current feedback does not show the selected strength")
	same_physical_state(before, scene.lab.get_state())
	for tick in 1200:
		scene.state = scene.lab.step()
		if tick % 100 == 0:
			scene._poll()
	scene.previous = scene.state.duplicate(true)
	scene._poll()
	check(scene.state.eaten > before.eaten and scene.state.elapsed > 60, "Live swimmer did not collect food after the current was enabled")
	check(scene.stats.background_swimmers == 0 and scene.stats.selector_update == 320, "Current control started training or changed the policy")
	before = scene.lab.get_state().duplicate(true)
	var samples: int = scene.curve.points.size()
	# Verify mouse routing in rendered Windows runs; use keyboard in headless.
	if DisplayServer.get_name() == "headless":
		key_toggle(scene.current_switch)
	else:
		await pointer_toggle(scene.current_switch)
	await settle()
	check(not scene.current_switch.button_pressed and scene.stats.water_current_cm_s == 0, "Current control did not restore calm water")
	check(scene.water_status.text == "Calm water", "Calm feedback is stale")
	same_physical_state(before, scene.lab.get_state())
	check(scene.curve.points.size() == samples, "Changing water reset the session graph")
	key_toggle(scene.current_switch)
	scene._toggle_untrained(true)
	check(scene.stats.showcase_untrained and absf(scene.stats.water_current_cm_s - 0.5) < 0.0001, "Random-weight comparison lost the water condition")
	scene._toggle_untrained(false)
	check(not scene.stats.showcase_untrained and scene.stats.selector_update == 320 and absf(scene.stats.water_current_cm_s - 0.5) < 0.0001, "Restoring trained weights lost the current or policy")
	for tick in 300:
		scene.state = scene.lab.step()
		if tick % 100 == 0:
			scene._poll()
	scene.previous = scene.state.duplicate(true)
	scene._poll()
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		await settle()
		check(scene.layout.get_global_rect().end.x <= dimensions.x + 1, "Current layout overflows horizontally")
		check(scene.viewport_container.size.y >= 100, "Current viewport collapsed")
		check(scene.current_switch.size.y >= 44, "Current control is too small")
		check(scene.current_switch.get_global_rect().end.x <= dimensions.x + 1, "Current control is outside the viewport")
		if scene.layout.vertical:
			check(scene.water_status.get_global_rect().end.y <= scene.sidebar_scroll.get_global_rect().end.y, "Current feedback is clipped on a narrow screen")
		if DisplayServer.get_name() != "headless":
			await RenderingServer.frame_post_draw
			var path := ProjectSettings.globalize_path("res://../build/gentle-current-control/current-%dx%d.png" % [dimensions.x, dimensions.y])
			DirAccess.make_dir_recursive_absolute(path.get_base_dir())
			check(root.get_texture().get_image().save_png(path) == OK, "Current screenshot failed")
	check(FileAccess.get_sha256(policy) == policy_hash and FileAccess.get_sha256(original) == original_hash, "Current switch changed a saved controller")
	var backend: String = scene.stats.compute_backend
	scene.find_child("BackToDemos", true, false).pressed.emit()
	await settle()
	check(current_scene.name == "DemoMenu", "Current scene could not return to the menu")
	print("GENTLE_CURRENT backend=%s checks=%d failures=%d" % [backend, checks, failures])
	quit(1 if failures else 0)

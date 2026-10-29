extends SceneTree

var checks := 0
var failures := 0
const POLICY := "res://../docs/experiments/swimmer/gentle-currents/bank-recovery/selected-policy.bin"
var folder := "res://../build/current-strength-checks"

func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)

func settle() -> void:
	for frame in 8:
		await process_frame

func key(keycode: Key) -> void:
	for pressed in [true, false]:
		var event := InputEventKey.new()
		event.keycode = keycode
		event.pressed = pressed
		root.push_input(event)

func _initialize() -> void:
	Engine.max_fps = 60
	_run.call_deferred()

func _run() -> void:
	create_timer(90).timeout.connect(func(): quit(1))
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(folder))
	var policy_hash := FileAccess.get_sha256(POLICY)
	var scene = load("res://pond/latest_agent.tscn").instantiate()
	root.add_child(scene)
	current_scene = scene
	scene.set_physics_process(false)
	await settle()
	check(scene.lab.get_error().is_empty(), "Live pond could not start")
	check(scene.current_strength.item_count == 2 and scene.current_strength.disabled, "Calm water must retain two disabled strength choices")
	scene.state = scene.lab.step()
	scene.previous = scene.state.duplicate(true)
	scene._poll()
	var before: Dictionary = scene.lab.get_state().duplicate(true)
	scene.current_switch.button_pressed = true
	check(scene.lab.get_state() == before, "Turning flow on moved or reset the world")
	check(not scene.current_strength.disabled and absf(scene.stats.water_current_cm_s - 0.5) < 0.0001, "Default live strength is not 0.5")
	# Exercise the actual OptionButton keyboard popup and selection route.
	scene.current_strength.grab_focus()
	key(KEY_SPACE)
	await settle()
	check(scene.current_strength.get_popup().visible, "Strength choices cannot be opened with the keyboard")
	key(KEY_DOWN)
	key(KEY_ENTER)
	await settle()
	check(scene.current_strength.selected == 1 and absf(scene.stats.water_current_cm_s - 0.75) < 0.0001, "Keyboard did not select the stronger physical current")
	check(scene.lab.get_state() == before, "Changing strength reset the body, food, clock or counters")
	check(scene.water_status.text.contains("0.75") and scene.stats.food_drifting, "Strength feedback or food-drift state is stale")
	var phase: Vector3 = scene.flow.displacement
	scene.flow.advance(2)
	check((scene.flow.displacement - phase).is_equal_approx(Vector3(0.375, 0, 0)), "Visual transport is not using 0.75 cm/s and the shared physical scale")
	# Validate the native live API rejects unsupported requests atomically.
	for invalid in [-0.1, 0.7501, INF, NAN]:
		check(not scene.lab.set_steady_current(invalid, true), "Invalid live current was accepted")
		check(scene.lab.get_state() == before and absf(scene.lab.get_stats().water_current_cm_s - 0.75) < 0.0001, "Rejected flow request mutated the world or current")
	check(scene.lab.set_steady_current(0.75, true), "A valid request did not clear the transient API error")
	scene._poll()
	# Food and flow speed use the same SI velocity. Initial food is in the interior.
	scene.state = scene.lab.step()
	for slot in 3:
		var movement: Vector3 = scene.state.foods[slot] - before.foods[slot]
		check(movement.is_equal_approx(Vector3(0.009375, 0, 0)), "Suspended food did not move at the selected physical speed")
	scene.previous = scene.state.duplicate(true)
	for tick in 1200:
		scene.state = scene.lab.step()
	scene.previous = scene.state.duplicate(true)
	scene._poll()
	check(scene.state.eaten > 0 and scene.stats.background_swimmers == 0 and scene.stats.selector_update == 320 and scene.stats.policy_update == 64350, "Stronger flow changed controllers or did not collect food")
	var samples: int = scene.curve.points.size()
	before = scene.lab.get_state().duplicate(true)
	scene.current_switch.button_pressed = false
	check(scene.lab.get_state() == before and scene.current_strength.selected == 1 and scene.current_strength.disabled, "Calm did not preserve the selected strength and world")
	check(scene.stats.water_current_cm_s == 0 and not scene.stats.food_drifting, "Calm retained imposed transport")
	scene.current_switch.button_pressed = true
	check(absf(scene.stats.water_current_cm_s - 0.75) < 0.0001 and scene.lab.get_state() == before, "Turning current back on lost the chosen strength")
	check(scene.curve.points.size() == samples, "Changing water reset the graph")
	scene._toggle_untrained(true)
	check(scene.stats.showcase_untrained and absf(scene.stats.water_current_cm_s - 0.75) < 0.0001, "Untrained comparison lost the stronger flow")
	scene._toggle_untrained(false)
	check(not scene.stats.showcase_untrained and absf(scene.stats.water_current_cm_s - 0.75) < 0.0001 and scene.stats.food_drifting, "Restoring the saved agent lost the flow or food mode")
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		await settle()
		scene.sidebar_scroll.ensure_control_visible(scene.current_strength)
		await settle()
		check(scene.current_strength.size.y >= 44 and scene.current_switch.size.y >= 44, "Current targets are too small")
		check(scene.current_strength.get_global_rect().end.x <= dimensions.x and scene.current_switch.get_global_rect().end.x <= scene.current_strength.get_global_rect().position.x, "Water controls overlap or clip")
		check(scene.viewport_container.size.y >= 100, "Current UI collapsed the pond viewport")
		if DisplayServer.get_name() != "headless":
			await RenderingServer.frame_post_draw
			check(root.get_texture().get_image().save_png(ProjectSettings.globalize_path(folder.path_join("strength-%dx%d.png" % [dimensions.x, dimensions.y]))) == OK, "Screenshot failed")
	check(FileAccess.get_sha256(POLICY) == policy_hash, "The strength control changed saved weights")
	print("CURRENT_STRENGTH backend=%s checks=%d failures=%d" % [scene.stats.compute_backend, checks, failures])
	quit(1 if failures else 0)

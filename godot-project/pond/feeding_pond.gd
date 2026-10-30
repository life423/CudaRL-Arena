extends "res://pond/deep_pond.gd"
const LATEST_POLICY := "res://../docs/experiments/swimmer/gentle-currents/bank-recovery/selected-policy.bin"
const CurrentControls = preload("res://pond/pond_current_controls.gd")
const WaterController = preload("res://pond/pond_water_controller.gd")
var water_controller := WaterController.new()
var current_switch: CheckButton
var current_strength: OptionButton
var water_status: Label
var flow: Node3D
@export var feeding_checkpoint_path := "user://pond-policies/feeding-learning-v1/training.bin"
var pellet_pool: MultiMeshInstance3D
var feed_feedback: Label
var feed_surface: Control
var drop_point := Vector2.ZERO
var feedback_until := 0
var last_touch_msec := -1000
var auto_feed_button: CheckButton
var auto_feed_enabled := false
var auto_feed_next_time := -1.0
var auto_feed_last_time := -1.0
var auto_feed_count := 0
var auto_feed_rng := RandomNumberGenerator.new()
var auto_feed_seeded := false

func _start_lab() -> bool:
	paused = false
	var backend := "auto"
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--pond-backend="):
			backend = argument.get_slice("=", 1)
	var policy: String = ProjectSettings.get_setting("cudarl/feeding_policy", LATEST_POLICY)
	if ProjectSettings.get_setting("cudarl/release_build", false):
		policy = OS.get_executable_path().get_base_dir().path_join("data/selected-policy.bin")
	return lab.start_feeding_learning(policy, feeding_checkpoint_path, population, backend)

func _build_ui() -> void:
	super._build_ui()
	var water: Dictionary = CurrentControls.build(self, sidebar_controls, camera_button)
	current_switch = water.switch
	current_strength = water.strength
	water_status = water.status
	water_controller.configure(lab, water)
	water_controller.changed.connect(_poll)
	untrained_button.hide()
	current_switch.tooltip_text = "Apply steady flow to this pond. Background practice independently includes calm, steady and varying currents."
	current_strength.tooltip_text = "Choose this pond's steady current. Background practice varies between calm water and 0.75 cm/s."
	feed_feedback = _label("Click or tap open water to drop food.", 14, MUTED)
	feed_feedback.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	var controls := camera_button.get_parent()
	controls.add_child(feed_feedback)
	controls.move_child(feed_feedback, 4)
	var accessible := _button("Drop food at pond center")
	accessible.name = "DropFoodAtCenter"
	accessible.pressed.connect(func(): _drop_at(Vector3.ZERO))
	controls.add_child(accessible)
	controls.move_child(accessible, 5)
	auto_feed_button = CheckButton.new()
	auto_feed_button.name = "AutoFeed"
	auto_feed_button.text = "Auto feed"
	auto_feed_button.custom_minimum_size.y = 44
	auto_feed_button.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	auto_feed_button.add_theme_font_size_override("font_size", 15)
	for color in ["font_color", "font_hover_color", "font_pressed_color", "font_focus_color"]:
		auto_feed_button.add_theme_color_override(color, TEXT)
	auto_feed_button.add_theme_stylebox_override("focus", camera_button.get_theme_stylebox("focus"))
	auto_feed_button.tooltip_text = "Drop one pellet every 2–6 seconds of pond time. You can still feed by hand; existing pellets are kept."
	auto_feed_button.toggled.connect(_set_auto_feed)
	controls.add_child(auto_feed_button)
	controls.move_child(auto_feed_button, 5)
	feed_surface = Control.new()
	feed_surface.name = "FeedingSurface"
	feed_surface.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	feed_surface.mouse_default_cursor_shape = Control.CURSOR_POINTING_HAND
	feed_surface.focus_mode = Control.FOCUS_ALL
	feed_surface.tooltip_text = "Click/tap open water. Keyboard: focus the pond, use arrow keys to position the drop, then Space."
	feed_surface.gui_input.connect(_feed_input)
	feed_surface.draw.connect(func():
		if feed_surface.has_focus() and drop_point != Vector2.ZERO:
			feed_surface.draw_arc(drop_point, 8, 0, TAU, 24, Color("ffc45d"), 2, true))
	feed_surface.focus_entered.connect(feed_surface.queue_redraw)
	feed_surface.focus_exited.connect(feed_surface.queue_redraw)
	viewport_container.add_child(feed_surface)

func _build_world() -> void:
	super._build_world()
	flow = preload("res://pond/current_flow.gd").new()
	flow.configure(water_surface.material_override)
	world.add_child(flow)
	water_controller.flow = flow
	for nodes in [foods, markers, lines, depth_labels]:
		for node: Node3D in nodes:
			node.hide()
	var mesh := SphereMesh.new()
	mesh.radius = 0.055
	mesh.height = 0.11
	mesh.radial_segments = 12
	mesh.rings = 6
	var material := _material(Color("ffc45d"))
	material.emission_enabled = true
	material.emission = Color("c89a43")
	material.emission_energy_multiplier = 0.6
	mesh.material = material
	var pool := MultiMesh.new()
	pool.transform_format = MultiMesh.TRANSFORM_3D
	pool.mesh = mesh
	pool.instance_count = 128
	pool.visible_instance_count = 0
	pellet_pool = MultiMeshInstance3D.new()
	pellet_pool.multimesh = pool
	world.add_child(pellet_pool)

func _after_lab_started() -> void:
	super._after_lab_started()
	if lab and lab.get_error().is_empty():
		released = true
	get_window().title = "Tadpole Pond · Feed and learn · CudaRL"
	if "--auto-feed" in OS.get_cmdline_user_args():
		auto_feed_button.button_pressed = true

func _set_auto_feed(enabled: bool) -> void:
	auto_feed_enabled = enabled
	if enabled:
		if not auto_feed_seeded:
			auto_feed_rng.seed = training_seed ^ 0xA17F00D
			auto_feed_seeded = true
		auto_feed_last_time = float(state.get("elapsed", 0.0))
		auto_feed_next_time = auto_feed_last_time + auto_feed_rng.randf_range(2.0, 6.0)
	else:
		auto_feed_next_time = -1.0
	feedback_until = 0
	_poll()

func _physics_process(delta: float) -> void:
	var before: float = state.get("elapsed", 0)
	super._physics_process(delta)
	if flow:
		flow.advance(float(state.get("elapsed", 0)) - before)
	_advance_auto_feed()

func _set_current(enabled: bool) -> void:
	water_controller.set_enabled(enabled)

func _select_strength(index: int) -> void:
	water_controller.select_strength(index, stats)

func _fit() -> void:
	super._fit()
	if not layout.vertical and size.y < 500:
		curve.custom_minimum_size.y = 148
	if layout.vertical:
		sidebar.custom_minimum_size.y = 242
		sidebar_scroll.custom_minimum_size.y = 206

func _advance_auto_feed() -> void:
	if not auto_feed_enabled or not lab or state.is_empty() or not stats.get("feeding_mode", false):
		return
	var elapsed: float = state.elapsed
	if elapsed <= auto_feed_last_time:
		if elapsed < auto_feed_last_time:
			auto_feed_next_time = elapsed + auto_feed_rng.randf_range(2.0, 6.0)
		auto_feed_last_time = elapsed
		return
	auto_feed_last_time = elapsed
	if elapsed < auto_feed_next_time:
		return
	# Consume a complete seeded event even when the pond is full, without deleting food.
	var angle := auto_feed_rng.randf_range(0.0, TAU)
	var radius := 0.86 * sqrt(auto_feed_rng.randf())
	var position := Vector3(RX * radius * cos(angle), 0, RZ * radius * sin(angle))
	auto_feed_next_time = elapsed + auto_feed_rng.randf_range(2.0, 6.0)
	if state.foods.size() >= int(stats.get("food_capacity", 128)):
		feed_feedback.text = "Auto feed is waiting for space in the pond (128 pellets)."
		feedback_until = Time.get_ticks_msec() + 4000
		return
	var slot: int = lab.drop_food(position)
	if slot < 0:
		feed_feedback.text = "Auto feed could not place food in open water."
		feedback_until = Time.get_ticks_msec() + 4000
		return
	auto_feed_count += 1
	state = lab.get_state()
	_poll()
	feed_feedback.text = "Food dropped automatically. You can still feed by hand."
	feedback_until = Time.get_ticks_msec() + 2500

func _feed_input(event: InputEvent) -> void:
	var point := Vector2.ZERO
	var requested := false
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
		if event.device < 0 and Time.get_ticks_msec() - last_touch_msec < 150:
			return
		point = event.position
		requested = true
	elif event is InputEventScreenTouch and event.pressed:
		last_touch_msec = Time.get_ticks_msec()
		point = event.position
		requested = true
	elif event is InputEventKey and event.pressed and not event.echo:
		if drop_point == Vector2.ZERO:
			drop_point = feed_surface.size * 0.5
		match event.keycode:
			KEY_LEFT: drop_point.x -= 24
			KEY_RIGHT: drop_point.x += 24
			KEY_UP: drop_point.y -= 24
			KEY_DOWN: drop_point.y += 24
			KEY_SPACE, KEY_ENTER: requested = true
		point = drop_point.clamp(Vector2.ZERO, feed_surface.size)
		drop_point = point
		feed_surface.queue_redraw()
		if event.keycode in [KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN]:
			feed_surface.accept_event()
	if requested:
		drop_point = point
		var pixel := point * Vector2(viewport.size) / feed_surface.size
		var hit: Variant = Plane(Vector3.UP, 0).intersects_ray(camera.project_ray_origin(pixel), camera.project_ray_normal(pixel))
		if hit is Vector3:
			_drop_at(hit)
		else:
			feed_feedback.text = "Use the pond overview to drop food on the surface."
			feedback_until = Time.get_ticks_msec() + 4000
		feed_surface.accept_event()

func _drop_at(position: Vector3) -> void:
	if not lab or not stats.get("feeding_mode", false):
		return
	var slot: int = lab.drop_food(position)
	if slot < 0:
		feed_feedback.text = "The pond has 128 pellets; let the creature eat some first." if stats.food_count >= stats.food_capacity else "Choose open water, clear of the shore."
		feedback_until = Time.get_ticks_msec() + 4000
		return
	state = lab.get_state()
	_poll()
	feed_feedback.text = "Food dropped."
	feedback_until = Time.get_ticks_msec() + 2500

func _apply_pose(alpha: float) -> void:
	if state.is_empty() or not pellet_pool:
		return
	var links: Array = []
	for i in 3:
		links.append({"position": (previous.links[i].position as Vector3).lerp(state.links[i].position, alpha), "orientation": (previous.links[i].orientation as Quaternion).slerp(state.links[i].orientation, alpha)})
	creature.pose(links)
	rendered_swimmer_position = links[0].position
	visibility_focus = (links[0].position as Vector3).lerp(links[2].position, 0.5)
	var positions: PackedVector3Array = state.foods
	pellet_pool.multimesh.visible_instance_count = positions.size()
	for i in positions.size():
		pellet_pool.multimesh.set_instance_transform(i, Transform3D(Basis.IDENTITY, positions[i]))

func _swimmer_text() -> String:
	return "Food collected  %d\nPellets in pond  %d\nDepth  %.1f cm\nSpeed  %.1f cm/s" % [state.eaten, state.foods.size(), state.depth_cm, state.speed_cm_s]

func _poll() -> void:
	super._poll()
	if lab and not lab.get_error().is_empty():
		title_status.text = "Feeding pond could not start"
		feed_feedback.text = lab.get_error() + " Return to the menu and try again."
		auto_feed_button.disabled = true
		return
	if not stats.get("feeding_mode", false):
		return
	water_controller.refresh(stats)
	untrained_button.hide()
	if Time.get_ticks_msec() > feedback_until:
		if auto_feed_enabled and state.foods.size() >= int(stats.get("food_capacity", 128)):
			feed_feedback.text = "Auto feed is waiting for space in the pond (128 pellets)."
		else:
			feed_feedback.text = "Auto feed · one pellet every 2–6 s. You can still drop food." if auto_feed_enabled else "Click or tap open water to drop food."
	var learning: bool = stats.learning_active
	title_status.text = "Learning live · PPO update %d" % stats.update if learning else ("Learning paused" if stats.paused else "Preparing background PPO…")
	if not str(stats.learning_error).is_empty():
		title_status.text = "Learning stopped: " + str(stats.learning_error)
	pause_button.show()
	pause_button.text = "Resume learning" if stats.paused else "Pause learning"
	find_child("SaveTrainingNow", true, false).show()
	training_heading.text = "Local PPO · swimming adaptation"
	var comparison := "First matched candidate check pending."
	if stats.evaluation_count > 0:
		comparison = "Matched check: candidate %.1f / min\nPreserved reference %.1f / min" % [stats.current_food_per_minute, stats.original_food_per_minute]
	training_stats.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	training_stats.text = "%s · %d background swimmers\nLearner update %d · adopted %d\nInteraction version %d · practiced %d\n%s\nPractice: timed food; calm, steady and varying currents (0–0.75 cm/s).\nThis pond's current uses your controls." % [stats.compute_backend, stats.background_swimmers, stats.update, stats.adopted_update, stats.context_version, stats.trained_context_version, comparison]
	curve.status_caption = "PPO %d · adopted %d · %d background swimmers" % [stats.update, stats.adopted_update, stats.background_swimmers]
	save_status.text = "Swimming retained from %d. Original policies preserved.\nSaved at %d · autosaves every 30 s and on exit." % [stats.policy_update, stats.saved_update]
	if not str(stats.save_error).is_empty():
		save_status.text = "Save failed: " + str(stats.save_error)
		feed_feedback.text = "Learning save failed. Open AI details."
	water_status.tooltip_text = "Food continues sinking under gravity. Current controls horizontal water-relative drag and drift."

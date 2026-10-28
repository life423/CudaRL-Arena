extends Control

const BG := Color("0b2429")
const TEXT := Color("eef4e9")
const MUTED := Color("b8cbc4")
const ACCENT := Color("faa640")
const LearningCurve = preload("res://pond/pond_curve.gd")
const Creature = preload("res://pond/deep_creature.gd")
@export var checkpoint_path := "user://pond-policies/deep-v1/training.bin"
@export var population := 1024
@export var training_seed := 7
var lab: RefCounted
var camera_direction := Vector3(0.57, 0.40, 0.72).normalized()
var overview_distance := 17.0
var layout: BoxContainer
var sidebar: PanelContainer
var sidebar_scroll: ScrollContainer
var visual: VBoxContainer
var viewport_container: SubViewportContainer
var viewport: SubViewport
var world: Node3D
var creature: Node3D
var food: MeshInstance3D
var food_marker: MeshInstance3D
var food_line: MeshInstance3D
var camera: Camera3D
var curve: Control
var title_status: Label
var swimmer_stats: Label
var training_stats: Label
var training_heading: Label
var save_status: Label
var pause_button: Button
var camera_button: Button
var speed_button: CheckButton
var state: Dictionary = {}
var previous: Dictionary = {}
var stats: Dictionary = {}
var accumulator := 0.0
var poll_time := 0.0
var startup := 0.0
var released := false
var paused := false
var overview := false
var evaluation_count := 0
var error_reported := false

func _ready() -> void:
	Engine.max_fps = 60
	DisplayServer.window_set_title("CudaRL · Deep-water swimming")
	_build_ui()
	_build_world()
	resized.connect(_fit)
	_fit()
	lab = _create_lab()
	if not _start_lab():
		title_status.text = lab.get_error()
		return
	state = lab.get_state()
	previous = state.duplicate(true)
	_poll()
	_apply_pose(1.0)

func _create_lab() -> RefCounted:
	return ClassDB.instantiate("DeepWaterLab")

func _start_lab() -> bool:
	return lab.start_training(population, training_seed, checkpoint_path)

func _exit_tree() -> void:
	if lab:
		lab.stop()

func _label(value: String, font_size := 16, color := TEXT) -> Label:
	var label := Label.new()
	label.text = value
	label.add_theme_font_size_override("font_size", font_size)
	label.add_theme_color_override("font_color", color)
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	return label

func _button(value: String, primary := false) -> Button:
	var button := Button.new()
	button.text = value
	button.custom_minimum_size = Vector2(0, 44)
	for kind in ["normal", "hover", "pressed"]:
		var style := StyleBoxFlat.new()
		style.bg_color = (ACCENT if primary else Color("263b3a")).lightened(0.08 if kind == "hover" else 0)
		style.set_corner_radius_all(5)
		style.content_margin_left = 12
		style.content_margin_right = 12
		button.add_theme_stylebox_override(kind, style)
	var focus := StyleBoxFlat.new()
	focus.bg_color = Color.TRANSPARENT
	focus.border_color = TEXT
	focus.set_border_width_all(2)
	button.add_theme_stylebox_override("focus", focus)
	for key in ["font_color", "font_hover_color", "font_pressed_color", "font_focus_color"]:
		button.add_theme_color_override(key, Color("17201e") if primary else TEXT)
	return button

func _build_ui() -> void:
	var background := ColorRect.new()
	background.color = BG
	background.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(background)
	layout = BoxContainer.new()
	layout.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	layout.add_theme_constant_override("separation", 0)
	add_child(layout)
	sidebar = PanelContainer.new()
	var panel := StyleBoxFlat.new()
	panel.bg_color = Color("1b2b29")
	panel.set_content_margin_all(18)
	sidebar.add_theme_stylebox_override("panel", panel)
	layout.add_child(sidebar)
	sidebar_scroll = ScrollContainer.new()
	sidebar_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	sidebar.add_child(sidebar_scroll)
	var controls := VBoxContainer.new()
	controls.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	controls.add_theme_constant_override("separation", 10)
	sidebar_scroll.add_child(controls)
	controls.add_child(_label("Deep-water lab", 25))
	title_status = _label("Preparing native PPO…", 14, MUTED)
	controls.add_child(title_status)
	var back := _button("Back to demos")
	back.name = "BackToDemos"
	back.pressed.connect(func(): DemoLaunch.show_menu())
	var navigation := HBoxContainer.new()
	navigation.add_theme_constant_override("separation", 8)
	controls.add_child(navigation)
	navigation.add_child(back)
	pause_button = _button("Pause learning", true)
	pause_button.pressed.connect(_toggle_pause)
	navigation.add_child(pause_button)
	for button: Button in [back, pause_button]:
		button.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		button.add_theme_font_size_override("font_size", 14)
	controls.add_child(HSeparator.new())
	swimmer_stats = _label("", 16)
	controls.add_child(swimmer_stats)
	camera_button = _button("Whole tank view")
	camera_button.pressed.connect(func(): overview = not overview; camera_button.text = "Follow swimmer" if overview else "Whole tank view")
	controls.add_child(camera_button)
	controls.add_child(HSeparator.new())
	training_heading = _label("Background training", 18)
	controls.add_child(training_heading)
	training_stats = _label("1,024 swimmers · native CUDA", 14, MUTED)
	controls.add_child(training_stats)
	speed_button = CheckButton.new()
	speed_button.text = "Maximum speed"
	speed_button.custom_minimum_size.y = 44
	speed_button.button_pressed = true
	speed_button.toggled.connect(func(enabled: bool): lab.set_updates_per_second(0 if enabled else 5))
	controls.add_child(speed_button)
	var save := _button("Save training now")
	save.name = "SaveTrainingNow"
	save.pressed.connect(func(): lab.save_training_now(); save_status.text = "Saving after this update…")
	controls.add_child(save)
	save_status = _label("Autosaves every 30 seconds and on exit.", 13, MUTED)
	controls.add_child(save_status)
	visual = VBoxContainer.new()
	visual.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	visual.size_flags_vertical = Control.SIZE_EXPAND_FILL
	visual.add_theme_constant_override("separation", 0)
	layout.add_child(visual)
	viewport_container = SubViewportContainer.new()
	viewport_container.stretch = true
	viewport_container.size_flags_vertical = Control.SIZE_EXPAND_FILL
	viewport_container.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	visual.add_child(viewport_container)
	viewport = SubViewport.new()
	viewport.own_world_3d = true
	viewport.msaa_3d = Viewport.MSAA_2X
	viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	viewport_container.add_child(viewport)
	curve = LearningCurve.new()
	curve.live_mode = false
	curve.empty_message = "Measuring the untrained swimmer…"
	visual.add_child(curve)

func _fit() -> void:
	var narrow := size.x < 700 or (size.x < 1000 and size.y > size.x)
	layout.vertical = narrow
	sidebar.custom_minimum_size = Vector2(0, 218) if narrow else Vector2(266, 0)
	sidebar.size_flags_vertical = Control.SIZE_FILL if narrow else Control.SIZE_EXPAND_FILL
	sidebar_scroll.custom_minimum_size.y = 180 if narrow else 0
	viewport_container.custom_minimum_size.y = 240 if narrow else 0
	curve.custom_minimum_size.y = 176 if narrow else 204

func _material(color: Color, roughness := 0.7) -> StandardMaterial3D:
	var material := StandardMaterial3D.new()
	material.albedo_color = color
	material.roughness = roughness
	return material

func _mesh(mesh: Mesh, material: Material, at: Vector3) -> MeshInstance3D:
	var instance := MeshInstance3D.new()
	instance.mesh = mesh
	instance.material_override = material
	instance.position = at
	world.add_child(instance)
	return instance

func _box(dimensions: Vector3, color: Color, at: Vector3) -> MeshInstance3D:
	var mesh := BoxMesh.new()
	mesh.size = dimensions
	return _mesh(mesh, _material(color), at)

func _sphere(radius: float, color: Color, at: Vector3) -> MeshInstance3D:
	var mesh := SphereMesh.new()
	mesh.radius = radius
	mesh.height = radius * 2
	return _mesh(mesh, _material(color, 0.3), at)

func _build_world() -> void:
	world = Node3D.new()
	viewport.add_child(world)
	var environment := Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = BG
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.ambient_light_color = Color("8ac4c0")
	environment.ambient_light_energy = 0.6
	environment.tonemap_mode = Environment.TONE_MAPPER_AGX
	environment.ssao_enabled = true
	var settings := WorldEnvironment.new()
	settings.environment = environment
	world.add_child(settings)
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-60, -30, 0)
	sun.light_color = Color("fff1cf")
	sun.light_energy = 1.5
	sun.shadow_enabled = true
	world.add_child(sun)
	var fill := OmniLight3D.new()
	fill.position = Vector3(2, -1, 2)
	fill.light_color = Color("7dc8db")
	fill.light_energy = 0.5
	fill.omni_range = 14
	world.add_child(fill)
	var sand := ShaderMaterial.new()
	sand.shader = preload("res://pond/sand.gdshader")
	var floor_mesh := BoxMesh.new()
	floor_mesh.size = Vector3(9.25, 0.22, 7.25)
	_mesh(floor_mesh, sand, Vector3(0, -4.11, 0))
	_box(Vector3(9.25, 4.15, 0.16), Color("1c4846"), Vector3(0, -1.95, -3.58))
	_box(Vector3(0.16, 4.15, 7.25), Color("22524c"), Vector3(-4.58, -1.95, 0))
	# Open front/right cutaway keeps the actual water volume visible.
	for height in [0.0, -1.0, -2.0, -3.0, -4.0]:
		_box(Vector3(9, 0.014, 0.014), Color("477c77"), Vector3(0, height, -3.48))
		_box(Vector3(0.014, 0.014, 7), Color("477c77"), Vector3(-4.48, height, 0))
		var label := Label3D.new()
		label.text = "%d cm" % roundi(-height * 4)
		label.font_size = 36
		label.pixel_size = 0.005
		label.position = Vector3(-4.0, height + 0.12, -3.42)
		label.modulate = Color("b2ccc0")
		world.add_child(label)
	var water_material := _material(Color(0.12, 0.49, 0.5, 0.10), 0.2)
	water_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	water_material.cull_mode = BaseMaterial3D.CULL_DISABLED
	var surface := PlaneMesh.new()
	surface.size = Vector2(9, 7)
	var water := _mesh(surface, water_material, Vector3.ZERO)
	water.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	for z in [-3.5, 3.5]:
		_box(Vector3(9.1, 0.055, 0.055), Color("73aaa1"), Vector3(0, 0, z))
	for x in [-4.5, 4.5]:
		_box(Vector3(0.055, 0.055, 7), Color("73aaa1"), Vector3(x, 0, 0))
	var rng := RandomNumberGenerator.new()
	rng.seed = 41
	for i in 22:
		var at := Vector3(rng.randf_range(-4.1, 4.1), -3.94, rng.randf_range(-3.1, -2.4))
		var rock := _sphere(rng.randf_range(0.12, 0.27), Color("768275"), at)
		rock.scale = Vector3(1.3, 0.6, 0.8)
		# Decorative rocks are beneath the floor collision envelope.
		rock.position.y = -4.10
	creature = Creature.new()
	world.add_child(creature)
	food = _sphere(0.09, Color("ffc45d"), Vector3.ZERO)
	var food_material := food.material_override as StandardMaterial3D
	food_material.emission_enabled = true
	food_material.emission = Color("c89a43")
	food_material.emission_energy_multiplier = 0.65
	var marker := CylinderMesh.new()
	marker.top_radius = 0.10
	marker.bottom_radius = 0.10
	marker.height = 0.008
	food_marker = _mesh(marker, _material(Color("aa9961")), Vector3(0, -3.99, 0))
	var line := CylinderMesh.new()
	line.top_radius = 0.007
	line.bottom_radius = 0.007
	line.height = 1
	food_line = _mesh(line, _material(Color(0.85, 0.76, 0.45, 0.30)), Vector3.ZERO)
	(food_line.material_override as StandardMaterial3D).transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	food_line.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	camera = Camera3D.new()
	camera.fov = 44
	camera.near = 0.05
	camera.far = 50
	camera.position = Vector3(9.0, 5.5, 12.5)
	world.add_child(camera)
	camera.look_at(Vector3(0, -1.8, 0))
	camera.current = true

func _toggle_pause() -> void:
	paused = not paused
	lab.set_paused(paused)
	pause_button.text = "Resume learning" if paused else "Pause learning"

func _physics_process(delta: float) -> void:
	if not lab or state.is_empty() or not released:
		return
	accumulator += delta
	while accumulator >= 0.05:
		previous = state.duplicate(true)
		state = lab.step()
		accumulator -= 0.05

func _process(delta: float) -> void:
	startup += delta
	poll_time += delta
	if poll_time >= 0.25 and lab:
		poll_time = 0
		_poll()
	if not released and startup >= 2 and stats.get("training_ready", false):
		released = true
		lab.set_paused(paused)
	if state.is_empty():
		return
	_apply_pose(clampf(accumulator / 0.05, 0, 1))
	_update_camera(delta)

func _update_camera(delta: float) -> void:
	var target: Vector3 = _camera_target()
	var distance := overview_distance if overview else maxf(7, (state.position as Vector3).distance_to(state.food) * 1.8 + 3)
	if size.x < 700:
		distance *= 1.35
	var offset := camera_direction * distance
	camera.position = camera.position.lerp(target + offset, 1 - exp(-delta * 3))
	camera.look_at(target)

func _camera_target() -> Vector3:
	return Vector3(0, -1.8, 0) if overview else (state.position + state.food) * 0.5

func _apply_pose(alpha: float) -> void:
	var links: Array = []
	for i in 3:
		var a: Dictionary = previous.links[i]
		var b: Dictionary = state.links[i]
		links.append({"position": (a.position as Vector3).lerp(b.position, alpha), "orientation": (a.orientation as Quaternion).slerp(b.orientation, alpha)})
	creature.pose(links)
	food.position = state.food
	food_marker.position = Vector3(food.position.x, -3.985, food.position.z)
	food_line.position = (food.position + food_marker.position) * 0.5
	food_line.scale.y = food.position.y - food_marker.position.y

func _poll() -> void:
	stats = lab.get_stats()
	var error: String = lab.get_error()
	if not error.is_empty():
		title_status.text = "Training could not start."
		save_status.text = error + "\nIf headless training is running, stop it with Ctrl+C first, then reopen this demo."
		pause_button.disabled = true
		return
	title_status.text = "%s · update %s" % ["Paused" if paused else ("Learning" if released else "Preparing"), str(stats.update)]
	if not state.is_empty():
		swimmer_stats.text = _swimmer_text()
	var update_ms: float = stats.update_ms
	var steps_per_second := population * 64 * 1000.0 / update_ms if update_ms > 0 else 0.0
	training_stats.text = "%s swimmers\n%s simulation steps\n%.2fM steps / second\n%.1f× simulation time / world" % [String.num_int64(population), String.num_int64(stats.env_steps), steps_per_second / 1e6, steps_per_second * 0.05 / population]
	save_status.text = "%s · saved at %d\nAutosaves every 30 s and on exit." % ["Resumed from %d" % stats.loaded_update if stats.resumed else "New 3D controller", stats.saved_update]
	if not str(stats.save_error).is_empty():
		save_status.text = "Save failed: " + str(stats.save_error)
	for point: Dictionary in lab.get_evaluation_history(evaluation_count):
		curve.points.append(Vector2(point.update, point.food_per_minute))
		var best := maxf(point.food_per_minute, curve.best_points[-1].y if not curve.best_points.is_empty() else 0)
		curve.best_points.append(Vector2(point.update, best))
		evaluation_count += 1
	curve.current_score = stats.current_food_per_minute if evaluation_count > 0 else -1
	curve.best_score = stats.best_food_per_minute if evaluation_count > 0 else -1
	curve.best_update = stats.best_update
	curve.best_saved = stats.saved_update >= stats.best_update and stats.best_update >= 0
	curve.queue_redraw()

func _swimmer_text() -> String:
	return "One food · free 3D swimming\n\nFood eaten  %d\nDepth  %.1f cm\nFood depth  %.1f cm\nSpeed  %.1f cm/s" % [state.eaten, state.depth_cm, state.food_depth_cm, state.speed_cm_s]

func _unhandled_key_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_cancel"):
		DemoLaunch.show_menu()
		get_viewport().set_input_as_handled()

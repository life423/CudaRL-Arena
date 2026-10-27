extends Node3D

const RX := 4.6
const RZ := 3.5
const WATER_Y := 0.48
const SWIM_Y := 0.25
const STEP := 0.05
const TEXT := Color("eef4e9")
const MUTED := Color("b8cbc4")
const ACCENT := Color("faa640")
@export var live_learning := true
@export var training_updates_per_second := 10.0
var lab := TadpoleLab.new()
var current: Dictionary = {}
var previous: Dictionary = {}
var blend := 1.0
var camera: Camera3D
var creature: Node3D
var tail: MeshInstance3D
var membrane: MeshInstance3D
var food: MeshInstance3D
var foods: Array[MeshInstance3D] = []
var food_tweens: Array[Tween] = []
var food_times: Array[float] = []
var food_rate_value: Label
var eyes: Array[Node3D] = []
var pupils: Array[MeshInstance3D] = []
var food_count: Label
var status: Label
var move_button: Button
var restart_button: Button
var sparks: Array[MeshInstance3D] = []
var spark_origin := Vector3.ZERO
var spark_age := 1.0
var old_tick_rate := 60
var ready_policy := false
var warmup_timer: Timer
var last_reported_update := -50
var ui_layout: BoxContainer
var side_panel: PanelContainer
var panel_scroll: ScrollContainer
var pond_container: SubViewportContainer
var pond_viewport: SubViewport
var curve_view
var speed_value: Label
var meal_value: Label
var background_value: Label
var steps_value: Label
var throughput_value: Label
var pause_button: Button
var speed_slider: HSlider
var speed_label: Label
var maximum_speed: CheckBox
var user_paused := false
var evaluation_points := 0
var rate_timestamp := 0
var rate_steps := 0
var metric_layout: BoxContainer
var settings_button: Button
var settings_panel: VBoxContainer
var settings_open := false
var pond_actions: BoxContainer
var back_button: Button

func _ready() -> void:
	get_window().title = "CudaRL · Tadpole pond"
	old_tick_rate = Engine.physics_ticks_per_second
	Engine.physics_ticks_per_second = 20
	get_viewport().msaa_3d = Viewport.MSAA_4X
	_build_world()
	_build_tadpole()
	_build_ui()
	warmup_timer = Timer.new()
	warmup_timer.one_shot = true
	warmup_timer.timeout.connect(_resume_learning)
	add_child(warmup_timer)
	get_viewport().size_changed.connect(_fit_camera)
	_fit_camera()
	_start_pond.call_deferred()

func _exit_tree() -> void:
	lab.stop()
	Engine.physics_ticks_per_second = old_tick_rate

func _unhandled_key_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_ESCAPE:
		get_viewport().set_input_as_handled()
		DemoLaunch.show_menu()

func _material(color: Color, roughness := 0.7) -> StandardMaterial3D:
	var material := StandardMaterial3D.new()
	material.albedo_color = color
	material.roughness = roughness
	return material

func _mesh(mesh: Mesh, material: Material, parent: Node3D, position := Vector3.ZERO) -> MeshInstance3D:
	var instance := MeshInstance3D.new()
	instance.mesh = mesh
	instance.material_override = material
	instance.position = position
	parent.add_child(instance)
	return instance

func _sphere(radius: float, color: Color, parent: Node3D, position := Vector3.ZERO) -> MeshInstance3D:
	var sphere := SphereMesh.new()
	sphere.radius = radius
	sphere.height = radius * 2.0
	sphere.radial_segments = 24
	sphere.rings = 12
	return _mesh(sphere, _material(color, 0.28), parent, position)

func _surface(vertices: PackedVector3Array, normals: PackedVector3Array, uv: PackedVector2Array, indices: PackedInt32Array) -> ArrayMesh:
	var arrays: Array = []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_NORMAL] = normals
	arrays[Mesh.ARRAY_TEX_UV] = uv
	arrays[Mesh.ARRAY_INDEX] = indices
	var result := ArrayMesh.new()
	result.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	return result

func _floor_y(radius: float) -> float:
	return -0.18 + 0.22 * radius * radius + 0.58 * smoothstep(0.85, 1.08, radius)

func _radial_mesh(rings: int, segments: int, extent: float, water: bool) -> ArrayMesh:
	var vertices := PackedVector3Array()
	var normals := PackedVector3Array()
	var uv := PackedVector2Array()
	var indices := PackedInt32Array()
	for ring in rings + 1:
		var radius := float(ring) / rings * extent
		for i in segments + 1:
			var angle := TAU * float(i) / segments
			var x := cos(angle) * radius
			var z := sin(angle) * radius
			vertices.append(Vector3(x * RX, WATER_Y if water else _floor_y(radius), z * RZ))
			var slope := 0.0 if water else (_floor_y(radius + 0.001) - _floor_y(maxf(0.0, radius - 0.001))) / 0.002
			normals.append(Vector3(-slope * cos(angle) / RX, 1.0, -slope * sin(angle) / RZ).normalized())
			uv.append(Vector2(x, z) * 0.5 + Vector2.ONE * 0.5)
	for ring in rings:
		for i in segments:
			var a := ring * (segments + 1) + i
			var b := a + segments + 1
			indices.append_array(PackedInt32Array([a, b, a + 1, a + 1, b, b + 1]))
	return _surface(vertices, normals, uv, indices)

func _build_world() -> void:
	var environment := Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = Color("15302e")
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.ambient_light_color = Color("91b9b5")
	environment.ambient_light_energy = 0.65
	environment.tonemap_mode = Environment.TONE_MAPPER_AGX
	environment.ssao_enabled = true
	environment.ssao_radius = 0.7
	environment.ssao_intensity = 0.7
	var world_environment := WorldEnvironment.new()
	world_environment.environment = environment
	add_child(world_environment)
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-58, -28, 0)
	sun.light_color = Color("fff0d2")
	sun.light_energy = 1.8
	sun.shadow_enabled = true
	add_child(sun)
	var fill := DirectionalLight3D.new()
	fill.rotation_degrees = Vector3(-35, 150, 0)
	fill.light_color = Color("8fc9d1")
	fill.light_energy = 0.35
	add_child(fill)
	var sand := ShaderMaterial.new()
	sand.shader = preload("res://pond/sand.gdshader")
	_mesh(_radial_mesh(36, 96, 1.25, false), sand, self)
	var base := CylinderMesh.new()
	base.top_radius = 1.25
	base.bottom_radius = 1.20
	base.height = 1.49
	base.cap_top = false
	base.radial_segments = 96
	var bank := _mesh(base, _material(Color("343b2d")), self, Vector3(0, -0.005, 0))
	bank.scale = Vector3(RX, 1, RZ)
	var water_material := ShaderMaterial.new()
	water_material.shader = preload("res://pond/water.gdshader")
	var water := _mesh(_radial_mesh(12, 96, 1.0, true), water_material, self)
	water.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	var rng := RandomNumberGenerator.new()
	rng.seed = 182
	for i in 28:
		var angle := rng.randf_range(0, TAU)
		var radius := rng.randf_range(1.015, 1.17)
		var rock := _sphere(rng.randf_range(0.11, 0.23), Color("68776b").lerp(Color("9a9b7b"), rng.randf()), self, Vector3(cos(angle) * radius * RX, _floor_y(radius) + 0.05, sin(angle) * radius * RZ))
		rock.scale = Vector3(rng.randf_range(1.1, 1.8), rng.randf_range(0.5, 0.8), rng.randf_range(0.7, 1.3))
		rock.rotation.y = rng.randf_range(0, TAU)
	for cluster in 4:
		var angle := PI * (1.12 + cluster * 0.22)
		var radius := 1.08
		var center := Vector3(cos(angle) * RX * radius, _floor_y(radius), sin(angle) * RZ * radius)
		for stem in 7:
			var height := rng.randf_range(0.35, 0.8)
			var stalk := CylinderMesh.new()
			stalk.top_radius = 0.008
			stalk.bottom_radius = 0.018
			stalk.height = height
			stalk.radial_segments = 6
			var reed := _mesh(stalk, _material(Color("436d48")), self, center + Vector3(rng.randf_range(-0.2, 0.2), height * 0.5, rng.randf_range(-0.16, 0.16)))
			reed.rotation.z = rng.randf_range(-0.16, 0.16)
	for i in (3 if live_learning else 1):
		var piece := _sphere(0.09, Color("ffc25c"), self)
		var food_material := piece.material_override as StandardMaterial3D
		food_material.emission_enabled = true
		food_material.emission = Color("e6a23e")
		food_material.emission_energy_multiplier = 0.5
		var light := OmniLight3D.new()
		light.light_color = Color("ffc25c")
		light.light_energy = 0.35
		light.omni_range = 0.85
		piece.add_child(light)
		foods.append(piece)
		food_tweens.append(null)
	food = foods[0]
	for i in 8:
		var spark := _sphere(0.018, Color("ffd98b"), self)
		spark.visible = false
		spark.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		sparks.append(spark)
	camera = Camera3D.new()
	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	camera.position = Vector3(0, 8.3, 8.7)
	camera.near = 0.1
	camera.far = 40.0
	add_child(camera)
	camera.look_at(Vector3(0, 0.1, 0))
	camera.current = true

func _head_radius(t: float) -> float:
	return (0.06 + 0.17 * sin(PI * t) * (0.6 + 0.4 * t)) * sqrt(maxf(0.0, 1.0 - pow(absf(2.0 * t - 1.0), 12)))

func _head_mesh() -> ArrayMesh:
	var vertices := PackedVector3Array()
	var normals := PackedVector3Array()
	var uv := PackedVector2Array()
	var indices := PackedInt32Array()
	for ring in 25:
		var t := float(ring) / 24.0
		var radius := _head_radius(t)
		var slope := (_head_radius(minf(1, t + 0.001)) - _head_radius(maxf(0, t - 0.001))) / 0.001
		for side in 25:
			var angle := TAU * float(side) / 24.0
			vertices.append(Vector3((t - 0.5) * 0.5, sin(angle) * radius * 0.8, cos(angle) * radius))
			normals.append(Vector3(-slope, sin(angle) / 0.8, cos(angle)).normalized())
			uv.append(Vector2(t, float(side) / 24.0))
	for ring in 24:
		for side in 24:
			var a := ring * 25 + side
			indices.append_array(PackedInt32Array([a, a + 25, a + 1, a + 1, a + 25, a + 26]))
	return _surface(vertices, normals, uv, indices)

func _build_tadpole() -> void:
	creature = Node3D.new()
	add_child(creature)
	var skin := ShaderMaterial.new()
	skin.shader = preload("res://pond/skin.gdshader")
	_mesh(_head_mesh(), skin, creature)
	tail = MeshInstance3D.new()
	tail.material_override = skin
	creature.add_child(tail)
	membrane = MeshInstance3D.new()
	var fin_material := _material(Color(0.17, 0.64, 0.55, 0.68), 0.38)
	fin_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	fin_material.cull_mode = BaseMaterial3D.CULL_DISABLED
	membrane.material_override = fin_material
	membrane.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	creature.add_child(membrane)
	for side in [-1, 1]:
		var eye := Node3D.new()
		eye.position = Vector3(0.12, 0.085, side * 0.13)
		creature.add_child(eye)
		_sphere(0.059, Color("e7eee0"), eye)
		var pupil := _sphere(0.033, Color("102824"), eye)
		_sphere(0.010, Color("ffffff"), pupil, Vector3(0.012, 0.018, 0.012))
		eyes.append(eye)
		pupils.append(pupil)

func _curve(p0: Vector3, p1: Vector3, p2: Vector3, p3: Vector3, t: float) -> Vector3:
	return 0.5 * ((2.0 * p1) + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t)

func _tail_point(t: float, joint0: float, joint1: float) -> Vector3:
	var a := Vector3(-0.25, 0, 0)
	var b := a - Vector3(cos(joint0), 0, sin(joint0)) * 0.65
	var c := b - Vector3(cos(joint0 + joint1), 0, sin(joint0 + joint1)) * 0.65
	if t < 0.5:
		return _curve(a + Vector3(0.65, 0, 0), a, b, c, t * 2.0)
	return _curve(a, b, c, c + c - b, (t - 0.5) * 2.0)

func _update_tail(joint0: float, joint1: float) -> void:
	var vertices := PackedVector3Array()
	var normals := PackedVector3Array()
	var uv := PackedVector2Array()
	var indices := PackedInt32Array()
	var fin_vertices := PackedVector3Array()
	var fin_normals := PackedVector3Array()
	var fin_uv := PackedVector2Array()
	var fin_indices := PackedInt32Array()
	for ring in 29:
		var t := float(ring) / 28.0
		var center := _tail_point(t, joint0, joint1)
		var tangent := (_tail_point(minf(1, t + 0.001), joint0, joint1) - _tail_point(maxf(0, t - 0.001), joint0, joint1)).normalized()
		var side := Vector3(-tangent.z, 0, tangent.x)
		for spoke in 11:
			var angle := TAU * float(spoke) / 10.0
			var normal := Vector3.UP * sin(angle) + side * cos(angle)
			vertices.append(center + normal * lerpf(0.055, 0.018, t))
			normals.append(normal)
			uv.append(Vector2(t, float(spoke) / 10.0))
		var fin_height := 0.025 + 0.19 * sin(PI * clampf((t - 0.35) / 0.65, 0, 1))
		for sign_y in [-1, 1]:
			fin_vertices.append(center + Vector3.UP * fin_height * sign_y)
			fin_normals.append(side)
			fin_uv.append(Vector2(t, (sign_y + 1) * 0.5))
	for ring in 28:
		for spoke in 10:
			var a := ring * 11 + spoke
			indices.append_array(PackedInt32Array([a, a + 11, a + 1, a + 1, a + 11, a + 12]))
		var b := ring * 2
		fin_indices.append_array(PackedInt32Array([b, b + 2, b + 1, b + 1, b + 2, b + 3]))
	tail.mesh = _surface(vertices, normals, uv, indices)
	membrane.mesh = _surface(fin_vertices, fin_normals, fin_uv, fin_indices)

func _label(text: String, size: int, color := TEXT) -> Label:
	var label := Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", size)
	label.add_theme_color_override("font_color", color)
	return label

func _button(text: String) -> Button:
	var button := Button.new()
	button.text = text
	button.custom_minimum_size = Vector2(104, 44)
	return button

func _build_ui() -> void:
	var canvas := CanvasLayer.new()
	add_child(canvas)
	var root_ui := Control.new()
	root_ui.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	canvas.add_child(root_ui)
	var theme := Theme.new()
	theme.default_font_size = 14
	for kind in ["Label", "Button", "CheckBox"]:
		theme.set_color("font_color", kind, TEXT)
	var normal := StyleBoxFlat.new()
	normal.bg_color = Color("263631")
	normal.border_color = Color("50645c")
	normal.set_border_width_all(1)
	normal.set_corner_radius_all(5)
	var hover := normal.duplicate() as StyleBoxFlat
	hover.bg_color = Color("354a42")
	var focus := StyleBoxFlat.new()
	focus.bg_color = Color.TRANSPARENT
	focus.border_color = ACCENT
	focus.set_border_width_all(2)
	focus.set_corner_radius_all(5)
	theme.set_stylebox("normal", "Button", normal)
	theme.set_stylebox("hover", "Button", hover)
	theme.set_stylebox("pressed", "Button", hover)
	theme.set_stylebox("focus", "Button", focus)
	theme.set_stylebox("focus", "CheckBox", focus)
	theme.set_stylebox("focus", "HSlider", focus)
	root_ui.theme = theme
	ui_layout = BoxContainer.new()
	ui_layout.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	ui_layout.add_theme_constant_override("separation", 0)
	root_ui.add_child(ui_layout)
	side_panel = PanelContainer.new()
	var surface := StyleBoxFlat.new()
	surface.bg_color = Color("1b2825")
	side_panel.add_theme_stylebox_override("panel", surface)
	ui_layout.add_child(side_panel)
	panel_scroll = ScrollContainer.new()
	panel_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	side_panel.add_child(panel_scroll)
	var margin := MarginContainer.new()
	margin.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	for side in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 16)
	panel_scroll.add_child(margin)
	var text := VBoxContainer.new()
	text.add_theme_constant_override("separation", 10)
	text.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	margin.add_child(text)
	text.add_child(_label("Tadpole lab", 23))
	status = _label("Preparing a fresh learner…" if live_learning else "Trained controller · shallow water", 14, MUTED)
	status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	text.add_child(status)
	var navigation := HBoxContainer.new()
	navigation.add_theme_constant_override("separation", 8)
	text.add_child(navigation)
	back_button = _button("Back to demos")
	back_button.name = "BackToDemos"
	back_button.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	back_button.pressed.connect(DemoLaunch.show_menu)
	navigation.add_child(back_button)
	pause_button = _button("Pause learning")
	pause_button.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	var primary := normal.duplicate() as StyleBoxFlat
	primary.bg_color = ACCENT
	primary.border_color = ACCENT
	pause_button.add_theme_stylebox_override("normal", primary)
	var primary_hover := primary.duplicate() as StyleBoxFlat
	primary_hover.bg_color = ACCENT.lightened(0.12)
	pause_button.add_theme_stylebox_override("hover", primary_hover)
	pause_button.add_theme_stylebox_override("pressed", primary_hover)
	for state in ["font_color", "font_hover_color", "font_pressed_color", "font_focus_color"]:
		pause_button.add_theme_color_override(state, Color("20251e"))
	pause_button.pressed.connect(_toggle_pause)
	pause_button.visible = live_learning
	navigation.add_child(pause_button)
	text.add_child(HSeparator.new())
	metric_layout = BoxContainer.new()
	metric_layout.add_theme_constant_override("separation", 16)
	text.add_child(metric_layout)
	var visible_group := VBoxContainer.new()
	visible_group.add_theme_constant_override("separation", 10)
	visible_group.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	metric_layout.add_child(visible_group)
	visible_group.add_child(_label("Three-food pond" if live_learning else "Visible tadpole", 16))
	food_count = _label("Food eaten  0", 18)
	visible_group.add_child(food_count)
	food_rate_value = _label("Food / min  —", 14, MUTED)
	food_rate_value.tooltip_text = "Collections in the last 60 simulated seconds. The first full minute is needed before this value appears."
	food_rate_value.visible = live_learning
	visible_group.add_child(food_rate_value)
	speed_value = _label("Swimming speed  0.0 cm/s", 14, MUTED)
	speed_value.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	visible_group.add_child(speed_value)
	meal_value = _label("Since last food  0.0 s", 14, MUTED)
	visible_group.add_child(meal_value)
	var actions := BoxContainer.new()
	pond_actions = actions
	actions.add_theme_constant_override("separation", 8)
	visible_group.add_child(actions)
	restart_button = _button("Start over" if live_learning else "Restart")
	restart_button.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	restart_button.tooltip_text = "Restart from random weights with 1,024 background swimmers." if live_learning else "Reset the pond."
	restart_button.pressed.connect(_restart)
	actions.add_child(restart_button)
	move_button = _button("Move food")
	move_button.tooltip_text = "Offer a different food position without moving the tadpole."
	move_button.pressed.connect(_move_food)
	move_button.visible = false
	actions.add_child(move_button)
	var training_section := VBoxContainer.new()
	training_section.add_theme_constant_override("separation", 10)
	training_section.visible = live_learning
	training_section.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	metric_layout.add_child(training_section)
	training_section.add_child(_label("Background training", 16))
	background_value = _label("1,024 swimmers", 14)
	training_section.add_child(background_value)
	steps_value = _label("0 simulation steps", 14, MUTED)
	steps_value.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	training_section.add_child(steps_value)
	throughput_value = _label("0 steps / second", 14, MUTED)
	throughput_value.tooltip_text = "Actual training steps per wall-clock second, including pacing and evaluation. This is not GPU utilization."
	training_section.add_child(throughput_value)
	settings_button = _button("Training speed ▾")
	settings_button.pressed.connect(func():
		settings_open = not settings_open
		settings_panel.visible = settings_open
		settings_button.text = "Training speed ▴" if settings_open else "Training speed ▾")
	training_section.add_child(settings_button)
	settings_panel = VBoxContainer.new()
	settings_panel.add_theme_constant_override("separation", 10)
	training_section.add_child(settings_panel)
	speed_label = _label("Training speed · 10 updates/s", 14)
	speed_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	settings_panel.add_child(speed_label)
	speed_slider = HSlider.new()
	speed_slider.min_value = 1
	speed_slider.max_value = 30
	speed_slider.step = 1
	speed_slider.value = clampf(training_updates_per_second, 1, 30)
	speed_slider.custom_minimum_size.y = 44
	speed_slider.tooltip_text = "Training updates per second. Left and right arrow keys adjust the speed."
	speed_slider.value_changed.connect(_set_training_speed)
	settings_panel.add_child(speed_slider)
	maximum_speed = CheckBox.new()
	maximum_speed.text = "Maximum speed"
	maximum_speed.tooltip_text = "Train as fast as the hardware allows, without the viewing throttle."
	maximum_speed.custom_minimum_size.y = 44
	maximum_speed.button_pressed = training_updates_per_second <= 0
	maximum_speed.toggled.connect(func(_enabled: bool): _set_training_speed(speed_slider.value))
	settings_panel.add_child(maximum_speed)
	var main := VBoxContainer.new()
	main.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	main.size_flags_vertical = Control.SIZE_EXPAND_FILL
	main.add_theme_constant_override("separation", 0)
	ui_layout.add_child(main)
	pond_container = SubViewportContainer.new()
	pond_container.stretch = true
	pond_container.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	pond_container.size_flags_vertical = Control.SIZE_EXPAND_FILL
	pond_container.custom_minimum_size = Vector2(0, 220)
	pond_container.mouse_filter = Control.MOUSE_FILTER_IGNORE
	main.add_child(pond_container)
	pond_viewport = SubViewport.new()
	pond_viewport.world_3d = get_world_3d()
	pond_viewport.msaa_3d = Viewport.MSAA_4X
	pond_viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	pond_container.add_child(pond_viewport)
	camera.reparent(pond_viewport)
	camera.current = true
	pond_viewport.size_changed.connect(_fit_camera)
	curve_view = preload("res://pond/pond_curve.gd").new()
	curve_view.tooltip_text = "The same 64 separate three-food ponds are tested for 60 simulated seconds every 25 updates. The score is average food collected per minute. Trials never train the policy."
	if not live_learning:
		curve_view.empty_message = "Live evaluation is available while learning."
	main.add_child(curve_view)
	_fit_layout()
	get_viewport().size_changed.connect(_fit_layout)
	_set_training_speed(speed_slider.value)

func _fit_layout() -> void:
	var dimensions := get_viewport().get_visible_rect().size
	var compact := dimensions.x < 900 and dimensions.y > dimensions.x
	ui_layout.vertical = compact
	metric_layout.vertical = not compact
	pond_actions.vertical = compact
	side_panel.custom_minimum_size = Vector2(0, 372) if compact else Vector2(264, 0)
	side_panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL if compact else Control.SIZE_FILL
	side_panel.size_flags_vertical = Control.SIZE_FILL if compact else Control.SIZE_EXPAND_FILL
	settings_button.visible = compact and live_learning
	settings_panel.visible = (not compact or settings_open) and live_learning
	curve_view.custom_minimum_size.y = 136 if dimensions.y < 600 else 168
	_fit_camera()

func _toggle_pause() -> void:
	user_paused = not user_paused
	warmup_timer.stop()
	lab.set_paused(user_paused)
	_update_learning_status()

func _set_training_speed(value: float) -> void:
	if maximum_speed == null:
		return
	training_updates_per_second = 0.0 if maximum_speed.button_pressed else value
	speed_slider.editable = not maximum_speed.button_pressed
	speed_label.text = "Training speed · maximum" if maximum_speed.button_pressed else "Training speed · %d updates/s" % int(value)
	lab.set_updates_per_second(training_updates_per_second)

func _start_pond() -> void:
	if live_learning:
		_begin_learning()
	else:
		_load_saved_policy()

func _begin_learning() -> void:
	last_reported_update = -50
	user_paused = false
	evaluation_points = 0
	curve_view.points.clear()
	curve_view.queue_redraw()
	rate_timestamp = Time.get_ticks_msec()
	rate_steps = 0
	food_times.clear()
	food_rate_value.text = "Food / min  —"
	throughput_value.text = "0 steps / second"
	ready_policy = false
	status.text = "Preparing a fresh learner…"
	ready_policy = lab.start_training(1024, 1, training_updates_per_second)
	if not ready_policy:
		status.text = lab.get_error()
		return
	current = lab.get_state()
	previous = current.duplicate()
	blend = 1.0
	food_count.text = "Food eaten  0"
	move_button.visible = false
	_apply_pose(current)
	_place_food_visual(true)
	_update_learning_status()
	# Show the genuine random update-zero policy before learning starts.
	warmup_timer.start(2.0)

func _resume_learning() -> void:
	if live_learning and ready_policy and not user_paused:
		lab.set_paused(false)

func _load_saved_policy() -> void:
	var file := FileAccess.open("res://pond/policies/tadpole.json", FileAccess.READ)
	if file == null:
		status.text = "The saved tadpole policy could not be opened."
		return
	var data: Variant = JSON.parse_string(file.get_as_text())
	if not data is Dictionary or data.get("inputs", 0) != 9 or data.get("hidden", 0) != 32 or data.get("outputs", 0) != 2 or data.get("body_id", -1) != 6:
		status.text = "The saved policy does not match this tadpole."
		return
	ready_policy = lab.watch_policy(PackedFloat32Array(data.parameters), 1)
	if not ready_policy:
		status.text = lab.get_error()
		return
	current = lab.get_state()
	previous = current.duplicate()
	_apply_pose(current)
	_place_food_visual(false)

func _restart() -> void:
	if live_learning:
		_begin_learning()
		return
	if not ready_policy:
		return
	lab.reset(1)
	current = lab.get_state()
	previous = current.duplicate()
	blend = 1.0
	food_count.text = "Food eaten  0"
	move_button.visible = false
	_apply_pose(current)
	_place_food_visual(true)

func _move_food() -> void:
	lab.move_food()
	current = lab.get_state()
	_place_food_visual(true)

func _place_food_visual(animate: bool, changed_slot := -1) -> void:
	var positions: PackedVector2Array = current.foods
	for i in foods.size():
		foods[i].position = Vector3(positions[i].x, SWIM_Y, positions[i].y)
		if changed_slot >= 0 and i != changed_slot:
			continue
		if food_tweens[i] != null:
			food_tweens[i].kill()
		foods[i].scale = Vector3.ONE * (0.15 if animate else 1.0)
		if animate:
			food_tweens[i] = create_tween()
			food_tweens[i].tween_property(foods[i], "scale", Vector3.ONE, 0.22).set_trans(Tween.TRANS_SINE)

func _physics_process(_delta: float) -> void:
	if not ready_policy:
		return
	previous = current.duplicate()
	current = lab.step()
	var error := lab.get_error()
	if not error.is_empty():
		ready_policy = false
		status.text = error
		return
	blend = 0.0
	if current.get("ate", false):
		var slot := maxi(0, int(current.get("eaten_slot", 0)))
		spark_origin = foods[slot].position
		spark_age = 0.0
		food_times.append(float(current.elapsed))
		_place_food_visual(true, slot)
	while not food_times.is_empty() and food_times[0] <= float(current.elapsed) - 60.0:
		food_times.pop_front()
	food_rate_value.text = "Food / min  %d" % food_times.size() if float(current.elapsed) >= 60.0 else "Food / min  —"
	food_count.text = "Food eaten  %d" % int(current.eaten)
	speed_value.text = ("Speed  %.1f cm/s" if ui_layout.vertical else "Swimming speed  %.1f cm/s") % float(current.speed_cm_s)
	meal_value.text = "Since last food  %.1f s" % float(current.seconds_since_food)
	move_button.visible = not live_learning and float(current.seconds_since_food) >= 12.0
	_update_learning_status()

func _update_learning_status() -> void:
	if not live_learning:
		status.text = "Try another food position" if move_button.visible else "Trained controller · shallow water"
		return
	var training := lab.get_stats()
	var update := int(training.get("policy_update", 0))
	var phase := "Paused" if user_paused else ("Untrained" if training.get("paused", true) else "Learning")
	status.text = "%s · update %d" % [phase, update]
	pause_button.text = "Resume learning" if user_paused else "Pause learning"
	steps_value.text = ("%s steps" if ui_layout.vertical else "%s simulation steps") % _format_count(int(training.env_steps))
	var now := Time.get_ticks_msec()
	if now - rate_timestamp >= 1000:
		var rate := float(int(training.env_steps) - rate_steps) * 1000.0 / float(now - rate_timestamp)
		throughput_value.text = ("%.2fM steps / second" % (rate / 1000000.0)) if rate >= 1000000 else ("%.0f steps / second" % rate)
		rate_steps = int(training.env_steps)
		rate_timestamp = now
	var samples := lab.get_evaluation_history(evaluation_points)
	for sample in samples:
		curve_view.points.append(Vector2(int(sample.update), float(sample.food_per_minute)))
		evaluation_points += 1
	if not samples.is_empty():
		curve_view.queue_redraw()
	status.tooltip_text = "Actual CUDA training continues until you close the scene. Start over begins fresh learning. More updates do not guarantee better swimming."
	if update >= last_reported_update + 50:
		last_reported_update = update
		print("POND_LIVE update=%d background=%d food=%d finished=%s" % [update, int(training.background_swimmers), int(current.get("eaten", 0)), str(training.finished)])

func _format_count(value: int) -> String:
	var digits := str(value)
	var grouped := ""
	for i in digits.length():
		if i > 0 and (digits.length() - i) % 3 == 0:
			grouped += ","
		grouped += digits[i]
	return grouped

func _process(delta: float) -> void:
	if not ready_policy:
		return
	blend = minf(1.0, blend + delta / STEP)
	var pose := current.duplicate()
	for key in ["x", "y", "joint0", "joint1"]:
		pose[key] = lerpf(float(previous[key]), float(current[key]), blend)
	pose.heading = lerp_angle(float(previous.heading), float(current.heading), blend)
	_apply_pose(pose)
	spark_age += delta
	for i in sparks.size():
		var spark := sparks[i]
		spark.visible = spark_age < 0.4
		if spark.visible:
			var angle := TAU * float(i) / sparks.size()
			spark.position = spark_origin + Vector3(cos(angle), 0.6, sin(angle)) * spark_age * 0.65
			spark.scale = Vector3.ONE * (1.0 - spark_age / 0.4)

func _apply_pose(pose: Dictionary) -> void:
	creature.position = Vector3(float(pose.x), SWIM_Y, float(pose.y))
	creature.rotation.y = -float(pose.heading)
	_update_tail(float(pose.joint0), float(pose.joint1))
	for i in eyes.size():
		# Cosmetic gaze follows the head, never implies a target chosen in code.
		var look := Vector3(0.7, 0.3, 0).normalized()
		pupils[i].position = look * 0.045

func _fit_camera() -> void:
	var dimensions := Vector2(pond_viewport.size) if pond_viewport != null else get_viewport().get_visible_rect().size
	var aspect := dimensions.x / maxf(1, dimensions.y)
	camera.size = maxf(8.2, 12.5 / maxf(0.1, aspect))

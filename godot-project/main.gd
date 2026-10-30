extends Control

const INK := Color("eef4ec")
const MUTED := Color("b6c8c1")
const ACCENT := Color("ffb24a")
const CreatureMesh = preload("res://creature_mesh.gd")
const AXES := ["Joint 1 · yaw", "Joint 1 · pitch", "Joint 2 · yaw", "Joint 2 · pitch"]
var lab: RefCounted
var state: Dictionary = {}
var layout: BoxContainer
var sidebar: PanelContainer
var sidebar_scroll: ScrollContainer
var viewport_container: SubViewportContainer
var viewport: SubViewport
var creature: Node3D
var camera: Camera3D
var run_button: Button
var step_button: Button
var reset_button: Button
var save_button: Button
var load_button: Button
var mode: OptionButton
var current: OptionButton
var sliders: Array[HSlider] = []
var slider_values: Array[Label] = []
var metrics: Label
var notice: Label
var status: Label
var accumulator := 0.0
var updating_controls := false

func label(value: String, size := 16, muted := false) -> Label:
	var result := Label.new()
	result.text = value
	result.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	result.add_theme_font_size_override("font_size", size)
	result.add_theme_color_override("font_color", MUTED if muted else INK)
	return result

func button(value: String, action: Callable, primary := false) -> Button:
	var result := Button.new()
	result.text = value
	result.custom_minimum_size.y = 44
	result.focus_mode = Control.FOCUS_ALL
	for kind in ["normal", "hover", "pressed"]:
		var style := StyleBoxFlat.new()
		style.bg_color = (ACCENT if primary else Color("2c4540")).lightened(0.08 if kind == "hover" else 0)
		style.set_corner_radius_all(5)
		style.content_margin_left = 12
		style.content_margin_right = 12
		result.add_theme_stylebox_override(kind, style)
	var focus := StyleBoxFlat.new()
	focus.bg_color = Color.TRANSPARENT
	focus.border_color = INK
	focus.set_border_width_all(2)
	focus.set_corner_radius_all(5)
	result.add_theme_stylebox_override("focus", focus)
	if primary:
		for color in ["font_color", "font_hover_color", "font_pressed_color", "font_focus_color"]:
			result.add_theme_color_override(color, Color("182822"))
	result.pressed.connect(action)
	return result

func _ready() -> void:
	get_window().title = "CudaRL · Creature foundation · diagnostic control"
	_build_ui()
	_build_world()
	resized.connect(_fit)
	_fit()
	if not ClassDB.class_exists("CreatureLab"):
		status.text = "The native creature foundation extension is unavailable. Build it before opening this scene."
		_set_available(false)
		return
	lab = ClassDB.instantiate("CreatureLab")
	state = lab.get_state()
	_sync_controls()
	_apply_state()
	status.text = "Paused. Select joint commands, then Run or Single step."

func _build_ui() -> void:
	var background := ColorRect.new()
	background.color = Color("10231f")
	background.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	background.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(background)
	layout = BoxContainer.new()
	layout.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	layout.add_theme_constant_override("separation", 0)
	add_child(layout)
	sidebar = PanelContainer.new()
	sidebar.custom_minimum_size.x = 330
	var surface := StyleBoxFlat.new()
	surface.bg_color = Color("192e28")
	surface.set_content_margin_all(18)
	sidebar.add_theme_stylebox_override("panel", surface)
	layout.add_child(sidebar)
	sidebar_scroll = ScrollContainer.new()
	sidebar_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	sidebar_scroll.follow_focus = true
	sidebar.add_child(sidebar_scroll)
	var controls := VBoxContainer.new()
	controls.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	controls.add_theme_constant_override("separation", 10)
	sidebar_scroll.add_child(controls)
	controls.add_child(label("Creature foundation", 24))
	controls.add_child(label("Diagnostic control · not learned", 15, true))
	notice = label("Contact gate unresolved", 15)
	notice.add_theme_color_override("font_color", ACCENT)
	controls.add_child(notice)
	var playback := HBoxContainer.new()
	playback.add_theme_constant_override("separation", 8)
	run_button = button("Run", _toggle_run, true)
	run_button.name = "RunPause"
	step_button = button("Single step", _single_step)
	step_button.name = "SingleStep"
	for item: Button in [run_button, step_button]:
		item.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		playback.add_child(item)
	controls.add_child(playback)
	reset_button = button("Reset physical state", _reset)
	reset_button.name = "ResetState"
	controls.add_child(reset_button)
	controls.add_child(HSeparator.new())
	controls.add_child(label("Actuation", 17))
	mode = OptionButton.new()
	mode.name = "ActuationMode"
	mode.custom_minimum_size.y = 44
	mode.add_item("Joint targets · physical servo", 1)
	mode.add_item("Direct joint torque", 0)
	mode.tooltip_text = "Changing mode pauses and clears commands. The servo applies limited internal joint torques; it does not stabilize or propel the whole body."
	mode.item_selected.connect(_set_mode)
	controls.add_child(mode)
	controls.add_child(label("Two universal joints · two axes each", 13, true))
	for axis in range(4):
		var row := HBoxContainer.new()
		var caption := label(AXES[axis], 14)
		caption.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		row.add_child(caption)
		var value := label("0.00 rad", 14, true)
		value.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		value.custom_minimum_size.x = 88
		row.add_child(value)
		slider_values.append(value)
		controls.add_child(row)
		var slider := HSlider.new()
		slider.name = "Actuator%d" % axis
		slider.min_value = -1
		slider.max_value = 1
		slider.step = 0.01
		slider.value = 0
		slider.custom_minimum_size.y = 44
		slider.focus_mode = Control.FOCUS_ALL
		slider.value_changed.connect(func(_value): _commands_changed())
		controls.add_child(slider)
		sliders.append(slider)
	controls.add_child(label("Water", 17))
	current = OptionButton.new()
	current.name = "WaterCurrent"
	current.custom_minimum_size.y = 44
	current.add_item("Calm water", 0)
	current.add_item("Steady +X · 0.5 cm/s", 1)
	current.add_item("Steady +X · 0.75 cm/s", 2)
	current.item_selected.connect(_set_current)
	controls.add_child(current)
	controls.add_child(HSeparator.new())
	var saving := HBoxContainer.new()
	saving.add_theme_constant_override("separation", 8)
	save_button = button("Save state", _save)
	save_button.name = "SaveDiagnostic"
	save_button.tooltip_text = "Save this diagnostic physical state and last applied commands in its separate local slot."
	load_button = button("Load state", _load)
	load_button.name = "LoadDiagnostic"
	load_button.tooltip_text = "Load the separate diagnostic state, paused for inspection. Original learning saves are not opened."
	for item: Button in [save_button, load_button]:
		item.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		saving.add_child(item)
	controls.add_child(saving)
	status = label("Loading native physical world…", 13, true)
	status.name = "Status"
	controls.add_child(status)
	var visual := VBoxContainer.new()
	visual.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	visual.size_flags_vertical = Control.SIZE_EXPAND_FILL
	visual.add_theme_constant_override("separation", 0)
	layout.add_child(visual)
	viewport_container = SubViewportContainer.new()
	viewport_container.stretch = true
	viewport_container.custom_minimum_size = Vector2(200, 200)
	viewport_container.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	viewport_container.size_flags_vertical = Control.SIZE_EXPAND_FILL
	viewport_container.resized.connect(_fit_camera)
	visual.add_child(viewport_container)
	viewport = SubViewport.new()
	viewport.own_world_3d = true
	viewport.msaa_3d = Viewport.MSAA_2X
	viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	viewport_container.add_child(viewport)
	var metric_pad := MarginContainer.new()
	for edge in ["left", "right", "top", "bottom"]:
		metric_pad.add_theme_constant_override("margin_" + edge, 16)
	visual.add_child(metric_pad)
	metrics = label("Physical state is unavailable.", 14, true)
	metrics.name = "PhysicalMetrics"
	metric_pad.add_child(metrics)

func _fit() -> void:
	var narrow := size.x < 800
	layout.vertical = narrow
	sidebar.custom_minimum_size = Vector2(0 if narrow else 330, 310 if narrow else 0)
	sidebar_scroll.custom_minimum_size.y = 274 if narrow else 0
	sidebar.size_flags_vertical = Control.SIZE_FILL if narrow else Control.SIZE_EXPAND_FILL
	_fit_camera.call_deferred()

func _fit_camera() -> void:
	if not camera or viewport_container.size.x < 1 or viewport_container.size.y < 1:
		return
	# Keep a fixed direction and center; fit the physical tank rather than follow
	# the creature. The stretched container alone owns the viewport's pixel size.
	var center := Vector3(0, -2, 0)
	var aspect := viewport_container.size.x / viewport_container.size.y
	var vertical_tangent := tan(deg_to_rad(camera.fov) * 0.5)
	var horizontal_tangent := vertical_tangent * aspect
	var inverse_basis := camera.basis.transposed()
	var distance := 1.0
	for x in [-4.5, 4.5]:
		for y in [-4.0, 0.0]:
			for z in [-3.5, 3.5]:
				var local: Vector3 = inverse_basis * (Vector3(x, y, z) - center)
				var projected_extent := maxf(absf(local.x) / horizontal_tangent, absf(local.y) / vertical_tangent)
				distance = maxf(distance, local.z + projected_extent * 1.08)
				distance = maxf(distance, local.z + camera.near * 2)
	camera.position = center + camera.basis.z * distance

func _node(mesh: Mesh, color: Color, at := Vector3.ZERO) -> MeshInstance3D:
	var node := MeshInstance3D.new()
	var material := StandardMaterial3D.new()
	material.albedo_color = color
	material.roughness = 0.8
	if color.a < 1:
		material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		material.cull_mode = BaseMaterial3D.CULL_DISABLED
	node.mesh = mesh
	node.material_override = material
	node.position = at
	return node

func _build_world() -> void:
	var world := Node3D.new()
	viewport.add_child(world)
	var settings := WorldEnvironment.new()
	settings.environment = Environment.new()
	settings.environment.background_mode = Environment.BG_COLOR
	settings.environment.background_color = Color("12352f")
	settings.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	settings.environment.ambient_light_color = Color("cfe2d7")
	settings.environment.ambient_light_energy = 0.7
	world.add_child(settings)
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-55, -30, 0)
	sun.light_energy = 1.3
	world.add_child(sun)
	var floor_mesh := BoxMesh.new()
	floor_mesh.size = Vector3(9, 0.04, 7)
	world.add_child(_node(floor_mesh, Color("596659"), Vector3(0, -4.02, 0)))
	for axis in range(-4, 5):
		var grid := BoxMesh.new()
		grid.size = Vector3(0.012, 0.008, 7)
		world.add_child(_node(grid, Color("81917b"), Vector3(axis, -3.995, 0)))
	for index in range(4):
		var waterline := BoxMesh.new()
		waterline.size = Vector3(9 if index < 2 else 0.025, 0.025, 0.025 if index < 2 else 7)
		world.add_child(_node(waterline, Color("8cbdb0"), Vector3(0 if index < 2 else (-4.5 if index == 2 else 4.5), 0, -3.5 if index == 0 else (3.5 if index == 1 else 0))))
	creature = CreatureMesh.new()
	world.add_child(creature)
	camera = Camera3D.new()
	camera.position = Vector3(6.4, 3.7, 8.2)
	camera.fov = 43
	camera.keep_aspect = Camera3D.KEEP_HEIGHT
	world.add_child(camera)
	camera.look_at(Vector3(0, -2, 0), Vector3.UP)
	_fit_camera()

func _commands_changed() -> void:
	if updating_controls:
		return
	var values := PackedFloat32Array()
	for axis in range(4):
		values.append(float(sliders[axis].value))
		slider_values[axis].text = "%.2f rad" % sliders[axis].value if mode.get_selected_id() == 1 else "%.1f µN·m" % (sliders[axis].value * 10)
	if lab and not lab.set_commands(values):
		status.text = lab.get_error()

func _set_mode(_index: int) -> void:
	if not lab:
		return
	lab.set_mode(mode.get_selected_id())
	for slider in sliders:
		slider.set_value_no_signal(0)
	_commands_changed()
	state = lab.get_state()
	accumulator = 0
	_apply_state()
	status.text = "Mode changed. Paused with zero commands."

func _set_current(index: int) -> void:
	if not lab:
		return
	var speed: float = [0.0, 0.005, 0.0075][index]
	if not lab.set_current(float(speed)):
		status.text = lab.get_error()
	state = lab.get_state()
	_apply_state()

func _toggle_run() -> void:
	if not lab or state.is_empty():
		return
	lab.set_paused(not bool(state.paused))
	state = lab.get_state()
	accumulator = 0
	_apply_state()
	status.text = "Paused." if state.paused else "Running manual diagnostic commands. No neural policy is active."

func _single_step() -> void:
	if not lab:
		return
	state = lab.single_step()
	accumulator = 0
	_apply_state()
	status.text = lab.get_error() if not lab.get_error().is_empty() else "Advanced one 50 ms control interval. Paused."

func _reset() -> void:
	if not lab:
		return
	lab.reset_world()
	state = lab.get_state()
	_sync_controls()
	accumulator = 0
	_apply_state()
	status.text = "Physical state reset. Paused; water and actuation mode retained."

func _save() -> void:
	if not lab:
		return
	status.text = "Diagnostic physical state saved locally." if lab.save_state() else lab.get_error()
	load_button.disabled = not lab.has_saved_state()

func _load() -> void:
	if not lab:
		return
	if not lab.load_state():
		status.text = lab.get_error()
		return
	state = lab.get_state()
	_sync_controls()
	accumulator = 0
	_apply_state()
	status.text = "Diagnostic state loaded and paused for inspection."

func _sync_controls() -> void:
	updating_controls = true
	mode.select(0 if int(state.get("mode", 1)) == 1 else 1)
	var speed := float(state.get("current_x", 0))
	current.select(0 if speed < 0.001 else (1 if speed < 0.006 else 2))
	var commands: PackedFloat32Array = state.get("commands", PackedFloat32Array([0, 0, 0, 0]))
	for axis in range(4):
		sliders[axis].set_value_no_signal(commands[axis])
	updating_controls = false
	_commands_changed()

func _set_available(enabled: bool) -> void:
	for control in [run_button, step_button, reset_button, save_button, load_button, mode, current]:
		control.disabled = not enabled
	for slider in sliders:
		slider.editable = enabled

func _apply_state() -> void:
	if state.is_empty():
		return
	if state.get("finite", false):
		creature.pose(state.links)
	var failed: bool = state.get("unsafe", false) or not state.get("finite", false)
	run_button.text = "Run" if state.paused else "Pause"
	run_button.disabled = failed
	step_button.disabled = failed or not state.paused
	load_button.disabled = not lab.has_saved_state()
	var position: Vector3 = state.position_cm
	var torques: PackedFloat32Array = state.actual_torques_uNm
	metrics.text = "%.2f s · %d control steps · %s\nPosition (%.2f, %.2f, %.2f) cm · heading %.1f°\nHead %.3f cm/s · COM %.3f cm/s · floor gap %.3f mm\nApplied torque [%.2f, %.2f, %.2f, %.2f] µN·m" % [state.elapsed, int(state.steps), "paused" if state.paused else "running", position.x, position.y, position.z, state.heading_degrees, state.head_speed_cm_s, state.com_speed_cm_s, state.floor_gap_mm, torques[0], torques[1], torques[2], torques[3]]
	if failed:
		status.text = "Diagnostic stepping stopped on an invalid or excessive-contact state. Reset to inspect another start."

func _physics_process(delta: float) -> void:
	if not lab or state.is_empty() or state.paused:
		return
	accumulator += minf(delta, 0.2)
	while accumulator >= 0.05:
		accumulator -= 0.05
		state = lab.step()
		_apply_state()
		if state.paused or state.unsafe or not state.finite:
			accumulator = 0
			break

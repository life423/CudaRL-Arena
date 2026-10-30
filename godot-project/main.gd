extends Control

const INK := Color("eef4ec")
const MUTED := Color("b6c8c1")
const ACCENT := Color("ffb24a")
const CreatureMesh = preload("res://creature_mesh.gd")
const AXES := ["Joint 1 · yaw", "Joint 1 · pitch", "Joint 2 · yaw", "Joint 2 · pitch"]
const REQUIRED_METHODS := ["get_config", "get_state", "set_commands", "set_mode", "set_current", "set_paused", "step", "single_step", "reset_world", "save_state", "load_state", "has_saved_state", "get_error"]
var lab: RefCounted
var config: Dictionary = {}
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
var current_speeds: Array[float] = [0.0, 0.005, 0.0075]
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
	_set_available(false)
	resized.connect(_fit)
	_fit()
	var instance: Variant = _create_lab()
	if not instance is RefCounted:
		_startup_error("The native creature foundation extension is unavailable. Build the current extension, then reopen this scene.")
		return
	for method in REQUIRED_METHODS:
		if not instance.has_method(method):
			_startup_error("The native extension is outdated: missing %s. Build the current extension, then reopen this scene." % method)
			return
	var metadata: Variant = instance.get_config()
	var problem := _config_error(metadata)
	if not problem.is_empty():
		_startup_error("Native configuration is incompatible: " + problem)
		return
	config = metadata
	lab = instance
	if not _capture_state(lab.get_state()):
		return
	_build_world()
	_set_available(true)
	_sync_controls()
	_refresh_save_availability()
	if _apply_state():
		status.text = "Paused. Select joint commands, then Run or Single step."

func _create_lab() -> Variant:
	return ClassDB.instantiate("CreatureLab") if ClassDB.class_exists("CreatureLab") else null

func _startup_error(message: String) -> void:
	lab = null
	state = {}
	accumulator = 0
	_set_available(false)
	viewport_container.hide()
	metrics.text = "Native physical state is unavailable."
	status.text = message

func _number(value: Variant) -> bool:
	return (typeof(value) == TYPE_FLOAT or typeof(value) == TYPE_INT) and is_finite(float(value))

func _config_error(value: Variant) -> String:
	if not value is Dictionary:
		return "configuration must be a dictionary."
	for key in ["version", "actuator_count"]:
		if not value.has(key) or typeof(value[key]) != TYPE_INT:
			return "missing integer " + key
	if int(value.version) != 1 or int(value.actuator_count) != AXES.size():
		return "unsupported configuration version or actuator count."
	for key in ["joint_limit_rad", "torque_limit_uNm", "control_seconds", "substep_seconds"]:
		if not value.has(key) or not _number(value[key]) or float(value[key]) <= 0:
			return "missing positive physical quantity " + key
	if float(value.substep_seconds) > float(value.control_seconds):
		return "the substep exceeds the control interval."
	for key in ["bounds_min_model", "bounds_max_model"]:
		if not value.has(key) or typeof(value[key]) != TYPE_VECTOR3 or not value[key].is_finite():
			return "missing finite tank bounds " + key
	var lower: Vector3 = value.bounds_min_model
	var upper: Vector3 = value.bounds_max_model
	if lower.x >= upper.x or lower.y >= upper.y or lower.z >= upper.z:
		return "tank bounds have no positive volume."
	return ""

func _state_error(value: Variant) -> String:
	if not value is Dictionary:
		return "state must be a dictionary."
	for key in ["paused", "unsafe", "finite", "learned", "contact_gate_passed", "pending_commands_clipped"]:
		if not value.has(key) or typeof(value[key]) != TYPE_BOOL:
			return "missing boolean " + key
	for key in ["mode", "steps", "substeps", "physics_version", "contact_samples", "first_unsafe_substep"]:
		if not value.has(key) or typeof(value[key]) != TYPE_INT:
			return "missing integer " + key
	if int(value.mode) not in [0, 1] or int(value.steps) < 0 or int(value.substeps) < 0:
		return "invalid mode or physical clock."
	if value.learned:
		return "this view accepts diagnostic control, not a learned policy."
	for key in ["elapsed", "current_x", "heading_degrees", "head_speed_cm_s", "com_speed_cm_s", "floor_gap_mm", "floor_spring_force_mN"]:
		if not value.has(key) or (typeof(value[key]) != TYPE_FLOAT and typeof(value[key]) != TYPE_INT):
			return "missing physical quantity " + key
		if value.finite and not _number(value[key]):
			return "nonfinite physical quantity " + key
	if not value.has("position_cm") or typeof(value.position_cm) != TYPE_VECTOR3:
		return "missing physical position."
	if value.finite and not value.position_cm.is_finite():
		return "nonfinite physical position."
	if not value.has("current_vector_m_s") or typeof(value.current_vector_m_s) != TYPE_VECTOR3:
		return "missing water-current vector."
	if value.finite and not value.current_vector_m_s.is_finite():
		return "nonfinite water-current vector."
	for key in ["commands", "last_raw_commands", "actual_torques_uNm", "joint_angles_rad", "joint_rates_rad_s"]:
		if not value.has(key) or typeof(value[key]) != TYPE_PACKED_FLOAT32_ARRAY or value[key].size() != int(config.actuator_count):
			return "invalid actuator array " + key
		for number in value[key]:
			if value.finite and not is_finite(number):
				return "nonfinite actuator value in " + key
	for command in value.commands:
		if not is_finite(command) or absf(command) > 1:
			return "commands exceed the normalized input bounds."
	if not value.has("links") or typeof(value.links) != TYPE_ARRAY:
		return "missing physical links."
	if value.finite:
		if value.links.size() != 3:
			return "the physical body must contain three links."
		for link in value.links:
			if not link is Dictionary or not link.has("position") or typeof(link.position) != TYPE_VECTOR3 or not link.position.is_finite():
				return "invalid link position."
			if not link.has("orientation") or typeof(link.orientation) != TYPE_QUATERNION or not link.orientation.is_finite():
				return "invalid link orientation."
			for key in ["length_model", "half_height_model", "half_width_model"]:
				if not link.has(key) or not _number(link[key]) or float(link[key]) <= 0:
					return "invalid link dimension " + key
	return ""

func _capture_state(value: Variant) -> bool:
	var problem := _state_error(value)
	if not problem.is_empty():
		_startup_error("Native state is incompatible: " + problem)
		return false
	state = value
	return true

func _refresh_save_availability() -> void:
	load_button.disabled = not lab or not lab.has_saved_state()

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
	for axis in range(AXES.size()):
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
	for index in range(current_speeds.size()):
		current.add_item("Calm water" if current_speeds[index] == 0 else "Steady +X · %s cm/s" % _speed_text(current_speeds[index]), index)
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
	if not camera or config.is_empty() or viewport_container.size.x < 1 or viewport_container.size.y < 1:
		return
	# Keep a fixed direction and center; fit the physical tank rather than follow
	# the creature. The stretched container alone owns the viewport's pixel size.
	var lower: Vector3 = config.bounds_min_model
	var upper: Vector3 = config.bounds_max_model
	var center := (lower + upper) * 0.5
	var aspect := viewport_container.size.x / viewport_container.size.y
	var vertical_tangent := tan(deg_to_rad(camera.fov) * 0.5)
	var horizontal_tangent := vertical_tangent * aspect
	var inverse_basis := camera.basis.transposed()
	var distance := 1.0
	for x in [lower.x, upper.x]:
		for y in [lower.y, upper.y]:
			for z in [lower.z, upper.z]:
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
	var lower: Vector3 = config.bounds_min_model
	var upper: Vector3 = config.bounds_max_model
	var extent := upper - lower
	var center := (lower + upper) * 0.5
	var world := Node3D.new()
	world.name = "PhysicalView"
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
	floor_mesh.size = Vector3(extent.x, 0.04, extent.z)
	var floor := _node(floor_mesh, Color("596659"), Vector3(center.x, lower.y - 0.02, center.z))
	floor.name = "TankFloor"
	world.add_child(floor)
	for axis in range(ceili(lower.x), floori(upper.x) + 1):
		var grid := BoxMesh.new()
		grid.size = Vector3(0.012, 0.008, extent.z)
		world.add_child(_node(grid, Color("81917b"), Vector3(axis, lower.y + 0.005, center.z)))
	for index in range(4):
		var waterline := BoxMesh.new()
		waterline.size = Vector3(extent.x if index < 2 else 0.025, 0.025, 0.025 if index < 2 else extent.z)
		var at := Vector3(center.x if index < 2 else (lower.x if index == 2 else upper.x), upper.y, lower.z if index == 0 else (upper.z if index == 1 else center.z))
		world.add_child(_node(waterline, Color("8cbdb0"), at))
	creature = CreatureMesh.new()
	world.add_child(creature)
	camera = Camera3D.new()
	camera.position = center + Vector3(6.4, 5.7, 8.2)
	camera.fov = 43
	camera.keep_aspect = Camera3D.KEEP_HEIGHT
	world.add_child(camera)
	camera.look_at(center, Vector3.UP)
	_fit_camera()

func _commands_changed() -> void:
	if updating_controls:
		return
	_refresh_command_labels()
	var values := PackedFloat32Array()
	for slider in sliders:
		values.append(float(slider.value))
	if lab and not lab.set_commands(values):
		status.text = lab.get_error()

func _refresh_command_labels() -> void:
	if config.is_empty():
		return
	for axis in range(sliders.size()):
		var value := float(sliders[axis].value)
		slider_values[axis].text = "%.2f rad" % (value * float(config.joint_limit_rad)) if mode.get_selected_id() == 1 else "%.1f µN·m" % (value * float(config.torque_limit_uNm))

func _set_mode(_index: int) -> void:
	if not lab:
		return
	if not lab.set_mode(mode.get_selected_id()):
		status.text = lab.get_error()
		return
	if not _capture_state(lab.get_state()):
		return
	_sync_controls()
	accumulator = 0
	if _apply_state():
		status.text = "Mode changed. Paused with zero commands."

func _set_current(index: int) -> void:
	if not lab:
		return
	# The extra saved-flow entry is readback, not another flow command.
	if index >= 3:
		return
	if not lab.set_current(current_speeds[index]):
		status.text = lab.get_error()
	if not _capture_state(lab.get_state()):
		return
	_sync_current()
	_apply_state()

func _speed_text(speed: float) -> String:
	return ("%.2f" % (speed * 100)).trim_suffix("0").trim_suffix("0").trim_suffix(".")

func _sync_current() -> void:
	while current.item_count > 3:
		current.remove_item(current.item_count - 1)
	current_speeds.resize(3)
	var flow: Vector3 = state.current_vector_m_s
	if not flow.is_finite():
		return
	var selected := -1
	if is_zero_approx(flow.y) and is_zero_approx(flow.z):
		for index in range(current_speeds.size()):
			if is_equal_approx(flow.x, current_speeds[index]):
				selected = index
	if selected < 0:
		current_speeds.append(flow.length())
		var text := "%s cm/s (saved)" % _speed_text(flow.x)
		if not is_zero_approx(flow.y) or not is_zero_approx(flow.z):
			text = "Saved flow · X %s · Z %s cm/s" % [_speed_text(flow.x), _speed_text(flow.z)]
		current.add_item(text, 3)
		current.set_item_disabled(3, true)
		selected = 3
	current.select(selected)

func _toggle_run() -> void:
	if not lab or state.is_empty():
		return
	lab.set_paused(not bool(state.paused))
	var error: String = lab.get_error()
	if not _capture_state(lab.get_state()):
		return
	accumulator = 0
	if _apply_state():
		status.text = error if not error.is_empty() else ("Paused." if state.paused else "Running manual diagnostic commands. No neural policy is active.")

func _single_step() -> void:
	if not lab:
		return
	if not _capture_state(lab.single_step()):
		return
	accumulator = 0
	if _apply_state():
		status.text = lab.get_error() if not lab.get_error().is_empty() else "Advanced one %.0f ms control interval. Paused." % (float(config.control_seconds) * 1000)

func _reset() -> void:
	if not lab:
		return
	lab.reset_world()
	if not _capture_state(lab.get_state()):
		return
	_sync_controls()
	accumulator = 0
	if _apply_state():
		status.text = "Physical state reset. Paused; water and actuation mode retained."

func _save() -> void:
	if not lab:
		return
	if not lab.save_state():
		status.text = lab.get_error()
	elif _apply_state():
		status.text = "Diagnostic physical state saved locally."
	_refresh_save_availability()

func _load() -> void:
	if not lab:
		return
	if not lab.load_state():
		status.text = lab.get_error()
		_refresh_save_availability()
		return
	if not _capture_state(lab.get_state()):
		return
	_sync_controls()
	accumulator = 0
	_refresh_save_availability()
	if _apply_state():
		status.text = "Diagnostic state loaded and paused for inspection."
	if state.pending_commands_clipped:
		status.text += " Next inputs use the actuator bounds; original saved raw values are retained."

func _sync_controls() -> void:
	updating_controls = true
	mode.select(0 if int(state.get("mode", 1)) == 1 else 1)
	_sync_current()
	var commands: PackedFloat32Array = state.get("commands", PackedFloat32Array([0, 0, 0, 0]))
	for axis in range(sliders.size()):
		sliders[axis].set_value_no_signal(commands[axis])
	updating_controls = false
	_refresh_command_labels()

func _set_available(enabled: bool) -> void:
	for control in [run_button, step_button, reset_button, save_button, load_button, mode, current]:
		control.disabled = not enabled
	for slider in sliders:
		slider.editable = enabled

func _apply_state() -> bool:
	if state.is_empty():
		return false
	if state.get("finite", false):
		creature.show()
		creature.pose(state.links)
	else:
		creature.hide()
	var failed: bool = state.get("unsafe", false) or not state.get("finite", false)
	run_button.text = "Run" if state.paused else "Pause"
	run_button.disabled = failed
	step_button.disabled = failed or not state.paused
	var position: Vector3 = state.position_cm
	var torques: PackedFloat32Array = state.actual_torques_uNm
	metrics.text = "%.2f s · %d control steps · %s\nPosition (%.2f, %.2f, %.2f) cm · heading %.1f°\nHead %.3f cm/s · COM %.3f cm/s · floor gap %.3f mm\nLast substep torque [%.2f, %.2f, %.2f, %.2f] µN·m" % [state.elapsed, int(state.steps), "paused" if state.paused else "running", position.x, position.y, position.z, state.heading_degrees, state.head_speed_cm_s, state.com_speed_cm_s, state.floor_gap_mm, torques[0], torques[1], torques[2], torques[3]]
	if not state.finite:
		metrics.text = "Native physical state is invalid. Movement is disabled."
	if failed:
		status.text = "Diagnostic stepping stopped on an invalid or excessive-contact state. Reset to inspect another start."
	return not failed

func _physics_process(delta: float) -> void:
	if not lab or state.is_empty() or state.paused:
		return
	var interval := float(config.control_seconds)
	accumulator += minf(delta, interval * 4)
	while accumulator >= interval:
		accumulator -= interval
		if not _capture_state(lab.step()):
			return
		_apply_state()
		if state.paused or state.unsafe or not state.finite:
			accumulator = 0
			break

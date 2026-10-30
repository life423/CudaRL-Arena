extends Control

const STUDY_ROOT := "res://../build/motor-v2/2026-10-30-v1"
const STEP := 0.05
const INK := Color("edf4ef")
const MUTED := Color("a9bfb7")
const ACCENT := Color("ffb24a")
var study_root := STUDY_ROOT
var labs: Array = []
var infos: Array = [{}, {}]
var states: Array = [{}, {}]
var bodies: Array = []
var goals: Array[Node3D] = []
var views: Array[SubViewportContainer] = []
var titles: Array[Label] = []
var summaries: Array[Label] = []
var messages: Array[Label] = []
var actuator_labels: Array[Label] = []
var comparison: BoxContainer
var trial: OptionButton
var resume_button: Button
var restart_button: Button
var next_button: Button
var notice: Label
var running := false
var accumulator := 0.0

func label(text: String, size := 16, muted := false) -> Label:
	var result := Label.new()
	result.text = text
	result.add_theme_font_size_override("font_size", size)
	result.add_theme_color_override("font_color", MUTED if muted else INK)
	return result

func button(text: String, action: Callable) -> Button:
	var result := Button.new()
	result.text = text
	result.custom_minimum_size.y = 44
	result.pressed.connect(action)
	return result

func material(color: Color) -> StandardMaterial3D:
	var result := StandardMaterial3D.new()
	result.albedo_color = color
	result.roughness = 0.45
	if color.a < 1:
		result.transparency = StandardMaterial3D.TRANSPARENCY_ALPHA
		result.cull_mode = StandardMaterial3D.CULL_DISABLED
	return result

func mesh_node(mesh: Mesh, color: Color) -> MeshInstance3D:
	var result := MeshInstance3D.new()
	result.mesh = mesh
	result.material_override = material(color)
	return result

func _ready() -> void:
	get_window().title = "Tadpole Pond · Motor v2 comparison · frozen live physics"
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--motor-v2-root="):
			study_root = argument.trim_prefix("--motor-v2-root=")
	var background := ColorRect.new()
	background.color = Color("101d1b")
	background.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	background.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(background)
	var margin := MarginContainer.new()
	margin.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	for edge in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + edge, 20)
	add_child(margin)
	var column := VBoxContainer.new()
	column.add_theme_constant_override("separation", 10)
	margin.add_child(column)
	column.add_child(label("Motor v2 comparison", 25))
	column.add_child(label("Contact physics v2 · saved PPO weights · live simulation · no learning updates", 15, true))
	var actions := HBoxContainer.new()
	actions.add_theme_constant_override("separation", 10)
	trial = OptionButton.new()
	trial.custom_minimum_size = Vector2(300, 44)
	trial.item_selected.connect(func(_index): _restart())
	actions.add_child(trial)
	resume_button = button("Resume", _toggle)
	actions.add_child(resume_button)
	restart_button = button("Restart paused", _restart)
	actions.add_child(restart_button)
	next_button = button("Next case", _next)
	actions.add_child(next_button)
	column.add_child(actions)
	comparison = BoxContainer.new()
	comparison.add_theme_constant_override("separation", 14)
	comparison.size_flags_vertical = Control.SIZE_EXPAND_FILL
	column.add_child(comparison)
	for arm in range(2):
		_build_view(arm)
	notice = label("", 14, true)
	notice.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	column.add_child(notice)
	resized.connect(_layout)
	_layout()
	if not ClassDB.class_exists("MotorV2Lab"):
		notice.text = "Build the isolated motor_v2_bench extension to inspect the new experiment. The production pond is unchanged."
		_set_controls(false)
		return
	for arm in range(2):
		var native = ClassDB.instantiate("MotorV2Lab")
		labs.append(native)
		if not native.configure(study_root):
			messages[arm].text = native.get_error()
		infos[arm] = native.get_controller_info(arm)
		var arm_name := "Direct torque" if arm == 0 else "Joint targets + physical servo"
		titles[arm].text = arm_name + (" · %d/64 updates" % int(infos[arm].update) if infos[arm].available else " · not trained")
	for item: Dictionary in labs[0].get_cases():
		trial.add_item(item.name, int(item.id))
	notice.text = _status()
	_restart()

func _set_controls(enabled: bool) -> void:
	resume_button.disabled = not enabled
	restart_button.disabled = not enabled
	next_button.disabled = not enabled
	trial.disabled = trial.item_count == 0

func _status() -> String:
	var inventory: PackedStringArray = []
	for arm in range(2):
		var name := "Torque" if arm == 0 else "Joint-target"
		inventory.append(name + (" saved at %d/64" % int(infos[arm].update) if infos[arm].get("available", false) else " not trained"))
	var text := "; ".join(inventory) + ". "
	for path in [study_root.path_join("status.json"), study_root.path_join("run/status.json")]:
		if FileAccess.file_exists(path):
			var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
			if parsed is Dictionary:
				text += str(parsed.get("reason", parsed.get("message", ""))) + " "
				break
	return text + "Experimental only; this scene never updates weights or replaces the production swimmer."

func _layout() -> void:
	if comparison:
		comparison.vertical = size.x < 900
	for view in views:
		view.custom_minimum_size.y = 120 if size.x < 900 else 160

func _body_mesh(link: int) -> ArrayMesh:
	var length := (0.020 if link == 0 else 0.026) / 0.04
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	var vertices := PackedVector3Array()
	var normals := PackedVector3Array()
	var indices := PackedInt32Array()
	for axial in range(9):
		var fraction := (axial - 4) * 0.25
		var profile := sqrt(maxf(0.0, 1.0 - fraction * fraction)) if link == 0 else 1.0
		var ry := (0.0064 if link == 0 else (0.006 if link == 2 else 0.0015)) / 0.04
		var rz := (0.008 if link == 0 else 0.0015) / 0.04
		for rim in range(16):
			var angle := rim * TAU / 16.0
			vertices.append(Vector3(fraction * length * 0.5, profile * ry * cos(angle), profile * rz * sin(angle)))
			var normal := Vector3(fraction / (length * 0.5) if link == 0 else 0.0, profile * cos(angle) / ry, profile * sin(angle) / rz)
			normals.append(normal.normalized())
	for axial in range(8):
		for rim in range(16):
			var a := axial * 16 + rim
			var b := (axial + 1) * 16 + rim
			var c := (axial + 1) * 16 + (rim + 1) % 16
			var d := axial * 16 + (rim + 1) % 16
			indices.append_array(PackedInt32Array([a, c, b, a, d, c]))
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_NORMAL] = normals
	arrays[Mesh.ARRAY_INDEX] = indices
	var result := ArrayMesh.new()
	result.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	return result

func _build_view(arm: int) -> void:
	var panel := PanelContainer.new()
	panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	panel.size_flags_vertical = Control.SIZE_EXPAND_FILL
	comparison.add_child(panel)
	var pad := MarginContainer.new()
	for edge in ["left", "right", "top", "bottom"]:
		pad.add_theme_constant_override("margin_" + edge, 12)
	panel.add_child(pad)
	var column := VBoxContainer.new()
	pad.add_child(column)
	var heading := label(("Direct torque" if arm == 0 else "Joint targets + physical servo") + " · not trained", 17)
	column.add_child(heading)
	titles.append(heading)
	var container := SubViewportContainer.new()
	container.stretch = true
	container.custom_minimum_size = Vector2(200, 160)
	container.size_flags_vertical = Control.SIZE_EXPAND_FILL
	column.add_child(container)
	views.append(container)
	container.hide()
	var viewport := SubViewport.new()
	viewport.size = Vector2i(640, 420)
	viewport.msaa_3d = Viewport.MSAA_4X
	viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	viewport.own_world_3d = true
	container.add_child(viewport)
	var world := Node3D.new()
	viewport.add_child(world)
	var environment := WorldEnvironment.new()
	environment.environment = Environment.new()
	environment.environment.background_mode = Environment.BG_COLOR
	environment.environment.background_color = Color("15342f")
	environment.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.environment.ambient_light_color = Color("cfe4dc")
	environment.environment.ambient_light_energy = 0.65
	world.add_child(environment)
	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-45, -25, 0)
	light.light_energy = 1.3
	world.add_child(light)
	var floor_mesh := BoxMesh.new()
	floor_mesh.size = Vector3(9, 0.04, 7)
	var floor_node := mesh_node(floor_mesh, Color("354b43"))
	floor_node.position.y = -4.02
	world.add_child(floor_node)
	for axis in range(-4, 5):
		var line_mesh := BoxMesh.new()
		line_mesh.size = Vector3(0.012, 0.008, 7)
		var line := mesh_node(line_mesh, Color("607b6a"))
		line.position = Vector3(axis, -3.995, 0)
		world.add_child(line)
	var camera := Camera3D.new()
	camera.position = Vector3(5, 3.5, 7)
	camera.fov = 40
	world.add_child(camera)
	camera.look_at(Vector3(0, -2, 0), Vector3.UP)
	var links: Array[MeshInstance3D] = []
	for link in range(3):
		var piece := mesh_node(_body_mesh(link), Color("68bec0") if link < 2 else Color(0.36, 0.72, 0.68, 0.82))
		world.add_child(piece)
		links.append(piece)
	for side in [-1, 1]:
		var eye_mesh := SphereMesh.new()
		eye_mesh.radius = 0.052
		eye_mesh.height = 0.104
		var eye := mesh_node(eye_mesh, Color("f7f5e8"))
		eye.position = Vector3(0.16, 0.09, 0.12 * side)
		links[0].add_child(eye)
		var pupil_mesh := SphereMesh.new()
		pupil_mesh.radius = 0.028
		pupil_mesh.height = 0.056
		var pupil := mesh_node(pupil_mesh, Color("13262b"))
		pupil.position = Vector3(0.018, 0.005, 0.035 * side)
		eye.add_child(pupil)
	bodies.append(links)
	var goal := Node3D.new()
	world.add_child(goal)
	var ring := TorusMesh.new()
	ring.inner_radius = 0.17
	ring.outer_radius = 0.19
	ring.rings = 32
	goal.add_child(mesh_node(ring, ACCENT))
	goals.append(goal)
	var summary := label("No trained policy available.", 15)
	column.add_child(summary)
	summaries.append(summary)
	var message := label("No older controller is substituted.", 14, true)
	message.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	column.add_child(message)
	messages.append(message)
	var details := PopupPanel.new()
	add_child(details)
	var detail_pad := MarginContainer.new()
	for edge in ["left", "right", "top", "bottom"]:
		detail_pad.add_theme_constant_override("margin_" + edge, 16)
	details.add_child(detail_pad)
	var actuator := label("No physical commands are being applied.", 14)
	actuator.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	actuator.custom_minimum_size = Vector2(380, 150)
	detail_pad.add_child(actuator)
	actuator_labels.append(actuator)
	column.add_child(button("Actuators and effort", func(): details.popup_centered(Vector2i(430, 230))))

func _restart() -> void:
	running = false
	accumulator = 0.0
	resume_button.text = "Resume"
	var any := false
	for arm in range(labs.size()):
		states[arm] = {}
		views[arm].hide()
		if trial.item_count > 0 and infos[arm].get("available", false) and labs[arm].start_trial(trial.get_selected_id(), arm):
			states[arm] = labs[arm].get_state()
			views[arm].show()
			any = true
			_apply_state(arm)
		else:
			summaries[arm].text = "Not trained · no live replacement"
			messages[arm].text = str(infos[arm].get("error", labs[arm].get_error()))
	_set_controls(any)

func _toggle() -> void:
	if resume_button.disabled:
		return
	running = not running
	accumulator = 0.0
	resume_button.text = "Pause" if running else "Resume"
	for arm in range(2):
		if not states[arm].is_empty():
			_apply_state(arm)

func _next() -> void:
	if trial.item_count > 0:
		trial.select((trial.selected + 1) % trial.item_count)
		_restart()

func _physics_process(delta: float) -> void:
	if not running:
		return
	accumulator += minf(delta, 0.2)
	while accumulator >= STEP:
		accumulator -= STEP
		var complete := true
		for arm in range(labs.size()):
			if states[arm].is_empty():
				continue
			states[arm] = labs[arm].step()
			_apply_state(arm)
			complete = complete and bool(states[arm].complete)
		if complete:
			running = false
			resume_button.text = "Trial complete"
			resume_button.disabled = true
			break

func _apply_state(arm: int) -> void:
	var state: Dictionary = states[arm]
	if state.is_empty():
		return
	if state.finite:
		for link in range(3):
			var pose: Dictionary = state.links[link]
			bodies[arm][link].transform = Transform3D(Basis(pose.quaternion), pose.position)
		goals[arm].position = state.goal
	var water: Vector3 = state.water_velocity * 1000.0
	var contact := "Bottom contact" if state.floor_contact else "No bottom contact"
	summaries[arm].text = "%.2f s · %s\nPosition %.1f mm · heading %.1f°\nHead / COM %.2f / %.2f mm/s · angular %.3f rad/s\nFloor gap %.3f mm · %s\nBest qualifying hold %.2f / 10 s\nWater (%+.1f, %+.1f, %+.1f) mm/s" % [state.elapsed, state.kind, state.goal_error_mm, state.heading_error_deg, state.head_mm_s, state.com_mm_s, state.angular_rad_s, state.floor_gap_mm, contact, state.best_hold_seconds, water.x, water.y, water.z]
	var status := "Running" if running else "Paused"
	if state.safety_failed:
		status = "Physical safety failed · simulation stopped"
	elif state.complete:
		status = "Passed this replay" if state.passed_trial else "Did not pass this replay"
	var evaluation := "Final suite recorded" if infos[arm].get("evaluation_complete", false) else "Final suite not completed"
	messages[arm].text = "%s · %d/64 saved updates · %s\n%s · tail movement is allowed during holding" % [status, int(state.policy_update), evaluation, "Accepted experiment" if state.accepted else "Unaccepted experimental actor"]
	actuator_labels[arm].text = "Two universal joints × two axes = four outputs\nDeterministic neural mean before clipping: %s\n%s\nActual last applied torque (µN·m): %s\nControl torque RMS %.3f µN·m\nPositive work %.3f µJ · peak interval floor force %.3f µN\nAllowed overlap: %.3f mm; unsupported hold requires ≥2 mm floor clearance" % [_four(state.raw_actions), ("Held targets (rad): " + _four(state.joint_targets_rad)) if arm == 1 else "Direct internal torques · no joint-angle targets", _four(state.actual_torque_uNm), state.torque_rms_uNm, state.positive_work_J * 1000000.0, state.floor_force_N * 1000000.0, state.overlap_limit_mm]

func _four(values) -> String:
	var text: PackedStringArray = []
	for value in values:
		text.append("%+.3f" % float(value))
	return "[" + ", ".join(text) + "]"

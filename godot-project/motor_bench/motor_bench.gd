extends Control

const STEP := 0.05
const INK := Color("edf4ef")
const MUTED := Color("a9bfb7")
const ACCENT := Color("ffb24a")
const ORIGINAL := "res://../build/predator-shelter/source-boulder.bin"
const CALIBRATED := "res://../build/predator-shelter/motor-v5-training.bin"
const CANDIDATE := "res://../build/motor-control/2026-10-29/pilot-v1/final-policy.bin"
const PAIRED_ROOT := "res://../build/paired-motor/2026-10-29-v1/run"
var labs: Array = []
var selectors: Array[OptionButton] = []
var summaries: Array[Label] = []
var messages: Array[Label] = []
var bodies: Array = []
var views: Array[SubViewportContainer] = []
var actuator_panels: Array[PopupPanel] = []
var actuator_labels: Array[Label] = []
var goals: Array[Node3D] = []
var states: Array = [{}, {}]
var comparison: BoxContainer
var trial: OptionButton
var pause_button: Button
var run_button: Button
var notice: Label
var running := false
var accumulator := 0.0
var paired_root := PAIRED_ROOT
var paired_available := false

func label(text: String, size := 16, muted := false) -> Label:
	var result := Label.new()
	result.text = text
	result.add_theme_font_size_override("font_size", size)
	result.add_theme_color_override("font_color", MUTED if muted else INK)
	return result

func button(text: String, callable: Callable) -> Button:
	var result := Button.new()
	result.text = text
	result.custom_minimum_size.y = 44
	result.pressed.connect(callable)
	return result

func material(color: Color, transparent := false) -> StandardMaterial3D:
	var result := StandardMaterial3D.new()
	result.albedo_color = color
	result.roughness = 0.45
	if transparent:
		result.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		result.cull_mode = BaseMaterial3D.CULL_DISABLED
	return result

func mesh_node(mesh: Mesh, color: Color) -> MeshInstance3D:
	var result := MeshInstance3D.new()
	result.mesh = mesh
	result.material_override = material(color, color.a < 1.0)
	return result

func _ready() -> void:
	get_window().title = "Tadpole Pond · Motor control bench · live physics"
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
	column.add_theme_constant_override("separation", 12)
	margin.add_child(column)
	var header := HBoxContainer.new()
	header.add_child(label("Motor control bench", 25))
	var flexible := Control.new()
	flexible.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	header.add_child(flexible)
	header.add_child(button("Pond menu", func(): get_tree().change_scene_to_file("res://launcher/launcher.tscn")))
	column.add_child(header)
	column.add_child(label("Live physical simulation · frozen controllers · no learning updates", 15, true))
	var actions := HBoxContainer.new()
	actions.add_theme_constant_override("separation", 10)
	trial = OptionButton.new()
	trial.custom_minimum_size = Vector2(240, 44)
	trial.item_selected.connect(func(_id): _run_trial())
	actions.add_child(trial)
	run_button = button("Run same trial", _run_trial)
	actions.add_child(run_button)
	pause_button = button("Pause", _pause)
	actions.add_child(pause_button)
	actions.add_child(button("Next trial", _next_trial))
	column.add_child(actions)
	comparison = BoxContainer.new()
	comparison.add_theme_constant_override("separation", 14)
	comparison.size_flags_vertical = Control.SIZE_EXPAND_FILL
	column.add_child(comparison)
	for index in range(2):
		_build_view(index)
	notice = label("The conventional controller is a diagnostic reference. Experimental motors remain separate from the pond.", 14, true)
	notice.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	column.add_child(notice)
	resized.connect(_layout)
	_layout()
	if not ClassDB.class_exists("MotorControlLab") or not ClassDB.class_has_method("MotorControlLab", "get_controller_info"):
		notice.text = "Build the motor_bench extension before opening this diagnostic scene. Your pond is unchanged."
		run_button.disabled = true
		pause_button.disabled = true
		return
	var candidate := CANDIDATE
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--motor-candidate="):
			candidate = argument.trim_prefix("--motor-candidate=")
		elif argument.begins_with("--paired-root="):
			paired_root = argument.trim_prefix("--paired-root=")
	for index in range(2):
		var native = ClassDB.instantiate("MotorControlLab")
		labs.append(native)
		if not native.configure(ORIGINAL, CALIBRATED, candidate, paired_root):
			messages[index].text = native.get_error()
		else:
			selectors[index].set_item_disabled(4, not native.has_candidate())
			if not native.has_candidate():
				selectors[index].set_item_text(4, "Command motor · not trained")
			for controller in [5, 6]:
				var info: Dictionary = native.get_controller_info(controller)
				selectors[index].set_item_disabled(controller, not info.available)
				selectors[index].set_item_text(controller, ("Torque PPO" if controller == 5 else "Joint-target PPO") + (" · %d/64" % int(info.update) if info.available else " · not trained"))
				selectors[index].set_item_tooltip(controller, _policy_description(info))
	for item: Dictionary in labs[0].get_cases(0):
		trial.add_item(item.name, int(item.id))
	paired_available = bool(labs[0].get_controller_info(5).available) and bool(labs[1].get_controller_info(6).available)
	var any_trained: bool = labs[0].get_controller_info(5).available or labs[1].get_controller_info(6).available
	selectors[0].select(5)
	selectors[1].select(6)
	if paired_available:
		notice.text = "Two saved PPO experiments · identical physical starts · live frozen inference. The working pond and its learner are unchanged."
		_run_trial()
	elif any_trained:
		notice.text = _paired_status()
		_run_trial()
		running = false
		accumulator = 0.0
		pause_button.text = "Resume saved arm"
		run_button.text = "Restart saved arm"
		for index in range(2):
			if not states[index].is_empty():
				_apply_state(index)
	else:
		notice.text = _paired_status()
		run_button.text = "Select an available reference"
		run_button.disabled = true
		pause_button.disabled = true
		for index in range(2):
			views[index].hide()
			if labs[index].get_controller_info(2).available:
				summaries[index].text = "Paired experiment not trained" if not labs[index].get_controller_info(5 + index).available else "Saved trained arm available"
				messages[index].text = "Legacy references remain explicitly selectable above."
			else:
				summaries[index].text = "Frozen sources unavailable"
				messages[index].text = labs[index].get_error()

func _policy_description(info: Dictionary) -> String:
	if not info.get("available", false):
		return str(info.get("error", "No saved trained policy"))
	var evaluation := "Final evaluation not run (partial checkpoint)" if int(info.update) < 64 else ("Final gate passed" if info.passed_final_gate else "Final gate not passed")
	return "Saved PPO weights: %d/64 updates. %s. Live simulation makes no learning updates. Source %s; contract %s." % [int(info.update), evaluation, str(info.source_fingerprint), str(info.environment_fingerprint)]

func _paired_status() -> String:
	var inventory: PackedStringArray = []
	var any_trained := false
	for arm in range(2):
		var info: Dictionary = labs[arm].get_controller_info(5 + arm)
		var arm_name := "Torque" if arm == 0 else "Joint-target"
		if info.available:
			any_trained = true
			inventory.append("%s saved at %d/64" % [arm_name, int(info.update)])
		else:
			inventory.append(arm_name + " not trained")
	var prefix := ("Partial paired experiment · " if any_trained else "Paired experiment not trained · ") + "; ".join(inventory) + ". "
	for path in [paired_root.path_join("status.json"), paired_root.get_base_dir().path_join("status.json")]:
		if FileAccess.file_exists(path):
			var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
			if parsed is Dictionary:
				var reason := str(parsed.get("reason", parsed.get("message", "")))
				if not reason.is_empty():
					return prefix + reason + ". Saved arms use frozen weights; the working pond is unchanged."
	return prefix + "No matched trained pair is available. Saved arms are experimental; old references are separate choices."

func _layout() -> void:
	if comparison:
		comparison.vertical = size.x < 900

func _body_mesh(link: int) -> ArrayMesh:
	var length := 0.020 / 0.04 if link == 0 else 0.026 / 0.04
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
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	return mesh

func _build_view(index: int) -> void:
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
	var selector := OptionButton.new()
	selector.custom_minimum_size.y = 44
	selector.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	for text in ["Passive zero torque", "Conventional reference", "Original saved swimmer", "Calibrated saved motor", "Legacy command motor", "Torque PPO · not trained", "Joint-target PPO · not trained"]:
		selector.add_item(text)
	for controller in [4, 5, 6]:
		selector.set_item_disabled(controller, true)
	selector.item_selected.connect(func(_id): _run_trial())
	var selector_row := HBoxContainer.new()
	column.add_child(selector_row)
	selector_row.add_child(selector)
	var actuator_panel := PopupPanel.new()
	add_child(actuator_panel)
	actuator_panels.append(actuator_panel)
	var actuator_margin := MarginContainer.new()
	for edge in ["left", "right", "top", "bottom"]:
		actuator_margin.add_theme_constant_override("margin_" + edge, 16)
	actuator_panel.add_child(actuator_margin)
	var actuator_label := label("No trial is running.", 15)
	actuator_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	actuator_label.custom_minimum_size = Vector2(380, 150)
	actuator_margin.add_child(actuator_label)
	actuator_labels.append(actuator_label)
	selector_row.add_child(button("Actuators", func(): actuator_panel.popup_centered(Vector2i(420, 210))))
	selectors.append(selector)
	var container := SubViewportContainer.new()
	container.stretch = true
	container.size_flags_vertical = Control.SIZE_EXPAND_FILL
	container.custom_minimum_size = Vector2(200, 180)
	column.add_child(container)
	views.append(container)
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
	floor_mesh.size = Vector3(9.0, 0.04, 7.0)
	var floor_node := mesh_node(floor_mesh, Color("354b43"))
	floor_node.position.y = -4.02
	world.add_child(floor_node)
	for axis in range(-4, 5):
		var line_mesh := BoxMesh.new()
		line_mesh.size = Vector3(0.012, 0.008, 7.0)
		var line_node := mesh_node(line_mesh, Color("607b6a"))
		line_node.position = Vector3(axis, -3.995, 0)
		world.add_child(line_node)
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
	var marker := Node3D.new()
	world.add_child(marker)
	var ring_mesh := TorusMesh.new()
	ring_mesh.inner_radius = 0.17
	ring_mesh.outer_radius = 0.19
	ring_mesh.rings = 32
	marker.add_child(mesh_node(ring_mesh, ACCENT))
	var direction_mesh := CylinderMesh.new()
	direction_mesh.height = 0.55
	direction_mesh.top_radius = 0.012
	direction_mesh.bottom_radius = 0.012
	var direction := mesh_node(direction_mesh, ACCENT)
	direction.rotation.z = -PI / 2
	direction.position.x = 0.27
	marker.add_child(direction)
	goals.append(marker)
	var summary := label("", 16)
	column.add_child(summary)
	summaries.append(summary)
	var message := label("", 14, true)
	message.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	column.add_child(message)
	messages.append(message)

func _run_trial() -> void:
	if labs.size() != 2 or trial.item_count == 0:
		return
	running = false
	accumulator = 0.0
	for index in range(2):
		if not labs[index].start_trial(0, trial.get_selected_id(), selectors[index].selected):
			messages[index].text = labs[index].get_error()
			states[index] = {}
			summaries[index].text = ("Torque arm unavailable" if selectors[index].selected == 5 else "Joint-target arm unavailable") if selectors[index].selected >= 5 else "Trial unavailable"
			views[index].visible = false
			actuator_labels[index].text = "Trial unavailable. " + labs[index].get_error()
		else:
			running = true
			views[index].visible = true
			states[index] = labs[index].get_state()
			_apply_state(index)
	run_button.disabled = false
	pause_button.disabled = not running
	pause_button.text = "Pause"
	run_button.text = "Restart same trial"
	print("MOTOR_BENCH_LIVE case=%d A=%d B=%d training=0" % [trial.get_selected_id(), selectors[0].selected, selectors[1].selected])

func _pause() -> void:
	if pause_button.disabled:
		return
	running = not running
	accumulator = 0.0
	pause_button.text = "Pause" if running else "Resume"
	for index in range(2):
		if not states[index].is_empty():
			_apply_state(index)

func _next_trial() -> void:
	if trial.item_count > 0:
		trial.select((trial.selected + 1) % trial.item_count)
		_run_trial()

func _physics_process(delta: float) -> void:
	if not running:
		return
	accumulator += minf(delta, 0.2)
	while accumulator >= STEP:
		accumulator -= STEP
		var complete := true
		for index in range(labs.size()):
			if states[index].is_empty():
				continue
			states[index] = labs[index].step()
			_apply_state(index)
			complete = complete and bool(states[index].get("complete", true))
		if complete:
			running = false
			pause_button.text = "Trial complete"
			pause_button.disabled = true
			break

func _apply_state(index: int) -> void:
	var data: Dictionary = states[index]
	if data.is_empty():
		return
	if data.get("finite", false):
		for link in range(3):
			var pose: Dictionary = data.links[link]
			bodies[index][link].transform = Transform3D(Basis(pose.quaternion), pose.position)
		goals[index].position = data.goal
		goals[index].quaternion = Quaternion(Vector3.RIGHT, data.heading)
	summaries[index].text = "%.2f s    Distance %.1f mm    Heading %.1f°\nHead %.2f mm/s    Whole body %.2f mm/s\nClearance %.2f mm\nBest quiet hold %.2f / 10 s" % [data.elapsed, data.goal_error_mm, data.heading_error_deg, data.speed_mm_s, data.body_speed_mm_s, data.clearance_mm, data.best_hold_seconds]
	var result := ("Running" if running else "Paused") + " · no rescue or automatic restart"
	if data.get("safety_failed", false):
		result = "Safety failed · continuing to show the physical outcome"
	if data.complete:
		result = "Passed this trial" if data.passed_trial else "Did not pass this trial"
	if data.experimental:
		result += " · experimental, separate from the pond"
	if data.get("paired", false):
		var evaluation := "final evaluation not run" if not data.training_complete else ("final gate passed" if data.qualified_candidate else "final gate not passed")
		result = "PPO weights %d/%d · %s · %s\n" % [int(data.policy_update), int(data.policy_budget), "budget reached" if data.training_complete else "partial checkpoint", evaluation] + result
	messages[index].text = result
	var torque: Array[float] = []
	for value in data.get("actual_torque_uNm", []):
		torque.append(float(value))
	if torque.is_empty():
		for value in data.commands:
			torque.append(float(value) * 10.0)
	var targets: Array[float] = []
	for value in data.get("joint_targets_rad", []):
		targets.append(float(value))
	var raw_caption := "Raw neural action (before actuator clipping)" if int(data.controller_id) >= 2 else "Reference controller command (before actuator clipping)"
	actuator_labels[index].text = "%s:\n%s\n%s\nActual torque, last 5 ms (µN·m):\n%s\n50 ms RMS torque: %s" % [raw_caption, _four_values(data.raw_actions), ("Held target (rad): " + _four_values(targets)) if not targets.is_empty() else "Direct torque interface · no joint-angle target", _four_values(torque), ("%.3f µN·m" % float(data.torque_rms_uNm)) if data.has("torque_rms_uNm") else "constant commands over this control tick"]
	if not targets.is_empty():
		actuator_labels[index].text += "\nRequested PD torque, before physical clipping (µN·m):\n" + _four_values(data.get("requested_torque_uNm", []))

func _four_values(values) -> String:
	var text: PackedStringArray = []
	for value in values:
		text.append("%+.3f" % float(value))
	return "[" + ", ".join(text) + "]"

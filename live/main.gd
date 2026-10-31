extends Node3D
# LIVE native model6, frozen actor inference. No CSV pose playback or training.
var swimmer: RefCounted
var links: Array[Node3D] = []
var geometry: Dictionary = {}
var status: Label
var start_button: Button
var pause_button: Button
var accumulator := 0.0
var state: Dictionary = {}
var speed_control: SpinBox
var output := ""
var capture_path := ""
var capture_queued := false
var capture_saved := false
var capture_paused := false
var total_substeps := 0
var started_ms := 0
func _material(color: Color) -> StandardMaterial3D:
	var result := StandardMaterial3D.new()
	result.albedo_color = color
	result.roughness = 0.5
	return result

func _piece(parent: Node3D, mesh: Mesh, color: Color, at := Vector3.ZERO) -> MeshInstance3D:
	var piece := MeshInstance3D.new()
	piece.mesh = mesh
	piece.material_override = _material(color)
	piece.position = at
	parent.add_child(piece)
	return piece

func _label(value: String, size: int) -> Label:
	var label := Label.new()
	label.text = value
	label.add_theme_font_size_override("font_size", size)
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	return label

func _button(value: String, action: Callable) -> Button:
	var button := Button.new()
	button.text = value
	button.custom_minimum_size = Vector2(100, 44)
	button.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	button.pressed.connect(action)
	return button


func _layout_camera(camera: Camera3D) -> void:
	var narrow := get_window().size.x < 500
	camera.fov = 72 if narrow else 42
	camera.v_offset = 1.0 if narrow else 0.0

func _ready() -> void:
	started_ms = Time.get_ticks_msec()
	get_window().title = "CudaRL - LIVE model6 - frozen learned controller"
	for argument in OS.get_cmdline_user_args():
		if argument == "--capture-paused": capture_paused = true
		if argument.begins_with("--output="): output = argument.trim_prefix("--output=")
		if argument.begins_with("--capture="): capture_path = argument.trim_prefix("--capture=")
	var environment := WorldEnvironment.new()
	var setting := Environment.new()
	setting.background_mode = Environment.BG_COLOR
	setting.background_color = Color("122d3b")
	setting.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	setting.ambient_light_color = Color("c1e6ee")
	setting.ambient_light_energy = 0.85
	environment.environment = setting
	add_child(environment)
	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-55, -30, 0)
	add_child(light)
	var camera := Camera3D.new()
	camera.position = Vector3(0.5, 0, 3.6)
	_layout_camera(camera)
	add_child(camera)
	camera.look_at(Vector3(-0.35, -2, 0))
	get_window().size_changed.connect(_layout_camera.bind(camera))
	var floor_mesh := PlaneMesh.new()
	floor_mesh.size = Vector2(9, 7)
	_piece(self, floor_mesh, Color("294b4d"), Vector3(0, -4, 0))
	var canvas := CanvasLayer.new()
	add_child(canvas)
	var margin := MarginContainer.new()
	margin.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	for side in ["left", "right", "top", "bottom"]: margin.add_theme_constant_override("margin_"+side,18)
	canvas.add_child(margin)
	var column := VBoxContainer.new()
	column.add_theme_constant_override("separation",10)
	margin.add_child(column)
	column.add_child(_label("Live swimming / frozen controller",25))
	column.add_child(_label("Frozen learned strength and frequency; supplied stroke. Real native model6 physics. No training. Auto-pauses after4 seconds.",16))
	status = _label("Loading verified actor...",16)
	column.add_child(status)
	speed_control = SpinBox.new()
	speed_control.custom_minimum_size = Vector2(140,44)
	speed_control.step = 0.5
	speed_control.suffix = " mm/s"
	speed_control.value_changed.connect(func(value: float):
		if swimmer != null:
			swimmer.call("set_speed",value/1000)
			_refresh())
	column.add_child(speed_control)
	var spacer := Control.new()
	spacer.size_flags_vertical = Control.SIZE_EXPAND_FILL
	column.add_child(spacer)
	var row := HBoxContainer.new()
	column.add_child(row)
	start_button = _button("Start",_start)
	pause_button = _button("Pause",_pause)
	row.add_child(start_button)
	row.add_child(pause_button)
	row.add_child(_button("Reset",_reset))
	start_button.disabled = true
	pause_button.disabled = true
	if not ClassDB.class_exists("FrozenSwimmer"):
		var loaded := GDExtensionManager.load_extension("res://addons/frozen_swimmer/frozen_swimmer.gdextension")
		if loaded != GDExtensionManager.LOAD_STATUS_OK and loaded != GDExtensionManager.LOAD_STATUS_ALREADY_LOADED:
			_fail("Native extension could not load; build frozen_swimmer first.")
			return
	swimmer = ClassDB.instantiate("FrozenSwimmer") as RefCounted
	if swimmer == null or not bool(swimmer.call("load_controller")):
		_fail("Frozen actor load failed: " + (str(swimmer.call("get_error")) if swimmer else "class missing"))
		return
	state = swimmer.call("get_state")
	speed_control.min_value = float(state.minimum_speed_m_s)*1000
	speed_control.max_value = float(state.maximum_speed_m_s)*1000
	speed_control.set_value_no_signal(float(state.target_speed_m_s)*1000)
	speed_control.editable = speed_control.min_value < speed_control.max_value
	geometry = swimmer.call("get_geometry")
	_build_creature()
	_refresh(state)
	if not capture_path.is_empty():
		if capture_paused: _capture_still()
		else: _start()

func _start() -> void:
	if swimmer != null and bool(swimmer.call("start")):
		accumulator = 0
		_refresh()

func _pause() -> void:
	if swimmer != null: swimmer.call("pause")
	accumulator = 0
	_refresh()

func _reset() -> void:
	if swimmer == null: return
	accumulator = 0
	if not bool(swimmer.call("reset",0)):
		_fail(str(swimmer.call("get_error")))
		return
	_refresh()

func _process(delta: float) -> void:
	if swimmer == null or state.is_empty() or not bool(state.loaded): return
	if not capture_path.is_empty() and Time.get_ticks_msec()-started_ms > 10000:
		_fail("Live UI verification wall cap")
		return
	if not bool(state.paused):
		accumulator += delta
		var tick_count := 0
		while accumulator >= 0.05 and tick_count < 4 and not bool(state.paused):
			var before_substeps := int(state.substeps)
			if not capture_path.is_empty() and total_substeps+10 > 800:
				_fail("Live UI substep cap")
				return
			var advanced := bool(swimmer.call("step"))
			state = swimmer.call("get_state")
			total_substeps += int(state.substeps)-before_substeps
			if not advanced:
				_fail(str(state.error))
				return
			accumulator -= 0.05
			tick_count += 1
		_refresh(state)
	if not capture_path.is_empty() and int(state.get("controls",0)) >= 50 and not capture_queued:
		capture_queued = true
		_capture()
	if not capture_path.is_empty() and bool(state.get("finished",false)) and capture_saved:
		var saved := _write_result({"ui_passed":bool(state.passed),"physical_substeps":total_substeps,"controls":state.controls,"auto_paused":state.paused,"screenshot":capture_path})
		get_tree().quit(0 if saved and bool(state.passed) else 3)

func _refresh(snapshot: Dictionary = {}) -> void:
	if swimmer == null: return
	state = swimmer.call("get_state") if snapshot.is_empty() else snapshot
	var active := not bool(state.paused)
	start_button.disabled = not bool(state.loaded) or bool(state.finished) or active or not str(state.error).is_empty()
	pause_button.disabled = not active
	speed_control.set_value_no_signal(float(state.target_speed_m_s)*1000)
	var prefix := "RUNNING" if active else ("COMPLETE / auto-paused" if bool(state.finished) else "PAUSED")
	status.text = "%s | %.2f / 4.00 s\nCOM speed %.2f mm/s | target %.2f | steady %.2f mm/s\nStrength %.3f rad | cadence %.2f Hz | positive work %.2f microJ" % [prefix,float(state.segment_time_s),float(state.speed_m_s)*1000,float(state.target_speed_m_s)*1000,float(state.steady_speed_m_s)*1000,float(state.amplitude),float(state.frequency),float(state.positive_work_J)*1e6]
	if not str(state.error).is_empty(): status.text += "\nERROR: " + str(state.error)
	var poses: Array = state.links
	for i in range(mini(links.size(),poses.size())):
		links[i].position = poses[i].position
		links[i].quaternion = poses[i].orientation

func _capture_still() -> void:
	await _capture()
	if not capture_saved: return
	var saved := _write_result({"paused_layout_passed":bool(state.loaded) and bool(state.paused),"physical_substeps":total_substeps})
	get_tree().quit(0 if saved else 3)

func _capture() -> void:
	await RenderingServer.frame_post_draw
	var code := get_viewport().get_texture().get_image().save_png(capture_path)
	if code != OK:
		_fail("Could not save verified live screenshot")
		return
	capture_saved = true

func _write_result(result: Dictionary) -> bool:
	result["wall_seconds"] = float(Time.get_ticks_msec()-started_ms)/1000
	if not output.is_empty():
		var file := FileAccess.open(output,FileAccess.WRITE)
		var error := FileAccess.get_open_error()
		if file:
			file.store_string(JSON.stringify(result))
			file.flush()
			error = file.get_error()
		if error != OK:
			var reason := "Could not write verification result: " + error_string(error)
			push_error(reason)
			print(JSON.stringify({"error":reason,"physical_substeps":total_substeps}))
			return false
	print(JSON.stringify(result))
	return true

func _fail(reason: String) -> void:
	if swimmer != null:
		swimmer.call("pause")
		_refresh()
	accumulator = 0
	if start_button != null: start_button.disabled = true
	if pause_button != null: pause_button.disabled = true
	if status != null: status.text = reason
	push_error(reason)
	if not capture_path.is_empty():
		_write_result({"error":reason,"physical_substeps":total_substeps})
		get_tree().quit(3)

func _build_creature() -> void:
	for index in range(3):
		var link := Node3D.new()
		add_child(link)
		links.append(link)
	var scale_m: float = geometry.scale_m
	var dimensions: Array = geometry.links
	var head := SphereMesh.new()
	head.radius = 1
	head.height = 2
	head.radial_segments = 40
	head.rings = 24
	var head_piece := _piece(links[0], head, Color("81d6bd"))
	head_piece.scale = Vector3(float(dimensions[0].length) / 2, float(dimensions[0].half_height), float(dimensions[0].half_width)) / scale_m
	for index in [1, 2]:
		var rod := CylinderMesh.new()
		rod.height = float(dimensions[index].length) / scale_m
		rod.top_radius = float(geometry.rod_radius_m) / scale_m
		rod.bottom_radius = rod.top_radius
		var rod_piece := _piece(links[index], rod, Color("64bda8"))
		rod_piece.rotation.z = PI / 2
	var wing_height := float(geometry.fin_extension_m) / 2
	for sign_value in [-1, 1]:
		var wing := BoxMesh.new()
		wing.size = Vector3(float(dimensions[2].length), wing_height, float(geometry.fin_thickness_m)) / scale_m
		_piece(links[2], wing, Color("62d8c6"), Vector3(0, sign_value * (float(geometry.rod_radius_m) + wing_height / 2) / scale_m, 0))
	for sign_value in [-1, 1]:
		var eye := SphereMesh.new()
		eye.radius = 0.034
		eye.height = 0.068
		_piece(links[0], eye, Color("10282b"), Vector3(0.14, 0.06, sign_value * 0.18))

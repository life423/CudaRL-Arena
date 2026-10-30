extends Node3D
# Recorded CSV poses only. No native world, physics stepping, inference or saves.
# Model6 dimensions are read from the accompanying frozen-run geometry artifact.
var samples: Array[Dictionary] = []
var links: Array[Node3D] = []
var geometry: Dictionary = {}
var time := 0.0
var playing := true
var status: Label
var explanation: Label
var playback: Button
var scrub: HSlider
var mode: OptionButton
var updating := false
var capture_path := ""
var capture_width := 1100

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

func _ready() -> void:
	get_window().title = "CudaRL - recorded model6 movement, not live simulation"
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--capture="): capture_path = argument.trim_prefix("--capture=")
		if argument.begins_with("--width="): capture_width = int(argument.trim_prefix("--width="))
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
	camera.position = Vector3(0.5, 0.0, 3.6)
	camera.fov = 42
	add_child(camera)
	camera.look_at(Vector3(-0.35, -2, 0))
	get_window().size_changed.connect(func(): camera.fov = 72 if get_window().size.x < 500 else 42)
	var floor_mesh := PlaneMesh.new()
	floor_mesh.size = Vector2(9, 7)
	_piece(self, floor_mesh, Color("294b4d"), Vector3(0, -4, 0))
	var canvas := CanvasLayer.new()
	add_child(canvas)
	var margin := MarginContainer.new()
	margin.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	for side in ["left", "right", "top", "bottom"]: margin.add_theme_constant_override("margin_" + side, 18)
	canvas.add_child(margin)
	var column := VBoxContainer.new()
	column.add_theme_constant_override("separation", 10)
	margin.add_child(column)
	column.add_child(_label("Recorded straight swim", 25))
	explanation = _label("Selected learned controls: 4/4 straight-speed checks. Supplied stroke; recorded poses only. Head sway remains about 27 degrees.", 16)
	column.add_child(explanation)
	status = _label("Loading measured poses...", 16)
	column.add_child(status)
	var spacer := Control.new()
	spacer.size_flags_vertical = Control.SIZE_EXPAND_FILL
	column.add_child(spacer)
	mode = OptionButton.new()
	mode.custom_minimum_size.y = 44
	mode.add_item("Selected learned controls - frozen evaluation (4/4)")
	mode.add_item("Original default - frozen baseline (4/4)")
	mode.item_selected.connect(func(_index: int): _load_trace())
	column.add_child(mode)
	scrub = HSlider.new()
	scrub.custom_minimum_size.y = 44
	scrub.step = 0.01
	scrub.focus_mode = Control.FOCUS_ALL
	scrub.value_changed.connect(func(value: float):
		if not updating:
			time = value
			playing = false
			playback.text = "Play"
			_show())
	column.add_child(scrub)
	var row := HBoxContainer.new()
	column.add_child(row)
	playback = _button("Pause", func():
		playing = not playing
		playback.text = "Pause" if playing else "Play")
	row.add_child(playback)
	row.add_child(_button("Replay", func():
		time = 0
		playing = true
		playback.text = "Pause"))
	var case_label := _label("Case 1 of 4", 15)
	case_label.autowrap_mode = TextServer.AUTOWRAP_OFF
	case_label.custom_minimum_size.x = 100
	case_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row.add_child(case_label)
	var data_path := _run_path("view_geometry.json")
	geometry = JSON.parse_string(FileAccess.get_file_as_string(data_path))
	if geometry.is_empty():
		status.text = "Missing recorded-run geometry: " + data_path
		return
	_build_creature()
	_load_trace()
	if not capture_path.is_empty():
		get_window().size = Vector2i(capture_width, 760 if capture_width > 500 else 720)
		playing = false
		playback.text = "Play"
		time = 2.5
		_show()
		await get_tree().process_frame
		await RenderingServer.frame_post_draw
		get_viewport().get_texture().get_image().save_png(capture_path)
		get_tree().quit()

func _run_path(name: String) -> String:
	return ProjectSettings.globalize_path("res://").path_join("../build/straight-motor/2026-10-30-task4-steady-reward-v2/" + name).simplify_path()

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

func _load_trace() -> void:
	samples.clear()
	var file := FileAccess.open(_run_path("after.csv" if mode.selected == 0 else "before.csv"), FileAccess.READ)
	explanation.text = "Selected learned controls: 4/4 straight-speed checks. Supplied stroke; recorded poses only. Head sway remains about 27 degrees." if mode.selected == 0 else "Original default: 4/4 straight-speed checks. Compare speed, drift and effort with the selected learned controls. No live learning."
	if file == null:
		status.text = "Recorded trace unavailable; no simulation will be started."
		return
	var header := file.get_csv_line()
	while not file.eof_reached():
		var values := file.get_csv_line()
		if values.size() != header.size() or values[0] != "0": continue
		var sample := {}
		for index in range(header.size()): sample[header[index]] = float(values[index])
		samples.append(sample)
	time = 0
	scrub.max_value = samples.size() * 0.05
	_show()

func _process(delta: float) -> void:
	if samples.is_empty(): return
	if playing:
		time = minf(time + delta, samples.size() * 0.05)
		if time >= samples.size() * 0.05:
			playing = false
			playback.text = "Play"
	_show()

func _show() -> void:
	if samples.is_empty() or links.is_empty(): return
	var index := clampi(int(time / 0.05) - 1, 0, samples.size() - 1)
	var sample: Dictionary = samples[index]
	var q := Quaternion(float(sample.qx), float(sample.qy), float(sample.qz), float(sample.qw))
	links[0].position = Vector3(float(sample.head_x), float(sample.head_y), float(sample.head_z))
	links[0].quaternion = q
	for link in [1, 2]:
		var previous := links[link - 1]
		var parent_q := previous.quaternion
		var angle0: float = sample["j" + str((link - 1) * 2)]
		var angle1: float = sample["j" + str((link - 1) * 2 + 1)]
		var next_q := parent_q * Quaternion(Vector3.UP, angle0) * Quaternion(Vector3(0, 0, 1), angle1)
		var parent_length := float(geometry.links[link - 1].length) / float(geometry.scale_m)
		var length_model := float(geometry.links[link].length) / float(geometry.scale_m)
		links[link].position = previous.position + parent_q * Vector3(-parent_length / 2, 0, 0) + next_q * Vector3(-length_model / 2, 0, 0)
		links[link].quaternion = next_q
	status.text = "t %.2f s  |  COM travel %.2f mm  |  strength %.3f rad  |  cadence %.2f Hz\nRecorded 20 Hz poses; playback performs no physics or learning." % [time, float(sample.forward_m) * 1000, float(sample.amplitude), float(sample.frequency)]
	updating = true
	scrub.value = time
	updating = false

extends Node3D
## Isolated playback of native positions, attitude and flexible-tail centerline.

const METRES_PER_UNIT := 0.01
const SPINE_POINTS := 7
const RING_SIDES := 12
const SAMPLES_PER_SPAN := 3
const TRACE_ARG := "--flex-tail-trace="
const COURSE_OFFSET := Vector3(0, 0.7, 0)
const MOVEMENT_PHASES := {
	"swim": [{"time_s": 0, "label": "Swimming straight"}],
	"turn": [{"time_s": 0, "label": "Swimming straight"}, {"time_s": 6, "label": "Turning · 30° request"}],
}

class TraceFrame:
	var time_s: float
	var body: Vector3
	var course_position: Vector3
	var velocity: Vector3
	var rotation: Quaternion
	var spine: PackedVector3Array

class TracePhase:
	var time_s: float
	var label: String

@onready var camera: Camera3D = $Camera3D
@onready var header: PanelContainer = %Header
@onready var metrics: Label = %Metrics
@onready var message: PanelContainer = %Message
@onready var status: Label = %Status
@onready var transport: PanelContainer = %Transport
@onready var play: Button = %Play
@onready var timeline: HSlider = %Timeline
@onready var clock: Label = %Time
@onready var trace_dialog: FileDialog = %TraceDialog

var frames: Array[TraceFrame] = []
var phases: Array[TracePhase] = []
var anchor := Vector3.INF
var elapsed := 0.0
var frame_index := 0
var playing := false
var head := Node3D.new()
var tail_view := MeshInstance3D.new()
var tail := ImmediateMesh.new()
var course_view := MeshInstance3D.new()
var anchor_view := MeshInstance3D.new()
var floor_view := MeshInstance3D.new()

func _ready() -> void:
	get_window().title = "Tadpole Pond | Flexible Tail Preview"
	get_viewport().msaa_3d = Viewport.MSAA_4X
	_build_world()
	for button in [%Open, play, %Swim, %Turn, %Hold]:
		_style_button(button)
	_style_timeline()
	%Open.pressed.connect(func(): trace_dialog.popup_centered_clamped())
	trace_dialog.file_selected.connect(_open_trace)
	play.pressed.connect(_toggle_play)
	timeline.value_changed.connect(_seek)
	%Swim.pressed.connect(_open_recording.bind("swim"))
	%Turn.pressed.connect(_open_recording.bind("turn"))
	%Hold.pressed.connect(_open_recording.bind("hold"))
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with(TRACE_ARG):
			_open_trace(arg.trim_prefix(TRACE_ARG))
			return
	_open_recording("swim")

func _build_world() -> void:
	var world := WorldEnvironment.new()
	world.environment = Environment.new()
	world.environment.background_mode = Environment.BG_COLOR
	world.environment.background_color = Color("204843")
	world.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	world.environment.ambient_light_color = Color("d8f2e5")
	world.environment.ambient_light_energy = 0.66
	add_child(world)
	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-54, -34, 0)
	light.light_color = Color("fff5d8")
	light.light_energy = 0.95
	add_child(light)

	var floor_mesh := PlaneMesh.new()
	floor_mesh.size = Vector2(72, 72)
	floor_view.mesh = floor_mesh
	var water := ShaderMaterial.new()
	water.shader = Shader.new()
	water.shader.code = """
shader_type spatial;
render_mode unshaded;
varying vec2 world_xz;
void vertex() {
	world_xz = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xz;
}
void fragment() {
	vec2 cell = abs(fract(world_xz + 0.5) - 0.5);
	vec2 line = 1.0 - smoothstep(vec2(0.0), max(fwidth(world_xz) * 1.4, vec2(0.015)), cell);
	float dapple = sin(world_xz.x * 0.26 + sin(world_xz.y * 0.4)) *
		sin(world_xz.y * 0.39 - world_xz.x * 0.17);
	ALBEDO = mix(vec3(0.10, 0.24, 0.22), vec3(0.15, 0.32, 0.27), 0.5 + 0.5 * dapple)
		+ 0.024 * max(line.x, line.y);
}
"""
	floor_view.material_override = water
	add_child(floor_view)

	add_child(head)
	_sphere(head, Vector3.ZERO, Vector3(1, 0.64, 0.8), Color("87cbb0"))
	for side in [-1.0, 1.0]:
		_sphere(head, Vector3(0.66, 0.24, side * 0.53), Vector3(0.155, 0.153, 0.13), Color("173d3d"))
		_sphere(head, Vector3(0.72, 0.31, side * 0.592), Vector3.ONE * 0.043, Color("e0f4df"))
	tail_view.mesh = tail
	var skin := StandardMaterial3D.new()
	skin.albedo_color = Color("70bd9f")
	skin.roughness = 0.61
	skin.cull_mode = BaseMaterial3D.CULL_DISABLED
	tail_view.material_override = skin
	add_child(tail_view)
	var course_material := StandardMaterial3D.new()
	course_material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	course_material.albedo_color = Color("69a78b")
	course_view.material_override = course_material
	add_child(course_view)
	var anchor_material := StandardMaterial3D.new()
	anchor_material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	anchor_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	anchor_material.no_depth_test = true
	anchor_material.albedo_color = Color("f6cb86d9")
	anchor_view.material_override = anchor_material
	anchor_view.mesh = _anchor_mesh()
	add_child(anchor_view)
	_set_subject_visible(false)
	_place_camera(Vector3.ZERO)
	get_viewport().size_changed.connect(_fit_camera)
	_fit_camera()

func _style_button(button: Button) -> void:
	for state in ["normal", "hover", "pressed", "focus"]:
		var surface := StyleBoxFlat.new()
		surface.set_corner_radius_all(9)
		surface.bg_color = Color("285e50")
		if state == "hover":
			surface.bg_color = Color("326d5d")
		elif state == "pressed":
			surface.bg_color = Color("214d42")
		if button.toggle_mode and state in ["normal", "hover"]:
			surface.bg_color = Color("e5ece2") if state == "normal" else Color("d4e4da")
		if state == "focus":
			surface.draw_center = false
			surface.set_border_width_all(2)
			surface.border_color = Color("c2edd7") if button == %Open else Color("173d3d")
			surface.set_expand_margin_all(2)
		button.add_theme_stylebox_override(state, surface)
	for state in ["font_color", "font_hover_color", "font_pressed_color", "font_focus_color"]:
		button.add_theme_color_override(state, Color("f9fff9"))
	if button.toggle_mode:
		for state in ["font_color", "font_hover_color", "font_focus_color"]:
			button.add_theme_color_override(state, Color("173d3d"))
		button.add_theme_color_override("font_hover_pressed_color", Color("f9fff9"))

func _style_timeline() -> void:
	for part in ["slider", "grabber_area", "grabber_area_highlight"]:
		var track := StyleBoxFlat.new()
		track.set_corner_radius_all(3)
		track.content_margin_top = 3.0
		track.content_margin_bottom = 3.0
		track.bg_color = Color("a1b8a9") if part == "slider" else Color("326d5d")
		timeline.add_theme_stylebox_override(part, track)
	var dot := Gradient.new()
	dot.offsets = PackedFloat32Array([0.0, 0.72, 0.88, 1.0])
	dot.colors = PackedColorArray([Color("285e50"), Color("285e50"), Color("285e5000"), Color("285e5000")])
	var thumb := GradientTexture2D.new()
	thumb.gradient = dot
	thumb.width = 19
	thumb.height = 19
	thumb.fill = GradientTexture2D.FILL_RADIAL
	thumb.fill_from = Vector2(0.5, 0.5)
	thumb.fill_to = Vector2(0.5, 1.0)
	timeline.add_theme_icon_override("grabber", thumb)
	timeline.add_theme_icon_override("grabber_highlight", thumb)
	var focus := StyleBoxFlat.new()
	focus.set_corner_radius_all(5)
	focus.draw_center = false
	focus.set_border_width_all(2)
	focus.border_color = Color("173d3d")
	timeline.add_theme_stylebox_override("focus", focus)

func _sphere(parent: Node3D, at: Vector3, dimensions: Vector3, color: Color) -> void:
	var shape := SphereMesh.new()
	shape.radius = 1.0
	shape.height = 2.0
	shape.radial_segments = 32
	shape.rings = 16
	var model := MeshInstance3D.new()
	model.mesh = shape
	var material := StandardMaterial3D.new()
	material.albedo_color = color
	material.roughness = 0.53
	model.material_override = material
	parent.add_child(model)
	model.position = at
	model.scale = dimensions

func _open_recording(recording: String) -> void:
	_open_trace("res://recordings/%s.json" % recording, recording)

func _open_trace(path: String, recording: String = "") -> void:
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		_show_problem("Could not open the selected trace.")
		return
	var data: Variant = JSON.parse_string(file.get_as_text())
	if not data is Dictionary or not data.get("frames") is Array:
		_show_problem("The trace is missing its physics frames.")
		return
	var decoded: Array[TraceFrame] = []
	var previous_time := -INF
	for sample in data.frames:
		var frame := _decode_frame(sample, previous_time)
		if frame == null:
			_show_problem("The trace has an invalid physics frame.")
			return
		decoded.append(frame)
		previous_time = frame.time_s
	if decoded.size() < 2:
		_show_problem("The trace needs at least two physics frames.")
		return
	frames = decoded
	phases = _read_phases(data.get("phases", MOVEMENT_PHASES.get(recording)))
	anchor = _read_vector(data.get("anchor_m"))
	if anchor.is_finite():
		anchor_view.position = anchor - COURSE_OFFSET
	elapsed = 0.0
	frame_index = 0
	playing = not recording.is_empty()
	timeline.max_value = frames[-1].time_s - frames[0].time_s
	timeline.set_value_no_signal(0.0)
	play.text = "Pause" if playing else "Play"
	play.disabled = false
	timeline.editable = true
	for button in [%Swim, %Turn, %Hold]:
		button.set_pressed_no_signal(String(button.name).to_lower() == recording)
	message.hide()
	_set_subject_visible(true)
	_build_course()
	_render_frame()
	print("Flexible tail preview loaded %d frames (%.3f s) from %s" % [frames.size(), timeline.max_value, path])

func _decode_frame(sample: Variant, previous_time: float) -> TraceFrame:
	if not sample is Dictionary:
		return null
	var time: Variant = sample.get("time_s")
	if not _is_number(time) or float(time) <= previous_time:
		return null
	var origin := _read_vector(sample.get("body_position_m"))
	var raw_rotation: Variant = sample.get("body_quaternion")
	if not origin.is_finite() or not raw_rotation is Array or raw_rotation.size() != 4:
		return null
	for n in raw_rotation:
		if not _is_number(n):
			return null
	var rotation := Quaternion(raw_rotation[0], raw_rotation[1], raw_rotation[2], raw_rotation[3])
	if rotation.length_squared() < 0.000001 or not rotation.is_finite():
		return null
	var points: Variant = sample.get("spine_m")
	if not points is Array or points.size() != SPINE_POINTS:
		return null
	var spine := PackedVector3Array()
	for point in points:
		var position := _read_vector(point)
		if not position.is_finite():
			return null
		if not spine.is_empty() and position.distance_squared_to(spine[-1]) < 1e-8:
			return null
		spine.append(position)
	var frame := TraceFrame.new()
	frame.time_s = float(time)
	frame.body = origin
	frame.course_position = _read_vector(sample.get("com_m", sample.get("body_position_m")))
	if not frame.course_position.is_finite():
		return null
	frame.velocity = _read_vector(sample.get("com_velocity_m_s"))
	frame.rotation = rotation.normalized()
	frame.spine = spine
	return frame

func _read_phases(raw: Variant) -> Array[TracePhase]:
	var result: Array[TracePhase] = []
	if not raw is Array:
		return result
	var previous_time := -INF
	for value in raw:
		if not value is Dictionary:
			continue
		var time: Variant = value.get("time_s")
		var label: Variant = value.get("label")
		if not _is_number(time) or not label is String:
			continue
		if float(time) <= previous_time or float(time) < frames[0].time_s or float(time) > frames[-1].time_s:
			continue
		var phase_label: String = label.strip_edges()
		if phase_label.is_empty() or phase_label.length() > 32:
			continue
		var phase := TracePhase.new()
		phase.time_s = float(time)
		phase.label = phase_label
		result.append(phase)
		previous_time = phase.time_s
	return result

func _is_number(value: Variant) -> bool:
	return (value is float or value is int) and is_finite(float(value))

func _read_vector(value: Variant) -> Vector3:
	if value is Array and value.size() == 3:
		if _is_number(value[0]) and _is_number(value[1]) and _is_number(value[2]):
			return Vector3(value[0], value[1], value[2]) / METRES_PER_UNIT
	return Vector3.INF

func _show_problem(problem: String) -> void:
	playing = false
	play.text = "Play"
	play.disabled = true
	timeline.editable = false
	clock.text = ""
	metrics.hide()
	_set_subject_visible(false)
	status.text = problem
	message.show()
	push_warning(problem)

func _set_subject_visible(visible: bool) -> void:
	head.visible = visible
	tail_view.visible = visible
	course_view.visible = visible
	anchor_view.visible = visible and anchor.is_finite()

func _anchor_mesh() -> ImmediateMesh:
	var marker := ImmediateMesh.new()
	marker.surface_begin(Mesh.PRIMITIVE_LINE_STRIP)
	for i in range(33):
		var angle := TAU * i / 32.0
		marker.surface_add_vertex(Vector3(cos(angle), 0, sin(angle)) * 0.23)
	marker.surface_end()
	marker.surface_begin(Mesh.PRIMITIVE_LINES)
	marker.surface_add_vertex(Vector3.LEFT * 0.12)
	marker.surface_add_vertex(Vector3.RIGHT * 0.12)
	marker.surface_add_vertex(Vector3.FORWARD * 0.12)
	marker.surface_add_vertex(Vector3.BACK * 0.12)
	marker.surface_end()
	return marker

func _build_course() -> void:
	var course := ImmediateMesh.new()
	course.surface_begin(Mesh.PRIMITIVE_LINE_STRIP)
	for frame in frames:
		course.surface_add_vertex(frame.course_position - COURSE_OFFSET)
	course.surface_end()
	course_view.mesh = course

func _toggle_play() -> void:
	if not playing and elapsed >= timeline.max_value:
		_seek(0.0)
	playing = not playing
	play.text = "Pause" if playing else "Play"

func _seek(value: float) -> void:
	elapsed = value
	timeline.set_value_no_signal(value)
	_render_frame()

func _process(delta: float) -> void:
	if not playing:
		return
	_seek(minf(timeline.max_value, elapsed + delta))
	if elapsed >= timeline.max_value:
		playing = false
		play.text = "Replay"

func _render_frame() -> void:
	var time := elapsed + frames[0].time_s
	while frame_index < frames.size() - 2 and frames[frame_index + 1].time_s < time:
		frame_index += 1
	while frame_index > 0 and frames[frame_index].time_s > time:
		frame_index -= 1
	var first := frames[frame_index]
	var next := frames[frame_index + 1]
	var weight := (time - first.time_s) / (next.time_s - first.time_s)
	head.position = first.body.lerp(next.body, weight)
	head.quaternion = first.rotation.slerp(next.rotation, weight).normalized()
	var spine := PackedVector3Array()
	for i in range(SPINE_POINTS):
		spine.append(first.spine[i].lerp(next.spine[i], weight))
	_draw_tail(spine, head.quaternion)
	_place_camera(head.position.lerp(spine[-1], 0.43))
	clock.text = "%.1f / %.1f s" % [elapsed, timeline.max_value]
	_render_metrics(time, first, next, weight)

func _render_metrics(time: float, first: TraceFrame, next: TraceFrame, weight: float) -> void:
	var readout := PackedStringArray()
	var phase_label := ""
	for phase in phases:
		if time < phase.time_s:
			break
		phase_label = phase.label
	if not phase_label.is_empty():
		readout.append(phase_label)
	if first.velocity.is_finite() and next.velocity.is_finite():
		var speed_mm_s := first.velocity.lerp(next.velocity, weight).length() * METRES_PER_UNIT * 1000.0
		readout.append("COM speed %.2f mm/s" % speed_mm_s)
	if anchor.is_finite():
		var offset := first.course_position.lerp(next.course_position, weight) - anchor
		var mm_from_anchor := Vector2(offset.x, offset.z).length() * METRES_PER_UNIT * 1000.0
		readout.append("%.1f mm from anchor" % mm_from_anchor)
	metrics.text = "\n".join(readout)
	metrics.visible = not readout.is_empty()

func _place_camera(center: Vector3) -> void:
	camera.position = center + Vector3(0.7, 8.4, 5.9)
	camera.look_at(center)
	floor_view.position = center - Vector3.UP * 3.0

func _fit_camera() -> void:
	var window := get_viewport().get_visible_rect().size
	camera.size = maxf(8.8, 8.7 * window.y / maxf(window.x, 1.0))
	header.offset_right = minf(252.0, window.x - 148.0)
	var dock_half_width := minf(300.0, window.x / 2.0 - 16.0)
	transport.offset_left = -dock_half_width
	transport.offset_right = dock_half_width

func _draw_tail(spine: PackedVector3Array, rotation: Quaternion) -> void:
	var centerline := PackedVector3Array()
	for i in range(SPINE_POINTS - 1):
		for j in range(SAMPLES_PER_SPAN):
			var fraction := float(j) / float(SAMPLES_PER_SPAN)
			centerline.append(spine[i].cubic_interpolate(
				spine[i + 1], spine[maxi(0, i - 1)], spine[mini(SPINE_POINTS - 1, i + 2)], fraction
			))
	centerline.append(spine[-1])

	var rings := PackedVector3Array()
	var normals := PackedVector3Array()
	for i in range(centerline.size()):
		var axis := (centerline[mini(i + 1, centerline.size() - 1)] - centerline[maxi(i - 1, 0)]).normalized()
		if axis.is_zero_approx():
			axis = rotation * Vector3.LEFT
		var across := axis.cross(rotation * Vector3.UP)
		if across.length_squared() < 0.00001:
			across = axis.cross(rotation * Vector3.RIGHT)
		across = across.normalized()
		var up := across.cross(axis)
		var fraction := float(i) / float(centerline.size() - 1)
		var half_width := lerpf(0.15, 0.015, fraction)
		var half_height := lerpf(0.15, 0.025, fraction) + 0.29 * sin(PI * fraction)
		for j in range(RING_SIDES):
			var angle := TAU * float(j) / float(RING_SIDES)
			var lateral := across * cos(angle)
			var vertical := up * sin(angle)
			rings.append(centerline[i] + lateral * half_width + vertical * half_height)
			normals.append((lateral / half_width + vertical / half_height).normalized())

	tail.clear_surfaces()
	tail.surface_begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in range(centerline.size() - 1):
		for j in range(RING_SIDES):
			var a := i * RING_SIDES + j
			var b := i * RING_SIDES + (j + 1) % RING_SIDES
			var c := a + RING_SIDES
			var d := b + RING_SIDES
			for index in [a, b, c, b, d, c]:
				tail.surface_set_normal(normals[index])
				tail.surface_add_vertex(rings[index])
	tail.surface_end()

extends Node3D
## Presents frozen native physics and recorded flexible-tail trajectories.

const METRES_PER_UNIT := 0.01
const SPINE_POINTS := 7
const RING_SIDES := 12
const SAMPLES_PER_SPAN := 3
const TRACE_ARG := "--flex-tail-trace="
const LIVE_CLASS := &"FlexibleTailLive"
const LIVE_DATA := "res://live_data/"
const LIVE_CONTROL_S := 0.02
const MAX_CONTROLS_PER_FRAME := 5
const COURSE_OFFSET := Vector3(0, 0.7, 0)
const ZOOM_STEP := 1.16
const MIN_FOLLOW_SCALE := 0.65
const MAX_FOLLOW_SCALE := 3.0
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
@onready var title: Label = %Title
@onready var metrics: Label = %Metrics
@onready var source: Label = %Source
@onready var review_mode: Button = %ReviewMode
@onready var live_mode: Button = %LiveMode
@onready var live_play: Button = %LivePlay
@onready var turn_left: Button = %Left
@onready var turn_right: Button = %Right
@onready var live_status: Label = %LiveStatus
@onready var live_controls: HBoxContainer = %LiveControls
@onready var playback_controls: VBoxContainer = %Playback
@onready var open_trace: Button = %Open
@onready var message: PanelContainer = %Message
@onready var status: Label = %Status
@onready var transport: PanelContainer = %Transport
@onready var play: Button = %Play
@onready var timeline: HSlider = %Timeline
@onready var clock: Label = %Time
@onready var trace_dialog: FileDialog = %TraceDialog
@onready var pond_view: Button = %PondView
@onready var view_controls: VBoxContainer = %ViewControls
@onready var zoom_in: Button = %ZoomIn
@onready var zoom_out: Button = %ZoomOut

var frames: Array[TraceFrame] = []
var phases: Array[TracePhase] = []
var anchor := Vector3.INF
var elapsed := 0.0
var frame_index := 0
var playing := false
var reviewing := false
var live_motor: Object
var live_snapshot: Dictionary = {}
var live_playing := false
var live_accumulator := 0.0
var showing_pond := false
var follow_scale := 1.0
var follow_center := Vector3.ZERO
var habitat := preload("res://pond_landscape.gd").new()
var head := Node3D.new()
var tail_view := MeshInstance3D.new()
var tail := ImmediateMesh.new()
var course_view := MeshInstance3D.new()
var anchor_view := MeshInstance3D.new()
var creature_marker := MeshInstance3D.new()

func _ready() -> void:
	get_window().title = "Tadpole Pond | Flexible Tail"
	get_viewport().msaa_3d = Viewport.MSAA_4X
	_build_world()
	for button in [open_trace, pond_view, zoom_in, zoom_out, play, live_play]:
		_style_button(button)
	for button in [live_mode, review_mode, turn_left, turn_right, %Swim, %Turn, %Hold]:
		_style_button(button, true)
	_style_mode_tabs()
	_style_timeline()
	open_trace.pressed.connect(func(): trace_dialog.popup_centered_clamped())
	live_mode.pressed.connect(_show_live)
	review_mode.pressed.connect(_show_recordings)
	live_play.pressed.connect(_toggle_live_play)
	turn_left.pressed.connect(_request_live_turn.bind(1))
	turn_right.pressed.connect(_request_live_turn.bind(-1))
	transport.minimum_size_changed.connect(_fit_dock_height.call_deferred)
	pond_view.pressed.connect(_toggle_pond_view)
	zoom_in.pressed.connect(_zoom_by.bind(1.0 / ZOOM_STEP))
	zoom_out.pressed.connect(_zoom_by.bind(ZOOM_STEP))
	trace_dialog.file_selected.connect(_open_trace)
	play.pressed.connect(_toggle_play)
	timeline.value_changed.connect(_seek)
	%Swim.pressed.connect(_open_recording.bind("swim"))
	%Turn.pressed.connect(_open_recording.bind("turn"))
	%Hold.pressed.connect(_open_recording.bind("hold"))
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with(TRACE_ARG):
			_set_reviewing(true)
			_open_trace(arg.trim_prefix(TRACE_ARG))
			return
	_show_live()

func _set_reviewing(value: bool) -> void:
	reviewing = value
	live_mode.set_pressed_no_signal(not reviewing)
	review_mode.set_pressed_no_signal(reviewing)
	live_controls.visible = not reviewing
	live_status.visible = not reviewing
	playback_controls.visible = reviewing
	open_trace.visible = reviewing
	source.text = "Recorded physics" if reviewing else "Live · Frozen policy"
	_fit_camera()

func _show_live() -> void:
	if not reviewing and not live_snapshot.is_empty():
		return
	playing = false
	_set_reviewing(false)
	if live_motor == null and not _load_live_motor():
		_update_live_controls()
		return
	message.hide()
	_render_live_snapshot()
	_update_live_controls()

func _show_recordings() -> void:
	if reviewing:
		return
	_set_live_playing(false)
	_set_reviewing(true)
	if frames.is_empty():
		_open_recording("swim")
	else:
		message.hide()
		play.text = "Replay" if elapsed >= timeline.max_value else "Play"
		_set_subject_visible(true)
		_render_frame()

func _load_live_motor() -> bool:
	if not ClassDB.class_exists(LIVE_CLASS):
		_show_live_problem("Live simulation is unavailable. You can still watch the recordings.")
		return false
	live_motor = ClassDB.instantiate(LIVE_CLASS)
	if live_motor == null:
		_show_live_problem("The live simulator could not start.")
		return false
	var loaded := bool(live_motor.call("load_motor",
		ProjectSettings.globalize_path(LIVE_DATA + "fitted-gait.bin"),
		ProjectSettings.globalize_path(LIVE_DATA + "yaw-offset-update-8.bin"),
		ProjectSettings.globalize_path(LIVE_DATA + "straight-evaluation-starts.bin")))
	if not loaded:
		var detail := String(live_motor.call("get_error"))
		live_motor = null
		push_warning(detail)
		_show_live_problem("The saved live policy could not load. You can still watch the recordings.")
		return false
	live_snapshot = live_motor.call("snapshot")
	return true

func _show_live_problem(text: String) -> void:
	metrics.hide()
	_set_subject_visible(false)
	source.text = "Live unavailable"
	status.text = text
	message.show()

func _toggle_live_play() -> void:
	if live_play.disabled:
		return
	_set_live_playing(not live_playing)
	_update_live_controls()

func _set_live_playing(value: bool) -> void:
	live_playing = value
	live_accumulator = 0.0

func _request_live_turn(direction: int) -> void:
	var button := turn_left if direction > 0 else turn_right
	if reviewing or button.disabled:
		return
	if bool(live_motor.call("request_turn", direction)):
		live_snapshot = live_motor.call("snapshot")
		_update_live_controls()

func _update_live_controls() -> void:
	var ready := live_motor != null and not live_snapshot.is_empty()
	var fault := int(live_snapshot.get("fault", 0))
	var queued := int(live_snapshot.get("queued_direction", 0))
	live_play.disabled = not ready or fault != 0
	turn_left.disabled = live_play.disabled or queued != 0
	turn_right.disabled = turn_left.disabled
	live_play.text = "Pause" if live_playing else "Play"
	live_status.text = "Paused"
	if live_playing:
		match String(live_snapshot.get("motion", "swim")):
			"turn_left": live_status.text = "Turning left"
			"turn_right": live_status.text = "Turning right"
			_: live_status.text = "Swimming"
	if not ready:
		live_status.text = "Live unavailable"
	elif fault != 0:
		var error := String(live_motor.call("get_error"))
		live_status.text = error if not error.is_empty() else "Simulation stopped"
	elif queued != 0:
		var cue := "Left turn queued" if queued > 0 else "Right turn queued"
		live_status.text = cue if live_playing else cue + " · Paused"

func _advance_live(delta: float) -> void:
	live_accumulator = minf(live_accumulator + delta, LIVE_CONTROL_S * MAX_CONTROLS_PER_FRAME)
	if live_accumulator < LIVE_CONTROL_S:
		return
	while live_accumulator >= LIVE_CONTROL_S:
		live_accumulator -= LIVE_CONTROL_S
		if not bool(live_motor.call("step_control")):
			_set_live_playing(false)
			break
	live_snapshot = live_motor.call("snapshot")
	_render_live_snapshot()
	_update_live_controls()

func _render_live_snapshot() -> void:
	if live_snapshot.is_empty():
		return
	var head_m: Vector3 = live_snapshot["body_position_m"]
	var orientation: Quaternion = live_snapshot["body_quaternion"]
	var spine_m: PackedVector3Array = live_snapshot["spine_m"]
	if spine_m.size() != SPINE_POINTS or not head_m.is_finite() or not orientation.is_finite():
		return
	var spine := PackedVector3Array()
	for point in spine_m:
		if not point.is_finite():
			return
		spine.append(point / METRES_PER_UNIT)
	var com_m: Vector3 = live_snapshot["com_m"]
	var velocity_m_s: Vector3 = live_snapshot["com_velocity_m_s"]
	_set_subject_visible(true)
	_render_subject(head_m / METRES_PER_UNIT, orientation, spine, com_m / METRES_PER_UNIT)
	var horizontal_speed := Vector2(velocity_m_s.x, velocity_m_s.z).length()
	metrics.text = "Speed %.1f mm/s" % (horizontal_speed * 1000.0)
	metrics.show()

func _build_world() -> void:
	var world := WorldEnvironment.new()
	world.environment = Environment.new()
	world.environment.background_mode = Environment.BG_COLOR
	world.environment.background_color = Color("5c7043")
	world.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	world.environment.ambient_light_color = Color("d8f2e5")
	world.environment.ambient_light_energy = 0.66
	add_child(world)
	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-54, -34, 0)
	light.light_color = Color("fff5d8")
	light.light_energy = 0.95
	add_child(light)

	add_child(habitat)
	if not habitat.build(METRES_PER_UNIT):
		pond_view.disabled = true
		world.environment.background_color = Color("204843")
		push_warning("The pond geometry could not be loaded.")

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
	var locator_material := anchor_material.duplicate() as StandardMaterial3D
	locator_material.albedo_color = Color("fff1b4")
	creature_marker.material_override = locator_material
	creature_marker.mesh = _anchor_mesh()
	creature_marker.scale = Vector3.ONE * 11.5
	add_child(creature_marker)
	_set_subject_visible(false)
	_place_camera(Vector3.ZERO)
	get_viewport().size_changed.connect(_fit_camera)
	_fit_camera()

func _style_button(button: Button, secondary: bool = false) -> void:
	for state in ["normal", "hover", "pressed", "focus", "disabled"]:
		var surface := StyleBoxFlat.new()
		surface.set_corner_radius_all(9)
		surface.content_margin_left = 10.0
		surface.content_margin_right = 10.0
		match state:
			"normal": surface.bg_color = Color("e5ece2" if secondary else "285e50")
			"hover": surface.bg_color = Color("d4e4da" if secondary else "326d5d")
			"pressed": surface.bg_color = Color("214d42")
			"disabled": surface.bg_color = Color("e8ebe2")
			"focus":
				surface.draw_center = false
				surface.set_border_width_all(2)
				surface.border_color = Color("e0ffe9") if view_controls.is_ancestor_of(button) else Color("173d3d")
				surface.set_expand_margin_all(2)
		button.add_theme_stylebox_override(state, surface)
	var text_color := Color("173d3d" if secondary else "f9fff9")
	button.add_theme_color_override("font_color", text_color)
	button.add_theme_color_override("font_hover_color", text_color)
	button.add_theme_color_override("font_pressed_color", Color("f9fff9"))
	button.add_theme_color_override("font_hover_pressed_color", Color("f9fff9"))
	button.add_theme_color_override("font_disabled_color", Color("59685e"))

func _style_mode_tabs() -> void:
	var selected := StyleBoxFlat.new()
	selected.set_corner_radius_all(9)
	selected.content_margin_left = 10.0
	selected.content_margin_right = 10.0
	selected.bg_color = Color("d2e5d8")
	selected.set_border_width_all(1)
	selected.border_color = Color("7b9f8d")
	for button in [live_mode, review_mode]:
		button.add_theme_color_override("font_pressed_color", Color("173d3d"))
		button.add_theme_color_override("font_hover_pressed_color", Color("173d3d"))
		button.add_theme_stylebox_override("pressed", selected)

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
	course_view.visible = visible and reviewing
	anchor_view.visible = visible and reviewing and anchor.is_finite()
	creature_marker.visible = visible and showing_pond

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
	if not reviewing:
		if live_playing:
			_advance_live(delta)
		return
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
	var spine := PackedVector3Array()
	for i in range(SPINE_POINTS):
		spine.append(first.spine[i].lerp(next.spine[i], weight))
	_render_subject(first.body.lerp(next.body, weight),
		first.rotation.slerp(next.rotation, weight).normalized(), spine,
		first.course_position.lerp(next.course_position, weight))
	clock.text = "%.1f / %.1f s" % [elapsed, timeline.max_value]
	_render_metrics(time, first, next, weight)

func _render_subject(position: Vector3, orientation: Quaternion, spine: PackedVector3Array, com: Vector3) -> void:
	head.position = position
	head.quaternion = orientation
	_draw_tail(spine, orientation)
	follow_center = head.position.lerp(spine[-1], 0.43)
	creature_marker.position = com
	_place_camera(follow_center)

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
	if showing_pond:
		center = Vector3.ZERO
		var radius := habitat.water_radii_m.x / METRES_PER_UNIT
		camera.position = center + Vector3(0.0, radius * 1.37, radius * 1.02) * 1.65
	else:
		camera.position = center + Vector3(0.85, 11.1, 7.1)
	camera.look_at(center)

func _toggle_pond_view() -> void:
	showing_pond = not showing_pond
	pond_view.text = "Zoom to tadpole" if showing_pond else "Pond view"
	creature_marker.visible = head.visible and showing_pond
	_place_camera(follow_center)
	_fit_camera()

func _zoom_by(factor: float) -> void:
	if not is_finite(factor) or factor <= 0.0:
		return
	if showing_pond:
		if factor < 1.0:
			_toggle_pond_view()
		return
	follow_scale = clampf(follow_scale * factor, MIN_FOLLOW_SCALE, MAX_FOLLOW_SCALE)
	_fit_camera()

func _unhandled_key_input(event: InputEvent) -> void:
	if trace_dialog.visible or not event is InputEventKey or not event.pressed or event.echo:
		return
	if event.keycode in [KEY_PLUS, KEY_EQUAL, KEY_KP_ADD]:
		_zoom_by(1.0 / ZOOM_STEP)
	elif event.keycode in [KEY_MINUS, KEY_KP_SUBTRACT]:
		_zoom_by(ZOOM_STEP)
	elif not reviewing and event.keycode == KEY_P:
		_toggle_live_play()
	elif not reviewing and event.keycode == KEY_Q:
		_request_live_turn(1)
	elif not reviewing and event.keycode == KEY_E:
		_request_live_turn(-1)
	else:
		return
	get_viewport().set_input_as_handled()

func _unhandled_input(event: InputEvent) -> void:
	if trace_dialog.visible:
		return
	var factor := 0.0
	if event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			factor = 1.0 / ZOOM_STEP
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			factor = ZOOM_STEP
	elif event is InputEventMagnifyGesture and event.factor > 0.0:
		factor = 1.0 / event.factor
	if factor == 0.0:
		return
	for control in [view_controls, transport, header]:
		if control.get_global_rect().has_point(event.position):
			return
	_zoom_by(factor)
	get_viewport().set_input_as_handled()

func _fit_camera() -> void:
	var window := get_viewport().get_visible_rect().size
	var aspect := maxf(window.x, 1.0) / maxf(window.y, 1.0)
	if showing_pond:
		var radii := habitat.water_radii_m / METRES_PER_UNIT
		camera.size = maxf(radii.x * 2.42 / aspect, (radii.y * .87 + 8.0) * 2.65)
	else:
		# Leave room for the full tail at the closest zoom on narrow windows.
		camera.size = maxf(9.8, 11.8 / aspect) * follow_scale
	zoom_in.disabled = not showing_pond and follow_scale <= MIN_FOLLOW_SCALE
	zoom_out.disabled = showing_pond or follow_scale >= MAX_FOLLOW_SCALE
	title.add_theme_font_size_override("font_size", 20 if window.x <= 420.0 else 22)
	header.offset_right = minf(252.0, window.x - view_controls.get_combined_minimum_size().x - 24.0)
	var dock_half_width := minf(280.0 if reviewing else 230.0, window.x / 2.0 - 16.0)
	transport.offset_left = -dock_half_width
	transport.offset_right = dock_half_width
	_fit_dock_height.call_deferred()

func _fit_dock_height() -> void:
	transport.offset_top = transport.offset_bottom - transport.get_combined_minimum_size().y

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

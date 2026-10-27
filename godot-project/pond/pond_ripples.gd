extends MultiMeshInstance3D

# Presentation only: observed positions/joints -> bounded surface disturbances.
# This component never writes the animal, food, controller or training state.
const CAPACITY := 32
const LIFETIME := 1.35
var enabled := true
var water_radii := Vector2(4.6, 3.5)
var records: Array[Dictionary] = []
var emitted := 0
var wake_emitted := 0
var tail_emitted := 0
var cursor := 0
var previous_time := -1.0
var previous_head := Vector3.ZERO
var previous_tip := Vector3.ZERO
var previous_joints := Vector2.ZERO
var head_travel := 0.0
var tail_travel := 0.0
var last_wake := -10.0
var last_tail := -10.0
var material: ShaderMaterial

func _ready() -> void:
	var plane := PlaneMesh.new()
	plane.size = Vector2(2, 2)
	multimesh = MultiMesh.new()
	multimesh.transform_format = MultiMesh.TRANSFORM_3D
	multimesh.use_custom_data = true
	multimesh.mesh = plane
	multimesh.instance_count = CAPACITY
	multimesh.custom_aabb = AABB(Vector3(-6, -0.1, -5), Vector3(12, 0.2, 10))
	material = ShaderMaterial.new()
	material.shader = preload("res://pond/ripples.gdshader")
	material.set_shader_parameter("pond_radii", water_radii)
	material.set_shader_parameter("lifetime", LIFETIME)
	material.render_priority = 1
	material_override = material
	cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	clear()
	visible = enabled

func clear() -> void:
	records.resize(CAPACITY)
	for i in CAPACITY:
		records[i] = {}
		if multimesh != null:
			multimesh.set_instance_transform(i, Transform3D.IDENTITY)
			multimesh.set_instance_custom_data(i, Color(-10, 0, 0, 0))
	cursor = 0
	emitted = 0
	wake_emitted = 0
	tail_emitted = 0
	previous_time = -1
	head_travel = 0
	tail_travel = 0
	last_wake = -10
	last_tail = -10

func set_enabled(value: bool) -> void:
	enabled = value
	visible = value
	clear()

func observe(head: Vector3, tip: Vector3, joints: Vector2, heading: float, seconds: float) -> void:
	if not enabled:
		return
	var dt := seconds - previous_time
	if previous_time < 0 or dt <= 0 or dt > 0.2 or head.distance_to(previous_head) > 1.0:
		if previous_time >= 0:
			clear()
		previous_head = head
		previous_tip = tip
		previous_joints = joints
		previous_time = seconds
		return
	var distance := head.distance_to(previous_head)
	var tip_distance := tip.distance_to(previous_tip)
	var speed := distance / dt
	var joint_speed := (joints - previous_joints).length() / dt
	head_travel += distance
	# Joint activity distinguishes a stroke from the tail simply coasting with
	# the body. Actual tip displacement supplies its visual intensity.
	if joint_speed > 0.35:
		tail_travel += tip_distance
	if speed > 0.03 and head_travel >= 0.10 and seconds - last_wake >= 0.16:
		var forward := Vector3(cos(heading), 0, sin(heading))
		_emit(head - forward * 0.10, heading, seconds, clampf(speed * 0.55, 0.08, 0.55), false)
		head_travel = 0
		last_wake = seconds
	if joint_speed > 0.35 and tail_travel >= 0.10 and seconds - last_tail >= 0.14:
		_emit(tip, heading, seconds, clampf(tip_distance / dt * 0.18, 0.08, 0.65), true)
		tail_travel = 0
		last_tail = seconds
	previous_head = head
	previous_tip = tip
	previous_joints = joints
	previous_time = seconds

func _emit(point: Vector3, heading: float, seconds: float, strength: float, is_tail: bool) -> void:
	var normalized := Vector2(point.x, point.z) / water_radii
	if normalized.length_squared() >= 1.0:
		return
	var radius := 0.65 if is_tail else 0.90
	var kind := 1.0 if is_tail else 0.0
	var transform := Transform3D(Basis(Vector3.UP, -heading), Vector3(point.x, 0, point.z))
	multimesh.set_instance_transform(cursor, transform)
	multimesh.set_instance_custom_data(cursor, Color(seconds, strength, radius, kind))
	records[cursor] = {"position": Vector2(point.x, point.z), "born": seconds, "strength": strength, "tail": is_tail}
	cursor = (cursor + 1) % CAPACITY
	emitted += 1
	if is_tail:
		tail_emitted += 1
	else:
		wake_emitted += 1

func render_at(seconds: float) -> void:
	if material != null and enabled:
		material.set_shader_parameter("ripple_clock", seconds)

func active_count(seconds: float) -> int:
	var count := 0
	for record in records:
		if not record.is_empty() and seconds >= float(record.born) and seconds - float(record.born) < LIFETIME:
			count += 1
	return count

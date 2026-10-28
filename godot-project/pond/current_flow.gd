extends Node3D

const METRES_PER_UNIT := 0.04
const COUNT := 96
var current_m_s := Vector3.ZERO
var displacement := Vector3.ZERO
var strength := 0.0
var target_strength := 0.0
var surface: ShaderMaterial
var tracer_material: ShaderMaterial
var tracers: MultiMeshInstance3D

func configure(water_material: ShaderMaterial) -> void:
	surface = water_material
	tracer_material = ShaderMaterial.new()
	tracer_material.shader = preload("res://pond/current_tracers.gdshader")
	tracer_material.render_priority = 2
	var particle := SphereMesh.new()
	particle.radius = 0.012
	particle.height = 0.024
	particle.radial_segments = 8
	particle.rings = 4
	var pool := MultiMesh.new()
	pool.transform_format = MultiMesh.TRANSFORM_3D
	pool.mesh = particle
	pool.instance_count = COUNT
	var random := RandomNumberGenerator.new()
	random.seed = 0xC0DECAFE
	for index in COUNT:
		var position := Vector3(random.randf_range(-4.6, 4.6), random.randf_range(-3.1, -0.15), random.randf_range(-3.5, 3.5))
		pool.set_instance_transform(index, Transform3D(Basis.from_scale(Vector3(2.2, 0.7, 0.7)), position))
	tracers = MultiMeshInstance3D.new()
	tracers.multimesh = pool
	tracers.material_override = tracer_material
	tracers.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(tracers)
	_render()

func set_current(value: Vector3) -> void:
	current_m_s = value
	target_strength = clampf(value.length() / 0.005, 0, 1)
	if value.length_squared() > 0:
		surface.set_shader_parameter("flow_direction", Vector2(value.x, value.z).normalized())

func advance(simulated_seconds: float) -> void:
	# The phase follows actual physical time, not a free-running shader TIME.
	displacement += current_m_s * maxf(0, simulated_seconds) / METRES_PER_UNIT
	_render()

func _process(delta: float) -> void:
	strength = lerpf(strength, target_strength, 1.0 - exp(-delta / 0.16))
	_render()

func _render() -> void:
	if surface == null:
		return
	surface.set_shader_parameter("flow_offset", Vector2(displacement.x, displacement.z))
	surface.set_shader_parameter("flow_strength", strength)
	tracer_material.set_shader_parameter("flow_offset", displacement)
	tracer_material.set_shader_parameter("flow_strength", strength)

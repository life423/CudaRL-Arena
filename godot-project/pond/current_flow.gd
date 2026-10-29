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
var habitat_radii := Vector2(4.6, 3.5)
var native_advection := Callable()
var native_sampling := Callable()
var tracer_positions := PackedVector3Array()

func configure_habitat(radii: Vector2, advection: Callable, sampling: Callable, geometry: Dictionary, terrain_sampling := Callable()) -> void:
	habitat_radii = radii
	native_advection = advection
	native_sampling = sampling
	if terrain_sampling.is_valid():
		_configure_cove_tracers(geometry, terrain_sampling)
		return
	# This shader reads each native-advected instance directly. The original
	# uniform wrap shader remains unchanged for existing pond demonstrations.
	var native_shader := Shader.new()
	native_shader.code = """shader_type spatial;
render_mode unshaded, cull_disabled, depth_draw_never, world_vertex_coords;
uniform float flow_strength = 0.0;
uniform vec2 habitat_radii = vec2(4.6, 3.5);
varying vec3 point;
void vertex() { point = VERTEX; }
void fragment() {
    float radius = length(point.xz / habitat_radii);
    float bed = -4.0 + 4.5 * pow(min(radius, 1.0), 6.0);
    float wet = (1.0 - smoothstep(0.78, 0.96, radius)) * step(bed + 0.025, point.y) * step(point.y, -0.025);
    ALBEDO = vec3(0.66, 0.83, 0.74);
    ALPHA = 0.48 * flow_strength * wet;
}
"""
	tracer_material.shader = native_shader
	tracer_material.set_shader_parameter("habitat_radii", radii)
	var random := RandomNumberGenerator.new()
	random.seed = 0xC0DECAFE
	tracer_positions.resize(COUNT)
	for index in COUNT:
		var position: Vector3 = geometry.get("center", Vector3(0, -2.25, 0))
		for attempt in 128:
			var angle := random.randf() * TAU
			var radius := sqrt(random.randf()) * 0.86
			var x := cos(angle) * radii.x * radius
			var z := sin(angle) * radii.y * radius
			var bed := -4.0 + 4.5 * pow(radius, 6.0)
			var candidate := Vector3(x, random.randf_range(bed + 0.12, -0.15), z)
			var local := Basis(Vector3.UP, -float(geometry.get("yaw", 0.0))) * (candidate - (geometry.get("center", Vector3(0, -2.25, 0)) as Vector3))
			var ring := Vector2(local.x, local.y).length() - float(geometry.get("major_radius", 1.35))
			var distance := Vector2(ring, local.z).length() - float(geometry.get("tube_radius", 0.5))
			if distance > 0.05:
				position = candidate
				break
		tracer_positions[index] = position
		tracers.multimesh.set_instance_transform(index, Transform3D(Basis.from_scale(Vector3(2.2, 0.7, 0.7)), position))
	_render()

func _configure_cove_tracers(geometry: Dictionary, terrain_sampling: Callable) -> void:
	# Every instance starts in native-sampled water and is advected by that
	# native habitat. There is no second circular shader mask hiding the cove.
	var shader := Shader.new()
	shader.code = """shader_type spatial;
render_mode unshaded, cull_disabled, depth_draw_never, world_vertex_coords;
uniform float flow_strength = 0.0;
void fragment() {
    ALBEDO = vec3(0.66, 0.83, 0.74);
    ALPHA = 0.48 * flow_strength;
}
"""
	tracer_material.shader = shader
	var random := RandomNumberGenerator.new()
	random.seed = 0xC0DECAFE
	var minimum: Vector2 = geometry.get("bounds_min", Vector2(-6.9, -5.25))
	var maximum: Vector2 = geometry.get("bounds_max", Vector2(9, 5.25))
	var candidates := PackedVector3Array()
	for index in COUNT * 6:
		candidates.append(Vector3(random.randf_range(minimum.x, maximum.x), 0, random.randf_range(minimum.y, maximum.y)))
	var samples: PackedVector3Array = terrain_sampling.call(candidates)
	tracer_positions.clear()
	for index in candidates.size():
		if samples[index].x >= -0.3:
			continue
		var candidate := candidates[index]
		candidate.y = random.randf_range(samples[index].x + 0.12, -0.15)
		var local := Basis(Vector3.UP, -float(geometry.get("yaw", 0.0))) * (candidate - (geometry.get("center", Vector3(0, -2.25, 0)) as Vector3))
		var ring := Vector2(local.x, local.y).length() - float(geometry.get("major_radius", 1.35))
		if Vector2(ring, local.z).length() - float(geometry.get("tube_radius", 0.5)) <= 0.05:
			continue
		tracer_positions.append(candidate)
		if tracer_positions.size() == COUNT:
			break
	while tracer_positions.size() < COUNT:
		tracer_positions.append(geometry.get("center", Vector3(0, -2.25, 0)))
	for index in COUNT:
		tracers.multimesh.set_instance_transform(index, Transform3D(Basis.from_scale(Vector3(2.2, 0.7, 0.7)), tracer_positions[index]))
	_render()

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
	var seconds := maxf(0, simulated_seconds)
	displacement += current_m_s * seconds / METRES_PER_UNIT
	if seconds > 0 and native_advection.is_valid() and native_sampling.is_valid():
		var remaining := seconds
		while remaining > 0.000001:
			var interval := minf(remaining, 0.05)
			var advanced: PackedVector3Array = native_advection.call(tracer_positions, interval)
			if advanced.size() != COUNT:
				break
			tracer_positions = advanced
			remaining -= interval
		if tracer_positions.size() == COUNT:
			var velocities: PackedVector3Array = native_sampling.call(tracer_positions)
			for index in COUNT:
				var basis := Basis.from_scale(Vector3(2.2, 0.7, 0.7))
				if velocities.size() == COUNT and velocities[index].length_squared() > 0.0000000001:
					var direction := velocities[index].normalized()
					var up := Vector3.UP if absf(direction.dot(Vector3.UP)) < 0.9 else Vector3.FORWARD
					var side := direction.cross(up).normalized()
					basis = Basis(direction * 2.2, side.cross(direction) * 0.7, side * 0.7)
				tracers.multimesh.set_instance_transform(index, Transform3D(basis, tracer_positions[index]))
	_render()

func _process(delta: float) -> void:
	strength = lerpf(strength, target_strength, 1.0 - exp(-delta / 0.16))
	_render()

func _render() -> void:
	if surface == null:
		return
	surface.set_shader_parameter("flow_offset", Vector2(displacement.x, displacement.z))
	surface.set_shader_parameter("flow_strength", strength)
	if not native_advection.is_valid():
		tracer_material.set_shader_parameter("flow_offset", displacement)
	tracer_material.set_shader_parameter("flow_strength", strength)

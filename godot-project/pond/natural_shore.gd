extends Node3D

# Dry-bank decoration only; the native swimming basin and obstacles are unchanged.
const RX := 4.6
const RZ := 3.5
var vegetation: MultiMeshInstance3D
var rock_count := 0
var materials: Array[ShaderMaterial] = []

func _ground_y(x: float, z: float) -> float:
	var radius := sqrt(x * x / (RX * RX) + z * z / (RZ * RZ))
	return 0.5 + smoothstep(1.02, 1.65, radius) * 0.10 * sin(x * 0.7) * cos(z * 0.8)

func _material(color: Color) -> ShaderMaterial:
	var material := ShaderMaterial.new()
	material.shader = preload("res://pond/shore_detail.gdshader")
	material.set_shader_parameter("base_color", color)
	materials.append(material)
	return material

func update_visibility(uv: Vector2, depth: float, radius: float, enabled: bool) -> void:
	for material in materials:
		material.set_shader_parameter("focus_uv", uv)
		material.set_shader_parameter("focus_depth", depth)
		material.set_shader_parameter("focus_radius", radius)
		material.set_shader_parameter("visibility_fade", enabled)

func _rock(rng: RandomNumberGenerator, position_value: Vector3, radius: float) -> void:
	var sphere := SphereMesh.new()
	sphere.radius = radius
	sphere.height = radius * 1.6
	sphere.radial_segments = 12
	sphere.rings = 6
	var arrays := sphere.get_mesh_arrays()
	var points: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
	for i in points.size():
		var p := points[i]
		# Position-based noise preserves shared seam vertices.
		var variation := 1.0 + 0.12 * sin(p.x * 23 + p.y * 17) * cos(p.z * 29 - p.y * 11)
		points[i] *= variation
	arrays[Mesh.ARRAY_VERTEX] = points
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	var stone := MeshInstance3D.new()
	stone.mesh = mesh
	stone.position = position_value
	stone.position.y = _ground_y(position_value.x, position_value.z) + 0.04
	stone.scale = Vector3(rng.randf_range(1.0, 1.8), rng.randf_range(0.6, 0.95), rng.randf_range(0.8, 1.4))
	stone.rotation = Vector3(rng.randf_range(-0.2, 0.2), rng.randf_range(0, TAU), rng.randf_range(-0.15, 0.15))
	stone.material_override = _material(Color("626a50").lerp(Color("969080"), rng.randf()))
	add_child(stone)
	rock_count += 1

func _leaf_mesh() -> ArrayMesh:
	var tool := SurfaceTool.new()
	tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	var vertices := [Vector3(-0.035, 0, 0), Vector3(0.035, 0, 0), Vector3(-0.025, 0.45, 0.055), Vector3(0.025, 0.45, 0.055), Vector3(0, 0.85, 0.24)]
	for index in [0, 2, 1, 1, 2, 3, 2, 4, 3]:
		tool.add_vertex(vertices[index])
	tool.generate_normals()
	return tool.commit()

func _ready() -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = 8271
	# Uneven clusters and open muddy gaps replace the evenly spaced stone rim.
	for cluster in 12:
		var angle := rng.randf_range(0, TAU)
		var r := rng.randf_range(1.01, 1.28)
		var center := Vector3(cos(angle) * RX * r, 0.54, sin(angle) * RZ * r)
		for stone in rng.randi_range(1, 4):
			var p := center + Vector3(rng.randf_range(-0.4, 0.4), 0, rng.randf_range(-0.3, 0.3))
			if p.x * p.x / (RX * RX) + p.z * p.z / (RZ * RZ) < 1.0:
				continue
			_rock(rng, p, rng.randf_range(0.09, 0.24))
	var instances: Array[Transform3D] = []
	var colors: Array[Color] = []
	for cluster in 36:
		var angle := rng.randf_range(0, TAU)
		var r := rng.randf_range(1.01, 1.7)
		var center := Vector3(cos(angle) * RX * r, 0.50, sin(angle) * RZ * r)
		for blade in rng.randi_range(8, 17):
			var p := center + Vector3(rng.randf_range(-0.33, 0.33), 0, rng.randf_range(-0.25, 0.25))
			if p.x * p.x / (RX * RX) + p.z * p.z / (RZ * RZ) < 1.0:
				continue
			var height := rng.randf_range(0.28, 0.85)
			p.y = _ground_y(p.x, p.z) - 0.015
			var basis := Basis(Vector3.UP, rng.randf_range(0, TAU)).scaled(Vector3(rng.randf_range(0.65, 1.1), height, 1))
			instances.append(Transform3D(basis, p))
			colors.append(Color("375523").lerp(Color("86934c"), rng.randf()))
	vegetation = MultiMeshInstance3D.new()
	var multi := MultiMesh.new()
	multi.transform_format = MultiMesh.TRANSFORM_3D
	multi.use_colors = true
	multi.mesh = _leaf_mesh()
	multi.instance_count = instances.size()
	for i in instances.size():
		multi.set_instance_transform(i, instances[i])
		multi.set_instance_color(i, colors[i])
	vegetation.multimesh = multi
	var leaves := _material(Color.WHITE)
	vegetation.material_override = leaves
	vegetation.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(vegetation)

extends RefCounted

# Presentation of geometry supplied by the native habitat. These functions
# neither create sessions nor change physical state or checkpoint contents.
static func outline(geometry: Dictionary) -> PackedVector3Array:
	var result: PackedVector3Array = geometry.get("shore_outline", PackedVector3Array())
	if result.size() > 1 and result[0].is_equal_approx(result[-1]):
		result.resize(result.size() - 1)
	return result

static func terrain(native_outline: PackedVector3Array, sampler: Callable) -> Dictionary:
	var positions := PackedVector3Array()
	var spokes := native_outline.size()
	for ring in 97:
		var radius := float(ring) / 64.0 * 1.08 if ring <= 64 else 1.08 + float(ring - 64) / 32.0 * 10.92
		for spoke in spokes:
			positions.append(native_outline[spoke] * radius)
	var samples: PackedVector3Array = sampler.call(positions)
	if samples.size() != positions.size():
		return {"mesh": ArrayMesh.new(), "positions": positions, "samples": samples}
	var tool := SurfaceTool.new()
	tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	for index in positions.size():
		var p := positions[index]
		var sample := samples[index]
		tool.set_uv(Vector2(p.x, p.z) * 0.05 + Vector2.ONE * 0.5)
		tool.set_normal(Vector3(-sample.y, 1, -sample.z).normalized())
		tool.add_vertex(Vector3(p.x, sample.x, p.z))
	for ring in 96:
		for spoke in spokes:
			var next := (spoke + 1) % spokes
			var a := ring * spokes + spoke
			var b := ring * spokes + next
			var c := (ring + 1) * spokes + spoke
			var d := (ring + 1) * spokes + next
			for index in [a, c, b, b, c, d]:
				tool.add_index(index)
	return {"mesh": tool.commit(), "positions": positions, "samples": samples}

static func water_mesh(native_outline: PackedVector3Array) -> ArrayMesh:
	var polygon := PackedVector2Array()
	for p in native_outline:
		polygon.append(Vector2(p.x, p.z))
	var tool := SurfaceTool.new()
	tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	for index in Geometry2D.triangulate_polygon(polygon):
		var p := native_outline[index]
		tool.set_normal(Vector3.UP)
		tool.set_uv(Vector2(p.x, p.z) * 0.05 + Vector2.ONE * 0.5)
		tool.add_vertex(Vector3(p.x, 0, p.z))
	return tool.commit()

static func stone_material() -> StandardMaterial3D:
	var noise := FastNoiseLite.new()
	noise.seed = 614
	noise.frequency = 0.045
	noise.fractal_octaves = 4
	var colors := Gradient.new()
	colors.colors = PackedColorArray([Color("686d60"), Color("a0a18b"), Color("c2b99a")])
	colors.offsets = PackedFloat32Array([0.0, 0.58, 1.0])
	var texture := NoiseTexture2D.new()
	texture.width = 256
	texture.height = 256
	texture.seamless = true
	texture.noise = noise
	texture.color_ramp = colors
	var normal := NoiseTexture2D.new()
	normal.width = 256
	normal.height = 256
	normal.seamless = true
	normal.noise = noise
	normal.as_normal_map = true
	normal.bump_strength = 1.4
	var material := StandardMaterial3D.new()
	material.albedo_texture = texture
	material.normal_enabled = true
	material.normal_texture = normal
	material.normal_scale = 0.45
	material.roughness = 0.94
	material.metallic_specular = 0.2
	material.uv1_scale = Vector3(3, 2, 1)
	return material

static func solids(world: Node3D, geometry: Dictionary, material: Material) -> Dictionary:
	var result := {}
	var major: float = geometry.get("major_radius", 1.35)
	var tube: float = geometry.get("tube_radius", 0.5)
	var torus := TorusMesh.new()
	torus.inner_radius = major - tube
	torus.outer_radius = major + tube
	torus.rings = 96
	torus.ring_segments = 32
	var arch := MeshInstance3D.new()
	arch.name = "PhysicalRockArch"
	arch.mesh = torus
	arch.material_override = material
	arch.basis = Basis(Vector3.UP, float(geometry.get("yaw", 0.0))) * Basis(Vector3.RIGHT, PI * 0.5)
	arch.position = geometry.get("center", Vector3(0, -2.25, 0))
	world.add_child(arch)
	result.arch = arch
	if geometry.has("boulder_radius"):
		var radius: float = geometry.boulder_radius
		var sphere := SphereMesh.new()
		sphere.radius = radius
		sphere.height = radius * 2
		sphere.radial_segments = 64
		sphere.rings = 32
		var boulder := MeshInstance3D.new()
		boulder.name = "PhysicalBoulder"
		boulder.mesh = sphere
		boulder.position = geometry.get("boulder_center", Vector3.ZERO)
		boulder.material_override = material
		boulder.set_meta("layout_id", 2)
		world.add_child(boulder)
		result.boulder = boulder
	if geometry.has("overhang_parts"):
		result.overhang = preload("res://pond/overhang_rock.gd").build(world, geometry, material)
	return result

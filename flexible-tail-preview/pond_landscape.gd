extends Node3D
## Visual mesh sampled from the same habitat parameters and profile as the native basin.

const SPOKES := 120
const WET_RINGS := 54
const BANK_RINGS := 20
const MEADOW_RINGS := 24
const RINGS := WET_RINGS + BANK_RINGS + MEADOW_RINGS
const MEADOW_START := 1.50
const WORLD_EXTENT := 8.5
const DESCRIPTOR := "res://habitat.json"
const PARAMETERS := ["water_radius_x_m", "water_radius_z_m", "flat_radius_x_m",
	"flat_radius_z_m", "floor_y_m", "reference_scale_m"]

var water_radii_m: Vector2
var flat_radii_m: Vector2
var floor_y_m: float
var reference_scale_m: float
var metres_per_unit: float

func build(scale_m: float) -> bool:
	if not FileAccess.file_exists(DESCRIPTOR):
		return false
	var descriptor: Variant = JSON.parse_string(FileAccess.get_file_as_string(DESCRIPTOR))
	if not descriptor is Dictionary or descriptor.get("schema") != 1:
		return false
	for parameter in PARAMETERS:
		var value: Variant = descriptor.get(parameter)
		if not (value is float or value is int) or not is_finite(value):
			return false
	water_radii_m = Vector2(descriptor.water_radius_x_m, descriptor.water_radius_z_m)
	flat_radii_m = Vector2(descriptor.flat_radius_x_m, descriptor.flat_radius_z_m)
	floor_y_m = descriptor.floor_y_m
	reference_scale_m = descriptor.reference_scale_m
	if water_radii_m.x <= 0.0 or water_radii_m.y <= 0.0 or flat_radii_m.x <= 0.0 or flat_radii_m.y <= 0.0:
		return false
	if flat_radii_m.x >= 0.955 * water_radii_m.x or flat_radii_m.y >= 0.955 * water_radii_m.y:
		return false
	if floor_y_m >= 0.0 or reference_scale_m <= 0.0:
		return false
	metres_per_unit = scale_m

	var basin := MeshInstance3D.new()
	basin.mesh = _basin()
	var bed_material := ShaderMaterial.new()
	bed_material.shader = preload("res://pond_bed.gdshader")
	bed_material.set_shader_parameter("metres_per_unit", metres_per_unit)
	bed_material.set_shader_parameter("floor_y_m", floor_y_m)
	basin.material_override = bed_material
	add_child(basin)

	var surface := MeshInstance3D.new()
	surface.mesh = _water()
	var water_material := ShaderMaterial.new()
	water_material.shader = preload("res://pond_surface.gdshader")
	water_material.set_shader_parameter("water_radii", water_radii_m / metres_per_unit)
	surface.material_override = water_material
	surface.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(surface)
	_build_shore()
	return true

func _edge(angle: float) -> float:
	return 1.0 + .022 * sin(3.0 * angle + .8) + .014 * cos(5.0 * angle - .4) + .009 * sin(7.0 * angle)

func _shore_point(angle: float, radius: float) -> Vector2:
	return Vector2(cos(angle), sin(angle)) * water_radii_m * (_edge(angle) * radius)

func _height_m(point_m: Vector2) -> float:
	var normalized := point_m / water_radii_m
	var angle := normalized.angle()
	var edge := _edge(angle)
	var radius := normalized.length() / edge
	var direction := Vector2(cos(angle), sin(angle))
	var flat_radius := 1.0 / ((direction * water_radii_m / flat_radii_m).length() * edge)
	if radius <= flat_radius:
		return floor_y_m
	if radius <= 1.0:
		var t := (radius - flat_radius) / (1.0 - flat_radius)
		var rise := t * t * t * (t * (6.0 * t - 15.0) + 10.0)
		return floor_y_m * (1.0 - rise)
	var point := point_m / reference_scale_m
	var lip := smoothstep(1.0, 1.075, radius)
	var ripple := smoothstep(1.02, 1.2, radius)
	var meadow := smoothstep(1.32, 2.4, radius)
	var undulation := sin(point.x * .57) * cos(point.y * .68) * .07
	var hills := 1.7 + 1.2 * sin(point.x * .055 + .6) * cos(point.y * .045 - .3) + .55 * sin(point.x * .11 - point.y * .08)
	var west := point + Vector2(36, 24)
	var east := point - Vector2(37, 18)
	hills += 3.8 * exp(-(west.x * west.x / 540.0 + west.y * west.y / 230.0))
	hills += 2.8 * exp(-(east.x * east.x / 400.0 + east.y * east.y / 330.0))
	return reference_scale_m * (.35 * lip + undulation * ripple + meadow * hills)

func _land_point(point_m: Vector2) -> Vector3:
	return Vector3(point_m.x, _height_m(point_m), point_m.y) / metres_per_unit

func _basin() -> ArrayMesh:
	var tool := SurfaceTool.new()
	tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	tool.add_vertex(Vector3(0, floor_y_m / metres_per_unit, 0))
	for ring in range(1, RINGS + 1):
		var radius := _ring_radius(ring)
		for spoke in range(SPOKES):
			var angle := float(spoke) * TAU / SPOKES
			tool.add_vertex(_land_point(_shore_point(angle, radius)))
	for spoke in range(SPOKES):
		var next := (spoke + 1) % SPOKES
		for index: int in [0, 1 + spoke, 1 + next]:
			tool.add_index(index)
	for ring in range(RINGS - 1):
		for spoke in range(SPOKES):
			var a := 1 + ring * SPOKES + spoke
			var b := 1 + ring * SPOKES + (spoke + 1) % SPOKES
			for index: int in [a, a + SPOKES, b, b, a + SPOKES, b + SPOKES]:
				tool.add_index(index)
	tool.generate_normals()
	return tool.commit()

func _ring_radius(ring: int) -> float:
	if ring <= WET_RINGS:
		return float(ring) / WET_RINGS
	var dry_ring := ring - WET_RINGS
	if dry_ring <= BANK_RINGS:
		return lerpf(1.0, MEADOW_START, float(dry_ring) / BANK_RINGS)
	return lerpf(MEADOW_START, WORLD_EXTENT, float(dry_ring - BANK_RINGS) / MEADOW_RINGS)

func _water() -> ArrayMesh:
	var tool := SurfaceTool.new()
	tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	for spoke in range(SPOKES):
		var points := [Vector2.ZERO, _shore_point(float(spoke) * TAU / SPOKES, 1.0),
			_shore_point(float(spoke + 1) * TAU / SPOKES, 1.0)]
		for point: Vector2 in points:
			tool.set_normal(Vector3.UP)
			tool.add_vertex(Vector3(point.x, .00005, point.y) / metres_per_unit)
	return tool.commit()

func _build_shore() -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = 8317
	var rocks := SphereMesh.new()
	rocks.radial_segments = 8
	rocks.rings = 4
	var blades := SurfaceTool.new()
	blades.begin(Mesh.PRIMITIVE_TRIANGLES)
	for point: Vector3 in [Vector3(-.10, 0, 0), Vector3(.10, 0, 0), Vector3(-.07, .65, .10),
		Vector3(.10, 0, 0), Vector3(.07, .65, .10), Vector3(-.07, .65, .10),
		Vector3(-.07, .65, .10), Vector3(.07, .65, .10), Vector3(.25, 1, .35)]:
		blades.add_vertex(point)
	blades.generate_normals()
	var rock_instances := _instances(rocks, 84)
	var reed_instances := _instances(blades.commit(), 624)
	for cluster in range(26):
		var at := _shore_point(cluster * TAU / 26.0 + rng.randf_range(-.18, .18), rng.randf_range(1.035, 1.37))
		for blade in range(24):
			var site := at + Vector2(rng.randf_range(-.029, .029), rng.randf_range(-.027, .027))
			var size := rng.randf_range(.021, .045) / metres_per_unit
			var color := Color("426537").lerp(Color("839558"), rng.randf())
			_add_scenery(reed_instances, cluster * 24 + blade, site, Vector3(size * .65, size, size), rng.randf_range(0, TAU), color, 0.0)
	for cluster in range(28):
		var at := _shore_point(cluster * TAU / 28.0 + rng.randf_range(-.19, .19), rng.randf_range(1.02, 1.44))
		for stone in range(3):
			var site := at + Vector2(rng.randf_range(-.025, .025), rng.randf_range(-.020, .020))
			var size := rng.randf_range(.010, .031) / metres_per_unit
			var color := Color("58605b").lerp(Color("958d74"), rng.randf())
			_add_scenery(rock_instances, cluster * 3 + stone, site, Vector3(size, size * .55, size * .84), rng.randf_range(0, TAU), color, size * .10)

func _instances(mesh: Mesh, count: int) -> MultiMesh:
	var instanced := MultiMesh.new()
	instanced.transform_format = MultiMesh.TRANSFORM_3D
	instanced.use_colors = true
	instanced.mesh = mesh
	instanced.instance_count = count
	var instance := MultiMeshInstance3D.new()
	instance.multimesh = instanced
	var material := StandardMaterial3D.new()
	material.vertex_color_use_as_albedo = true
	material.roughness = .94
	material.cull_mode = BaseMaterial3D.CULL_DISABLED
	instance.material_override = material
	add_child(instance)
	return instanced

func _add_scenery(instances: MultiMesh, index: int, point_m: Vector2, size: Vector3,
		yaw: float, color: Color, raise: float) -> void:
	var at := _land_point(point_m)
	if at.y <= 0.0:
		instances.set_instance_transform(index, Transform3D(Basis.from_scale(Vector3.ZERO), Vector3.ZERO))
		return
	at.y += raise
	var basis := Basis(Vector3.UP, yaw) * Basis.from_scale(size)
	instances.set_instance_transform(index, Transform3D(basis, at))
	instances.set_instance_color(index, color)

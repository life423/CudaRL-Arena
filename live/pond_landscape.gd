extends Node3D
## Presentation of the shared native pond geometry; no movement or contact logic.
var geometry_source: RefCounted
var geometry_descriptor: Dictionary = {}
var geometry_ready := false
var geometry_issue := ""
var native_query_batches := 0
var native_geometry_points := 0
var scale_m := .04
var floor_y := -4.0
var flat_radii := Vector2(.22,.14)/.04
var water_radii := Vector2(.52,.34)/.04
var flat_radii_m := Vector2(.22,.14)
var water_radii_m := Vector2(.52,.34)
var surface: MeshInstance3D
var scenery_count := 0
const SPOKES := 160
const RINGS := 128
const INNER_RINGS := 72
const WORLD_VIEW_RADIUS := 3.25
const NATIVE_GEOMETRY_VERSION := 1
var _height_cache: Dictionary = {}

func _geometry_failure(message: String) -> void:
	if geometry_issue.is_empty():
		geometry_issue=message
		push_error(message)
	geometry_ready=false

func _outline_points(samples: PackedVector2Array) -> PackedVector3Array:
	if geometry_source==null:
		_geometry_failure("The native pond geometry is unavailable.")
		return PackedVector3Array()
	var result: Variant=geometry_source.call("sample_pond_outline",samples)
	native_query_batches+=1
	native_geometry_points+=samples.size()
	if not result is Dictionary or not bool(result.get("valid",false)) or int(result.get("version",0))!=NATIVE_GEOMETRY_VERSION:
		_geometry_failure("The native shoreline query could not be verified.")
		return PackedVector3Array()
	var points: Variant=result.get("positions_m",null)
	if not points is PackedVector3Array or points.size()!=samples.size():
		_geometry_failure("The native shoreline query has an incompatible shape.")
		return PackedVector3Array()
	return points

func _surface_points(positions_m: PackedVector3Array) -> Dictionary:
	if geometry_source==null:
		_geometry_failure("The native pond geometry is unavailable.")
		return {}
	var positions := PackedVector2Array()
	for at in positions_m: positions.append(Vector2(at.x,at.z))
	var result: Variant=geometry_source.call("sample_pond_surface",positions)
	native_query_batches+=1
	native_geometry_points+=positions.size()
	if not result is Dictionary or not bool(result.get("valid",false)) or int(result.get("version",0))!=NATIVE_GEOMETRY_VERSION:
		_geometry_failure("The native pond surface query could not be verified.")
		return {}
	var heights: Variant=result.get("heights_m",null)
	var normals: Variant=result.get("normals",null)
	if not heights is PackedFloat64Array or not normals is PackedVector3Array or heights.size()!=positions.size() or normals.size()!=positions.size():
		_geometry_failure("The native pond surface query has an incompatible shape.")
		return {}
	return result

func fit_points(whole_world := false) -> PackedVector3Array:
	var points := PackedVector3Array([Vector3(0,floor_y,0)])
	if not geometry_ready: return points
	var samples := PackedVector2Array()
	var reed_indices: Array[int] = []
	for index in range(128):
		var angle := TAU*float(index)/128
		var radii := [WORLD_VIEW_RADIUS,1.30,1.0] if whole_world else [1.30,1.0]
		for radius in radii: samples.append(Vector2(angle,radius))
		reed_indices.append(samples.size())
		samples.append(Vector2(angle,1.18))
	var positions_m := _outline_points(samples)
	if positions_m.size()!=samples.size(): return points
	var surface_data := _surface_points(positions_m)
	if surface_data.is_empty(): return points
	var heights: PackedFloat64Array=surface_data.heights_m
	for index in positions_m.size():
		var at: Vector3=positions_m[index]/scale_m
		at.y=1.7 if index in reed_indices else heights[index]/scale_m
		points.append(at)
	return points

func outline(angle: float, radius := 1.0) -> Vector3:
	var points := _outline_points(PackedVector2Array([Vector2(angle,radius)]))
	return points[0]/scale_m if points.size()==1 else Vector3.ZERO

func terrain_height(x: float,z: float) -> float:
	var key := Vector2(x,z)
	if _height_cache.has(key): return float(_height_cache[key])
	var surface_data := _surface_points(PackedVector3Array([Vector3(x*scale_m,0,z*scale_m)]))
	if surface_data.is_empty(): return floor_y
	var heights: PackedFloat64Array=surface_data.heights_m
	var height := heights[0]/scale_m
	_height_cache[key]=height
	return height

func _ready() -> void:
	if geometry_source==null or not geometry_source.has_method("get_pond_geometry_descriptor") or not geometry_source.has_method("sample_pond_outline") or not geometry_source.has_method("sample_pond_surface"):
		_geometry_failure("The app components do not match. Reopen the updated Tadpole Pond app.")
		return
	var descriptor: Variant=geometry_source.call("get_pond_geometry_descriptor")
	if not descriptor is Dictionary or int(descriptor.get("version",0))!=NATIVE_GEOMETRY_VERSION or descriptor.get("units","")!="metres":
		_geometry_failure("The native pond geometry description is incompatible.")
		return
	geometry_descriptor=descriptor
	flat_radii_m=geometry_descriptor.flat_radii_m
	water_radii_m=geometry_descriptor.water_radii_m
	scale_m=float(geometry_descriptor.reference_scale_m)
	floor_y=float(geometry_descriptor.floor_y_m)/scale_m
	flat_radii = flat_radii_m/scale_m
	water_radii = water_radii_m/scale_m
	var terrain_mesh := _terrain()
	if terrain_mesh==null: return
	var water_mesh := _water()
	if water_mesh==null: return
	var land := MeshInstance3D.new()
	land.name = "OrganicPondBasin"
	land.mesh = terrain_mesh
	var basin_material := ShaderMaterial.new()
	basin_material.shader = preload("res://pond_basin.gdshader")
	land.material_override = basin_material
	land.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	land.add_to_group("pond_bed_visual")
	add_child(land)
	surface = MeshInstance3D.new()
	surface.name = "ClearPondSurface"
	surface.mesh = water_mesh
	var water_material := ShaderMaterial.new()
	water_material.shader = preload("res://pond_water.gdshader")
	surface.material_override = water_material
	surface.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	surface.add_to_group("pond_surface_visual")
	add_child(surface)
	_build_shore()
	geometry_ready=geometry_issue.is_empty()

func _terrain() -> ArrayMesh:
	var samples := PackedVector2Array()
	for ring in range(RINGS+1):
		var radius := float(ring)/INNER_RINGS*1.2 if ring<=INNER_RINGS else 1.2+float(ring-INNER_RINGS)/(RINGS-INNER_RINGS)*10.8
		for spoke in range(SPOKES+1):
			samples.append(Vector2(float(spoke)/SPOKES*TAU,radius))
	var positions_m := _outline_points(samples)
	if positions_m.size()!=samples.size(): return null
	var surface_data := _surface_points(positions_m)
	if surface_data.is_empty(): return null
	var heights: PackedFloat64Array=surface_data.heights_m
	var normals: PackedVector3Array=surface_data.normals
	var tool := SurfaceTool.new()
	tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	for index in positions_m.size():
		var at: Vector3=positions_m[index]/scale_m
		at.y=heights[index]/scale_m
		tool.set_normal(normals[index])
		tool.set_uv(Vector2(at.x,at.z)*.05)
		tool.add_vertex(at)
	for ring in range(RINGS):
		for spoke in range(SPOKES):
			var a := ring*(SPOKES+1)+spoke
			for index in [a,a+SPOKES+1,a+1,a+1,a+SPOKES+1,a+SPOKES+2]: tool.add_index(index)
	return tool.commit()

func _water() -> ArrayMesh:
	var samples := PackedVector2Array()
	for spoke in range(SPOKES+1): samples.append(Vector2(float(spoke)/SPOKES*TAU,1))
	var rim_m := _outline_points(samples)
	if rim_m.size()!=samples.size(): return null
	var tool := SurfaceTool.new()
	tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	for spoke in range(SPOKES):
		for p in [Vector3.ZERO,rim_m[spoke+1]/scale_m,rim_m[spoke]/scale_m]:
			tool.set_normal(Vector3.UP)
			tool.add_vertex(p)
	return tool.commit()

func _shore_material(color: Color,vertex_color := false) -> StandardMaterial3D:
	var material := StandardMaterial3D.new()
	material.albedo_color = color
	material.roughness = .95
	material.metallic_specular = .08
	material.vertex_color_use_as_albedo = vertex_color
	material.cull_mode = BaseMaterial3D.CULL_DISABLED
	return material

func _rock(rng: RandomNumberGenerator,at: Vector3,radius: float) -> void:
	var sphere := SphereMesh.new()
	sphere.radius = radius
	sphere.height = radius*1.6
	sphere.radial_segments = 12
	sphere.rings = 6
	var arrays := sphere.get_mesh_arrays()
	var points: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
	for i in range(points.size()):
		var p := points[i]
		points[i] *= 1+.10*sin(p.x*23+p.y*17)*cos(p.z*29-p.y*11)
	arrays[Mesh.ARRAY_VERTEX] = points
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES,arrays)
	var stone := MeshInstance3D.new()
	stone.mesh = mesh
	stone.position = Vector3(at.x,terrain_height(at.x,at.z)+radius*.38,at.z)
	stone.scale = Vector3(rng.randf_range(1,1.5),rng.randf_range(.65,.95),rng.randf_range(.8,1.3))
	stone.rotation = Vector3(rng.randf_range(-.15,.15),rng.randf_range(0,TAU),rng.randf_range(-.12,.12))
	stone.material_override = _shore_material(Color("3f493c").lerp(Color("68715d"),rng.randf()))
	stone.add_to_group("pond_shore_visual")
	add_child(stone)
	scenery_count += 1

func _leaf_mesh() -> ArrayMesh:
	var tool := SurfaceTool.new()
	tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	var points := [Vector3(-.035,0,0),Vector3(.035,0,0),Vector3(-.025,.45,.055),Vector3(.025,.45,.055),Vector3(0,.85,.24)]
	for index in [0,2,1,1,2,3,2,4,3]: tool.add_vertex(points[index])
	tool.generate_normals()
	return tool.commit()

func _build_shore() -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = 8271
	for cluster in range(12):
		var center := outline(rng.randf_range(0,TAU),rng.randf_range(1.06,1.22))
		for stone in range(rng.randi_range(1,3)):
			var p := center+Vector3(rng.randf_range(-.45,.45),0,rng.randf_range(-.35,.35))
			if terrain_height(p.x,p.z)<.13: continue
			_rock(rng,p,rng.randf_range(.18,.43))
	var transforms: Array[Transform3D] = []
	var colors: Array[Color] = []
	for cluster in range(27):
		var center := outline(rng.randf_range(0,TAU),rng.randf_range(1.05,1.28))
		for blade in range(rng.randi_range(8,17)):
			var p := center+Vector3(rng.randf_range(-.42,.42),0,rng.randf_range(-.34,.34))
			p.y = terrain_height(p.x,p.z)-.012
			if p.y<.1: continue
			var basis := Basis(Vector3.UP,rng.randf_range(0,TAU)).scaled(Vector3(rng.randf_range(.7,1.3),rng.randf_range(.65,1.55),1))
			transforms.append(Transform3D(basis,p))
			colors.append(Color("284e39").lerp(Color("526f43"),rng.randf()))
	var vegetation := MultiMeshInstance3D.new()
	vegetation.name = "ShoreReedClusters"
	var multi := MultiMesh.new()
	multi.transform_format = MultiMesh.TRANSFORM_3D
	multi.use_colors = true
	multi.mesh = _leaf_mesh()
	multi.instance_count = transforms.size()
	for i in range(transforms.size()):
		multi.set_instance_transform(i,transforms[i])
		multi.set_instance_color(i,colors[i])
	vegetation.multimesh = multi
	vegetation.material_override = _shore_material(Color.WHITE,true)
	vegetation.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	vegetation.add_to_group("pond_shore_visual")
	add_child(vegetation)
	scenery_count += transforms.size()
	_build_outer_land(rng)

func _build_outer_land(rng: RandomNumberGenerator) -> void:
	# Dry environment only: native movement remains inside the guarded water basin.
	for cluster in range(25):
		var center := outline(rng.randf_range(0,TAU),rng.randf_range(1.65,4.6))
		for stone in range(rng.randi_range(2,4)):
			var p := center+Vector3(rng.randf_range(-1.3,1.3),0,rng.randf_range(-1.1,1.1))
			_rock(rng,p,rng.randf_range(.32,.85))
	var transforms: Array[Transform3D] = []
	var colors: Array[Color] = []
	for cluster in range(100):
		var center := outline(rng.randf_range(0,TAU),rng.randf_range(1.45,4.5))
		for blade in range(rng.randi_range(5,13)):
			var p := center+Vector3(rng.randf_range(-.8,.8),0,rng.randf_range(-.65,.65))
			p.y = terrain_height(p.x,p.z)-.01
			var basis := Basis(Vector3.UP,rng.randf_range(0,TAU)).scaled(Vector3(rng.randf_range(.8,1.5),rng.randf_range(.55,1.25),1))
			transforms.append(Transform3D(basis,p))
			colors.append(Color("3d6141").lerp(Color("6f8250"),rng.randf()))
	var grass := MultiMeshInstance3D.new()
	grass.name = "MeadowGrassClusters"
	var multi := MultiMesh.new()
	multi.transform_format = MultiMesh.TRANSFORM_3D
	multi.use_colors = true
	multi.mesh = _leaf_mesh()
	multi.instance_count = transforms.size()
	for i in range(transforms.size()):
		multi.set_instance_transform(i,transforms[i])
		multi.set_instance_color(i,colors[i])
	grass.multimesh = multi
	grass.material_override = _shore_material(Color.WHITE,true)
	grass.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	grass.add_to_group("pond_dry_world_visual")
	add_child(grass)
	scenery_count += transforms.size()

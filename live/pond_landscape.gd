extends Node3D
## Organic presentation around an unchanged, flat native central basin.
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

func fit_points(whole_world := false) -> PackedVector3Array:
	var points := PackedVector3Array([Vector3(0,floor_y,0)])
	for index in range(128):
		var angle := TAU*float(index)/128
		var radii := [WORLD_VIEW_RADIUS,1.30,1.0] if whole_world else [1.30,1.0]
		for radius in radii:
			var p := outline(angle,radius)
			p.y = terrain_height(p.x,p.z)
			points.append(p)
		var reeds := outline(angle,1.18)
		reeds.y = 1.7
		points.append(reeds)
	return points

func outline(angle: float, radius := 1.0) -> Vector3:
	var edge := 1+.022*sin(angle*3+.8)+.014*cos(angle*5-.4)+.009*sin(angle*7)
	return Vector3(cos(angle)*water_radii.x*radius*edge,0,sin(angle)*water_radii.y*radius*edge)

func terrain_height(x: float,z: float) -> float:
	var angle := atan2(z/water_radii.y,x/water_radii.x)
	var edge := 1+.022*sin(angle*3+.8)+.014*cos(angle*5-.4)+.009*sin(angle*7)
	var radius := Vector2(x/water_radii.x,z/water_radii.y).length()/edge
	var flat_radius := 1.0/(Vector2(cos(angle)*water_radii.x/flat_radii.x,sin(angle)*water_radii.y/flat_radii.y).length()*edge)
	if radius<=flat_radius: return floor_y
	if radius<=1:
		var t := clampf((radius-flat_radius)/(1-flat_radius),0,1)
		var slope := t*t*t*(t*(t*6-15)+10)
		return lerpf(floor_y,0,slope)
	var rise := smoothstep(1.0,1.075,radius)*.35
	var undulation := sin(x*.57)*cos(z*.68)*.07*smoothstep(1.02,1.2,radius)
	var open_land := smoothstep(1.32,2.4,radius)
	var hills := 1.7+1.2*sin(x*.055+.6)*cos(z*.045-.3)+.55*sin(x*.11-z*.08)
	var west_ridge := 3.8*exp(-((x+36)*(x+36)/540+(z+24)*(z+24)/230))
	var east_ridge := 2.8*exp(-((x-37)*(x-37)/400+(z-18)*(z-18)/330))
	return rise+undulation+open_land*(hills+west_ridge+east_ridge)

func _ready() -> void:
	flat_radii = flat_radii_m/scale_m
	water_radii = water_radii_m/scale_m
	var land := MeshInstance3D.new()
	land.name = "OrganicPondBasin"
	land.mesh = _terrain()
	var basin_material := ShaderMaterial.new()
	basin_material.shader = preload("res://pond_basin.gdshader")
	land.material_override = basin_material
	land.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	land.add_to_group("pond_bed_visual")
	add_child(land)
	surface = MeshInstance3D.new()
	surface.name = "ClearPondSurface"
	surface.mesh = _water()
	var water_material := ShaderMaterial.new()
	water_material.shader = preload("res://pond_water.gdshader")
	surface.material_override = water_material
	surface.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	surface.add_to_group("pond_surface_visual")
	add_child(surface)
	_build_shore()

func _terrain() -> ArrayMesh:
	var tool := SurfaceTool.new()
	tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	for ring in range(RINGS+1):
		var radius := float(ring)/INNER_RINGS*1.2 if ring<=INNER_RINGS else 1.2+float(ring-INNER_RINGS)/(RINGS-INNER_RINGS)*10.8
		for spoke in range(SPOKES+1):
			var p := outline(float(spoke)/SPOKES*TAU,radius)
			p.y = terrain_height(p.x,p.z)
			var dx := (terrain_height(p.x+.01,p.z)-terrain_height(p.x-.01,p.z))/.02
			var dz := (terrain_height(p.x,p.z+.01)-terrain_height(p.x,p.z-.01))/.02
			tool.set_normal(Vector3(-dx,1,-dz).normalized())
			tool.set_uv(Vector2(p.x,p.z)*.05)
			tool.add_vertex(p)
	for ring in range(RINGS):
		for spoke in range(SPOKES):
			var a := ring*(SPOKES+1)+spoke
			for index in [a,a+SPOKES+1,a+1,a+1,a+SPOKES+1,a+SPOKES+2]: tool.add_index(index)
	return tool.commit()

func _water() -> ArrayMesh:
	var tool := SurfaceTool.new()
	tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	for spoke in range(SPOKES):
		for p in [Vector3.ZERO,outline(float(spoke+1)/SPOKES*TAU),outline(float(spoke)/SPOKES*TAU)]:
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

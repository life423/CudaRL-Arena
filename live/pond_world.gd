extends Node3D
## Presentation only. Native poses and physical pond bounds remain authoritative.
var geometry: Dictionary = {}
var links: Array[Node3D] = []
var camera: Camera3D
var target: Node3D
var connector: ImmediateMesh
var connector_view: MeshInstance3D
var anchor := Vector3.ZERO
var body_ready := false
var viewport_area := Rect2()
var overview := false
var zoom_factor := 1.0
var half_width := 15.0
var half_depth := 10.0
var floor_y := -4.0
var surface: MeshInstance3D
var environment_settings: Environment
var landscape: Node3D

func material(color: Color) -> StandardMaterial3D:
	var result := StandardMaterial3D.new()
	result.albedo_color = color
	result.roughness = .85
	result.metallic_specular = .12
	return result

func piece(parent: Node3D, mesh: Mesh, color: Color, at := Vector3.ZERO) -> MeshInstance3D:
	var result := MeshInstance3D.new()
	result.mesh = mesh
	result.material_override = material(color)
	result.position = at
	parent.add_child(result)
	return result

func _ready() -> void:
	var scale_m := float(geometry.get("scale_m",.04))
	half_width = float(geometry.get("arena_width_m",1.2))/(2*scale_m)
	half_depth = float(geometry.get("arena_depth_m",.8))/(2*scale_m)
	floor_y = float(geometry.get("floor_y_m",-.16))/scale_m
	_build_light()
	_build_pond()
	_build_body()
	camera = Camera3D.new()
	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	camera.near = .03
	camera.far = 500
	add_child(camera)
	target = Node3D.new()
	add_child(target)
	var ring := TorusMesh.new()
	ring.inner_radius = .045
	ring.outer_radius = .055
	ring.rings = 24
	ring.ring_segments = 8
	piece(target,ring,Color("c7e1ba"))
	for dimensions in [Vector3(.22,.004,.004),Vector3(.004,.004,.22)]:
		var mark := BoxMesh.new()
		mark.size = dimensions
		piece(target,mark,Color("c7e1ba"))
	connector = ImmediateMesh.new()
	connector_view = piece(self,connector,Color("8fae9c"))
	target.visible = false
	connector_view.visible = false
	for i in range(3): links[i].position = Vector3([0,-.575,-1.225][i],-2,0)
	fit_view()

func _build_light() -> void:
	var environment := WorldEnvironment.new()
	environment_settings = Environment.new()
	environment_settings.background_mode = Environment.BG_COLOR
	environment_settings.background_color = Color("f5f3ec")
	environment_settings.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment_settings.ambient_light_color = Color("d5e2cf")
	environment_settings.ambient_light_energy = .7
	environment_settings.fog_enabled = false
	environment.environment = environment_settings
	add_child(environment)
	var sunlight := DirectionalLight3D.new()
	sunlight.rotation_degrees = Vector3(-65,195,0)
	sunlight.light_color = Color("fff0da")
	sunlight.light_energy = .95
	sunlight.shadow_enabled = true
	sunlight.set_param(Light3D.PARAM_SIZE,2.0)
	sunlight.set_param(Light3D.PARAM_SHADOW_OPACITY,.24)
	sunlight.directional_shadow_max_distance = 65
	add_child(sunlight)

func _build_pond() -> void:
	landscape = preload("res://pond_landscape.gd").new()
	landscape.scale_m = float(geometry.get("scale_m",.04))
	landscape.floor_y = floor_y
	landscape.flat_radii_m = geometry.get("pond_flat_radii_m",Vector2(.22,.14))
	landscape.water_radii_m = geometry.get("pond_water_radii_m",Vector2(.52,.34))
	add_child(landscape)
	surface = landscape.surface

func _membrane(length: float, height: float, thickness: float) -> ArrayMesh:
	# Visual corner rounding stays inside the rectangular physical fin envelope.
	var rim := PackedVector2Array()
	var radius := height*.19
	for corner in range(4):
		var center := Vector2((length/2-radius)*(1 if corner in [0,3] else -1),(height/2-radius)*(1 if corner<2 else -1))
		for sample in range(5):
			var angle := float(corner)*PI/2+float(sample)*PI/8
			rim.append(center+Vector2(cos(angle),sin(angle))*radius)
	var vertices := PackedVector3Array()
	var normals := PackedVector3Array()
	for face in [-1,1]:
		for i in range(rim.size()):
			var a := rim[i]
			var b := rim[(i+1)%rim.size()]
			for vertex in [Vector3(0,0,face*thickness/2),Vector3(a.x,a.y,face*thickness/2),Vector3(b.x,b.y,face*thickness/2)]:
				vertices.append(vertex)
				normals.append(Vector3(0,0,face))
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_NORMAL] = normals
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES,arrays)
	return mesh

func _build_body() -> void:
	var scale_m := float(geometry.get("scale_m",.04))
	var dimensions: Array = geometry.get("links",[{"length":.020,"half_height":.0064,"half_width":.008},{"length":.026,"half_height":.0015,"half_width":.0015},{"length":.026,"half_height":.00635,"half_width":.0015}])
	for i in range(3):
		var link := Node3D.new()
		add_child(link)
		links.append(link)
	var head := SphereMesh.new()
	head.radius = 1
	head.height = 2
	head.radial_segments = 48
	head.rings = 24
	var head_piece := piece(links[0],head,Color("67ab93"))
	head_piece.scale = Vector3(float(dimensions[0].length)/2,float(dimensions[0].half_height),float(dimensions[0].half_width))/scale_m
	for i in [1,2]:
		var rod := CylinderMesh.new()
		rod.height = float(dimensions[i].length)/scale_m
		rod.top_radius = float(geometry.get("rod_radius_m",.0015))/scale_m
		rod.bottom_radius = rod.top_radius
		rod.radial_segments = 24
		piece(links[i],rod,Color("5d9e89")).rotation.z = PI/2
	var fin_height := float(geometry.get("fin_extension_m",.0097))/2
	for direction in [-1,1]:
		var fin := _membrane(float(dimensions[2].length)/scale_m,fin_height/scale_m,float(geometry.get("fin_thickness_m",.0001))/scale_m)
		var wing := piece(links[2],fin,Color("78bda0"),Vector3(0,direction*(float(geometry.get("rod_radius_m",.0015))+fin_height/2)/scale_m,0))
		var skin := wing.material_override as StandardMaterial3D
		skin.cull_mode = BaseMaterial3D.CULL_DISABLED
	for direction in [-1,1]:
		var eye := SphereMesh.new()
		eye.radius = .035
		eye.height = .07
		piece(links[0],eye,Color("173536"),Vector3(.15,.055,direction*.18))

func show_state(snapshot: Dictionary) -> void:
	var poses: Array = snapshot.get("links",[])
	body_ready = poses.size()==3
	for i in range(mini(3,poses.size())):
		links[i].position = poses[i].position
		links[i].quaternion = poses[i].orientation
	anchor = snapshot.get("anchor_position",Vector3.ZERO)
	var show_target := body_ready and bool(snapshot.get("holding",true))
	target.visible = show_target
	connector_view.visible = show_target
	if show_target:
		target.position = anchor
		connector.clear_surfaces()
		connector.surface_begin(Mesh.PRIMITIVE_LINES)
		connector.surface_add_vertex(anchor)
		connector.surface_add_vertex(snapshot.get("com_position",links[0].position))
		connector.surface_end()
	fit_view()

func set_overview(enabled: bool) -> void:
	overview = enabled
	zoom_factor = 1.0
	fit_view()

func set_zoom(value: float) -> void:
	if not is_finite(value): return
	zoom_factor = clampf(value,.55,24)
	fit_view()

func zoom_by(factor: float) -> void:
	if factor>0 and is_finite(factor): set_zoom(zoom_factor*factor)

func native_bounds_model() -> AABB:
	return AABB(Vector3(-half_width,floor_y,-half_depth),Vector3(half_width*2,-floor_y,half_depth*2))

func screen_to_pond_goal(screen_position: Vector2) -> Variant:
	# Coordinate projection only. No goal, request or native controller is changed.
	if camera==null or not body_ready or not viewport_area.has_point(screen_position): return null
	var origin := camera.project_ray_origin(screen_position)
	var direction := camera.project_ray_normal(screen_position)
	if absf(direction.y)<.00001: return null
	var distance := (links[0].position.y-origin.y)/direction.y
	if distance<=0: return null
	var point := origin+direction*distance
	var scale_m := float(geometry.get("scale_m",.04))
	var radii: Vector2 = geometry.get("pond_domain_radii_m",Vector2(.20,.12))
	radii = (radii-Vector2(.04,.015))/scale_m
	if Vector2(point.x/radii.x,point.z/radii.y).length_squared()>1: return null
	return point

func fit_view() -> void:
	if camera == null or links.is_empty(): return
	var window_size := get_viewport().get_visible_rect().size
	var area := viewport_area if viewport_area.size.x>0 else Rect2(Vector2.ZERO,window_size)
	var focus := links[0].position+Vector3(-.65,0,0)
	if overview:
		# Aerial creature camera: keep the native body readable, even when the
		# shoreline extends beyond the frame. Center actual link poses on turns.
		if body_ready: focus=(links[0].position+links[2].position)*.5
		camera.position = focus+Vector3(half_width*.2,half_width*1.6,half_width*.8)
		camera.look_at(focus)
		var readable_span := clampf(minf(area.size.x*.65,area.size.y*.85),160,260)
		camera.size = 1.8*window_size.y/readable_span
	else:
		camera.position = focus+Vector3(1.4,8,9)
		camera.look_at(focus)
		var desired_width := maxf(180,minf(area.size.x*.74,850))
		camera.size = maxf(4.7,2.4*window_size.y/desired_width)
	camera.size = minf(camera.size*zoom_factor,125)
	var center := area.get_center()
	camera.h_offset = (window_size.x*.5-center.x)/window_size.y*camera.size
	camera.v_offset = (center.y-window_size.y*.5)/window_size.y*camera.size
	if surface != null: surface.visible = true

func target_caption_position() -> Vector2:
	var point := camera.unproject_position(anchor)
	return Vector2(clampf(point.x-30,viewport_area.position.x,viewport_area.end.x-95),clampf(point.y+.18*get_viewport().get_visible_rect().size.y/camera.size+16,viewport_area.position.y,viewport_area.end.y-30))

func target_caption() -> String:
	var point := camera.unproject_position(anchor)
	if viewport_area.has_point(point): return "Target"
	return "← Target" if point.x<viewport_area.get_center().x else "Target →"

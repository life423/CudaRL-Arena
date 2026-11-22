extends Node3D
## Presentation only. Body transforms come from the one native world owner.
var geometry: Dictionary = {}
var links: Array[Node3D] = []
var camera: Camera3D
var target: Node3D
var connector: ImmediateMesh
var connector_view: MeshInstance3D
var anchor := Vector3.ZERO
var body_ready := false
var viewport_area := Rect2()

func material(color: Color) -> StandardMaterial3D:
	var result := StandardMaterial3D.new()
	result.albedo_color = color
	result.roughness = .72
	result.metallic_specular = .18
	return result

func piece(parent: Node3D, mesh: Mesh, color: Color, at := Vector3.ZERO) -> MeshInstance3D:
	var result := MeshInstance3D.new()
	result.mesh = mesh
	result.material_override = material(color)
	result.position = at
	parent.add_child(result)
	return result

func _ready() -> void:
	var environment := WorldEnvironment.new()
	var settings := Environment.new()
	settings.background_mode = Environment.BG_COLOR
	settings.background_color = Color("15363f")
	settings.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	settings.ambient_light_color = Color("bad8d0")
	settings.ambient_light_energy = .7
	settings.fog_enabled = true
	settings.fog_light_color = Color("254b51")
	settings.fog_density = .018
	environment.environment = settings
	add_child(environment)
	var sunlight := DirectionalLight3D.new()
	sunlight.rotation_degrees = Vector3(-48,-24,0)
	sunlight.light_color = Color("d5ebdd")
	sunlight.light_energy = .85
	add_child(sunlight)
	var bounce := OmniLight3D.new()
	bounce.position = Vector3(-1,.5,3)
	bounce.light_color = Color("78aeb0")
	bounce.light_energy = .35
	bounce.omni_range = 14
	add_child(bounce)
	camera = Camera3D.new()
	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	add_child(camera)
	var floor_mesh := PlaneMesh.new()
	var scale_m := float(geometry.get("scale_m",.04))
	floor_mesh.size = Vector2(float(geometry.get("arena_width_m",1.2)),float(geometry.get("arena_depth_m",.8)))/scale_m
	piece(self,floor_mesh,Color("1d4147"),Vector3(0,float(geometry.get("floor_y_m",-.16))/scale_m,0))
	_build_body()
	target = Node3D.new()
	add_child(target)
	var ring := TorusMesh.new()
	ring.inner_radius = .045
	ring.outer_radius = .055
	ring.rings = 24
	ring.ring_segments = 8
	piece(target,ring,Color("a9ccc3"))
	for dimensions in [Vector3(.22,.004,.004),Vector3(.004,.004,.22)]:
		var mark := BoxMesh.new()
		mark.size = dimensions
		piece(target,mark,Color("a9ccc3"))
	connector = ImmediateMesh.new()
	connector_view = piece(self,connector,Color("71968f"))
	target.visible = false
	connector_view.visible = false
	for i in range(3): links[i].position = Vector3([0,-.575,-1.225][i],-2,0)
	fit_view()

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
	var head_piece := piece(links[0],head,Color("87d1b4"))
	head_piece.scale = Vector3(float(dimensions[0].length)/2,float(dimensions[0].half_height),float(dimensions[0].half_width))/scale_m
	for i in [1,2]:
		var rod := CylinderMesh.new()
		rod.height = float(dimensions[i].length)/scale_m
		rod.top_radius = float(geometry.get("rod_radius_m",.0015))/scale_m
		rod.bottom_radius = rod.top_radius
		piece(links[i],rod,Color("73baa1")).rotation.z = PI/2
	var fin_height := float(geometry.get("fin_extension_m",.0097))/2
	for direction in [-1,1]:
		var fin := BoxMesh.new()
		fin.size = Vector3(float(dimensions[2].length),fin_height,float(geometry.get("fin_thickness_m",.0001)))/scale_m
		piece(links[2],fin,Color("79c6ae"),Vector3(0,direction*(float(geometry.get("rod_radius_m",.0015))+fin_height/2)/scale_m,0))
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

func fit_view() -> void:
	if camera == null or links.is_empty(): return
	var window_size := get_viewport().get_visible_rect().size
	var focus := links[0].position+Vector3(-.65,0,0)
	camera.position = focus+Vector3(.9,1.25,4.8)
	camera.look_at(focus)
	var desired_width := maxf(180,minf(viewport_area.size.x*.74,850))
	camera.size = maxf(2.75,1.8*window_size.y/desired_width)
	var center := viewport_area.get_center()
	camera.h_offset = (window_size.x*.5-center.x)/window_size.y*camera.size
	camera.v_offset = (center.y-window_size.y*.5)/window_size.y*camera.size

func target_caption_position() -> Vector2:
	var point := camera.unproject_position(anchor)
	return Vector2(clampf(point.x-30,viewport_area.position.x,viewport_area.end.x-95),clampf(point.y+.18*get_viewport().get_visible_rect().size.y/camera.size+16,viewport_area.position.y,viewport_area.end.y-30))

func target_caption() -> String:
	var point := camera.unproject_position(anchor)
	if viewport_area.has_point(point): return "Target"
	return "← Target" if point.x<viewport_area.get_center().x else "Target →"

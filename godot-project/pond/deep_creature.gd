extends Node3D

var head: Node3D
var tail: MeshInstance3D
var membrane: MeshInstance3D
var membrane_half_height := 0.215

func _ready() -> void:
	head = Node3D.new()
	add_child(head)
	var body := MeshInstance3D.new()
	body.mesh = preload("res://pond/pond_geometry.gd").head_mesh()
	var skin := ShaderMaterial.new()
	skin.shader = preload("res://pond/deep_skin.gdshader")
	body.material_override = skin
	head.add_child(body)
	for side in [-1, 1]:
		_sphere(head, Vector3(0.12, 0.085, side * 0.13), Vector3.ONE * 0.059, Color("eef4e5"))
		_sphere(head, Vector3(0.146, 0.096, side * 0.166), Vector3.ONE * 0.031, Color("102824"))
		_sphere(head, Vector3(0.153, 0.11, side * 0.183), Vector3.ONE * 0.009, Color.WHITE)
		for spot in 3:
			_sphere(head, Vector3(-0.02 - spot * 0.05, 0.02, side * 0.167), Vector3(0.019, 0.014, 0.009), Color("c8b879"))
	tail = MeshInstance3D.new()
	var tail_skin := skin.duplicate() as ShaderMaterial
	tail_skin.set_shader_parameter("tail_surface", true)
	tail.material_override = tail_skin
	add_child(tail)
	membrane = MeshInstance3D.new()
	var fin := StandardMaterial3D.new()
	fin.albedo_color = Color(0.08, 0.48, 0.41, 0.82)
	fin.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	fin.cull_mode = BaseMaterial3D.CULL_DISABLED
	fin.roughness = 0.4
	membrane.material_override = fin
	membrane.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(membrane)

func _sphere(parent: Node3D, at: Vector3, dimensions: Vector3, color: Color) -> void:
	var mesh := SphereMesh.new()
	mesh.radius = 1
	mesh.height = 2
	mesh.radial_segments = 16
	mesh.rings = 8
	var instance := MeshInstance3D.new()
	instance.mesh = mesh
	instance.position = at
	instance.scale = dimensions
	var material := StandardMaterial3D.new()
	material.albedo_color = color
	material.roughness = 0.35
	instance.material_override = material
	parent.add_child(instance)

func _surface(vertices: PackedVector3Array, indices: PackedInt32Array, uv: PackedVector2Array) -> ArrayMesh:
	var tool := SurfaceTool.new()
	tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in vertices.size():
		tool.set_uv(uv[i])
		tool.add_vertex(vertices[i])
	for index in indices:
		tool.add_index(index)
	tool.generate_normals()
	return tool.commit()

func pose(links: Array) -> void:
	if links.size() != 3 or head == null:
		return
	head.position = links[0].position
	head.quaternion = links[0].orientation
	var a: Vector3 = links[0].position - (links[0].orientation as Quaternion) * Vector3(0.25, 0, 0)
	var b: Vector3 = links[1].position - (links[1].orientation as Quaternion) * Vector3(0.325, 0, 0)
	var c: Vector3 = links[2].position - (links[2].orientation as Quaternion) * Vector3(0.325, 0, 0)
	var vertices := PackedVector3Array()
	var uv := PackedVector2Array()
	var indices := PackedInt32Array()
	var fin_vertices := PackedVector3Array()
	var fin_uv := PackedVector2Array()
	var fin_indices := PackedInt32Array()
	for ring in 25:
		var t := float(ring) / 24
		var center := a.lerp(b, t * 2) if t <= 0.5 else b.lerp(c, (t - 0.5) * 2)
		var orientation: Quaternion = (links[1].orientation as Quaternion).slerp(links[2].orientation, smoothstep(0.35, 0.65, t))
		var up := orientation * Vector3.UP
		var side := orientation * Vector3.BACK
		for spoke in 13:
			var angle := TAU * float(spoke) / 12
			vertices.append(center + (up * sin(angle) + side * cos(angle)) * lerpf(0.06, 0.018, t))
			uv.append(Vector2(t, float(spoke) / 12))
		var height := minf(membrane_half_height, 0.025 + 0.19 * sin(PI * clampf((t - 0.3) / 0.7, 0, 1)))
		for sign_y in [-1, 1]:
			fin_vertices.append(center + up * height * sign_y)
			fin_uv.append(Vector2(t, (sign_y + 1) * 0.5))
	for ring in 24:
		for spoke in 12:
			var index := ring * 13 + spoke
			indices.append_array(PackedInt32Array([index, index + 13, index + 1, index + 1, index + 13, index + 14]))
		var index := ring * 2
		fin_indices.append_array(PackedInt32Array([index, index + 2, index + 1, index + 1, index + 2, index + 3]))
	tail.mesh = _surface(vertices, indices, uv)
	membrane.mesh = _surface(fin_vertices, fin_indices, fin_uv)

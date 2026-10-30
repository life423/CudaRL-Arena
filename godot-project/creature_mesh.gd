extends Node3D

# Geometry dimensions and poses come from the native world. This node only
# renders them; it never synthesizes a gait or writes physical transforms back.
var pieces: Array[MeshInstance3D] = []

func _material(color: Color) -> StandardMaterial3D:
	var result := StandardMaterial3D.new()
	result.albedo_color = color
	result.roughness = 0.46
	if color.a < 1:
		result.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		result.cull_mode = BaseMaterial3D.CULL_DISABLED
	return result

func _mesh(length: float, half_height: float, half_width: float, rounded: bool) -> ArrayMesh:
	var tool := SurfaceTool.new()
	tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	for axial in range(9):
		var fraction := (axial - 4) * 0.25
		var profile := sqrt(maxf(0.0, 1.0 - fraction * fraction)) if rounded else 1.0
		for rim in range(16):
			var angle := rim * TAU / 16
			var normal := Vector3(fraction / (length * 0.5) if rounded else 0.0, profile * cos(angle) / half_height, profile * sin(angle) / half_width)
			tool.set_normal(normal.normalized())
			tool.add_vertex(Vector3(fraction * length * 0.5, profile * half_height * cos(angle), profile * half_width * sin(angle)))
	for axial in range(8):
		for rim in range(16):
			var a := axial * 16 + rim
			var b := (axial + 1) * 16 + rim
			var c := (axial + 1) * 16 + (rim + 1) % 16
			var d := axial * 16 + (rim + 1) % 16
			for index in [a, c, b, a, d, c]:
				tool.add_index(index)
	return tool.commit()

func _eye(parent: Node3D, at: Vector3, radius: float, color: Color) -> void:
	var node := MeshInstance3D.new()
	var sphere := SphereMesh.new()
	sphere.radius = radius
	sphere.height = radius * 2
	sphere.radial_segments = 12
	sphere.rings = 6
	node.mesh = sphere
	node.position = at
	node.material_override = _material(color)
	parent.add_child(node)

func pose(links: Array) -> void:
	if links.size() != 3:
		return
	if pieces.is_empty():
		for index in range(3):
			var geometry: Dictionary = links[index]
			var piece := MeshInstance3D.new()
			piece.mesh = _mesh(geometry.length_model, geometry.half_height_model, geometry.half_width_model, index == 0)
			piece.material_override = _material(Color("71c5bd") if index < 2 else Color(0.33, 0.70, 0.63, 0.80))
			add_child(piece)
			pieces.append(piece)
		for side in [-1, 1]:
			_eye(pieces[0], Vector3(0.14, 0.10, side * 0.14), 0.048, Color("f3f5e9"))
			_eye(pieces[0], Vector3(0.16, 0.105, side * 0.177), 0.024, Color("102824"))
	for index in range(3):
		var link: Dictionary = links[index]
		pieces[index].transform = Transform3D(Basis(link.orientation), link.position)

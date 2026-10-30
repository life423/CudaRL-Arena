extends RefCounted

# Geometry-only renderer. Core half extents + bevel are the native rounded-box
# descriptor; surface detail changes normals/color, never the solid silhouette.
static func _coordinates(half_extent: float, bevel: float) -> PackedFloat32Array:
	var values := PackedFloat32Array()
	if bevel <= 0.0:
		for index in 5:
			values.append(lerpf(-half_extent, half_extent, float(index) / 4.0))
		return values
	for index in 9:
		values.append(-half_extent - bevel + bevel * float(index) / 8.0)
	for index in range(1, 4):
		values.append(lerpf(-half_extent, half_extent, float(index) / 4.0))
	for index in 9:
		values.append(half_extent + bevel * float(index) / 8.0)
	return values

static func _surface_vertex(point: Vector3, half_extent: Vector3, bevel: float, face_normal: Vector3) -> Dictionary:
	var core := Vector3(clampf(point.x, -half_extent.x, half_extent.x), clampf(point.y, -half_extent.y, half_extent.y), clampf(point.z, -half_extent.z, half_extent.z))
	var offset := point - core
	var normal := offset.normalized() if offset.length_squared() > 0.00000001 else face_normal
	return {"position":core + normal * bevel, "normal":normal}

static func _vertex_color(position: Vector3, normal: Vector3) -> Color:
	# Restrained sediment bands and moss on upward faces; no displacement.
	var strata := 0.96 + 0.04 * sin(position.y * 17.0 + position.x * 0.7 + position.z * 0.3)
	var moss := clampf(normal.y, 0, 1) * (0.32 + 0.18 * sin(position.x * 1.8 + position.z * 1.4))
	return Color(1.0, 0.98, 0.94).lerp(Color(0.74, 0.82, 0.58), moss) * Color(strata, strata, strata)

static func _mesh(half_extent: Vector3, bevel: float) -> ArrayMesh:
	var outer := half_extent + Vector3.ONE * bevel
	var tool := SurfaceTool.new()
	tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	for axis in 3:
		var u_axis := (axis + 1) % 3
		var v_axis := (axis + 2) % 3
		var u := _coordinates(half_extent[u_axis], bevel)
		var v := _coordinates(half_extent[v_axis], bevel)
		for side in [-1.0, 1.0]:
			var face_normal := Vector3.ZERO
			face_normal[axis] = side
			var grid: Array[Dictionary] = []
			for row in v.size():
				for column in u.size():
					var point := Vector3.ZERO
					point[axis] = outer[axis] * side
					point[u_axis] = u[column]
					point[v_axis] = v[row]
					var vertex := _surface_vertex(point, half_extent, bevel, face_normal)
					vertex["uv"] = Vector2((u[column] / outer[u_axis] + 1.0) * 0.5, (v[row] / outer[v_axis] + 1.0) * 0.5)
					grid.append(vertex)
			for row in v.size() - 1:
				for column in u.size() - 1:
					var a := row * u.size() + column
					var b := a + 1
					var c := a + u.size()
					var d := c + 1
					for triangle in [[a, b, c], [b, d, c]]:
						var first: Vector3 = grid[triangle[0]].position
						var second: Vector3 = grid[triangle[1]].position
						var third: Vector3 = grid[triangle[2]].position
						# Godot front faces use clockwise winding. Normals remain
						# the outward analytic rounded-box normals on every patch.
						if (second - first).cross(third - first).dot(face_normal) > 0.0:
							var swap: int = triangle[1]
							triangle[1] = triangle[2]
							triangle[2] = swap
						for index in triangle:
							var vertex: Dictionary = grid[index]
							tool.set_normal(vertex.normal)
							tool.set_uv(vertex.uv)
							tool.set_color(_vertex_color(vertex.position, vertex.normal))
							tool.add_vertex(vertex.position)
	tool.generate_tangents()
	return tool.commit()

static func build(world: Node3D, geometry: Dictionary, material: Material) -> Node3D:
	var root := Node3D.new()
	root.name = "NativeOverhangingRock"
	root.set_meta("layout_id", int(geometry.get("layout_id", -1)))
	var stone: Material = material.duplicate() as Material if material != null else StandardMaterial3D.new()
	if stone is StandardMaterial3D:
		(stone as StandardMaterial3D).vertex_color_use_as_albedo = true
	var parts: Array = geometry.get("overhang_parts", [])
	for index in parts.size():
		var part: Dictionary = parts[index]
		var center: Vector3 = part.get("center", Vector3.INF)
		var half_extent: Vector3 = part.get("half_extent", Vector3.ZERO)
		var orientation: Quaternion = part.get("orientation", Quaternion.IDENTITY)
		var bevel := float(part.get("bevel", -1.0))
		if not center.is_finite() or not half_extent.is_finite() or minf(half_extent.x, minf(half_extent.y, half_extent.z)) <= 0.0 or not is_finite(bevel) or bevel < 0.0 or not orientation.is_normalized():
			push_error("Invalid native overhang part descriptor")
			continue
		var instance := MeshInstance3D.new()
		instance.name = "NativeOverhangPart%d" % index
		instance.mesh = _mesh(half_extent, bevel)
		instance.position = center
		instance.quaternion = orientation
		instance.material_override = stone
		instance.set_meta("core_half_extent", half_extent)
		instance.set_meta("bevel", bevel)
		instance.set_meta("native_part", index)
		root.add_child(instance)
	world.add_child(root)
	return root

extends RefCounted

static func head_radius(t: float) -> float:
	return (0.06 + 0.17 * sin(PI * t) * (0.6 + 0.4 * t)) * sqrt(maxf(0.0, 1.0 - pow(absf(2.0 * t - 1.0), 12)))

static func head_mesh() -> ArrayMesh:
	var vertices := PackedVector3Array()
	var normals := PackedVector3Array()
	var uv := PackedVector2Array()
	var indices := PackedInt32Array()
	for ring in 25:
		var t := float(ring) / 24.0
		var radius := head_radius(t)
		var slope := (head_radius(minf(1, t + 0.001)) - head_radius(maxf(0, t - 0.001))) / 0.001
		for side in 25:
			var angle := TAU * float(side) / 24.0
			vertices.append(Vector3((t - 0.5) * 0.5, sin(angle) * radius * 0.8, cos(angle) * radius))
			normals.append(Vector3(-slope, sin(angle) / 0.8, cos(angle)).normalized())
			uv.append(Vector2(t, float(side) / 24.0))
	for ring in 24:
		for side in 24:
			var a := ring * 25 + side
			indices.append_array(PackedInt32Array([a, a + 25, a + 1, a + 1, a + 25, a + 26]))
	var arrays: Array = []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_NORMAL] = normals
	arrays[Mesh.ARRAY_TEX_UV] = uv
	arrays[Mesh.ARRAY_INDEX] = indices
	var result := ArrayMesh.new()
	result.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	return result

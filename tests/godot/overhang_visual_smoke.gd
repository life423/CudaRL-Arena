extends SceneTree

const OverhangRock = preload("res://pond/overhang_rock.gd")
var checks := 0
var failures := 0
var output := ""
var measurements: Array[Dictionary] = []

func check(ok: bool, message: String) -> void:
	checks += 1
	if not ok:
		failures += 1
		push_error(message)

func _initialize() -> void:
	Engine.max_fps = 60
	run.call_deferred()

func settle(frames := 6) -> void:
	for tick in frames:
		await process_frame

func part_distance(point: Vector3, half_extent: Vector3, bevel: float) -> float:
	var q := point.abs() - half_extent
	return Vector3(maxf(q.x, 0), maxf(q.y, 0), maxf(q.z, 0)).length() + minf(maxf(q.x, maxf(q.y, q.z)), 0) - bevel

func freeze_material(material: Material, seen: Dictionary) -> void:
	if not material is ShaderMaterial or seen.has(material.get_instance_id()):
		return
	seen[material.get_instance_id()] = true
	var shader_material := material as ShaderMaterial
	if shader_material.shader and "TIME" in shader_material.shader.code:
		# Hold the actual shader at a single instant for pixel differencing. A
		# moving caustic must never be counted as a visible part of the creature.
		var snapshot := Shader.new()
		snapshot.code = shader_material.shader.code.replace("TIME", "0.0")
		shader_material.shader = snapshot

func freeze_visual_time(node: Node, seen: Dictionary) -> void:
	if node is MeshInstance3D:
		var instance := node as MeshInstance3D
		freeze_material(instance.material_override, seen)
		if instance.mesh:
			for surface in instance.mesh.get_surface_count():
				freeze_material(instance.mesh.surface_get_material(surface), seen)
	for child in node.get_children():
		freeze_visual_time(child, seen)

func inspect_geometry(pond, rendered: Node3D, geometry: Dictionary) -> void:
	var parts: Array = geometry.get("overhang_parts", [])
	check(int(geometry.get("layout_id", -1)) == 3, "The overhang must use its isolated native layout")
	check(parts.size() == 2 and rendered.get_child_count() == parts.size(), "Roof and bank support must match the two native solids")
	var points := PackedVector3Array()
	var maximum_part_error := 0.0
	var maximum_facet_error := 0.0
	var winding_errors := 0
	var bad_normals := 0
	var vertex_count := 0
	for child in rendered.get_children():
		var instance := child as MeshInstance3D
		check(instance != null and instance.mesh != null, "Native part has no rendered mesh")
		if instance == null or instance.mesh == null:
			continue
		var index := int(instance.get_meta("native_part", -1))
		check(index >= 0 and index < parts.size(), "Mesh part identity is invalid")
		if index < 0 or index >= parts.size():
			continue
		var descriptor: Dictionary = parts[index]
		var half_extent: Vector3 = descriptor.half_extent
		var bevel: float = descriptor.bevel
		check(instance.position == descriptor.center and instance.quaternion.is_equal_approx(descriptor.orientation), "Renderer changed a native solid transform")
		var arrays: Array = instance.mesh.surface_get_arrays(0)
		var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
		var normals: PackedVector3Array = arrays[Mesh.ARRAY_NORMAL]
		check(vertices.size() > 36 and vertices.size() == normals.size(), "Beveled part must have explicit surface normals")
		check(not (arrays[Mesh.ARRAY_TANGENT] as PackedFloat32Array).is_empty(), "Stone normal maps need generated tangent data")
		for vertex in vertices:
			maximum_part_error = maxf(maximum_part_error, absf(part_distance(vertex, half_extent, bevel)))
			points.append(instance.global_transform * vertex)
		vertex_count += vertices.size()
		for normal in normals:
			bad_normals += int(not normal.is_finite() or absf(normal.length() - 1.0) > 0.001)
		for triangle in range(0, vertices.size(), 3):
			var cross := (vertices[triangle + 1] - vertices[triangle]).cross(vertices[triangle + 2] - vertices[triangle])
			var normal := normals[triangle] + normals[triangle + 1] + normals[triangle + 2]
			winding_errors += int(cross.dot(normal) > 0.00000001)
			var centroid := (vertices[triangle] + vertices[triangle + 1] + vertices[triangle + 2]) / 3.0
			maximum_facet_error = maxf(maximum_facet_error, absf(part_distance(centroid, half_extent, bevel)))
	check(maximum_part_error <= 0.0002, "Mesh vertices do not lie on the native rounded-box surface")
	check(maximum_facet_error <= 0.002, "Bevel tessellation is too coarse for the physical solid")
	check(winding_errors == 0 and bad_normals == 0, "Rock face winding or normals are incorrect")
	var native: PackedFloat32Array = pond.lab.sample_overhang_solid(points)
	check(native.size() == points.size(), "Native union SDF query returned the wrong number of samples")
	var outside := 0
	var exposed := 0
	var maximum_native_distance := -INF
	for distance in native:
		outside += int(not is_finite(distance) or distance > 0.0003)
		exposed += int(absf(distance) <= 0.0003)
		maximum_native_distance = maxf(maximum_native_distance, distance)
	check(outside == 0 and exposed > vertex_count / 4, "Render geometry protrudes outside, or fails to expose, the native union")
	measurements.append({"mesh_vertices":vertex_count, "maximum_part_sdf_error":maximum_part_error, "maximum_facet_sdf_error":maximum_facet_error, "maximum_native_union_distance":maximum_native_distance, "exposed_union_vertices":exposed, "winding_errors":winding_errors})

func body_visibility(pond, dimensions: Vector2i, view: String) -> void:
	pond.creature.show()
	await settle(3)
	await RenderingServer.frame_post_draw
	var with_body: Image = pond.viewport.get_texture().get_image()
	pond.creature.hide()
	await settle(3)
	await RenderingServer.frame_post_draw
	var without_body: Image = pond.viewport.get_texture().get_image()
	pond.creature.show()
	await settle(2)
	check(not with_body.is_empty() and with_body.get_size() == without_body.get_size(), "Cannot inspect native body visibility")
	if with_body.is_empty() or with_body.get_size() != without_body.get_size():
		return
	var head: Vector3 = pond.state.position
	check(not pond.camera.is_position_behind(head), "The native shelter pose is behind the camera")
	var center: Vector2 = pond.camera.unproject_position(head)
	var rectangle := Rect2(center, Vector2.ZERO)
	for link: Dictionary in pond.state.links:
		var position: Vector3 = link.position
		for offset in [Vector3(0.4,0,0), Vector3(-0.4,0,0), Vector3(0,0.4,0), Vector3(0,-0.4,0), Vector3(0,0,0.4), Vector3(0,0,-0.4)]:
			rectangle = rectangle.expand(pond.camera.unproject_position(position + offset))
	rectangle = rectangle.grow(3).intersection(Rect2(Vector2.ZERO, Vector2(with_body.get_size())))
	var visible_pixels := 0
	var head_pixels := 0
	for y in range(int(rectangle.position.y), int(rectangle.end.y)):
		for x in range(int(rectangle.position.x), int(rectangle.end.x)):
			var a := with_body.get_pixel(x, y)
			var b := without_body.get_pixel(x, y)
			if absf(a.r-b.r) + absf(a.g-b.g) + absf(a.b-b.b) > 0.04:
				visible_pixels += 1
				if absf(x-center.x) <= 6 and absf(y-center.y) <= 6:
					head_pixels += 1
	check(visible_pixels >= 8 and head_pixels >= 3, "The creature cannot be read under the ledge at %s / %s" % [dimensions, view])
	measurements.append({"size":str(dimensions), "view":view, "body_pixels":visible_pixels, "head_pixels":head_pixels})

func run() -> void:
	create_timer(120).timeout.connect(func(): push_error("Overhang visual QA timed out"); quit(1))
	check(DisplayServer.get_name() != "headless", "Overhang visual QA requires a real renderer")
	check(ClassDB.class_has_method("DeepPondLab", "get_overhang_geometry") and ClassDB.class_has_method("DeepPondLab", "sample_overhang_solid"), "Native overhang descriptor/SDF API is unavailable")
	if failures:
		quit(1)
		return
	output = ProjectSettings.globalize_path("res://../build/predator-shelter/overhang-visual")
	check(DirAccess.make_dir_recursive_absolute(output) == OK, "Cannot create isolated QA output")
	# Runtime loading lets the project's autoloads initialize before parsing
	# the inherited pond scripts (the standalone --script entry loads first).
	var fixture: Script = load("res://../tests/godot/overhang_geometry_fixture.gd")
	check(fixture != null and fixture.can_instantiate(), "Geometry fixture could not load")
	if fixture == null or not fixture.can_instantiate():
		quit(1)
		return
	var pond = fixture.new()
	root.add_child(pond)
	current_scene = pond
	pond.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	pond.sidebar.hide()
	pond.curve.hide()
	pond.viewport_container.custom_minimum_size = Vector2.ZERO
	pond.viewport_container.size_flags_vertical = Control.SIZE_EXPAND_FILL
	get_root().title = "Overhang geometry QA · no learning"
	var geometry: Dictionary = pond.arch_geometry
	check(not geometry.get("hiding_validated", true), "Geometry QA must not claim learned hiding is validated")
	var rock := OverhangRock.build(pond.world, geometry, pond._stone_material())
	pond.water_surface.set_meta("layout_id", 3)
	freeze_visual_time(pond.world, {})
	await settle()
	inspect_geometry(pond, rock, geometry)
	var covers: Array = geometry.get("cover_poses", [])
	check(not covers.is_empty(), "Native cover poses are required to inspect the whole creature")
	if covers.is_empty():
		pond.queue_free()
		await process_frame
		quit(1)
		return
	var cover: Dictionary = covers[0]
	var links: Array = cover.get("links", [])
	check(links.size() == 3 and cover.get("valid", false), "Static cover fixture must have three native legal links")
	if links.size() != 3:
		quit(1)
		return
	pond.state = {"position":cover.position, "links":links}
	pond.previous = pond.state.duplicate(true)
	pond.creature.pose(links)
	pond.rendered_swimmer_position = links[0].position
	pond.visibility_focus = (links[0].position as Vector3).lerp(links[2].position, 0.5)
	var label := Label.new()
	label.text = "GEOMETRY QA · NO LEARNING"
	label.position = Vector2(12,12)
	label.add_theme_font_size_override("font_size", 14)
	label.add_theme_color_override("font_color", Color("eef4e9"))
	label.add_theme_color_override("font_outline_color", Color("0b2429"))
	label.add_theme_constant_override("outline_size", 4)
	label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	root.add_child(label)
	for dimensions in [Vector2i(1280,720), Vector2i(768,1024), Vector2i(390,844), Vector2i(844,390)]:
		root.size = dimensions
		await settle()
		pond._fit()
		for view in ["overview", "under-mouth"]:
			pond.overview = view == "overview"
			pond._update_camera(1.0)
			await settle()
			await body_visibility(pond, dimensions, view)
			await RenderingServer.frame_post_draw
			check(root.get_texture().get_image().save_png(output.path_join("overhang-%dx%d-%s.png" % [dimensions.x,dimensions.y,view])) == OK, "Geometry capture failed")
	check(pond.lab.get_stats().get("learning_active", false) == false, "Static geometry QA must never start a learner")
	var summary := {"checks":checks, "failures":failures, "static_geometry_only":true, "no_learning":true, "hiding_validated":false, "measurements":measurements}
	var report := FileAccess.open(output.path_join("summary.json"), FileAccess.WRITE)
	if report:
		report.store_string(JSON.stringify(summary, "\t"))
		report.close()
	print("OVERHANG_VISUAL_SMOKE " + JSON.stringify(summary))
	pond.queue_free()
	await process_frame
	quit(1 if failures else 0)

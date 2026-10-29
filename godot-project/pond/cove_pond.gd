extends "res://pond/arch_pond.gd"

@export var source_arch_checkpoint := "user://pond-policies/arch-navigation-v1/training.bin"
@export var cove_checkpoint := "user://pond-policies/cove-navigation-v1/training.bin"
var native_outline := PackedVector3Array()
var native_terrain_positions := PackedVector3Array()
var native_terrain_samples := PackedVector3Array()

func _load_habitat_geometry() -> Dictionary:
	var geometry: Dictionary = lab.get_cove_geometry()
	native_outline = geometry.get("shore_outline", PackedVector3Array())
	if native_outline.size() > 1 and native_outline[0].is_equal_approx(native_outline[-1]):
		native_outline.resize(native_outline.size() - 1)
	return geometry

func _ready() -> void:
	super._ready()
	get_window().title = "Tadpole Pond · Cove learning · CudaRL"
	(sidebar_scroll.get_child(0).get_child(0) as Label).text = "Natural pond"
	_poll()

func _start_lab() -> bool:
	paused = false
	var backend := "auto"
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--pond-backend="):
			backend = argument.get_slice("=", 1)
	return lab.start_cove_learning(source_arch_checkpoint, cove_checkpoint, population, backend)

func _terrain() -> ArrayMesh:
	# The outline and every underwater height/normal originate in native code.
	# Interior rings stay dense; exterior rings extend the dry ground for views.
	native_terrain_positions.clear()
	var spokes := native_outline.size()
	for ring in 97:
		var radius := float(ring) / 64.0 * 1.08 if ring <= 64 else 1.08 + float(ring - 64) / 32.0 * 10.92
		for spoke in spokes:
			native_terrain_positions.append(native_outline[spoke] * radius)
	native_terrain_samples = lab.sample_cove_terrain(native_terrain_positions)
	var tool := SurfaceTool.new()
	tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	for index in native_terrain_positions.size():
		var p := native_terrain_positions[index]
		var sample := native_terrain_samples[index]
		tool.set_uv(Vector2(p.x, p.z) * 0.05 + Vector2.ONE * 0.5)
		tool.set_normal(Vector3(-sample.y, 1, -sample.z).normalized())
		tool.add_vertex(Vector3(p.x, sample.x, p.z))
	for ring in 96:
		for spoke in spokes:
			var next := (spoke + 1) % spokes
			var a := ring * spokes + spoke
			var b := ring * spokes + next
			var c := (ring + 1) * spokes + spoke
			var d := (ring + 1) * spokes + next
			for index in [a, c, b, b, c, d]:
				tool.add_index(index)
	return tool.commit()

func _water_mesh() -> ArrayMesh:
	var polygon := PackedVector2Array()
	for p in native_outline:
		polygon.append(Vector2(p.x, p.z))
	var triangles := Geometry2D.triangulate_polygon(polygon)
	var tool := SurfaceTool.new()
	tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	for index in triangles:
		var p := native_outline[index]
		tool.set_normal(Vector3.UP)
		tool.set_uv(Vector2(p.x, p.z) * 0.05 + Vector2.ONE * 0.5)
		tool.add_vertex(Vector3(p.x, 0, p.z))
	return tool.commit()

func _create_shore() -> Node3D:
	var decoration = preload("res://pond/natural_shore.gd").new()
	decoration.configure_outline(native_outline, Callable(lab, "sample_cove_terrain"))
	return decoration

func _configure_flow() -> void:
	flow.configure_habitat(_pond_radii(), Callable(lab, "advect_arch_tracers"), Callable(lab, "get_arch_flow_vectors"), arch_geometry, Callable(lab, "sample_cove_terrain"))

func _build_world() -> void:
	super._build_world()
	# Native cove vertices/shore props already use actual world dimensions.
	shore.scale = Vector3.ONE
	water_surface.mesh = _water_mesh()
	water_surface.scale = Vector3.ONE
	water_surface.name = "NativeCoveWater"
	water_surface.set_meta("layout_id", 1)

func _pond_overview_target() -> Vector3:
	var minimum: Vector2 = arch_geometry.get("bounds_min", Vector2(-6.9, -5.25))
	var maximum: Vector2 = arch_geometry.get("bounds_max", Vector2(9, 5.25))
	var center := (minimum + maximum) * 0.5
	return Vector3(center.x, -1.3, center.y)

func _pond_fit_points() -> Array[Vector3]:
	var points: Array[Vector3] = []
	var center := _pond_overview_target()
	for p in native_outline:
		points.append(center + Vector3((p.x - center.x) * 1.14, 2.1, (p.z - center.z) * 1.14))
	points.append(Vector3(0, -4, 0))
	points.append(arch_geometry.get("cove_center", Vector3(5.8, -3, -1.5)))
	return points

func _poll() -> void:
	super._poll()
	if not lab:
		return
	if not lab.get_error().is_empty():
		title_status.text = "Cove learning could not start"
		return
	if int(stats.get("layout_id", 0)) != 1:
		return
	feed_feedback.text = "Food explores the main pond and cove."
	training_heading.text = "PPO · cove navigation"
	training_stats.text += "\nHabitat: main pool + outward cove. Existing arch swimming is retained and tested."
	var imported: int = stats.get("parent_stage_update", stats.get("loaded_update", 0))
	save_status.text = "Arch learning retained from update %d. Original arch save preserved.\nSeparate cove save at %d · autosaves every 30 s and on exit." % [imported, stats.get("saved_update", 0)]
	if not str(stats.get("save_error", "")).is_empty():
		save_status.text = "Save failed: " + str(stats.save_error)

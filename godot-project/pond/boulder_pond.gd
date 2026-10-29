extends "res://pond/cove_pond.gd"

@export var source_cove_checkpoint := "user://pond-policies/cove-navigation-v1/training.bin"
@export var boulder_checkpoint := "user://pond-policies/boulder-navigation-v1/training.bin"
var rock_boulder: MeshInstance3D
var visit_start_time := -1.0
var visit_start_eaten := 0

func _update_session_curve(_untrained: bool) -> void:
	var elapsed: float = state.get("elapsed", 0.0)
	var eaten: int = state.get("eaten", 0)
	if visit_start_time < 0.0 or elapsed < visit_start_time or eaten < visit_start_eaten:
		visit_start_time = elapsed
		visit_start_eaten = eaten
		curve.points.clear()
	var duration := elapsed - visit_start_time
	curve.best_points.clear()
	curve.best_score = -1
	curve.show_best_score = false
	curve.current_score = (eaten - visit_start_eaten) * 60.0 / duration if duration >= 1.0 else -1.0
	if duration >= 1.0 and (curve.points.is_empty() or floori(duration / 5.0) > floori(curve.points[-1].x / 5.0)):
		curve.points.append(Vector2(duration, curve.current_score))
		if curve.points.size() > 360:
			curve.points.remove_at(0)
	last_session_time = elapsed
	curve.queue_redraw()

func _load_habitat_geometry() -> Dictionary:
	var geometry: Dictionary = lab.get_boulder_geometry()
	native_outline = geometry.get("shore_outline", PackedVector3Array())
	if native_outline.size() > 1 and native_outline[0].is_equal_approx(native_outline[-1]):
		native_outline.resize(native_outline.size() - 1)
	return geometry

func _ready() -> void:
	super._ready()
	get_window().title = "Tadpole Pond · Boulder learning · CudaRL"
	(sidebar_scroll.get_child(0).get_child(0) as Label).text = "Boulder pond"
	_poll()

func _start_lab() -> bool:
	paused = false
	var backend := "auto"
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--pond-backend="):
			backend = argument.get_slice("=", 1)
	return lab.start_boulder_learning(source_cove_checkpoint, boulder_checkpoint, population, backend)

func _build_world() -> void:
	super._build_world()
	# The physical solid is a sphere. Surface color/normal variation adds stone
	# character without changing its silhouette away from the native contact.
	var radius: float = arch_geometry.get("boulder_radius", 0.0)
	var mesh := SphereMesh.new()
	mesh.radius = radius
	mesh.height = radius * 2.0
	mesh.radial_segments = 64
	mesh.rings = 32
	rock_boulder = MeshInstance3D.new()
	rock_boulder.name = "PhysicalBoulder"
	rock_boulder.mesh = mesh
	rock_boulder.position = arch_geometry.get("boulder_center", Vector3.ZERO)
	rock_boulder.material_override = _stone_material()
	rock_boulder.set_meta("layout_id", 2)
	world.add_child(rock_boulder)
	water_surface.set_meta("layout_id", 2)

func _poll() -> void:
	super._poll()
	if not lab:
		return
	if not lab.get_error().is_empty():
		title_status.text = "Boulder learning could not start"
		return
	if int(stats.get("layout_id", 0)) != 2:
		return
	feed_feedback.text = "Food explores the main pond, cove and routes around the boulder."
	curve.status_caption = "PPO %d · live actor %d · session collection" % [stats.get("update", 0), stats.get("adopted_update", 0)]
	curve.plot_caption = "This visit · actual food / minute"
	curve.score_label = "This visit"
	curve.axis_label = "Seconds in this visit"
	curve.tooltip_text = "Food collected since opening this pond, excluding inherited lifetime totals. The original physical clock and food counter are still saved. AI details show separate matched policy checks."
	water_status.tooltip_text = "Native boundary-aware flow and a bounded downstream wake drive body drag, food and tracers. This reduced-order model does not solve pressure, turbulence or surge propagation."
	current_strength.tooltip_text = "Choose 0.5 or 0.75 cm/s imposed flow. Native obstacles reduce downstream flow; shelter-seeking has not yet been trained."
	training_heading.text = "PPO · boulder navigation"
	training_stats.text += "\nHabitat: main pool + cove + one solid boulder. Native flow is reduced downstream of obstacles; shelter-seeking has not been trained. Existing skills are checked before new weights reach the swimmer."
	if float(stats.get("cove_retention_food_per_minute", -1.0)) >= 0.0:
		training_stats.text += "\nCove retention: %.1f / min · required %.1f / min." % [stats.cove_retention_food_per_minute, stats.cove_retention_reference_food_per_minute]
	var imported: int = stats.get("parent_stage_update", stats.get("loaded_update", 0))
	save_status.text = "Cove learning retained from update %d. Original cove save preserved.\nSeparate boulder save at %d · autosaves every 30 s and on exit." % [imported, stats.get("saved_update", 0)]
	if not str(stats.get("save_error", "")).is_empty():
		save_status.text = "Save failed: " + str(stats.save_error)

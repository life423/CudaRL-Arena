extends "res://pond/boulder_pond.gd"

@export var source_boulder_checkpoint := "user://pond-policies/boulder-navigation-v1/training.bin"
@export var predator_checkpoint := "user://pond-policies/predator-shelter-v1/training.bin"

var predator_button: Button
var predator_visual: MeshInstance3D
var predator_shadow: MeshInstance3D
var predator_shadow_material: ShaderMaterial
var predator_api_available := false
var readiness_message: Label

func _start_lab() -> bool:
	paused = false
	predator_api_available = lab.has_method("start_predator_learning") and lab.has_method("trigger_predator")
	if not predator_api_available:
		return false
	var backend := "auto"
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--pond-backend="):
			backend = argument.get_slice("=", 1)
	return lab.start_predator_learning(source_boulder_checkpoint, predator_checkpoint, population, backend)

func _ready() -> void:
	super._ready()
	get_window().title = "Tadpole Pond · Predator shelter prototype · CudaRL"
	(sidebar_scroll.get_child(0).get_child(0) as Label).text = "Tadpole pond"
	_poll()

func _build_ui() -> void:
	super._build_ui()
	# Reuse the inherited accessible action position; this scene does not drop
	# food or direct the creature from Godot.
	predator_button = find_child("DropFoodAtCenter", true, false) as Button
	for connection in predator_button.pressed.get_connections():
		predator_button.pressed.disconnect(connection.callable)
	predator_button.name = "CallPredator"
	predator_button.text = "Call predator"
	predator_button.tooltip_text = "Send one predator across the pond. The native agent chooses whether and where to shelter; food remains available."
	predator_button.pressed.connect(_call_predator)
	predator_button.disabled = true
	predator_button.show()
	feed_feedback.text = "Preparing predator shelter practice…"
	feed_feedback.tooltip_text = "The shadow and overhead silhouette show the same native predator. Shelter requires real solid cover, not simply reaching a shaded pixel."

func _call_predator() -> void:
	if not lab or not predator_api_available or not released or str(state.get("predator_phase", "calm")) != "calm":
		return
	lab.trigger_predator()
	state = lab.get_state()
	# A new event starts at its actual native pose rather than interpolating
	# across the last event's unrelated endpoint.
	previous = state.duplicate(true)
	_poll()
	_apply_pose(1.0)

func _disable_trial_controls() -> void:
	predator_button.disabled = true
	pause_button.disabled = true
	current_switch.disabled = true
	current_strength.disabled = true
	speed_button.disabled = true
	(find_child("SaveTrainingNow", true, false) as Button).disabled = true
	camera_button.hide()
	viewport_container.hide()
	curve.hide()
	if not readiness_message:
		readiness_message = Label.new()
		readiness_message.name = "ShelterReadinessMessage"
		readiness_message.text = "Shelter practice is not ready yet.\n\nThis controller has not passed its holding test.\nUse Pond menu to return to the working swimmer."
		readiness_message.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		readiness_message.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		readiness_message.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
		readiness_message.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		readiness_message.size_flags_vertical = Control.SIZE_EXPAND_FILL
		readiness_message.add_theme_font_size_override("font_size", 18)
		readiness_message.add_theme_color_override("font_color", TEXT)
		visual.add_child(readiness_message)

func _bird_mesh() -> ArrayMesh:
	# A small soaring silhouette. All points lie in the XZ plane, facing -Z.
	# Its pose is wholly supplied by the native source, with no local flight clock.
	var outline := PackedVector2Array([
		Vector2(0, -0.72), Vector2(0.12, -0.55), Vector2(0.18, -0.25),
		Vector2(0.6, -0.2), Vector2(1.5, 0.4), Vector2(1.65, 0.8),
		Vector2(1.1, 0.55), Vector2(0.6, 0.25), Vector2(0.2, 0.4),
		Vector2(0.18, 0.75), Vector2(0.3, 0.9), Vector2(0, 0.78),
		Vector2(-0.3, 0.9), Vector2(-0.18, 0.75), Vector2(-0.2, 0.4),
		Vector2(-0.6, 0.25), Vector2(-1.1, 0.55), Vector2(-1.65, 0.8),
		Vector2(-1.5, 0.4), Vector2(-0.6, -0.2), Vector2(-0.18, -0.25),
		Vector2(-0.12, -0.55),
	])
	var triangles := Geometry2D.triangulate_polygon(outline)
	var tool := SurfaceTool.new()
	tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	for index in triangles:
		var p := outline[index]
		tool.set_normal(Vector3.UP)
		tool.add_vertex(Vector3(p.x, 0, p.y))
	return tool.commit()

func _build_world() -> void:
	super._build_world()
	var silhouette_material := StandardMaterial3D.new()
	silhouette_material.albedo_color = Color("162221")
	silhouette_material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	silhouette_material.cull_mode = BaseMaterial3D.CULL_DISABLED
	predator_visual = MeshInstance3D.new()
	predator_visual.name = "NativePredator"
	predator_visual.mesh = _bird_mesh()
	predator_visual.material_override = silhouette_material
	# The explicit surface cue below follows the native overhead projection;
	# do not add a second sun-angle shadow suggesting a different threat source.
	predator_visual.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	predator_visual.hide()
	world.add_child(predator_visual)
	predator_shadow_material = ShaderMaterial.new()
	predator_shadow_material.shader = preload("res://pond/predator_shadow.gdshader")
	predator_shadow_material.render_priority = 2
	predator_shadow = MeshInstance3D.new()
	predator_shadow.name = "NativePredatorSurfaceCue"
	# The native water outline clips the cue at the shore and cove boundary.
	predator_shadow.mesh = _water_mesh()
	predator_shadow.position.y = 0.018
	predator_shadow.material_override = predator_shadow_material
	predator_shadow.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	predator_shadow.hide()
	world.add_child(predator_shadow)

func _apply_pose(alpha: float) -> void:
	super._apply_pose(alpha)
	if not predator_visual or not predator_shadow:
		return
	var phase := str(state.get("predator_phase", "calm"))
	var active := phase in ["warning", "passing"] and state.has("predator_position") and state.has("predator_shadow_position")
	predator_visual.visible = active
	predator_shadow.visible = active
	if not active:
		return
	var position: Vector3 = state.predator_position
	var last_position: Vector3 = previous.get("predator_position", position)
	position = last_position.lerp(position, alpha)
	var shadow_position: Vector3 = state.predator_shadow_position
	var last_shadow_position: Vector3 = previous.get("predator_shadow_position", shadow_position)
	shadow_position = last_shadow_position.lerp(shadow_position, alpha)
	var direction: Vector3 = state.get("predator_direction", Vector3.FORWARD)
	var last_direction: Vector3 = previous.get("predator_direction", direction)
	direction = last_direction.lerp(direction, alpha)
	direction.y = 0
	if direction.length_squared() < 0.0001:
		direction = Vector3.FORWARD
	direction = direction.normalized()
	predator_visual.position = position
	predator_visual.basis = Basis.looking_at(direction, Vector3.UP)
	# Oblique native sunlight projects the shadow away from the bird's XZ
	# location. Use that measured projection rather than inventing another cue.
	predator_shadow_material.set_shader_parameter("predator_center", Vector2(shadow_position.x, shadow_position.z))
	predator_shadow_material.set_shader_parameter("predator_direction", Vector2(direction.x, direction.z))

func _poll() -> void:
	super._poll()
	if not predator_button or not lab:
		return
	if not predator_api_available:
		title_status.text = "Predator prototype is unavailable"
		feed_feedback.text = "This extension has no predator practice API. Return to the pond menu."
		_disable_trial_controls()
		return
	if not lab.get_error().is_empty():
		title_status.text = "Shelter practice is not ready"
		feed_feedback.text = "The swimmer must learn to hold cover first."
		feed_feedback.tooltip_text = lab.get_error()
		save_status.text = "The working boulder pond and its training saves are preserved."
		_disable_trial_controls()
		curve.status_caption = "No shelter policy applied"
		return
	var phase := str(state.get("predator_phase", "calm"))
	var goal := str(state.get("goal_kind", "food"))
	var exposure := clampf(float(state.get("body_exposure", 1.0)), 0, 1)
	predator_button.disabled = not released or phase != "calm"
	title_status.text = "Live · predator shelter prototype"
	if str(stats.get("save_error", "")).is_empty() and str(stats.get("learning_error", "")).is_empty():
		match phase:
			"warning":
				feed_feedback.text = "Predator approaching · goal: %s" % goal
			"passing":
				feed_feedback.text = "Predator passing · %.0f%% exposed\nGoal: %s" % [exposure * 100.0, goal]
			_:
				feed_feedback.text = "Predator clear · food remains available."
	training_heading.text = "PPO · predator shelter prototype"
	training_stats.text = "%s · %d background swimmers\nLearner update %d · visible actor %d\nNative phase: %s\nChosen goal: %s\nThreat %.0f%% · body exposure %.0f%%\n\nThe native simulation measures danger and solid cover. Godot only displays the agent's decisions. Shelter learning is experimental; food collection alone does not demonstrate successful hiding." % [stats.get("compute_backend", ""), stats.get("background_swimmers", population), stats.get("update", 0), stats.get("adopted_update", 0), phase, goal, clampf(float(state.get("threat_level", 0.0)), 0, 1) * 100.0, exposure * 100.0]
	if not str(stats.get("learning_error", "")).is_empty():
		training_stats.text += "\nLearning stopped: " + str(stats.learning_error)
	curve.status_caption = "PPO %d · actor %d · shelter prototype" % [stats.get("update", 0), stats.get("adopted_update", 0)]
	curve.tooltip_text = "Food collected during this visit. Choosing safety can temporarily reduce this rate; collection is not a shelter-learning score. AI details report the native threat state."
	var imported: int = stats.get("parent_stage_update", stats.get("loaded_update", 0))
	save_status.text = "Boulder learning retained from update %d. Original boulder save preserved.\nSeparate shelter save at %d · autosaves every 30 s and on exit." % [imported, stats.get("saved_update", 0)]
	if not str(stats.get("save_error", "")).is_empty():
		save_status.text = "Save failed: " + str(stats.save_error)

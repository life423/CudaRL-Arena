extends "res://pond/boulder_pond.gd"

const HoldingCurve = preload("res://pond/holding_curve.gd")
@export var source_motor_run := "res://../build/predator-shelter/motor-v4-training.bin"
@export var holding_checkpoint := "user://pond-policies/calm-hold-v1/training.bin"
var hold_api_available := false
var goal_ring: MeshInstance3D
var holding_message: Label
var hold_visit_start := -1.0

func _start_lab() -> bool:
	paused = false
	hold_api_available = lab.has_method("start_hold_learning")
	if not hold_api_available:
		return false
	var backend := "auto"
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--pond-backend="):
			backend = argument.get_slice("=", 1)
	return lab.start_hold_learning(source_motor_run, holding_checkpoint, population, backend)

func _ready() -> void:
	super._ready()
	get_window().title = "Tadpole Pond · Calm holding practice · CudaRL"
	(sidebar_scroll.get_child(0).get_child(0) as Label).text = "Holding practice"
	# Follow the trajectory with the inherited fixed-angle camera. Nothing here
	# follows joint rotations or changes the native body/goal state.
	overview = false
	camera_button.text = "Pond overview"
	_poll()
	_update_camera(1.0)

func _build_ui() -> void:
	super._build_ui()
	var old_curve := curve
	var curve_index := old_curve.get_index()
	visual.remove_child(old_curve)
	old_curve.queue_free()
	curve = HoldingCurve.new()
	curve.name = "HoldingCurve"
	curve.live_mode = false
	curve.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	visual.add_child(curve)
	visual.move_child(curve, curve_index)
	curve.watch_button.hide()
	find_child("WaterControls", true, false).hide()
	current_switch.disabled = true
	current_strength.disabled = true
	feed_feedback.text = "Assigned task · hold the marked shelter spot."
	feed_feedback.tooltip_text = "This is a calm-water body-control lesson. The neural network produces all four joint commands. The target is assigned; choosing safety over food is a later lesson."
	swimmer_stats.tooltip_text = "Actual native measurements. Holding requires the body and tail to remain concealed with positive physical clearance. Reaching the ring alone does not count."

func _build_world() -> void:
	super._build_world()
	var mesh := TorusMesh.new()
	mesh.inner_radius = 0.39
	mesh.outer_radius = 0.43
	mesh.rings = 40
	mesh.ring_segments = 10
	var material := StandardMaterial3D.new()
	material.albedo_color = Color("ffc45d")
	material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	material.albedo_color.a = 0.75
	goal_ring = MeshInstance3D.new()
	goal_ring.name = "AssignedHoldGoal"
	goal_ring.mesh = mesh
	goal_ring.material_override = material
	goal_ring.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	goal_ring.hide()
	world.add_child(goal_ring)

func _apply_pose(alpha: float) -> void:
	if state.is_empty() or not state.has("links"):
		return
	super._apply_pose(alpha)
	if not goal_ring:
		return
	var goal: Vector3 = state.get("goal_position", Vector3.INF)
	goal_ring.visible = goal.is_finite()
	if goal_ring.visible:
		goal_ring.position = goal

func _swimmer_text() -> String:
	return "Current hold  %.1f s\nBest hold  %.1f s\nBody hidden  %.0f%%\nDistance to spot  %.1f cm\nSpeed  %.2f cm/s" % [state.get("safe_hold_seconds", 0.0), state.get("longest_safe_hold_seconds", 0.0), (1.0 - clampf(float(state.get("body_exposure", 1.0)), 0, 1)) * 100.0, state.get("goal_distance_cm", 0.0), state.get("speed_cm_s", 0.0)]

func _update_session_curve(_untrained: bool) -> void:
	var elapsed: float = state.get("elapsed", 0.0)
	if hold_visit_start < 0.0 or elapsed < hold_visit_start:
		hold_visit_start = elapsed
		curve.points.clear()
		curve.best_points.clear()
	var duration := elapsed - hold_visit_start
	var current: float = state.get("safe_hold_seconds", 0.0)
	var longest: float = state.get("longest_safe_hold_seconds", 0.0)
	curve.current_score = current
	curve.best_score = longest
	curve.show_best_score = true
	if curve.points.is_empty() or floori(duration * 2.0) > floori(curve.points[-1].x * 2.0):
		curve.points.append(Vector2(duration, current))
		curve.best_points.append(Vector2(duration, longest))
		if curve.points.size() > 360:
			curve.points.remove_at(0)
			curve.best_points.remove_at(0)
	curve.queue_redraw()

func _show_unavailable() -> void:
	pause_button.disabled = true
	(find_child("SaveTrainingNow", true, false) as Button).disabled = true
	camera_button.hide()
	viewport_container.hide()
	curve.hide()
	if not holding_message:
		holding_message = Label.new()
		holding_message.name = "HoldingReadinessMessage"
		holding_message.text = "Holding practice is not ready yet.\n\nUse Pond menu to return to the working swimmer."
		holding_message.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		holding_message.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		holding_message.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
		holding_message.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		holding_message.size_flags_vertical = Control.SIZE_EXPAND_FILL
		holding_message.add_theme_font_size_override("font_size", 18)
		holding_message.add_theme_color_override("font_color", TEXT)
		visual.add_child(holding_message)

func _poll() -> void:
	super._poll()
	if not lab:
		return
	if not hold_api_available or not lab.get_error().is_empty():
		title_status.text = "Holding practice is not ready"
		feed_feedback.text = "This exercise needs a qualified calm-water controller."
		feed_feedback.tooltip_text = lab.get_error() if hold_api_available else "The native extension has no holding-practice API."
		save_status.text = "The working boulder swimmer and its training remain separate."
		_show_unavailable()
		return
	if not stats.get("hold_practice_mode", false):
		return
	title_status.text = "Learning live · body PPO %d" % stats.get("update", 0) if stats.get("learning_active", false) else ("Learning paused · swimmer runs live" if stats.get("paused", false) else "Preparing body PPO…")
	feed_feedback.text = "Assigned task · hold the marked shelter spot."
	if not str(stats.get("learning_error", "")).is_empty():
		title_status.text = "Background learning stopped"
		feed_feedback.text = "The visible saved controller is still running. Open AI details."
	water_status.text = "Calm water · holding before currents"
	current_switch.disabled = true
	current_strength.disabled = true
	curve.watch_button.hide()
	curve.plot_title = "Continuous time in cover"
	curve.plot_caption = "Native whole-body cover · actual seconds"
	curve.axis_label = "Seconds in this visit"
	curve.score_label = "Current hold"
	curve.empty_message = "Measuring this live swimmer…"
	curve.status_caption = "Assigned body-control lesson · PPO %d · actor %d" % [stats.get("update", 0), stats.get("adopted_update", 0)]
	curve.tooltip_text = "Amber is the current continuous protected hold; green is the longest measured hold in this saved exercise. These are live physical measurements, not food rates or proof of learned predator avoidance."
	training_heading.text = "Body PPO · assigned task"
	training_stats.text = "%s · %d background swimmers\nLearner update %d · visible actor %d\n\nPractice starts with holding in calm water while ordinary feeding skills are checked. The target is assigned by this exercise; the creature has not learned to choose shelter over food here. Its neural body controller produces every joint command.\n\nQualified candidates reach this same live body without a reset." % [stats.get("compute_backend", "CUDA"), stats.get("background_swimmers", population), stats.get("update", 0), stats.get("adopted_update", 0)]
	if not str(stats.get("learning_error", "")).is_empty():
		training_stats.text += "\nLearning stopped: " + str(stats.learning_error)
	save_status.text = "Separate holding save at %d.\nAutosaves every 30 s and on exit. Original pond saves preserved." % stats.get("saved_update", 0)
	if not str(stats.get("save_error", "")).is_empty():
		save_status.text = "Save failed: " + str(stats.save_error)
		feed_feedback.text = "The learning save failed. Open AI details."
	_update_session_curve(false)

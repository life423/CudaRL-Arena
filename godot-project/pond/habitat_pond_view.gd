extends "res://pond/deep_pond.gd"

# Shared habitat presentation. Native session choice is explicit configuration;
# arch/cove/boulder/ledge scenes do not inherit another experiment's lifecycle.
const HabitatMode = preload("res://pond/habitat_mode.gd")
const HabitatRender = preload("res://pond/habitat_render.gd")
const CurrentControls = preload("res://pond/pond_current_controls.gd")
const CurrentFlow = preload("res://pond/current_flow.gd")
const CURRENT_SPEEDS := [0.5, 0.75]
# Preserve exported properties used by developer fixtures and saved scenes.
# Only the selected mode's source/destination pair is ever opened.
@export var feeding_checkpoint_path := "user://pond-policies/feeding-learning-v1/training.bin"
@export var source_feeding_checkpoint := "user://pond-policies/feeding-learning-v1/training.bin"
@export var arch_checkpoint := "user://pond-policies/arch-navigation-v1/training.bin"
@export var source_arch_checkpoint := "user://pond-policies/arch-navigation-v1/training.bin"
@export var cove_checkpoint := "user://pond-policies/cove-navigation-v1/training.bin"
@export var source_cove_checkpoint := "user://pond-policies/cove-navigation-v1/training.bin"
@export var boulder_checkpoint := "user://pond-policies/boulder-navigation-v1/training.bin"
var habitat_mode: Dictionary = {}
var arch_geometry: Dictionary = {}
var native_outline := PackedVector3Array()
var native_terrain_positions := PackedVector3Array()
var native_terrain_samples := PackedVector3Array()
var rock_arch: MeshInstance3D
var rock_boulder: MeshInstance3D
var rock_overhang: Node3D
var flow: Node3D
var current_switch: CheckButton
var current_strength: OptionButton
var water_status: Label
var feed_feedback: Label
var pellet_pool: MultiMeshInstance3D
var visit_start_time := -1.0
var visit_start_eaten := 0
var overhang_api_available := false
var overhang_started := false
var next_debug_elapsed := 0.0
var _habitat_readiness_message: Label
# Compatibility actions used by the separate predator/holding prototypes.
# Automatic-food habitats never enable manual feeding or an auto-feed clock.
var auto_feed_button: CheckButton
var auto_feed_enabled := false
var auto_feed_next_time := -1.0
var feed_surface: Control

func _habitat_kind() -> String:
	return "arch"

func _create_lab() -> RefCounted:
	habitat_mode = HabitatMode.describe(_habitat_kind())
	var methods: Array[String] = ["get_error", "get_stats", "get_state", "step", "stop", "set_paused", "set_steady_current", str(habitat_mode.geometry), str(habitat_mode.start), "get_arch_flow_vectors", "advect_arch_tracers"]
	if int(habitat_mode.id) >= 1:
		methods.append("sample_cove_terrain")
	return _create_native_lab("DeepPondLab", methods)

func _prepare_lab() -> bool:
	if not lab or habitat_mode.is_empty():
		return false
	arch_geometry = _load_habitat_geometry()
	native_outline = HabitatRender.outline(arch_geometry)
	overhang_api_available = habitat_mode.kind != "overhang" or lab.has_method("watch_overhang")
	if arch_geometry.is_empty() or (int(habitat_mode.id) >= 1 and native_outline.is_empty()):
		startup_error = "The native %s habitat did not return its required geometry." % str(habitat_mode.kind)
		return false
	return true

func _load_habitat_geometry() -> Dictionary:
	return lab.call(habitat_mode.geometry)

func _start_lab() -> bool:
	paused = false
	var started: bool = HabitatMode.start_session(self, lab, habitat_mode)
	if habitat_mode.kind == "overhang":
		overhang_started = started
	return started

func _after_lab_started() -> void:
	super._after_lab_started()
	get_window().title = habitat_mode.window
	sidebar_title.text = habitat_mode.title
	released = lab != null and lab.get_error().is_empty()
	if habitat_mode.frozen and released:
		next_debug_elapsed = float(state.get("elapsed", 0.0)) + 5.0
		print("OVERHANG_LIVE_STARTED actor=%d learner_source=%d layout=3 training=0" % [stats.get("adopted_update", 0), stats.get("loaded_update", 0)])

func _build_ui() -> void:
	super._build_ui()
	untrained_button.hide()
	untrained_button.disabled = true
	var water: Dictionary = CurrentControls.build(self, sidebar_controls, camera_button)
	current_switch = water.switch
	current_strength = water.strength
	water_status = water.status
	feed_feedback = _label("Food replenishes across the habitat and drifts with the current.", 14, MUTED)
	sidebar_controls.add_child(feed_feedback)
	sidebar_controls.move_child(feed_feedback, 4)
	var action := _button("Drop food at pond center")
	action.name = "DropFoodAtCenter"
	action.hide()
	sidebar_controls.add_child(action)
	auto_feed_button = CheckButton.new()
	auto_feed_button.name = "AutoFeed"
	auto_feed_button.hide()
	auto_feed_button.disabled = true
	sidebar_controls.add_child(auto_feed_button)
	feed_surface = Control.new()
	feed_surface.name = "FeedingSurface"
	feed_surface.hide()
	feed_surface.mouse_filter = Control.MOUSE_FILTER_IGNORE
	feed_surface.focus_mode = Control.FOCUS_NONE
	viewport_container.add_child(feed_surface)
	speed_button.hide()
	current_switch.tooltip_text = "Change the actual water current without resetting the creature. Background PPO retains and checks existing swimming skills."
	current_strength.tooltip_text = "Choose 0.5 or 0.75 cm/s imposed flow. Native geometry drives the physical local field."
	swimmer_stats.tooltip_text = "Live native physical measurements. Geometry shown here matches the native habitat."
	if habitat_mode.get("frozen", false):
		speed_button.disabled = true
		(find_child("SaveTrainingNow", true, false) as Button).hide()
		pause_button.text = "Pause pond"
		pause_button.tooltip_text = "Pause live simulation. This saved-controller view does not update neural weights."

func _set_auto_feed(_enabled: bool) -> void:
	auto_feed_enabled = false
	auto_feed_next_time = -1
	if auto_feed_button:
		auto_feed_button.set_pressed_no_signal(false)

func _set_current(enabled: bool) -> void:
	if not lab or not lab.set_steady_current(CURRENT_SPEEDS[current_strength.selected] if enabled else 0.0, true):
		current_switch.set_pressed_no_signal(not enabled)
		water_status.text = "Current could not change"
		water_status.tooltip_text = lab.get_error() if lab else "The native pond is unavailable."
		return
	_poll()

func _select_strength(index: int) -> void:
	if not current_switch.button_pressed:
		return
	if not lab.set_steady_current(CURRENT_SPEEDS[index], true):
		current_strength.select(1 if stats.get("water_current_cm_s", 0.0) > 0.625 else 0)
		water_status.text = "Current could not change"
		water_status.tooltip_text = lab.get_error()
		return
	_poll()

func _stone_material() -> StandardMaterial3D:
	return HabitatRender.stone_material()

func _pond_radii() -> Vector2:
	return arch_geometry.get("radii", Vector2(6.9, 5.25))

func _pond_overview_direction() -> Vector3:
	return Vector3(0.22, 0.67, 0.74)

func _terrain() -> ArrayMesh:
	if native_outline.is_empty():
		return super._terrain()
	var result: Dictionary = HabitatRender.terrain(native_outline, Callable(lab, "sample_cove_terrain"))
	native_terrain_positions = result.positions
	native_terrain_samples = result.samples
	return result.mesh

func _water_mesh() -> ArrayMesh:
	return HabitatRender.water_mesh(native_outline)

func _create_shore() -> Node3D:
	if native_outline.is_empty():
		return super._create_shore()
	var decoration = preload("res://pond/natural_shore.gd").new()
	decoration.configure_outline(native_outline, Callable(lab, "sample_cove_terrain"))
	return decoration

func _configure_flow() -> void:
	var terrain := Callable(lab, "sample_cove_terrain") if not native_outline.is_empty() else Callable()
	flow.configure_habitat(_pond_radii(), Callable(lab, "advect_arch_tracers"), Callable(lab, "get_arch_flow_vectors"), arch_geometry, terrain)

func _build_world() -> void:
	super._build_world()
	creature.membrane_half_height = 0.15
	camera_direction = _pond_overview_direction().normalized()
	shore.scale = Vector3.ONE if not native_outline.is_empty() else Vector3(float(arch_geometry.get("scale", 1.5)), 1, float(arch_geometry.get("scale", 1.5)))
	if not native_outline.is_empty():
		water_surface.mesh = _water_mesh()
		water_surface.scale = Vector3.ONE
		water_surface.name = "NativeCoveWater"
	water_surface.set_meta("layout_id", int(habitat_mode.id))
	var solids: Dictionary = HabitatRender.solids(world, arch_geometry, _stone_material())
	rock_arch = solids.get("arch")
	rock_boulder = solids.get("boulder")
	rock_overhang = solids.get("overhang")
	flow = CurrentFlow.new()
	flow.configure(water_surface.material_override)
	world.add_child(flow)
	_configure_flow()
	for nodes in [foods, markers, lines, depth_labels]:
		for node: Node3D in nodes:
			node.hide()
	var mesh := SphereMesh.new()
	mesh.radius = 0.055
	mesh.height = 0.11
	mesh.radial_segments = 12
	mesh.rings = 6
	var material := _material(Color("ffc45d"))
	material.emission_enabled = true
	material.emission = Color("c89a43")
	material.emission_energy_multiplier = 0.6
	mesh.material = material
	var pool := MultiMesh.new()
	pool.transform_format = MultiMesh.TRANSFORM_3D
	pool.mesh = mesh
	pool.instance_count = 128
	pool.visible_instance_count = 0
	pellet_pool = MultiMeshInstance3D.new()
	pellet_pool.multimesh = pool
	world.add_child(pellet_pool)

func _pond_overview_target() -> Vector3:
	if native_outline.is_empty():
		return super._pond_overview_target()
	var minimum: Vector2 = arch_geometry.get("bounds_min", Vector2(-6.9, -5.25))
	var maximum: Vector2 = arch_geometry.get("bounds_max", Vector2(9, 5.25))
	var center := (minimum + maximum) * 0.5
	return Vector3(center.x, -1.3, center.y)

func _pond_fit_points() -> Array[Vector3]:
	if native_outline.is_empty():
		return super._pond_fit_points()
	var points: Array[Vector3] = []
	var center := _pond_overview_target()
	for point in native_outline:
		points.append(center + Vector3((point.x - center.x) * 1.14, 2.1, (point.z - center.z) * 1.14))
	points.append(Vector3(0, -4, 0))
	points.append(arch_geometry.get("cove_center", Vector3(5.8, -3, -1.5)))
	return points

func _apply_pose(alpha: float) -> void:
	if state.is_empty() or not pellet_pool:
		return
	var links: Array = []
	for index in 3:
		links.append({"position": (previous.links[index].position as Vector3).lerp(state.links[index].position, alpha), "orientation": (previous.links[index].orientation as Quaternion).slerp(state.links[index].orientation, alpha)})
	creature.pose(links)
	rendered_swimmer_position = links[0].position
	visibility_focus = (links[0].position as Vector3).lerp(links[2].position, 0.5)
	var positions: PackedVector3Array = state.get("foods", PackedVector3Array())
	pellet_pool.multimesh.visible_instance_count = positions.size()
	for index in positions.size():
		pellet_pool.multimesh.set_instance_transform(index, Transform3D(Basis.IDENTITY, positions[index]))

func _physics_process(delta: float) -> void:
	if habitat_mode.get("frozen", false) and (not overhang_started or paused):
		return
	var before: float = state.get("elapsed", 0)
	super._physics_process(delta)
	if flow:
		flow.advance(float(state.get("elapsed", 0)) - before)
	if habitat_mode.get("frozen", false) and "--overhang-status" in OS.get_cmdline_user_args() and float(state.get("elapsed", 0)) >= next_debug_elapsed:
		print("OVERHANG_LIVE_TICK elapsed=%.2f food=%d position=%s training=0" % [state.elapsed, state.eaten, str(state.position)])
		next_debug_elapsed = float(state.elapsed) + 5

func _toggle_pause() -> void:
	if not habitat_mode.get("frozen", false):
		super._toggle_pause()
		return
	if not lab or state.is_empty():
		return
	paused = not paused
	lab.set_paused(paused)
	previous = state.duplicate(true)
	accumulator = 0
	_apply_pose(1)
	_poll()

func _swimmer_text() -> String:
	return "Food collected  %d\nArch passages  %d\nDepth  %.1f cm\nSpeed  %.1f cm/s" % [state.get("eaten", 0), stats.get("arch_passages", 0), state.get("depth_cm", 0), state.get("speed_cm_s", 0)]

func _update_session_curve(untrained: bool) -> void:
	if int(habitat_mode.get("id", 0)) < 2:
		super._update_session_curve(untrained)
		return
	var elapsed: float = state.get("elapsed", 0)
	var eaten: int = state.get("eaten", 0)
	if visit_start_time < 0 or elapsed < visit_start_time or eaten < visit_start_eaten:
		visit_start_time = elapsed
		visit_start_eaten = eaten
		curve.points.clear()
	var duration := elapsed - visit_start_time
	curve.best_points.clear()
	curve.best_score = -1
	curve.show_best_score = false
	curve.current_score = (eaten - visit_start_eaten) * 60.0 / duration if duration >= 1 else -1
	if duration >= 1 and (curve.points.is_empty() or floori(duration / 5) > floori(curve.points[-1].x / 5)):
		curve.points.append(Vector2(duration, curve.current_score))
		if curve.points.size() > 360:
			curve.points.remove_at(0)
	last_session_time = elapsed
	curve.queue_redraw()

func _show_readiness_error(message: String) -> void:
	overhang_started = false
	title_status.text = "Rock ledge view could not start"
	feed_feedback.text = message
	for control in [pause_button, current_switch, current_strength]:
		control.disabled = true
	for control in [camera_button, viewport_container, curve, details_button, details_group]:
		control.hide()
	if not _habitat_readiness_message:
		_habitat_readiness_message = _label("The live swimmer is unavailable.\n\nUse Pond menu to return to the working pond.", 18)
		_habitat_readiness_message.name = "OverhangReadinessMessage"
		_habitat_readiness_message.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		_habitat_readiness_message.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
		_habitat_readiness_message.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		_habitat_readiness_message.size_flags_vertical = Control.SIZE_EXPAND_FILL
		visual.add_child(_habitat_readiness_message)

func _poll() -> void:
	if not lab or not title_status:
		return
	if habitat_mode.get("frozen", false):
		_poll_frozen()
		return
	super._poll()
	if not lab.get_error().is_empty():
		title_status.text = "%s learning could not start" % str(habitat_mode.get("kind", "Habitat")).capitalize()
		feed_feedback.text = lab.get_error()
		return
	if not stats.get("arch_mode", false):
		return
	curve.watch_button.hide()
	untrained_button.hide()
	pause_button.show()
	(find_child("SaveTrainingNow", true, false) as Button).show()
	var learning: bool = stats.get("learning_active", false)
	title_status.text = "Learning live · PPO update %d" % stats.get("update", 0) if learning else ("Learning paused" if stats.get("paused", false) else "Preparing background PPO…")
	pause_button.text = "Resume learning" if stats.get("paused", false) else "Pause learning"
	_update_water_controls()
	var kind: String = habitat_mode.kind
	feed_feedback.text = "Food replenishes across the arch and drifts with the current." if kind == "arch" else ("Food explores the main pond and cove." if kind == "cove" else "Food explores the main pond, cove and routes around the boulder.")
	training_heading.text = "PPO · obstacle navigation" if kind == "arch" else "PPO · %s navigation" % kind
	var comparison := "Current settings awaiting a matched comparison."
	if int(stats.get("evaluation_count", 0)) > 0 and int(stats.get("evaluated_context_version", -1)) == int(stats.get("context_version", 0)):
		comparison = "Matched candidate %.1f / min\nCalm arch %.1f / min · reference %.1f" % [stats.get("current_food_per_minute", 0), stats.get("arch_calm_food_per_minute", 0), stats.get("arch_calm_reference_food_per_minute", 0)]
	var actor := "inherited swimmer" if int(stats.get("adopted_update", 0)) == 0 else "validated swimmer %d" % int(stats.adopted_update)
	training_stats.text = "%s · %d background swimmers\nLearner update %d\nVisible: %s\nCurrent change %d · practiced %d\n%d swimming observations\nRock contact steps  %d\n%s\nPractice retains calm water and earlier habitats; qualified updates reach this same swimmer." % [stats.get("compute_backend", ""), stats.get("background_swimmers", population), stats.get("update", 0), actor, stats.get("context_version", 0), stats.get("trained_context_version", 0), stats.get("policy_inputs", 55), stats.get("arch_contacts", 0), comparison]
	var imported: int = stats.get("parent_update", 0) if kind == "arch" else stats.get("parent_stage_update", stats.get("loaded_update", 0))
	save_status.text = "Swimming retained from update %d. Original %s save preserved.\nSeparate %s save at %d · autosaves every 30 s and on exit." % [imported, habitat_mode.parent, kind, stats.get("saved_update", 0)]
	curve.empty_message = "Watching inherited swimming around the habitat…"
	curve.plot_title = "Live collection rate"
	curve.plot_caption = "This session · cumulative food / minute" if kind != "boulder" else "This visit · actual food / minute"
	curve.axis_label = "Elapsed seconds" if kind != "boulder" else "Seconds in this visit"
	curve.score_label = "Session" if kind != "boulder" else "This visit"
	curve.show_best_score = false
	curve.status_caption = "PPO %d · %s · session collection" % [stats.get("update", 0), actor]
	curve.tooltip_text = "Actual food collection in this live session. Background PPO and its matched policy checks are separate; collection alone does not establish reliable navigation or shelter use."
	if not str(stats.get("save_error", "")).is_empty():
		save_status.text = "Save failed: " + str(stats.save_error)
		feed_feedback.text = "Learning save failed. Open AI details."
	if not str(stats.get("learning_error", "")).is_empty():
		feed_feedback.text = "Background learning stopped. Open AI details."
		training_stats.text += "\nLearning stopped: " + str(stats.learning_error)

func _update_water_controls() -> void:
	var speed := float(stats.get("water_current_cm_s", 0))
	flow.set_current(stats.get("water_current", Vector3.ZERO))
	current_switch.set_pressed_no_signal(speed > 0)
	current_strength.disabled = speed <= 0
	current_strength.select(1 if speed > 0.625 else 0)
	water_status.text = "%.2f cm/s · food drifts" % speed if speed > 0 else "Calm water"
	water_status.tooltip_text = "Native local flow drives body drag, food and displayed tracers. It is a prescribed reduced-order field, not CFD."

func _poll_frozen() -> void:
	stats = lab.get_stats()
	var error: String = lab.get_error()
	if not overhang_api_available or not error.is_empty() or state.is_empty():
		_show_readiness_error(error if not error.is_empty() else "The live rock ledge view is unavailable.")
		return
	title_status.text = "Pond paused" if paused else "Live · trained swimmer"
	pause_button.show()
	pause_button.disabled = false
	pause_button.text = "Resume pond" if paused else "Pause pond"
	swimmer_stats.text = _swimmer_text()
	speed_button.hide()
	(find_child("SaveTrainingNow", true, false) as Button).hide()
	curve.watch_button.hide()
	untrained_button.hide()
	_update_water_controls()
	current_switch.disabled = false
	current_switch.tooltip_text = "Change water flow without resetting the creature. Saved neural-network weights remain frozen."
	current_strength.tooltip_text = "Choose 0.5 or 0.75 cm/s steady flow. This scene does not perform PPO updates."
	feed_feedback.text = "New bank ledge · an open shelter under real rock."
	details_button.show()
	details_group.visible = details_expanded
	training_heading.text = "Saved swimming controller"
	training_stats.text = "Native %s physics and neural inference.\nInherited boulder swimmer · both networks frozen.\nNo background training or learning updates.\nThe ledge changes real geometry. Hiding behavior has not passed its learning test." % stats.get("compute_backend", "")
	save_status.text = "The source boulder training save is read only in this view."
	curve.plot_title = "Live collection rate"
	curve.status_caption = "Saved weights · live physics · no learning updates"
	curve.score_label = "This visit"
	curve.plot_caption = "Food collected during this visit · actual food / minute"
	curve.axis_label = "Seconds in this visit"
	curve.empty_message = "Collection appears after the first second."
	curve.tooltip_text = "Session collection is a physical observation, not evidence of learning or shelter use."
	_update_session_curve(false)

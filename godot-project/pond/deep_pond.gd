extends "res://pond/deep_water.gd"

const PondExperiments = preload("res://launcher/pond_experiments.gd")
@export var swimming_checkpoint := "user://pond-policies/deep-v1/training.bin"
@export var controller_policy_path := "res://pond/policies/deep_selector.bin"
var foods: Array[MeshInstance3D] = []
var markers: Array[MeshInstance3D] = []
var lines: Array[MeshInstance3D] = []
var depth_labels: Array[Label3D] = []
var water_surface: MeshInstance3D
var untrained_button: CheckButton
var terrain_material: ShaderMaterial
var visibility_focus := Vector3(0, -2, 0)
var rendered_swimmer_position := Vector3(0, -2, 0)
var shore: Node3D
var pond_sun: DirectionalLight3D
var underwater_environment: Environment
var follow_focus := Vector3.ZERO
var camera_mode_initialized := false
var previous_camera_overview := true
var fitted_viewport_size := Vector2i.ZERO
var fitted_overview_distance := 13.8
var experiments: AcceptDialog
var details_button: Button
var details_group: VBoxContainer
var details_expanded := false
var last_session_time := -1.0
const RX := 4.6
const RZ := 3.5
const NATURAL_OVERVIEW_DIRECTION := Vector3(0.20, 0.87, 0.45)

func _init() -> void:
	checkpoint_path = "user://pond-policies/deep-pond-v1/training.bin"
	overview = true
	camera_direction = Vector3(0.22, 0.55, 0.81).normalized()
	overview_distance = 13.8

func _ready() -> void:
	super._ready()
	get_window().title = "CudaRL · Original swimmer · live" if stats.get("playback_mode", false) else "CudaRL · Deep pond · training"
	(sidebar_scroll.get_child(0).get_child(0) as Label).text = "Tadpole pond"
	camera_button.text = "Closer view"
	camera_button.pressed.disconnect(camera_button.pressed.get_connections()[0].callable)
	camera_button.pressed.connect(func(): overview = not overview; camera_button.text = "Closer view" if overview else "Pond overview")
	camera_button.tooltip_text = "Pond overview is stationary. Closer view uses a steady three-quarter angle and fixed zoom with gentle tracking."
	curve.empty_message = "Evaluating transferred swimming on three foods…"
	curve.watch_button.visible = true
	curve.watch_button.disabled = false
	curve.watch_button.set_pressed_no_signal(true)
	curve.watch_button.text = "Watch latest"
	curve.watch_button.tooltip_text = "The pond shows the strongest evaluated controller while training continues. Switch to inspect the latest learner."
	curve.watch_button.toggled.connect(func(enabled: bool): lab.set_watch_best(enabled))
	_poll()

func _create_lab() -> RefCounted:
	return DeepPondLab.new()

func _build_ui() -> void:
	super._build_ui()
	untrained_button = CheckButton.new()
	untrained_button.name = "ShowUntrainedPolicy"
	untrained_button.text = "Show untrained policy"
	untrained_button.custom_minimum_size.y = 44
	untrained_button.add_theme_font_size_override("font_size", 14)
	untrained_button.add_theme_color_override("font_color", TEXT)
	untrained_button.add_theme_stylebox_override("focus", camera_button.get_theme_stylebox("focus"))
	untrained_button.tooltip_text = "Use random initial weights for swimming and food choice, with the same physics. Each switch restarts the same pond layout. No learning or saved weights change."
	untrained_button.toggled.connect(_toggle_untrained)
	var controls := camera_button.get_parent()
	controls.add_child(untrained_button)
	controls.move_child(camera_button, 3)
	controls.move_child(untrained_button, 4)
	swimmer_stats.tooltip_text = "Live physical measurements. Food depth markers appear in the pond overview."
	details_button = _button("AI details")
	details_button.name = "PondDetails"
	details_button.toggle_mode = true
	details_button.toggled.connect(func(enabled: bool):
		details_expanded = enabled
		details_button.text = "Hide AI details" if enabled else "AI details"
		details_group.visible = enabled)
	controls.add_child(details_button)
	details_group = VBoxContainer.new()
	details_group.add_theme_constant_override("separation", 8)
	controls.add_child(details_group)
	for item in [training_heading, training_stats, save_status]:
		item.reparent(details_group)
	details_group.hide()
	var more := _button("Experiments")
	more.name = "PondExperiments"
	more.tooltip_text = "Other live agents, the learning lab and recorded diagnostics."
	more.pressed.connect(_show_experiments)
	controls.add_child(more)

func _show_experiments() -> void:
	if experiments == null:
		experiments = PondExperiments.new()
		add_child(experiments)
	experiments.open()

func _toggle_untrained(enabled: bool) -> void:
	if not lab or not lab.set_untrained(enabled):
		untrained_button.set_pressed_no_signal(not enabled)
		return
	state = lab.get_state()
	previous = state.duplicate(true)
	accumulator = 0.0
	_poll()
	_apply_pose(1.0)

func _start_lab() -> bool:
	if checkpoint_path == "user://pond-policies/deep-pond-v1/training.bin" and not controller_policy_path.is_empty():
		paused = true
		var backend := "auto"
		for argument in OS.get_cmdline_user_args():
			if argument.begins_with("--pond-backend="):
				backend = argument.get_slice("=", 1)
		return lab.watch_controller(controller_policy_path, backend)
	return lab.start_training(population, training_seed, checkpoint_path, swimming_checkpoint)

func _floor_y(radius: float) -> float:
	return -4.0 + 4.5 * pow(minf(radius, 1), 6)

func _visual_height(x: float, z: float) -> float:
	var radius := sqrt(x * x / (RX * RX) + z * z / (RZ * RZ))
	return _floor_y(radius) + smoothstep(1.02, 1.65, radius) * 0.10 * sin(x * 0.7) * cos(z * 0.8)

func _terrain() -> ArrayMesh:
	var tool := SurfaceTool.new()
	tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	for ring in 129:
		# Retain the physical basin and extend the dry ground beyond the view.
		var r := float(ring) / 64 * 1.2 if ring <= 64 else 1.2 + float(ring - 64) / 64 * 10.8
		for spoke in 129:
			var angle := TAU * float(spoke) / 128
			tool.set_uv(Vector2(cos(angle), sin(angle)) * r * 0.5 + Vector2.ONE * 0.5)
			var x := cos(angle) * RX * r
			var z := sin(angle) * RZ * r
			# Exterior dry ground is decorative; underwater geometry stays exact.
			var height := _visual_height(x, z)
			var slope_x := (_visual_height(x + 0.005, z) - _visual_height(x - 0.005, z)) / 0.01
			var slope_z := (_visual_height(x, z + 0.005) - _visual_height(x, z - 0.005)) / 0.01
			tool.set_normal(Vector3(-slope_x, 1, -slope_z).normalized())
			tool.add_vertex(Vector3(x, height, z))
	for ring in 128:
		for spoke in 128:
			var a := ring * 129 + spoke
			for index in [a, a + 129, a + 1, a + 1, a + 129, a + 130]:
				tool.add_index(index)
	return tool.commit()

func _build_world() -> void:
	world = Node3D.new()
	viewport.add_child(world)
	var environment := Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = Color("15302e")
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.ambient_light_color = Color("91b9b5")
	environment.ambient_light_energy = 0.8
	var sky := Sky.new()
	sky.radiance_size = Sky.RADIANCE_SIZE_128
	var daylight := ShaderMaterial.new()
	daylight.shader = preload("res://pond/deep_daylight.gdshader")
	sky.sky_material = daylight
	environment.sky = sky
	environment.reflected_light_source = Environment.REFLECTION_SOURCE_SKY
	environment.tonemap_mode = Environment.TONE_MAPPER_AGX
	environment.ssao_enabled = RenderingServer.get_current_rendering_method() == "forward_plus"
	var settings := WorldEnvironment.new()
	settings.environment = environment
	world.add_child(settings)
	underwater_environment = environment.duplicate()
	underwater_environment.background_color = Color("12393d")
	underwater_environment.fog_enabled = true
	underwater_environment.fog_light_color = Color("277876")
	underwater_environment.fog_light_energy = 0.35
	underwater_environment.fog_density = 0.028
	underwater_environment.fog_sky_affect = 0.3
	var sun := DirectionalLight3D.new()
	pond_sun = sun
	sun.rotation_degrees = Vector3(-65, 195, 0)
	sun.light_color = Color("fff0d2")
	sun.light_energy = 1.5
	sun.shadow_enabled = true
	sun.set_param(Light3D.PARAM_SIZE, 2.0)
	sun.set_param(Light3D.PARAM_SPECULAR, 0.12)
	sun.set_param(Light3D.PARAM_SHADOW_OPACITY, 0.26)
	world.add_child(sun)
	var fill := DirectionalLight3D.new()
	fill.rotation_degrees = Vector3(-35, 150, 0)
	fill.light_color = Color("8fc9d1")
	fill.light_energy = 0.4
	world.add_child(fill)
	terrain_material = ShaderMaterial.new()
	terrain_material.shader = preload("res://pond/deep_basin.gdshader")
	var basin := _terrain()
	var terrain := _mesh(basin, terrain_material, Vector3.ZERO)
	# Receive real creature/shore shadows without coarse landscape self-shadow
	# artifacts obscuring the small swimmer.
	terrain.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	var surface := CylinderMesh.new()
	surface.top_radius = pow(4.0 / 4.5, 1.0 / 6.0)
	surface.bottom_radius = surface.top_radius
	surface.height = 0.008
	surface.radial_segments = 96
	var water_material := ShaderMaterial.new()
	water_material.shader = preload("res://pond/deep_surface.gdshader")
	water_material.render_priority = 1
	water_surface = _mesh(surface, water_material, Vector3.ZERO)
	water_surface.scale = Vector3(RX, 1, RZ)
	water_surface.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	shore = preload("res://pond/natural_shore.gd").new()
	world.add_child(shore)
	creature = Creature.new()
	world.add_child(creature)
	for i in 3:
		var piece := _sphere(0.09, Color("ffc45d"), Vector3.ZERO)
		var material := piece.material_override as StandardMaterial3D
		material.emission_enabled = true
		material.emission = Color("c89a43")
		material.emission_energy_multiplier = 0.6
		foods.append(piece)
		var marker := CylinderMesh.new()
		marker.top_radius = 0.075
		marker.bottom_radius = 0.075
		marker.height = 0.008
		markers.append(_mesh(marker, _material(Color("aa9961")), Vector3.ZERO))
		var line := CylinderMesh.new()
		line.top_radius = 0.006
		line.bottom_radius = 0.006
		line.height = 1
		var line_material := _material(Color(0.85, 0.76, 0.45, 0.20))
		line_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		var guide := _mesh(line, line_material, Vector3.ZERO)
		guide.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		lines.append(guide)
		var depth := Label3D.new()
		depth.font_size = 38
		depth.pixel_size = 0.006
		depth.billboard = BaseMaterial3D.BILLBOARD_ENABLED
		depth.no_depth_test = true
		depth.modulate = Color("efdba6")
		world.add_child(depth)
		depth_labels.append(depth)
	food = foods[0]
	camera = Camera3D.new()
	camera.fov = 44
	camera.near = 0.05
	camera.far = 120
	camera.position = Vector3(0.4, -1.4, 3.8)
	world.add_child(camera)
	camera.look_at(Vector3(0, -1.4, 0))
	camera.current = true

func _apply_pose(alpha: float) -> void:
	var links: Array = []
	for i in 3:
		links.append({"position": (previous.links[i].position as Vector3).lerp(state.links[i].position, alpha), "orientation": (previous.links[i].orientation as Quaternion).slerp(state.links[i].orientation, alpha)})
	creature.pose(links)
	rendered_swimmer_position = links[0].position
	visibility_focus = (links[0].position as Vector3).lerp(links[2].position, 0.5)
	for i in 3:
		foods[i].position = state.foods[i]
		var p := foods[i].position
		var radius := sqrt(p.x * p.x / (RX * RX) + p.z * p.z / (RZ * RZ))
		markers[i].position = Vector3(p.x, _floor_y(radius) + 0.012, p.z)
		lines[i].position = (p + markers[i].position) * 0.5
		lines[i].scale.y = maxf(0.01, p.y - markers[i].position.y)
		depth_labels[i].position = p + Vector3(0, 0.22, 0)
		depth_labels[i].text = "%.1f cm" % (-p.y * 4)
		markers[i].visible = overview
		lines[i].visible = overview
		depth_labels[i].visible = overview

func _swimmer_text() -> String:
	return "Food collected  %d\nDepth  %.1f cm\nSpeed  %.1f cm/s\nSince food  %.1f s" % [state.eaten, state.depth_cm, state.speed_cm_s, state.seconds_since_food]

func _camera_target() -> Vector3:
	return Vector3(0, -1.4, 0) if overview else state.position

func _fit_view_distance(points: Array[Vector3], target: Vector3, direction: Vector3) -> float:
	var aspect := maxf(0.25, float(viewport.size.x) / maxf(1, viewport.size.y))
	var basis := Basis.looking_at(-direction, Vector3.UP)
	var vertical := tan(deg_to_rad(camera.fov * 0.5)) * 0.88
	var horizontal := vertical * aspect
	var distance := 0.0
	for point in points:
		var delta := point - target
		distance = maxf(distance, delta.dot(direction) + maxf(absf(delta.dot(basis.x)) / horizontal, absf(delta.dot(basis.y)) / vertical))
	return distance + 0.35

func _update_camera(delta: float) -> void:
	if state.is_empty():
		return
	var target := Vector3(0, -1.3, 0)
	var direction := NATURAL_OVERVIEW_DIRECTION.normalized() if overview else camera_direction
	var mode_changed := not camera_mode_initialized or previous_camera_overview != overview
	if fitted_viewport_size != viewport.size:
		var points: Array[Vector3] = []
		for i in 32:
			var angle := TAU * float(i) / 32
			points.append(Vector3(cos(angle) * RX * 1.18, 0.8, sin(angle) * RZ * 1.18))
			for radius in [0.0, 0.35, 0.65, 0.85, 0.97]:
				points.append(Vector3(cos(angle) * RX * radius, _floor_y(radius) + 0.1, sin(angle) * RZ * radius))
		fitted_overview_distance = maxf(9.0, _fit_view_distance(points, target, NATURAL_OVERVIEW_DIRECTION.normalized()))
		fitted_viewport_size = viewport.size
	var distance := fitted_overview_distance
	if not overview:
		var desired_focus: Vector3 = rendered_swimmer_position + Vector3(0, -0.15, 0)
		if mode_changed:
			follow_focus = desired_focus
		var offset := desired_focus - follow_focus
		# Let small movements happen within the frame. Neither joint pose nor
		# heading affects camera angle, target offset or zoom.
		if offset.length() > 0.45:
			follow_focus += offset.normalized() * (offset.length() - 0.45) * (1 - exp(-delta / 0.4))
		if desired_focus.distance_to(follow_focus) > 1.0:
			follow_focus = desired_focus - (desired_focus - follow_focus).normalized()
		target = follow_focus
		var aspect := maxf(0.25, float(viewport.size.x) / maxf(1, viewport.size.y))
		distance = 7.2 * maxf(1.0, 0.75 / aspect)
	camera.position = target + direction * distance
	camera.look_at(target)
	var focus_pixel := camera.unproject_position(visibility_focus)
	var radius_pixel := camera.unproject_position(visibility_focus + camera.basis.y * 1.1)
	terrain_material.set_shader_parameter("focus_uv", focus_pixel / Vector2(viewport.size))
	terrain_material.set_shader_parameter("focus_depth", camera.to_local(visibility_focus).z)
	terrain_material.set_shader_parameter("focus_radius", focus_pixel.distance_to(radius_pixel) / maxf(1, viewport.size.y))
	terrain_material.set_shader_parameter("visibility_fade", not camera.is_position_behind(visibility_focus))
	shore.update_visibility(focus_pixel / Vector2(viewport.size), camera.to_local(visibility_focus).z, focus_pixel.distance_to(radius_pixel) / maxf(1, viewport.size.y), not camera.is_position_behind(visibility_focus))
	camera.environment = underwater_environment if camera.position.y < 0 else null
	camera_mode_initialized = true
	previous_camera_overview = overview

func _poll() -> void:
	super._poll()
	details_button.visible = stats.get("playback_mode", false)
	details_group.visible = details_expanded or not stats.get("playback_mode", false)
	if untrained_button:
		untrained_button.visible = stats.get("playback_mode", false)
		untrained_button.set_pressed_no_signal(stats.get("showcase_untrained", false))
	if stats.get("playback_mode", false):
		var untrained: bool = stats.get("showcase_untrained", false)
		title_status.text = "Live · random initial weights" if untrained else "Live · original saved agent"
		pause_button.visible = false
		speed_button.visible = false
		find_child("SaveTrainingNow", true, false).visible = false
		training_heading.text = "Untrained controller" if untrained else "Saved controller"
		training_stats.text = "AI and physics · %s\nRandom swimming + food choice\nNo learning updates" % stats.get("compute_backend", "CPU") if untrained else "AI and physics · %s\nSelector update %d\nSwimming retained from %d\nFood choice every %.1f s" % [stats.get("compute_backend", "CPU"), stats.selector_update, stats.policy_update, stats.decision_steps * 0.05]
		training_stats.tooltip_text = stats.get("compute_detail", "")
		save_status.text = "Both networks use random initial weights.\nSwitching restarts the same pond layout." if untrained else "Validated controller loaded.\nFull training checkpoints preserved."
		curve.watch_button.visible = false
		_update_session_curve(untrained)
		return
	if curve.watch_button:
		curve.watch_button.set_pressed_no_signal(stats.get("watching_best", true))
		curve.watch_button.text = "Watch latest" if stats.get("watching_best", true) else "Watch best"
	if stats.get("transferred", false):
		save_status.text = "Swimming retained from %d\nSaved at %d · autosaves every 30 s." % [stats.loaded_update, stats.saved_update]

func _update_session_curve(untrained: bool) -> void:
	var elapsed: float = state.get("elapsed", 0)
	if last_session_time < 0 or elapsed < last_session_time:
		curve.points.clear()
	curve.best_points.clear()
	curve.best_score = -1
	curve.show_best_score = false
	curve.plot_title = "Live collection rate"
	curve.status_caption = "Random weights · no learning updates" if untrained else "Saved PPO weights · no learning updates"
	curve.plot_caption = "This session · cumulative food / minute"
	curve.axis_label = "Elapsed seconds"
	curve.score_label = "Session"
	curve.empty_message = "Watching the live swimmer…"
	curve.tooltip_text = "Measured from food actually collected in this live session. This is not a training curve or a fixed validation score."
	curve.current_score = state.eaten * 60.0 / elapsed if elapsed >= 1 else -1
	if elapsed >= 1 and (curve.points.is_empty() or floori(elapsed / 5) > floori(curve.points[-1].x / 5)):
		curve.points.append(Vector2(elapsed, curve.current_score))
		if curve.points.size() > 360:
			curve.points.remove_at(0)
	last_session_time = elapsed
	curve.queue_redraw()

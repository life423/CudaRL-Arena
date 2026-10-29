extends "res://pond/feeding_pond.gd"

@export var source_feeding_checkpoint := "user://pond-policies/feeding-learning-v1/training.bin"
@export var arch_checkpoint := "user://pond-policies/arch-navigation-v1/training.bin"
var arch_geometry: Dictionary = {}
var rock_arch: MeshInstance3D

func _ready() -> void:
	# The native habitat also supplies the rendered dimensions. Fetch it before
	# the inherited setup builds the basin, surface, shore and camera.
	lab = DeepPondLab.new()
	arch_geometry = lab.get_arch_geometry()
	super._ready()
	get_window().title = "Tadpole Pond · Rock arch learning · CudaRL"
	(sidebar_scroll.get_child(0).get_child(0) as Label).text = "Rock arch pond"
	_poll()

func _create_lab() -> RefCounted:
	return lab

func _pond_radii() -> Vector2:
	return arch_geometry.get("radii", Vector2(6.9, 5.25))

func _pond_overview_direction() -> Vector3:
	return Vector3(0.22, 0.67, 0.74)

func _start_lab() -> bool:
	paused = false
	var backend := "auto"
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--pond-backend="):
			backend = argument.get_slice("=", 1)
	return lab.start_arch_learning(source_feeding_checkpoint, arch_checkpoint, population, backend)

func _build_ui() -> void:
	super._build_ui()
	# This first navigation task controls food and current in native simulation.
	# Keep the public feeding scene's interactions unchanged in its own scene.
	find_child("WaterControls", true, false).hide()
	find_child("DropFoodAtCenter", true, false).hide()
	speed_button.hide()
	auto_feed_button.hide()
	auto_feed_button.disabled = true
	feed_surface.hide()
	feed_surface.focus_mode = Control.FOCUS_NONE
	feed_surface.mouse_filter = Control.MOUSE_FILTER_IGNORE
	feed_feedback.text = "Food replenishes on both sides of the rock arch. Calm water."
	feed_feedback.tooltip_text = "Two suspended foods are placed by the native simulation, clear of solid rock. This controlled task preserves the original feeding pond."
	swimmer_stats.tooltip_text = "Live physical measurements. A passage counts only when the swimmer crosses the native arch opening. Rock contact is handled by the same CPU/CUDA physics that runs training."

func _set_auto_feed(_enabled: bool) -> void:
	auto_feed_enabled = false
	auto_feed_next_time = -1.0
	if auto_feed_button:
		auto_feed_button.set_pressed_no_signal(false)

func _set_current(_enabled: bool) -> void:
	# Ignore developer launch flags inherited from the gentle-current scene.
	# Currents are intentionally outside this first obstacle experiment.
	if current_switch:
		current_switch.set_pressed_no_signal(false)

func _stone_material() -> StandardMaterial3D:
	var noise := FastNoiseLite.new()
	noise.seed = 614
	noise.frequency = 0.045
	noise.fractal_octaves = 4
	var colors := Gradient.new()
	colors.colors = PackedColorArray([Color("686d60"), Color("a0a18b"), Color("c2b99a")])
	colors.offsets = PackedFloat32Array([0.0, 0.58, 1.0])
	var color_texture := NoiseTexture2D.new()
	color_texture.width = 256
	color_texture.height = 256
	color_texture.seamless = true
	color_texture.noise = noise
	color_texture.color_ramp = colors
	var normal_texture := NoiseTexture2D.new()
	normal_texture.width = 256
	normal_texture.height = 256
	normal_texture.seamless = true
	normal_texture.noise = noise
	normal_texture.as_normal_map = true
	normal_texture.bump_strength = 1.4
	var material := StandardMaterial3D.new()
	material.albedo_texture = color_texture
	material.normal_enabled = true
	material.normal_texture = normal_texture
	material.normal_scale = 0.45
	material.roughness = 0.94
	material.metallic_specular = 0.2
	material.uv1_scale = Vector3(3, 2, 1)
	return material

func _build_world() -> void:
	super._build_world()
	# Match the native tail membrane's 6 mm physical half-height at 4 cm/unit.
	creature.membrane_half_height = 0.15
	var landscape_scale: float = arch_geometry.get("scale", 1.5)
	shore.scale = Vector3(landscape_scale, 1, landscape_scale)
	camera_direction = _pond_overview_direction().normalized()
	var major: float = arch_geometry.get("major_radius", 1.35)
	var tube: float = arch_geometry.get("tube_radius", 0.5)
	var mesh := TorusMesh.new()
	mesh.inner_radius = major - tube
	mesh.outer_radius = major + tube
	mesh.rings = 96
	mesh.ring_segments = 32
	rock_arch = MeshInstance3D.new()
	rock_arch.name = "PhysicalRockArch"
	rock_arch.mesh = mesh
	rock_arch.material_override = _stone_material()
	# TorusMesh circles the Y axis. The native solid circles local Z; preserve
	# that exact torus and vary surface normals/color rather than its vertices.
	rock_arch.basis = Basis(Vector3.UP, float(arch_geometry.get("yaw", 0.0))) * Basis(Vector3.RIGHT, PI * 0.5)
	rock_arch.position = arch_geometry.get("center", Vector3(0, -2.25, 0))
	world.add_child(rock_arch)

func _swimmer_text() -> String:
	return "Food collected  %d\nArch passages  %d\nDepth  %.1f cm\nSpeed  %.1f cm/s" % [state.get("eaten", 0), stats.get("arch_passages", 0), state.get("depth_cm", 0.0), state.get("speed_cm_s", 0.0)]

func _poll() -> void:
	super._poll()
	if not lab:
		return
	if not lab.get_error().is_empty():
		title_status.text = "Arch learning could not start"
		feed_feedback.text = lab.get_error()
		return
	if not stats.get("arch_mode", false):
		return
	curve.watch_button.hide()
	feed_feedback.text = "Food replenishes on both sides of the rock arch. Calm water."
	training_heading.text = "PPO · obstacle navigation"
	var comparison := "First matched candidate check pending."
	if int(stats.get("evaluation_count", 0)) > 0:
		comparison = "Matched check: candidate %.1f / min\nInherited swimmer %.1f / min" % [stats.get("current_food_per_minute", 0.0), stats.get("original_food_per_minute", 0.0)]
	var actor_label := "inherited swimmer" if int(stats.get("adopted_update", 0)) == 0 else "validated arch %d" % int(stats.adopted_update)
	training_stats.text = "%s · %d background swimmers\nLearner update %d\nVisible: %s\n%d swimming observations\nRock contacts  %d\n%s\nCalm water; suspended food on opposite sides. The motor learns from inherited weights; food selection remains inherited." % [stats.get("compute_backend", ""), stats.get("background_swimmers", population), stats.get("update", 0), actor_label, stats.get("policy_inputs", 55), stats.get("arch_contacts", 0), comparison]
	save_status.text = "Swimming inherited from update %d. Original feeding save preserved.\nSeparate arch save at %d · autosaves every 30 s and on exit." % [stats.get("parent_update", 0), stats.get("saved_update", 0)]
	if not str(stats.get("save_error", "")).is_empty():
		save_status.text = "Save failed: " + str(stats.save_error)
		feed_feedback.text = "Learning save failed. Open AI details."
	curve.empty_message = "Watching inherited swimming around the arch…"
	curve.plot_title = "Live collection rate"
	curve.plot_caption = "This session · cumulative food / minute"
	curve.axis_label = "Elapsed seconds"
	curve.score_label = "Session"
	curve.show_best_score = false
	curve.tooltip_text = "Food actually collected by this live swimmer, averaged over this session. Background PPO practices separately; AI details report matched candidate checks. Collection alone does not establish reliable passage through the opening."
	curve.status_caption = "PPO %d · %s · session collection" % [stats.get("update", 0), actor_label]
	if not str(stats.get("learning_error", "")).is_empty():
		feed_feedback.text = "Background learning stopped. Open AI details."
		training_stats.text += "\nLearning stopped: " + str(stats.learning_error)

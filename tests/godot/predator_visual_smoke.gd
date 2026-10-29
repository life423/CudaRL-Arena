extends SceneTree

# Render-only QA. The failed research actor must remain rejected. The second
# stage borrows a real paused boulder pose and draws an explicitly static cue;
# it never represents a trained shelter agent or applies policy weights.
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

func freeze(node: Node) -> void:
	node.set_process(false)
	node.set_physics_process(false)
	for child in node.get_children():
		freeze(child)

func capture(name: String, pond) -> void:
	await RenderingServer.frame_post_draw
	check(root.get_texture().get_image().save_png(output.path_join(name + ".png")) == OK, "Root capture failed: " + name)
	if pond.viewport_container.visible:
		check(pond.viewport.get_texture().get_image().save_png(output.path_join(name + "-3d.png")) == OK, "Pond capture failed: " + name)
	else:
		check(pond.readiness_message.visible and not pond.curve.visible, "Unavailable trial must show clear feedback rather than a frozen game")

func controls_disabled(pond) -> void:
	check(pond.predator_button.disabled and pond.pause_button.disabled, "Unready event or learning action remains enabled")
	check(pond.current_switch.disabled and pond.current_strength.disabled and pond.speed_button.disabled, "Unready native water or speed action remains enabled")
	var save := pond.find_child("SaveTrainingNow", true, false) as Button
	check(save != null and save.disabled, "Unready scene still offers a learning save")

func menu_fits(pond, dimensions: Vector2i) -> void:
	var back := pond.find_child("BackToDemos", true, false) as Button
	check(back != null and not back.disabled, "Pond menu must remain available")
	if back == null:
		return
	pond.sidebar_scroll.ensure_control_visible(back)
	back.grab_focus()
	await settle(2)
	var bounds := back.get_global_rect()
	check(back.size.y >= 44 and bounds.position.x >= -1 and bounds.end.x <= dimensions.x + 1, "Pond menu target is too small or clips horizontally")
	check(bounds.position.y >= -1 and bounds.end.y <= dimensions.y + 1, "Pond menu cannot be reached at this height")
	check(root.gui_get_focus_owner() == back, "Pond menu lacks keyboard focus")

func banner(pond) -> PanelContainer:
	var layer := CanvasLayer.new()
	layer.name = "VisualFixtureOnly"
	layer.layer = 10
	pond.viewport.add_child(layer)
	var panel := PanelContainer.new()
	panel.position = Vector2(8, 8)
	panel.size = Vector2(240, 48)
	panel.size_flags_horizontal = Control.SIZE_SHRINK_BEGIN
	panel.size_flags_vertical = Control.SIZE_SHRINK_BEGIN
	panel.clip_contents = true
	panel.mouse_filter = Control.MOUSE_FILTER_IGNORE
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.02, 0.07, 0.07, 0.94)
	style.content_margin_left = 6
	style.content_margin_right = 6
	style.content_margin_top = 6
	style.content_margin_bottom = 6
	panel.add_theme_stylebox_override("panel", style)
	layer.add_child(panel)
	var label := Label.new()
	label.text = "VISUAL FIXTURE · NO LEARNING"
	label.autowrap_mode = TextServer.AUTOWRAP_OFF
	label.max_lines_visible = 1
	label.size_flags_vertical = Control.SIZE_SHRINK_BEGIN
	label.add_theme_font_size_override("font_size", 12)
	label.add_theme_color_override("font_color", Color("eef4e9"))
	label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	panel.add_child(label)
	return panel

func shadow_pixels(pond, dimensions: Vector2i) -> void:
	pond.predator_shadow.hide()
	await settle(3)
	await RenderingServer.frame_post_draw
	var before: Image = pond.viewport.get_texture().get_image()
	pond.predator_shadow.show()
	await settle(3)
	await RenderingServer.frame_post_draw
	var after: Image = pond.viewport.get_texture().get_image()
	check(not before.is_empty() and before.get_size() == after.get_size(), "Shadow renderer returned no matching images")
	if before.is_empty() or before.get_size() != after.get_size():
		return
	var center := Vector3(0, 0.018, -1.25)
	var rect := Rect2(pond.camera.unproject_position(center), Vector2.ZERO)
	# A conservative projected footprint contains every triangle of the cue.
	for x in [-2.1, 2.1]:
		for z in [-2.1, 2.1]:
			rect = rect.expand(pond.camera.unproject_position(center + Vector3(x, 0, z)))
	rect = rect.grow(4).intersection(Rect2(Vector2.ZERO, Vector2(after.get_size())))
	var inside := 0
	var outside := 0
	var outside_samples := 0
	for y in range(0, after.get_height(), 3):
		for x in range(0, after.get_width(), 3):
			var a := before.get_pixel(x, y)
			var b := after.get_pixel(x, y)
			var darkened := (a.r + a.g + a.b) - (b.r + b.g + b.b) > 0.06
			if rect.has_point(Vector2(x, y)):
				inside += int(darkened)
			else:
				outside_samples += 1
				outside += int(darkened)
	check(inside >= 3, "No visible predator shadow was rendered at %s; inspect shader compilation" % dimensions)
	check(outside <= maxi(12, int(outside_samples * 0.02)), "Shadow darkens the whole pond instead of its bounded silhouette")
	measurements.append({"size":str(dimensions), "shadow_dark_samples":inside, "outside_dark_samples":outside, "outside_samples":outside_samples})

func run() -> void:
	create_timer(180).timeout.connect(func(): push_error("Predator visual smoke timed out"); quit(1))
	check(DisplayServer.get_name() != "headless", "This fixture requires the real renderer, not --headless")
	if failures:
		quit(1)
		return
	var base := ProjectSettings.globalize_path("res://../build/predator-shelter")
	output = base.path_join("visual")
	check(DirAccess.make_dir_recursive_absolute(output) == OK, "Could not create visual QA directory")
	var private_dir := output.path_join("private-%d-%d" % [OS.get_process_id(), Time.get_ticks_usec()])
	check(DirAccess.make_dir_recursive_absolute(private_dir) == OK, "Could not isolate visual QA saves")
	var source := base.path_join("source-boulder.bin")
	var motor := base.path_join("motor-v2-training.bin")
	var cove := ProjectSettings.globalize_path("res://../build/boulder-navigation/source-cove.bin")
	var protected := {source:FileAccess.get_sha256(source), motor:FileAccess.get_sha256(motor), cove:FileAccess.get_sha256(cove)}
	for path: String in protected:
		check(not str(protected[path]).is_empty(), "Missing protected research source: " + path)
	var private_source := private_dir.path_join("source-boulder.bin")
	var private_motor := private_dir.path_join("motor-training.bin")
	var private_cove := private_dir.path_join("source-cove.bin")
	var private_boulder := private_dir.path_join("boulder-training.bin")
	for pair in [[source,private_source], [motor,private_motor], [cove,private_cove], [source,private_boulder]]:
		check(DirAccess.copy_absolute(pair[0], pair[1]) == OK, "Could not create private fixture copy: " + str(pair[1]))
	if failures:
		quit(1)
		return
	var pond = load("res://pond/predator_pond.tscn").instantiate()
	pond.source_boulder_checkpoint = private_source
	pond.predator_checkpoint = private_dir.path_join("training.bin")
	pond.population = 16
	pond.ready.connect(func(): freeze(pond), CONNECT_ONE_SHOT)
	root.add_child(pond)
	current_scene = pond
	freeze(pond)
	await settle()
	check("18 of 24" in pond.lab.get_error(), "Research actor was not rejected by its actual holding gate")
	check(pond.title_status.text == "Shelter practice is not ready", "Readiness error is unclear")
	check(pond.lab.get_stats().get("learning_active", false) == false, "Unready goal learner was started")
	check(not FileAccess.file_exists(pond.predator_checkpoint), "Unready trial created an approved shelter checkpoint")
	var sizes := [Vector2i(1280,720), Vector2i(768,1024), Vector2i(390,844), Vector2i(844,390)]
	for dimensions in sizes:
		root.size = dimensions
		await settle()
		controls_disabled(pond)
		await menu_fits(pond, dimensions)
		await capture("readiness-%dx%d" % [dimensions.x, dimensions.y], pond)
	# Only this separate private boulder session supplies a physical render pose.
	# It resumes its entire 1,024-world save; no personal path is ever opened.
	var pose_lab: RefCounted = DeepPondLab.new()
	var pose_started: bool = pose_lab.start_boulder_learning(private_cove, private_boulder, 1024, "cuda")
	check(pose_started and pose_lab.get_error().is_empty(), "Private boulder pose source failed: " + pose_lab.get_error())
	if not pose_started:
		pond.queue_free()
		await process_frame
		quit(1)
		return
	pose_lab.set_paused(true)
	var pose_stats: Dictionary = pose_lab.get_stats()
	var ready_deadline := Time.get_ticks_msec() + 5000
	while not pose_stats.get("learning_ready", false) and str(pose_stats.get("learning_error", "")).is_empty() and Time.get_ticks_msec() < ready_deadline:
		await create_timer(0.05).timeout
		pose_stats = pose_lab.get_stats()
	check(pose_stats.get("learning_ready", false) and str(pose_stats.get("learning_error", "")).is_empty(), "Paused pose source did not finish startup within five seconds")
	check(pose_stats.get("resumed", false) and int(pose_stats.get("background_swimmers", 0)) == 1024, "Fixture must resume full 1,024-world boulder state")
	pond.state = pose_lab.get_state().duplicate(true)
	# Only this explicitly labelled renderer fixture replaces the unavailable
	# view with static physical geometry. Production readiness shows no frozen game.
	pond.readiness_message.hide()
	pond.viewport_container.show()
	pond.curve.show()
	check(pond.state.has("links") and pond.state.links.size() == 3, "Fixture has no real articulated swimmer pose")
	pond.state.merge({"predator_position":Vector3(-8,4,-1.25), "predator_shadow_position":Vector3(0,0,-1.25), "predator_direction":Vector3(0.4,0,0.1), "predator_phase":"passing", "threat_level":1.0, "body_exposure":1.0, "goal_kind":"visual fixture", "goal_position":Vector3.ZERO}, true)
	pond.previous = pond.state.duplicate(true)
	pond.overview = true
	pond.released = false
	pond._apply_pose(1.0)
	pond.title_status.text = "VISUAL FIXTURE · no learning"
	pond.feed_feedback.text = "Static canonical predator pose. Not an approved agent."
	pond.training_heading.text = "Visual QA only"
	pond.training_stats.text = "A paused private boulder save supplies the swimmer pose. This fixture does not run or adopt the failed shelter actor."
	pond.save_status.text = "Private fixture copies only. Original training saves preserved."
	pond.curve.plot_title = "Visual fixture only"
	pond.curve.plot_caption = "No learning · no policy adoption"
	pond.curve.status_caption = "The shelter trial remains rejected"
	pond.curve.current_score = -1
	pond.curve.show_best_score = false
	pond.curve.points.clear()
	pond.curve.best_points.clear()
	pond.curve.empty_message = "Static renderer fixture; no performance result."
	pond.curve.queue_redraw()
	var fixture_banner := banner(pond)
	check(pond.predator_visual.visible and pond.predator_shadow.visible, "Native-position cue meshes were not displayed")
	check(pond.predator_visual.position == Vector3(-8,4,-1.25), "Bird source changed its canonical native eye")
	check(pond.predator_shadow_material.get_shader_parameter("predator_center") == Vector2(0,-1.25), "Shadow was incorrectly placed below the bird instead of its native projection")
	for dimensions in sizes:
		root.size = dimensions
		await settle()
		pond._fit()
		await settle()
		pond._update_camera(1.0)
		# Set both dimensions after wrapping changes. Setting width alone leaves
		# the old tall minimum-size allocation covering the narrow pond viewport.
		fixture_banner.size = Vector2(maxf(120, minf(330, pond.viewport.size.x - 16)), 48)
		await settle()
		check(fixture_banner.size.y <= 48 and fixture_banner.get_rect().end.y <= 56, "Visual-fixture banner expanded over the pond")
		controls_disabled(pond)
		await menu_fits(pond, dimensions)
		await shadow_pixels(pond, dimensions)
		await capture("fixture-%dx%d" % [dimensions.x, dimensions.y], pond)
	pose_lab.stop()
	pond.queue_free()
	await process_frame
	for path: String in protected:
		check(FileAccess.get_sha256(path) == str(protected[path]), "Protected source changed: " + path)
	check(FileAccess.get_sha256(private_source) == str(protected[source]), "Private read-only boulder reference changed")
	check(FileAccess.get_sha256(private_motor) == str(protected[motor]), "Rejected motor trial copy changed")
	check(FileAccess.get_sha256(private_cove) == str(protected[cove]), "Private read-only cove reference changed")
	var summary := {"checks":checks, "failures":failures, "rendered":true, "fixture_only":true, "measurements":measurements, "protected_sha256":protected, "private_directory":private_dir}
	var report := FileAccess.open(output.path_join("summary.json"), FileAccess.WRITE)
	check(report != null, "Could not write visual QA report")
	if report:
		report.store_string(JSON.stringify(summary, "\t"))
		report.close()
	print("PREDATOR_VISUAL_SMOKE " + JSON.stringify(summary))
	quit(1 if failures else 0)

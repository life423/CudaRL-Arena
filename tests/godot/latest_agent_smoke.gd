extends SceneTree
var checks := 0
var failures := 0
func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(message)
func settle() -> void:
	for tick in 8:
		await process_frame
func _initialize() -> void:
	Engine.max_fps = 60
	_run.call_deferred()
func _run() -> void:
	create_timer(90).timeout.connect(func(): quit(1))
	var source := "res://../docs/experiments/swimmer/gentle-currents/bank-recovery/selected-policy.bin"
	var source_hash := FileAccess.get_sha256(source)
	var original := "res://pond/policies/deep_selector.bin"
	var original_hash := FileAccess.get_sha256(original)
	var scene = load("res://pond/latest_agent.tscn").instantiate()
	root.add_child(scene)
	current_scene = scene
	scene.set_physics_process(false)
	await settle()
	check(scene.lab.get_error().is_empty(), "Live candidate did not start: " + scene.lab.get_error())
	check(scene.stats.progress_controller and scene.stats.selector_update == 320, "Wrong saved candidate loaded")
	check(scene.stats.policy_update == 64350 and scene.stats.background_swimmers == 0, "Unexpected motor or training session")
	check(scene.lab.get_evaluation_history(0).is_empty(), "Candidate inherited the original policy validation score")
	check(scene.title_status.text.contains("Live") and not scene.title_status.text.contains("Recorded"), "Live simulation mislabeled")
	check(not scene.curve.show_best_score and scene.curve.status_caption.contains("no learning"), "Live graph implies training")
	var first: Dictionary = scene.lab.get_state()
	for tick in 1200:
		scene.state = scene.lab.step()
		if tick % 100 == 0:
			scene._poll()
	scene.previous = scene.state.duplicate(true)
	scene._poll()
	check(scene.state.elapsed > 59.9 and scene.state.eaten > 10, "Live candidate did not simulate a minute of food collection")
	check(scene.state.position.distance_to(first.position) > 0.1, "Live body did not move")
	check(scene.curve.points.size() > 10, "Live collection graph did not sample the session")
	check(absf(scene.curve.current_score - scene.state.eaten) < 0.01, "One-minute graph score is not actual live food collection")
	var trained_first_step: Dictionary
	scene._toggle_untrained(true)
	check(scene.state.elapsed == 0 and scene.stats.showcase_untrained, "Random-weight comparison did not reset")
	check(scene.curve.points.is_empty(), "Graph retained the previous session after reset")
	scene._toggle_untrained(false)
	check(scene.state.foods == first.foods and scene.state.position == first.position, "Restoring latest weights changed the initial layout")
	trained_first_step = scene.lab.step()
	scene._toggle_untrained(true)
	scene._toggle_untrained(false)
	check(scene.lab.step().position.distance_to(trained_first_step.position) < 0.00001, "Restoring trained weights changed live actions")
	for tick in 300:
		scene.state = scene.lab.step()
		if tick % 100 == 0:
			scene._poll()
	scene.previous = scene.state.duplicate(true)
	scene._poll()
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844), Vector2i(844, 390)]:
		root.size = dimensions
		await settle()
		check(scene.layout.get_global_rect().end.x <= dimensions.x + 1, "Live layout overflows")
		check(scene.viewport_container.size.y >= 100, "Live viewport collapsed")
		check(scene.untrained_button.size.y >= 44 and scene.camera_button.size.y >= 44, "Live controls are too small")
		if DisplayServer.get_name() != "headless":
			await RenderingServer.frame_post_draw
			var path := ProjectSettings.globalize_path("res://../build/pond-reliability-study/live-%dx%d.png" % [dimensions.x, dimensions.y])
			check(root.get_texture().get_image().save_png(path) == OK, "Live screenshot failed")
	check(FileAccess.get_sha256(source) == source_hash and FileAccess.get_sha256(original) == original_hash, "Live viewer changed saved policies")
	var backend: String = scene.stats.compute_backend
	scene.find_child("BackToDemos", true, false).pressed.emit()
	await settle()
	check(current_scene.name == "DemoMenu", "Live view Back to demos failed")
	print("LATEST_LIVE_AGENT backend=%s checks=%d failures=%d" % [backend, checks, failures])
	quit(1 if failures else 0)

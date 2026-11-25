extends SceneTree

var checks := 0
var failures: Array[String] = []

func verify(ok: bool, label: String) -> void:
	checks += 1
	if not ok: failures.append(label)

func _initialize() -> void:
	call_deferred("_run")

func _run() -> void:
	var output := ""
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--output="): output = arg.trim_prefix("--output=")
	var scene = load("res://flexible_tail_preview.tscn").instantiate()
	root.add_child(scene)
	await process_frame
	root.title = "Tadpole Pond · UI verification"
	verify(scene.autonomous and not scene.live_snapshot.is_empty(), "qualified models load")
	verify(scene.get_node("%LivePlay").text == "Explore pond", "obvious primary action")
	scene.get_node("%LivePlay").pressed.emit()
	verify(scene.live_playing, "play signal starts exploration")
	scene.get_node("%LivePlay").pressed.emit()
	verify(not scene.live_playing, "pause signal pauses")
	scene._select_control_mode(1)
	verify(not scene.autonomous and not scene.live_snapshot.autonomous, "original motor remains accessible")
	verify(scene.get_node("%LivePlay").text == "Play" and scene.get_node("%Qualification").text.begins_with("Original"), "manual controls and explanation update")
	scene._request_live_turn(1)
	verify(int(scene.live_snapshot.queued_direction) == 1, "original manual request remains functional")
	scene.get_node("%Restart").pressed.emit()
	verify(int(scene.live_snapshot.queued_direction) == 0 and not scene.live_playing, "restart clears request and pauses")
	scene._select_control_mode(0)
	verify(scene.autonomous and scene.live_snapshot.autonomous, "return to qualified exploration")
	verify(scene.get_node("%LivePlay").text == "Explore pond" and scene.get_node("%Qualification").text.begins_with("Verified"), "exploration controls and qualification update")
	for dimensions in [Vector2i(390,844), Vector2i(768,1024), Vector2i(1100,760), Vector2i(844,390)]:
		root.size = dimensions
		for expanded in [false, true]:
			scene.live_details.set_pressed_no_signal(expanded)
			scene.live_options.visible = expanded
			scene._fit_camera()
			await process_frame
			await process_frame
			await RenderingServer.frame_post_draw
			var viewport := Rect2(Vector2.ZERO, Vector2(root.size))
			var dock: Control = scene.get_node("%Transport")
			verify(viewport.encloses(dock.get_global_rect()), "dock fits %s details%s" % [dimensions,expanded])
			for id in ["%LivePlay", "%Restart", "%LiveDetails"]:
				var button: Control = scene.get_node(id)
				verify(button.size.y >= 44 and button.size.x >= 44, "touch target %s %s" % [id,dimensions])
			if scene.header.visible:
				verify(not scene.header.get_global_rect().intersects(dock.get_global_rect()), "header/dock separation %s" % dimensions)
			root.get_texture().get_image().save_png(output.path_join("%dx%d-%s.png" % [dimensions.x,dimensions.y,"details" if expanded else "main"]))
	verify(float(scene.live_snapshot.time_s) == 0, "UI checks perform zero physics steps")
	var result := {"pass": failures.is_empty(), "checks": checks, "failures": failures, "physics_steps": 0, "optimizer_updates": 0, "rendered_sizes": [[390,844],[768,1024],[1100,760],[844,390]]}
	FileAccess.open(output.path_join("ui-result.json"),FileAccess.WRITE).store_string(JSON.stringify(result,"\t"))
	print(JSON.stringify(result))
	scene.queue_free()
	quit(0 if failures.is_empty() else 1)

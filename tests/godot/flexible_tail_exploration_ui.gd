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
	verify(not scene.get_node("%Restart").is_visible_in_tree(), "secondary actions hidden until Details")
	verify(not scene.get_node("%LiveMode").visible, "redundant live-mode tab removed")
	var camera_checks: Array[Dictionary] = []
	for dimensions in [Vector2i(390,844), Vector2i(768,1024), Vector2i(1100,760), Vector2i(844,390)]:
		root.size = dimensions
		await process_frame
		scene._fit_camera()
		scene.zoom_span = scene.close_span
		for expanded in [false, true]:
			scene.live_details.set_pressed_no_signal(expanded)
			scene.live_options.visible = expanded
			scene._fit_camera()
			await process_frame
			await process_frame
			await RenderingServer.frame_post_draw
			var viewport := Rect2(Vector2.ZERO, Vector2(root.size))
			var sidebar: ScrollContainer = scene.get_node("%Sidebar")
			verify(viewport.encloses(sidebar.get_global_rect()), "sidebar fits %s details%s" % [dimensions,expanded])
			for id in ["%LivePlay", "%Restart", "%LiveDetails", "%ZoomIn", "%ZoomOut", "%PondView"]:
				var button: Control = scene.get_node(id)
				verify(button.size.y >= 44 and button.size.x >= 44, "touch target %s %s" % [id,dimensions])
				verify(sidebar.is_ancestor_of(button), "control is in left column %s" % id)
				verify(button.get_global_rect().end.x <= sidebar.get_global_rect().end.x + 1, "no horizontal overflow %s %s" % [id,dimensions])
			verify(not scene.header.get_global_rect().intersects(scene.transport.get_global_rect()), "header/action separation %s" % dimensions)
			var subject: Vector2 = scene.camera.unproject_position(scene.follow_center)
			verify(subject.x > sidebar.get_global_rect().end.x and viewport.has_point(subject), "subject framed outside controls %s" % dimensions)
			root.get_texture().get_image().save_png(output.path_join("%dx%d-%s.png" % [dimensions.x,dimensions.y,"details" if expanded else "main"]))
			if expanded:
				sidebar.ensure_control_visible(scene.get_node("%ReviewMode"))
				await process_frame
				await process_frame
				verify(sidebar.get_global_rect().encloses(scene.get_node("%ReviewMode").get_global_rect()), "last action reachable by scrolling %s" % dimensions)
				sidebar.scroll_vertical = 0
		scene.live_options.visible = false
		scene.live_details.set_pressed_no_signal(false)
		var before: float = scene.camera.size
		scene._zoom_by(scene.ZOOM_STEP)
		verify(is_equal_approx(scene.camera.size, before * scene.ZOOM_STEP), "gradual close zoom out %s" % dimensions)
		for i in range(40): scene._zoom_by(scene.ZOOM_STEP)
		verify(scene.showing_pond and scene.zoom_out.disabled, "continuous range reaches full pond %s" % dimensions)
		before = scene.camera.size
		scene._zoom_by(1.0 / scene.ZOOM_STEP)
		verify(is_equal_approx(scene.camera.size, before / scene.ZOOM_STEP), "wide zoom in has no camera-mode jump %s" % dimensions)
		scene._toggle_pond_view()
		verify(is_equal_approx(scene.camera.size, scene.close_span), "follow preset %s" % dimensions)
		scene._toggle_pond_view()
		verify(is_equal_approx(scene.camera.size, scene.pond_preset_span), "closer pond preset %s" % dimensions)
		await process_frame
		await RenderingServer.frame_post_draw
		root.get_texture().get_image().save_png(output.path_join("%dx%d-pond.png" % [dimensions.x,dimensions.y]))
		camera_checks.append({"size": [dimensions.x, dimensions.y], "close_span": scene.close_span, "pond_preset_span": scene.camera.size})
		for i in range(60): scene._zoom_by(1.0 / scene.ZOOM_STEP)
		verify(scene.zoom_in.disabled, "closest limit %s" % dimensions)
	# Recordings remain available without permanent mode tabs on the pond.
	scene.get_node("%LiveDetails").grab_focus()
	var before_input: float = scene.camera.size
	var key := InputEventKey.new()
	key.keycode = KEY_MINUS
	key.pressed = true
	Input.parse_input_event(key)
	await process_frame
	verify(scene.camera.size > before_input, "keyboard minus works with sidebar focus")
	before_input = scene.camera.size
	var pinch := InputEventMagnifyGesture.new()
	pinch.position = Vector2(root.size.x - 40, root.size.y / 2)
	pinch.factor = 1.1
	Input.parse_input_event(pinch)
	await process_frame
	verify(is_equal_approx(scene.camera.size, before_input / 1.1), "pinch zoom uses continuous range: %f -> %f" % [before_input,scene.camera.size])
	before_input = scene.camera.size
	var wheel := InputEventMouseButton.new()
	wheel.position = pinch.position
	wheel.button_index = MOUSE_BUTTON_WHEEL_DOWN
	wheel.pressed = true
	Input.parse_input_event(wheel)
	await process_frame
	verify(is_equal_approx(scene.camera.size, before_input * scene.ZOOM_STEP), "wheel zoom over pond")
	before_input = scene.camera.size
	wheel.position = Vector2(50, 100)
	Input.parse_input_event(wheel)
	await process_frame
	verify(is_equal_approx(scene.camera.size, before_input), "scrolling sidebar does not zoom world")
	scene.live_details.pressed.emit()
	scene._show_recordings()
	verify(scene.reviewing and scene.get_node("%LiveMode").visible and not scene.live_controls.visible, "recordings isolated from live controls")
	scene._show_live()
	verify(not scene.reviewing and not scene.live_playing, "back to pond preserves paused live world")
	verify(float(scene.live_snapshot.time_s) == 0, "UI checks perform zero physics steps")
	var result := {"pass": failures.is_empty(), "checks": checks, "failures": failures, "physics_steps": 0, "optimizer_updates": 0, "camera": camera_checks, "rendered_sizes": [[390,844],[768,1024],[1100,760],[844,390]]}
	FileAccess.open(output.path_join("ui-result.json"),FileAccess.WRITE).store_string(JSON.stringify(result,"\t"))
	print(JSON.stringify(result))
	scene.queue_free()
	quit(0 if failures.is_empty() else 1)

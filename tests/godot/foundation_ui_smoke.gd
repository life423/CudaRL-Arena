extends SceneTree

# This UI check consumes exactly TWO explicitly requested 50ms control steps.
# All rendering/layout inspection is paused. No save or load is performed.
var failures: Array[String] = []
var checks := 0

func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures.append(message)
		push_error(message)

func _initialize() -> void:
	call_deferred("run")

func settle() -> void:
	await process_frame
	await process_frame

func run() -> void:
	if not ClassDB.class_exists("CreatureLab"):
		push_error("CreatureLab extension must be built and loaded before this smoke check.")
		quit(1)
		return
	var packed: PackedScene = load("res://main.tscn")
	var scene = packed.instantiate()
	root.add_child(scene)
	# Disable the frame scheduler: this test owns every physical step explicitly.
	scene.set_physics_process(false)
	await settle()
	var initial: Dictionary = scene.lab.get_state()
	check(initial.paused and int(initial.steps) == 0 and int(initial.substeps) == 0, "Construction must be paused at zero control/physical steps.")
	check(initial.finite and not initial.unsafe, "The default diagnostic pose must be finite and initially safe.")
	check(not initial.learned and not initial.contact_gate_passed, "The view must not claim learned behavior or resolved contact.")
	check(scene.sliders.size() == 4 and initial.links.size() == 3, "The view must have four controls for the three-link physical body.")
	check(scene.notice.text == "Contact gate unresolved", "Unresolved contact must remain visible.")
	check(scene.run_button.text == "Run" and not scene.run_button.disabled, "A paused world must offer Run.")
	check(not scene.step_button.disabled, "Single step must be available while paused.")
	var initial_position: Vector3 = initial.position_cm
	var layout_only := "--foundation-layout-only" in OS.get_cmdline_user_args()
	scene.sliders[0].value = 0.2
	var pending: Dictionary = scene.lab.get_state()
	check(is_equal_approx(float(pending.commands[0]), 0.2) and int(pending.steps) == 0, "Editing an actuator must change pending input without stepping.")
	check(pending.position_cm == initial_position, "Editing an actuator must not change the physical body.")
	scene.mode.select(1)
	scene.mode.item_selected.emit(1)
	var torque: Dictionary = scene.lab.get_state()
	check(int(torque.mode) == 0 and torque.paused and torque.commands == PackedFloat32Array([0, 0, 0, 0]), "Changing mode must pause and clear pending commands.")
	check(torque.position_cm == initial_position and int(torque.steps) == 0, "Changing mode must preserve the physical pose and clock.")
	scene.current.select(1)
	scene.current.item_selected.emit(1)
	var flowing: Dictionary = scene.lab.get_state()
	check(is_equal_approx(float(flowing.current_x), 0.005) and flowing.position_cm == initial_position and int(flowing.steps) == 0, "Changing water must preserve body/time and set the actual flow.")
	scene.run_button.pressed.emit()
	check(not scene.lab.get_state().paused, "Run must unpause without introducing an implicit step.")
	check(int(scene.lab.get_state().steps) == 0, "Run is not an automatic warmup.")
	scene.run_button.pressed.emit()
	check(scene.lab.get_state().paused and int(scene.lab.get_state().steps) == 0, "Pause must freeze before any requested step.")
	# The complete authorized physical budget for this smoke: TWO controls.
	# After it is spent, --foundation-layout-only permits render checks with ZERO.
	if not layout_only:
		scene.step_button.pressed.emit()
		var first: Dictionary = scene.lab.get_state()
		check(first.paused and int(first.steps) == 1 and int(first.substeps) == 10, "Single step must advance exactly one control/ten substeps and pause.")
		scene.step_button.pressed.emit()
		var second: Dictionary = scene.lab.get_state()
		check(second.paused and int(second.steps) == 2 and int(second.substeps) == 20, "The second explicit step must end at exactly two controls.")
		check(second.finite and not second.unsafe, "The two zero-torque open-water controls must remain inspectable.")
		check(absf(float(second.elapsed) - 0.1) < 0.000001, "Two controls must report 0.1 seconds of native integration.")
		check(second.actual_torques_uNm == PackedFloat32Array([0, 0, 0, 0]), "Zero direct-torque commands must report zero applied torques.")
	scene.reset_button.pressed.emit()
	var reset: Dictionary = scene.lab.get_state()
	check(reset.paused and int(reset.steps) == 0 and int(reset.substeps) == 0, "Reset must return to paused zero time without an integration step.")
	check(int(reset.mode) == 0 and is_equal_approx(float(reset.current_x), 0.005), "Reset must retain selected mode and water current.")
	check(reset.position_cm == initial_position, "Reset must restore the default physical pose.")
	# Optional screenshot output is a test-artifact directory, never user saves.
	var screenshot_directory := ""
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--foundation-screenshots="):
			screenshot_directory = argument.trim_prefix("--foundation-screenshots=")
	for dimensions in [Vector2i(1280, 720), Vector2i(768, 1024), Vector2i(390, 844)]:
		root.size = dimensions
		await settle()
		scene._fit()
		await settle()
		check(scene.size.x <= dimensions.x + 1 and scene.size.y <= dimensions.y + 1, "The root must fit %dx%d." % [dimensions.x, dimensions.y])
		check(scene.viewport_container.size.x >= 180 and scene.viewport_container.size.y >= 180, "The physical view must remain usable at %dx%d." % [dimensions.x, dimensions.y])
		check(scene.layout.get_minimum_size().x <= dimensions.x + 1, "The layout must not force horizontal overflow at %dx%d." % [dimensions.x, dimensions.y])
		check(scene.run_button.size.y >= 44 and scene.step_button.size.y >= 44, "Primary controls must retain 44px targets.")
		check(int(scene.lab.get_state().steps) == 0 and scene.lab.get_state().paused, "Layout inspection must not consume control steps.")
		if not screenshot_directory.is_empty() and DisplayServer.get_name() != "headless":
			DirAccess.make_dir_recursive_absolute(screenshot_directory)
			await RenderingServer.frame_post_draw
			var output := screenshot_directory.path_join("foundation-%dx%d.png" % [dimensions.x, dimensions.y])
			check(root.get_texture().get_image().save_png(output) == OK, "The paused screenshot must be written to the artifact directory.")
	check(scene.run_button.focus_mode == Control.FOCUS_ALL and scene.sliders[0].focus_mode == Control.FOCUS_ALL, "Buttons and actuators must be keyboard focusable.")
	scene.run_button.grab_focus()
	check(scene.run_button.has_focus(), "Keyboard focus must reach Run.")
	print("Foundation UI smoke: %d checks, %d failures; %d explicit controls; no saves, no learner." % [checks, failures.size(), 0 if layout_only else 2])
	quit(0 if failures.is_empty() else 1)

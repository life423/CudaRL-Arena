extends "res://pond/boulder_pond.gd"

const OverhangRock = preload("res://pond/overhang_rock.gd")
@export var source_boulder_checkpoint := "user://pond-policies/boulder-navigation-v1/training.bin"
var rock_overhang: Node3D
var overhang_api_available := false
var overhang_started := false
var readiness_message: Label
var next_debug_elapsed := 0.0

func _load_habitat_geometry() -> Dictionary:
	var geometry: Dictionary = lab.get_overhang_geometry() if lab.has_method("get_overhang_geometry") else lab.get_boulder_geometry()
	native_outline = geometry.get("shore_outline", PackedVector3Array())
	if native_outline.size() > 1 and native_outline[0].is_equal_approx(native_outline[-1]):
		native_outline.resize(native_outline.size() - 1)
	return geometry

func _start_lab() -> bool:
	paused = false
	overhang_api_available = lab.has_method("watch_overhang") and lab.has_method("get_overhang_geometry")
	if not overhang_api_available:
		return false
	var backend := "auto"
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--pond-backend="):
			backend = argument.get_slice("=", 1)
	overhang_started = lab.watch_overhang(source_boulder_checkpoint, backend)
	return overhang_started

func _build_ui() -> void:
	super._build_ui()
	speed_button.hide()
	speed_button.disabled = true
	find_child("SaveTrainingNow", true, false).hide()
	untrained_button.hide()
	untrained_button.disabled = true
	pause_button.text = "Pause pond"
	pause_button.tooltip_text = "Pause the live creature and food. This view does not train or change the saved controller."
	feed_feedback.text = "A new rock ledge extends from the bank. Its open side faces you."
	feed_feedback.tooltip_text = "The rock's visible surface matches native collision. This scene shows the inherited swimmer; learned hiding is not demonstrated yet."

func _build_world() -> void:
	super._build_world()
	if arch_geometry.has("overhang_parts"):
		rock_overhang = OverhangRock.build(world, arch_geometry, _stone_material())
		water_surface.set_meta("layout_id", int(arch_geometry.get("layout_id", 3)))

func _ready() -> void:
	super._ready()
	get_window().title = "Tadpole Pond · Rock ledge · live · CudaRL"
	(sidebar_scroll.get_child(0).get_child(0) as Label).text = "Rock ledge pond"
	_poll()
	if overhang_started:
		next_debug_elapsed = float(state.get("elapsed", 0.0)) + 5.0
		print("OVERHANG_LIVE_STARTED actor=%d learner_source=%d layout=3 training=0" % [stats.get("adopted_update", 0), stats.get("loaded_update", 0)])

func _toggle_pause() -> void:
	if not lab or state.is_empty():
		return
	paused = not paused
	lab.set_paused(paused)
	previous = state.duplicate(true)
	accumulator = 0.0
	_apply_pose(1.0)
	_poll()

func _physics_process(delta: float) -> void:
	if not overhang_started or paused or state.is_empty():
		return
	super._physics_process(delta)
	if "--overhang-status" in OS.get_cmdline_user_args() and float(state.get("elapsed", 0.0)) >= next_debug_elapsed:
		print("OVERHANG_LIVE_TICK elapsed=%.2f food=%d position=%s training=0" % [state.elapsed, state.eaten, str(state.position)])
		next_debug_elapsed = float(state.elapsed) + 5.0

func _show_readiness_error(message: String) -> void:
	overhang_started = false
	title_status.text = "Rock ledge view could not start"
	feed_feedback.text = message
	pause_button.disabled = true
	current_switch.disabled = true
	current_strength.disabled = true
	camera_button.hide()
	viewport_container.hide()
	curve.hide()
	details_button.hide()
	details_group.hide()
	if not readiness_message:
		readiness_message = _label("The live swimmer is unavailable.\n\nUse Pond menu to return to the working pond.", 18)
		readiness_message.name = "OverhangReadinessMessage"
		readiness_message.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		readiness_message.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
		readiness_message.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		readiness_message.size_flags_vertical = Control.SIZE_EXPAND_FILL
		visual.add_child(readiness_message)

func _poll() -> void:
	if not lab or not title_status:
		return
	stats = lab.get_stats()
	var error: String = lab.get_error()
	if not overhang_api_available:
		_show_readiness_error("This extension does not include the live rock ledge view yet.")
		return
	if not error.is_empty() or state.is_empty():
		_show_readiness_error(error if not error.is_empty() else "No physical swimmer state was returned.")
		return
	title_status.text = "Pond paused" if paused else "Live · trained swimmer"
	pause_button.show()
	pause_button.disabled = false
	pause_button.text = "Resume pond" if paused else "Pause pond"
	swimmer_stats.text = _swimmer_text()
	speed_button.hide()
	find_child("SaveTrainingNow", true, false).hide()
	curve.watch_button.hide()
	untrained_button.hide()
	var current_speed := float(stats.get("water_current_cm_s", 0.0))
	flow.set_current(stats.get("water_current", Vector3.ZERO))
	current_switch.set_pressed_no_signal(current_speed > 0)
	current_switch.disabled = false
	current_strength.disabled = current_speed <= 0
	current_strength.select(1 if current_speed > 0.625 else 0)
	water_status.text = "%.2f cm/s · food drifts" % current_speed if current_speed > 0 else "Calm water"
	water_status.tooltip_text = "Native flow drives body drag and food movement; the displayed ledge is a real solid obstacle."
	current_switch.tooltip_text = "Change water flow without resetting the creature. Saved neural-network weights remain frozen."
	current_strength.tooltip_text = "Choose 0.5 or 0.75 cm/s steady flow. This scene does not perform PPO updates."
	feed_feedback.text = "New bank ledge · an open shelter under real rock."
	details_button.show()
	details_group.visible = details_expanded
	training_heading.text = "Saved swimming controller"
	training_stats.text = "Native %s physics and neural inference.\nInherited boulder swimmer · both networks frozen.\nNo background training or learning updates.\nThe new ledge changes the real geometry. Hiding behavior has not passed its learning test." % stats.get("compute_backend", "")
	save_status.text = "The source boulder training save is read only in this view."
	curve.plot_title = "Live collection rate"
	curve.status_caption = "Saved weights · live physics · no learning updates"
	curve.score_label = "This visit"
	curve.plot_caption = "Food collected during this visit · actual food / minute"
	curve.axis_label = "Seconds in this visit"
	curve.empty_message = "Collection appears after the first second."
	curve.tooltip_text = "Session collection is a physical observation, not evidence of learning or shelter use."
	_update_session_curve(false)

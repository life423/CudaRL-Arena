extends "res://pond/deep_pond.gd"

const LATEST_POLICY := "res://../docs/experiments/swimmer/gentle-currents/bank-recovery/selected-policy.bin"
var current_switch: CheckButton
var water_status: Label

func _build_ui() -> void:
	super._build_ui()
	var water_controls := VBoxContainer.new()
	water_controls.name = "WaterControls"
	water_controls.add_theme_constant_override("separation", 0)
	current_switch = CheckButton.new()
	current_switch.name = "GentleCurrent"
	current_switch.text = "Gentle current"
	current_switch.custom_minimum_size.y = 44
	current_switch.add_theme_font_size_override("font_size", 15)
	current_switch.add_theme_color_override("font_color", TEXT)
	current_switch.add_theme_stylebox_override("focus", camera_button.get_theme_stylebox("focus"))
	current_switch.tooltip_text = "Switch between calm water and a steady 0.5 cm/s current across the pond. The creature and food stay in place when switched. Saved weights remain fixed."
	current_switch.toggled.connect(_set_current)
	water_controls.add_child(current_switch)
	water_status = _label("Calm water", 13, MUTED)
	water_controls.add_child(water_status)
	var controls := camera_button.get_parent()
	controls.add_child(water_controls)
	controls.move_child(water_controls, 3)

func _set_current(enabled: bool) -> void:
	if not lab or not lab.set_gentle_current(enabled):
		current_switch.set_pressed_no_signal(not enabled)
		water_status.text = "Current could not change"
		water_status.tooltip_text = lab.get_error() if lab else "Live pond is unavailable."
		return
	_poll()

func _start_lab() -> bool:
	paused = true
	var backend := "auto"
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--pond-backend="):
			backend = argument.get_slice("=", 1)
	return lab.watch_progress_controller(LATEST_POLICY, backend)

func _ready() -> void:
	super._ready()
	get_window().title = "CudaRL · Latest trained swimmer · live"
	(sidebar_scroll.get_child(0).get_child(0) as Label).text = "Tadpole pond"
	_poll()
	if "--gentle-current" in OS.get_cmdline_user_args():
		current_switch.button_pressed = true

func _fit() -> void:
	super._fit()
	if not layout.vertical and size.y < 500:
		curve.custom_minimum_size.y = 148
	if layout.vertical:
		sidebar.custom_minimum_size.y = 242
		sidebar_scroll.custom_minimum_size.y = 206

func _poll() -> void:
	super._poll()
	if not lab or not lab.get_error().is_empty() or state.is_empty():
		title_status.text = "Live agent could not start"
		return
	var untrained: bool = stats.get("showcase_untrained", false)
	title_status.text = "Live · random initial weights" if untrained else "Live · latest trained agent"
	training_heading.text = "Live inference"
	var flowing: bool = stats.get("water_current_cm_s", 0.0) > 0
	current_switch.set_pressed_no_signal(flowing)
	water_status.text = "0.5 cm/s · steady flow →" if flowing else "Calm water"
	water_status.tooltip_text = "Uniform flow along the pond's horizontal X axis; food stays at its existing underwater locations." if flowing else "No imposed water current."
	training_stats.text = "AI and physics · %s\nSelector update %d\nSwimming retained from %d\nThree underwater foods" % [stats.compute_backend, 0 if untrained else stats.selector_update, 0 if untrained else stats.policy_update]
	save_status.text = "Random initial networks · no training updates." if untrained else "Bank-trained candidate · weights fixed.\nPhysical simulation runs live.\nOriginal policy and saves preserved."

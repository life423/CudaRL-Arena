extends "res://pond/deep_pond.gd"

const LATEST_POLICY := "res://../docs/experiments/swimmer/gentle-currents/bank-recovery/selected-policy.bin"
const CURRENT_SPEEDS := [0.5, 0.75]
var current_switch: CheckButton
var current_strength: OptionButton
var water_status: Label
var flow: Node3D

func _build_world() -> void:
	super._build_world()
	flow = preload("res://pond/current_flow.gd").new()
	flow.configure(water_surface.material_override)
	world.add_child(flow)

func _physics_process(delta: float) -> void:
	var before: float = state.get("elapsed", 0)
	super._physics_process(delta)
	if flow:
		flow.advance(float(state.get("elapsed", 0)) - before)

func _build_ui() -> void:
	super._build_ui()
	var water_controls := VBoxContainer.new()
	water_controls.name = "WaterControls"
	water_controls.add_theme_constant_override("separation", 0)
	var current_row := HBoxContainer.new()
	current_row.add_theme_constant_override("separation", 8)
	water_controls.add_child(current_row)
	current_switch = CheckButton.new()
	current_switch.name = "GentleCurrent"
	current_switch.text = "Current"
	current_switch.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	current_switch.custom_minimum_size.y = 44
	current_switch.add_theme_font_size_override("font_size", 15)
	current_switch.add_theme_color_override("font_color", TEXT)
	current_switch.add_theme_stylebox_override("focus", camera_button.get_theme_stylebox("focus"))
	current_switch.tooltip_text = "Apply steady flow at the selected strength to the swimmer and suspended food. Switching preserves the physical state; saved weights remain fixed."
	current_switch.toggled.connect(_set_current)
	current_row.add_child(current_switch)
	current_strength = OptionButton.new()
	current_strength.name = "CurrentStrength"
	current_strength.custom_minimum_size = Vector2(112, 44)
	current_strength.add_theme_font_size_override("font_size", 14)
	for style in ["normal", "hover", "pressed", "focus"]:
		current_strength.add_theme_stylebox_override(style, camera_button.get_theme_stylebox(style))
	for color in ["font_color", "font_hover_color", "font_pressed_color", "font_focus_color"]:
		current_strength.add_theme_color_override(color, TEXT)
	current_strength.add_item("0.5 cm/s")
	current_strength.add_item("0.75 cm/s")
	current_strength.tooltip_text = "Choose a tested steady-current strength. Turn Current on to change it. Both neural networks stay frozen."
	current_strength.item_selected.connect(_select_strength)
	current_row.add_child(current_strength)
	water_status = _label("Calm water", 13, MUTED)
	water_controls.add_child(water_status)
	var controls := camera_button.get_parent()
	controls.add_child(water_controls)
	controls.move_child(water_controls, 3)

func _set_current(enabled: bool) -> void:
	if not lab or not lab.set_steady_current(CURRENT_SPEEDS[current_strength.selected] if enabled else 0.0, true):
		current_switch.set_pressed_no_signal(not enabled)
		water_status.text = "Current could not change"
		water_status.tooltip_text = lab.get_error() if lab else "Live pond is unavailable."
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

func _start_lab() -> bool:
	paused = true
	var backend := "auto"
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--pond-backend="):
			backend = argument.get_slice("=", 1)
	return lab.watch_progress_controller(LATEST_POLICY, backend)

func _ready() -> void:
	super._ready()
	get_window().title = "Tadpole Pond · Live trained creature · CudaRL"
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
	var drifting: bool = stats.get("food_drifting", false)
	flow.set_current(stats.get("water_current", Vector3.ZERO))
	current_switch.set_pressed_no_signal(flowing)
	current_strength.disabled = not flowing
	if flowing:
		current_strength.select(1 if stats.water_current_cm_s > 0.625 else 0)
	var strength := str(snappedf(stats.get("water_current_cm_s", 0.0), 0.01))
	water_status.text = (strength + " cm/s · food drifts →" if drifting else strength + " cm/s · steady flow →") if flowing else "Calm water"
	water_status.tooltip_text = ("Suspended food follows the horizontal current at fixed depth, with clearance at the bank." if drifting else "Stationary-food reference mode.") if flowing else "No imposed current; food remains at its present position."
	training_stats.text = "AI and physics · %s\nSelector update %d\nSwimming retained from %d\nThree underwater foods" % [stats.compute_backend, 0 if untrained else stats.selector_update, 0 if untrained else stats.policy_update]
	save_status.text = "Random initial networks · no training updates." if untrained else "Bank-trained candidate · weights fixed.\nPhysical simulation runs live.\nOriginal policy and saves preserved."

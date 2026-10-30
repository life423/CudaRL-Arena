extends "res://pond/deep_pond.gd"

const LATEST_POLICY := "res://../docs/experiments/swimmer/gentle-currents/bank-recovery/selected-policy.bin"
const CURRENT_SPEEDS := [0.5, 0.75]
const CurrentControls = preload("res://pond/pond_current_controls.gd")
const WaterController = preload("res://pond/pond_water_controller.gd")
var water_controller := WaterController.new()
var current_switch: CheckButton
var current_strength: OptionButton
var water_status: Label
var flow: Node3D

func _build_world() -> void:
	super._build_world()
	flow = preload("res://pond/current_flow.gd").new()
	flow.configure(water_surface.material_override)
	world.add_child(flow)
	water_controller.flow = flow

func _physics_process(delta: float) -> void:
	var before: float = state.get("elapsed", 0)
	super._physics_process(delta)
	if flow:
		flow.advance(float(state.get("elapsed", 0)) - before)

func _build_ui() -> void:
	super._build_ui()
	var water: Dictionary = CurrentControls.build(self, sidebar_controls, camera_button)
	current_switch = water.switch
	current_strength = water.strength
	water_status = water.status
	current_switch.tooltip_text = "Apply steady flow at the selected strength to the swimmer and suspended food. Switching preserves the physical state; saved weights remain fixed."
	current_strength.tooltip_text = "Choose a tested steady-current strength. Turn Current on to change it. Both neural networks stay frozen."
	water_controller.configure(lab, water)
	water_controller.changed.connect(_poll)

func _set_current(enabled: bool) -> void:
	water_controller.set_enabled(enabled)

func _select_strength(index: int) -> void:
	water_controller.select_strength(index, stats)

func _start_lab() -> bool:
	paused = true
	var backend := "auto"
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--pond-backend="):
			backend = argument.get_slice("=", 1)
	return lab.watch_progress_controller(LATEST_POLICY, backend)

func _after_lab_started() -> void:
	super._after_lab_started()
	get_window().title = "Tadpole Pond · Live trained creature · CudaRL"
	sidebar_title.text = "Tadpole pond"
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
	water_controller.refresh(stats)
	training_stats.text = "AI and physics · %s\nSelector update %d\nSwimming retained from %d\nThree underwater foods" % [stats.compute_backend, 0 if untrained else stats.selector_update, 0 if untrained else stats.policy_update]
	save_status.text = "Random initial networks · no training updates." if untrained else "Bank-trained candidate · weights fixed.\nPhysical simulation runs live.\nOriginal policy and saves preserved."

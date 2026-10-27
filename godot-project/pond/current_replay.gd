extends "res://pond/deep_pond.gd"

const Recording = preload("res://pond/current_replay_data.gd")
const REPLAY_PATH := "res://../build/pond-current-study/diagnosis/replay.json"
var replay_pause: Button
var scenario: OptionButton
var timeline: HSlider
var clock: Label
var before_button: Button
var case_ids := [0, 2, 9]

func _create_lab() -> RefCounted:
	return Recording.new()

func _start_lab() -> bool:
	paused = false
	if not lab.open(REPLAY_PATH):
		return false
	lab.seek(60)
	return true

func _ready() -> void:
	super._ready()
	get_window().title = "CudaRL · Current stall replay"
	(sidebar_scroll.get_child(0).get_child(0) as Label).text = "Current replay"
	curve.hide()
	if not state.is_empty():
		released = true
		_refresh_timeline()

func _build_ui() -> void:
	super._build_ui()
	var launch_replay := find_child("WatchCurrentTest", true, false)
	if launch_replay:
		launch_replay.hide()
	var controls := camera_button.get_parent()
	scenario = OptionButton.new()
	scenario.name = "ReplayScenario"
	scenario.custom_minimum_size.y = 44
	scenario.add_item("Original current trial")
	scenario.add_item("Remove current at 60 s")
	scenario.add_item("Target Food 3 for 1 s")
	scenario.fit_to_longest_item = false
	scenario.tooltip_text = "Recorded diagnostic branches share the identical body, food and frozen weights at 60 seconds. The food change is a test intervention, not a learned fix."
	scenario.item_selected.connect(_select_scenario)
	controls.add_child(scenario)
	controls.move_child(scenario, 3)
	replay_pause = _button("Pause replay")
	replay_pause.name = "PauseReplay"
	replay_pause.pressed.connect(_toggle_replay)
	var bar := HBoxContainer.new()
	bar.add_theme_constant_override("separation", 8)
	replay_pause.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	bar.add_child(replay_pause)
	for spec in [["Before stall", 25.0], ["Stalled", 60.0]]:
		var button := _button(spec[0])
		button.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		button.pressed.connect(func(): _seek_time(spec[1]))
		bar.add_child(button)
		if spec[0] == "Before stall":
			before_button = button
	var inset := MarginContainer.new()
	for side in ["left", "right", "top", "bottom"]:
		inset.add_theme_constant_override("margin_" + side, 16 if side in ["left", "right"] else 8)
	var transport := VBoxContainer.new()
	transport.add_theme_constant_override("separation", 8)
	clock = _label("Recorded CUDA trial · no training", 14, MUTED)
	transport.add_child(clock)
	timeline = HSlider.new()
	timeline.name = "ReplayTimeline"
	timeline.custom_minimum_size.y = 44
	timeline.step = 0.05
	timeline.tooltip_text = "Seek within the recorded trial using the slider, arrow keys or the jump buttons."
	timeline.value_changed.connect(_seek_time)
	transport.add_child(timeline)
	transport.add_child(bar)
	inset.add_child(transport)
	visual.add_child(inset)

func _select_scenario(choice: int) -> void:
	lab.select_case(case_ids[choice])
	lab.paused = false
	before_button.text = "Before stall" if choice == 0 else "Branch start"
	_seek_time(60)
	_refresh_timeline()

func _toggle_replay() -> void:
	if lab.index == lab.frames().size() - 1:
		_seek_time(lab.frames()[0].time)
		lab.paused = false
	else:
		lab.paused = not lab.paused
	replay_pause.text = "Play replay" if lab.paused else "Pause replay"

func _seek_time(seconds: float) -> void:
	if not lab or lab.data.is_empty():
		return
	lab.seek(seconds)
	state = lab.get_state()
	previous = state.duplicate(true)
	accumulator = 0
	_poll()
	_apply_pose(1)

func _refresh_timeline() -> void:
	timeline.set_block_signals(true)
	timeline.min_value = lab.frames()[0].time
	timeline.max_value = lab.frames()[-1].time
	timeline.set_value_no_signal(state.elapsed)
	timeline.set_block_signals(false)

func _process(delta: float) -> void:
	super._process(delta)
	if not state.is_empty():
		timeline.set_value_no_signal(state.elapsed)
		var ended: bool = lab.index == lab.frames().size() - 1
		replay_pause.text = "Replay" if ended else ("Play replay" if lab.paused else "Pause replay")
		clock.text = "Recorded CUDA · %.1f / %.1f s · %s" % [state.elapsed, timeline.max_value, "Finished" if ended else ("Paused" if lab.paused else "Playing")]

func _poll() -> void:
	super._poll()
	untrained_button.hide()
	curve.hide()
	if state.is_empty():
		return
	title_status.text = "Recorded trial · no training"
	training_heading.text = "Frozen controller"
	training_stats.text = "Original GPU test · world 24\nSame trained swimming + selector\nCurrent  %.1f cm/s" % lab.data.cases[lab.case_index].current_cm_s
	training_stats.tooltip_text = "This is recorded simulation output. The GPU is not running a training workload while you watch."
	save_status.text = "Current removal did not release it. A one-second food-target change did. These are diagnostic tests, not a trained improvement."

func _swimmer_text() -> String:
	return "Food eaten  %d\nSince food  %.1f s\nSpeed  %.3f cm/s\nTarget  Food %d\nDepth  %.1f cm" % [state.eaten, state.seconds_since_food, state.speed_cm_s, state.selected_food + 1, state.depth_cm]

func _apply_pose(alpha: float) -> void:
	super._apply_pose(alpha)
	for i in 3:
		depth_labels[i].text = "%s%d · %.1f cm" % ["Target " if i == state.selected_food else "", i + 1, -state.foods[i].y * 4]
		depth_labels[i].modulate = ACCENT if i == state.selected_food else MUTED

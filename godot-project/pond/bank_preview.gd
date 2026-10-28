extends "res://pond/current_replay.gd"

const BankRecording = preload("res://pond/bank_preview_data.gd")
const BANK_REPLAY := "res://../build/pond-reliability-study/bank-preview/replay.json"
var compare_button: Button

func _create_lab() -> RefCounted:
	return BankRecording.new()

func _start_lab() -> bool:
	paused = false
	if not lab.open(BANK_REPLAY):
		return false
	lab.select_case(1)
	lab.seek(0)
	return true

func _ready() -> void:
	super._ready()
	get_window().title = "CudaRL · Recovery comparison · recording"
	(sidebar_scroll.get_child(0).get_child(0) as Label).text = "Learned recovery"
	if not state.is_empty():
		scenario.select(1)
		_refresh_timeline()

func _build_ui() -> void:
	super._build_ui()
	var self_link := find_child("WatchBankRecovery", true, false)
	if self_link:
		self_link.hide()
	scenario.clear()
	scenario.add_item("Original food selector")
	scenario.add_item("Bank-trained food selector")
	scenario.tooltip_text = "Each policy starts with the identical body, food and current. The swimming controller and physics are unchanged."
	before_button.text = "Restart"
	for connection in before_button.pressed.get_connections():
		before_button.pressed.disconnect(connection.callable)
	before_button.pressed.connect(func(): _seek_time(0); lab.paused = false)
	compare_button = before_button.get_parent().get_child(2) as Button
	compare_button.text = "Compare from start"
	compare_button.tooltip_text = "Switch between the original and bank-trained food selector, restarting from the identical state."
	for connection in compare_button.pressed.get_connections():
		compare_button.pressed.disconnect(connection.callable)
	compare_button.pressed.connect(func(): _select_scenario(1 - lab.case_index); scenario.select(lab.case_index))

func _select_scenario(choice: int) -> void:
	scenario.select(choice)
	lab.select_case(choice)
	lab.paused = false
	_seek_time(0)
	_refresh_timeline()

func _process(delta: float) -> void:
	var was_playing: bool = lab != null and not lab.paused
	super._process(delta)
	if state.is_empty():
		return
	compare_button.text = "Compare" if visual.size.x < 550 else "Compare from start"
	if was_playing and lab.index == lab.frames().size() - 1:
		_seek_time(0)
		lab.paused = false
	clock.text = "Recorded CUDA · %.1f / %.1f s · %s" % [state.elapsed, timeline.max_value, "Paused" if lab.paused else "Playing · repeats"]

func _poll() -> void:
	super._poll()
	if state.is_empty():
		title_status.text = "Replay unavailable"
		save_status.text = lab.get_error()
		return
	title_status.text = "Recorded comparison · no live training"
	training_heading.text = "What changed"
	training_stats.text = "Food selection trained with PPO\nSame swimming controller and physics\nCurrent  %.1f cm/s" % lab.data.cases[lab.case_index].current_cm_s
	save_status.text = "A held-out bank case. Full test: 83 of 86 recovered within 20 seconds. Some cases still fail. Both replays use the same starting state."

func _swimmer_text() -> String:
	return "Food collected  %d\nSince food  %.1f s\nSpeed  %.2f cm/s\nTarget  Food %d\nDepth  %.1f cm" % [state.eaten, state.seconds_since_food, state.speed_cm_s, state.selected_food + 1, state.depth_cm]

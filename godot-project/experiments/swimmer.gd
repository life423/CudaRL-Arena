extends Control

# Isolated experiment viewer. The C++ process trains 1,024 CUDA environments;
# its live preview steps six CPU swimmers with the identical native physics and
# newest policy. Completed runs replay actual held-out evaluation trajectories.
const BG := Color("101c24")
const TEXT := Color("ecf4f1")
const MUTED := Color("afc5cb")
const ACCENT := Color("faa640")
const COLORS := [Color("66e3cf"), Color("7fc9ff"), Color("ffbe79"), Color("da9eff"), Color("ff96b3"), Color("d4e67e")]
@export var comparison_reference := ""
@export var initial_body := 6
var directory := ""
var worker := -1
var latest: Dictionary = {}
var baseline: Dictionary = {}
var preview: Dictionary = {}
var poll_time := 0.0
var replay_time := 0.0
var seen_update := -1
var history: Array[Vector2] = []
var status: Label
var metrics: Label
var explanation: Label
var restart: Button
var compare: Button
var world: Control
var root_box: VBoxContainer
var body_choice: OptionButton
var focused := -1
var reference: Dictionary = {}
var reference_baseline: Dictionary = {}
var heading: Label

func _ready() -> void:
	get_window().title = "CudaRL · Swimmer experiment"
	RenderingServer.set_default_clear_color(BG)
	_build()
	resized.connect(_fit)
	_fit()
	var replay_path := ""
	var reference_path := ProjectSettings.globalize_path(comparison_reference) if not comparison_reference.is_empty() else ""
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--swimmer-replay="):
			replay_path = argument.trim_prefix("--swimmer-replay=")
		if argument.begins_with("--swimmer-body="):
			body_choice.select(clampi(int(argument.trim_prefix("--swimmer-body=")), 0, 7))
		if argument.begins_with("--swimmer-reference="):
			reference_path = argument.trim_prefix("--swimmer-reference=")
	if not reference_path.is_empty():
		reference = _read_file(reference_path.path_join("seed-1-update-300.json"))
		reference_baseline = _read_file(reference_path.path_join("seed-1-update-0.json"))
		if reference.is_empty() or reference_baseline.is_empty():
			status.text = "The saved tadpole comparison could not be loaded."
			return
		heading.text = "Classic tadpole vs baby eel"
		body_choice.select(7)
		body_choice.visible = false
		restart.text = "Train eel from scratch"
		focused = 0
		_fit()
	if not replay_path.is_empty():
		directory = replay_path
		_poll()
	else:
		_start()

func _exit_tree() -> void:
	if worker > 0 and OS.is_process_running(worker):
		OS.kill(worker)

func _label(value: String, font_size: int, color := TEXT) -> Label:
	var label := Label.new()
	label.text = value
	label.add_theme_font_size_override("font_size", font_size)
	label.add_theme_color_override("font_color", color)
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	return label

func _button(value: String) -> Button:
	var button := Button.new()
	button.text = value
	button.custom_minimum_size.y = 48
	button.add_theme_font_size_override("font_size", 16)
	return button

func _build() -> void:
	var margin := MarginContainer.new()
	margin.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	for side in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 20)
	add_child(margin)
	root_box = VBoxContainer.new()
	root_box.add_theme_constant_override("separation", 10)
	margin.add_child(root_box)
	heading = _label("Can a swimmer learn to reach food?", 28)
	root_box.add_child(heading)
	status = _label("Starting from random weights…", 18, ACCENT)
	root_box.add_child(status)
	metrics = _label("1,024 unseen swimmers share one policy. Six independent attempts are shown.", 16, MUTED)
	root_box.add_child(metrics)
	body_choice = OptionButton.new()
	body_choice.fit_to_longest_item = false
	body_choice.clip_text = true
	body_choice.custom_minimum_size.y = 44
	for value in ["Next run: original body", "Next run: small head + long tail", "Next run: small head + broad fin", "Next run: original body + faster strokes", "Next run: broad fin + faster strokes", "Next run: original body + energetic strokes", "Next run: classic tadpole", "Next run: baby eel"]:
		body_choice.add_item(value)
	body_choice.select(initial_body)
	root_box.add_child(body_choice)
	var actions := HBoxContainer.new()
	actions.add_theme_constant_override("separation", 12)
	restart = _button("Start from scratch")
	restart.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	restart.pressed.connect(_start)
	actions.add_child(restart)
	compare = _button("Show untrained")
	compare.toggle_mode = true
	compare.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	compare.toggled.connect(func(on: bool):
		compare.text = "Return to latest" if on else "Show untrained"
		replay_time = 0.0)
	actions.add_child(compare)
	root_box.add_child(actions)
	explanation = _label("Live training is deliberately paced for viewing. The separate benchmark runs at full speed.", 14, MUTED)
	root_box.add_child(explanation)
	var scroll := ScrollContainer.new()
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root_box.add_child(scroll)
	world = Control.new()
	world.focus_mode = Control.FOCUS_ALL
	world.tooltip_text = "Select a pond to enlarge it. Keyboard: Enter toggles the view; Left and Right change swimmers."
	world.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	world.draw.connect(_draw_world)
	world.gui_input.connect(_pond_input)
	scroll.add_child(world)
	root_box.add_child(_label("Select a pond to enlarge it; select again for all six. Tail joints drive motion through simplified water drag.", 14, MUTED))

func _pond_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed:
		if event.is_action_pressed("ui_accept"):
			focused = -1 if focused >= 0 else 0
		elif event.is_action_pressed("ui_right"):
			focused = (focused + 1) % 6
		elif event.is_action_pressed("ui_left"):
			focused = 5 if focused <= 0 else focused - 1
		else:
			return
		world.accept_event()
		_fit()
		return
	var point := Vector2.ZERO
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		point = event.position
	elif event is InputEventScreenTouch and event.pressed:
		point = event.position
	else:
		return
	if focused >= 0:
		focused = -1
	else:
		var columns := _view_columns()
		var count := 12 if not reference.is_empty() else 6
		var cell := Vector2(world.size.x / columns, world.size.y / ceili(float(count) / columns))
		var slot := int(point.x / cell.x) + int(point.y / cell.y) * columns
		focused = clampi(slot / 2 if not reference.is_empty() else slot, 0, 5)
	_fit()

func _view_columns() -> int:
	if not reference.is_empty():
		return 2 if size.x >= 700 else 1
	return 1 if focused >= 0 else (3 if size.x >= 1000 else (2 if size.x >= 650 else 1))

func _fit() -> void:
	if world == null:
		return
	if not reference.is_empty():
		var count := 2 if focused >= 0 else 12
		var rows := ceili(float(count) / _view_columns())
		world.custom_minimum_size.y = rows * (350 if focused >= 0 and size.x >= 700 else 220)
		world.queue_redraw()
		return
	var columns := 3 if size.x >= 1000 else (2 if size.x >= 650 else 1)
	world.custom_minimum_size.y = 350 if focused >= 0 else (6 / columns) * (180 if size.y < 850 and size.x >= 1000 else (230 if size.x >= 650 else 220))
	world.queue_redraw()

func _start() -> void:
	if worker > 0 and OS.is_process_running(worker):
		return
	var executable := ProjectSettings.globalize_path("res://../build/Release/swimmer_experiment.exe")
	if not FileAccess.file_exists(executable):
		status.text = "Build swimmer_experiment first; its executable was not found."
		return
	directory = ProjectSettings.globalize_path("res://../build/swimmer-live/%d-%d" % [OS.get_process_id(), Time.get_ticks_usec()])
	DirAccess.make_dir_recursive_absolute(directory)
	latest.clear()
	baseline.clear()
	preview.clear()
	history.clear()
	seen_update = -1
	replay_time = 0.0
	compare.set_pressed_no_signal(false)
	compare.text = "Show untrained"
	# 50 ms per update is an explicit viewing throttle, not benchmark timing.
	worker = OS.create_process(executable, PackedStringArray([directory, "300", "1", "1024", "50", str(body_choice.selected)]), false)
	status.text = "Starting from random weights…" if worker > 0 else "Could not start the swimmer experiment."
	restart.disabled = worker > 0

func _read(name: String) -> Dictionary:
	return _read_file(directory.path_join(name))

func _read_file(path: String) -> Dictionary:
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		return {}
	var data: Variant = JSON.parse_string(file.get_as_text())
	return data if data is Dictionary else {}

func _poll() -> void:
	if directory.is_empty():
		return
	var data := _read("live.json")
	if not reference.is_empty() and worker == -1:
		data = _read("seed-1-update-300.json")
	if not data.is_empty():
		latest = data
		if int(data.get("update", -1)) == 0:
			baseline = data
		if baseline.is_empty():
			baseline = _read("seed-1-update-0.json")
		var update := int(data.get("update", 0))
		if update != seen_update:
			seen_update = update
			history.append(Vector2(float(data.get("training_seconds", 0)), float(data.get("success_rate", 0))))
	data = _read("preview.json")
	if not data.is_empty():
		preview = data
	var running := worker > 0 and OS.is_process_running(worker)
	restart.disabled = running
	body_choice.disabled = running
	compare.disabled = baseline.is_empty()
	if latest.is_empty():
		if worker > 0 and not running:
			status.text = "Experiment stopped before producing results."
		return
	var update := int(preview.get("update", latest.get("update", 0))) if running else int(latest.get("update", 0))
	status.text = "Untrained policy replay" if compare.button_pressed else (("Live training" if running else "Recorded policy evaluation") + " · update %d" % update)
	metrics.text = "Food reached: %.0f%% of 128 evaluation attempts · %.2f s training compute" % [float(latest.get("success_rate", 0)) * 100.0, float(latest.get("training_seconds", 0))]
	if float(latest.get("median_food_seconds", -1)) >= 0:
		metrics.text += " · median swim %.1f s" % float(latest.median_food_seconds)
	if not running:
		explanation.text = "Training finished. Replaying six recorded evaluation attempts; Start from scratch runs real training again."
	else:
		explanation.text = "Live CPU previews use the latest policy from 1,024 CUDA training copies. Viewing pace is deliberately slowed."
	if not reference.is_empty():
		var current_data := baseline if compare.button_pressed else latest
		var reference_data := reference_baseline if compare.button_pressed else reference
		status.text = "Untrained comparison · both update 0" if compare.button_pressed else (("Live eel training · latest recorded evaluations" if running else "Recorded comparison") + " · eel update %d / tadpole update 300" % int(latest.get("update", 0)))
		metrics.text = _comparison_metric("Tadpole", reference_data) + "\n" + _comparison_metric("Baby eel", current_data)
		explanation.text = "Same food, starting state, body length and motor limit. Arrow keys switch six matched starts; select a pond to toggle the overview."

func _comparison_metric(name: String, data: Dictionary) -> String:
	var median := float(data.get("median_food_seconds", -1))
	return "%s: %.0f%% reach food in 128 matched attempts%s" % [name, float(data.get("success_rate", 0)) * 100.0, " · median %.1f s" % median if median >= 0 else ""]

func _process(delta: float) -> void:
	poll_time += delta
	replay_time += delta
	if poll_time >= 0.05:
		poll_time = 0.0
		_poll()
	world.queue_redraw()

func _states() -> Array:
	var running := worker > 0 and OS.is_process_running(worker)
	if running and reference.is_empty() and not compare.button_pressed and preview.has("states"):
		return preview.states
	var data := baseline if compare.button_pressed else latest
	var traces: Array = data.get("traces", [])
	var result: Array = []
	for trace: Array in traces:
		if not trace.is_empty():
			var duration := float(trace[-1][7]) * 0.05
			var index := mini(int(fmod(replay_time, duration + 1.0) / 0.2), trace.size() - 1)
			result.append(trace[index])
	return result

func _display_entries() -> Array:
	var entries: Array = []
	var indices: Array = range(6) if focused < 0 else [focused]
	if reference.is_empty():
		var states := _states()
		for i: int in indices:
			entries.append({"state": states[i] if states.size() > i else [], "label": "Swimmer %d" % (i + 1), "color": COLORS[i]})
		return entries
	var current_data := baseline if compare.button_pressed else latest
	var reference_data := reference_baseline if compare.button_pressed else reference
	var a: Array = reference_data.get("traces", [])
	var b: Array = current_data.get("traces", [])
	for i: int in indices:
		var ref_trace: Array = a[i] if a.size() > i else []
		var eel_trace: Array = b[i] if b.size() > i else []
		var duration := 0.0
		for trace: Array in [ref_trace, eel_trace]:
			if not trace.is_empty():
				duration = maxf(duration, float(trace[-1][7]) * 0.05)
		var frame := int(fmod(replay_time, duration + 1.0) / 0.2)
		entries.append({"state": ref_trace[mini(frame, ref_trace.size() - 1)] if not ref_trace.is_empty() else [], "label": "Tadpole · start %d" % (i + 1), "color": COLORS[0]})
		entries.append({"state": eel_trace[mini(frame, eel_trace.size() - 1)] if not eel_trace.is_empty() else [], "label": "Baby eel · start %d" % (i + 1), "color": COLORS[1]})
	return entries

func _draw_world() -> void:
	var entries := _display_entries()
	var columns := _view_columns()
	var rows := ceili(float(entries.size()) / columns)
	var cell := Vector2(world.size.x / columns, world.size.y / rows)
	var font := ThemeDB.fallback_font
	for i in entries.size():
		var entry: Dictionary = entries[i]
		var creature_color: Color = entry.color
		var slot := i
		var rect := Rect2(Vector2(slot % columns, slot / columns) * cell + Vector2(4, 4), cell - Vector2(8, 8))
		var style := StyleBoxFlat.new()
		style.bg_color = Color("172d37")
		style.border_color = Color("33515c")
		if world.has_focus() and (focused >= 0 or i == 0):
			style.border_color = TEXT
		style.set_border_width_all(1)
		style.set_corner_radius_all(8)
		world.draw_style_box(style, rect)
		world.draw_string(font, rect.position + Vector2(14, 24), entry.label, HORIZONTAL_ALIGNMENT_LEFT, -1, 16, creature_color)
		var s: Array = entry.state
		if s.is_empty():
			continue
		var origin := rect.get_center() + Vector2(0, 12)
		var scale := minf(rect.size.x / 6.8, (rect.size.y - 40) / 5.5)
		var position := origin + Vector2(float(s[0]), float(s[1])) * scale
		var food := origin + Vector2(float(s[5]), float(s[6])) * scale
		world.draw_circle(food, 10, Color(1, 0.75, 0.3, 0.12))
		world.draw_circle(food, 5, ACCENT)
		var angle := float(s[2])
		var head_length := float(s[8]) if s.size() > 8 else 0.5
		var first_tail := float(s[9]) if s.size() > 9 else 0.65
		var second_tail := float(s[10]) if s.size() > 10 else 0.65
		var fin_drag := float(s[11]) if s.size() > 11 else 2.0
		var direction := Vector2(cos(angle), sin(angle))
		var hinge := position - direction * head_length * 0.5 * scale
		var tail_angle := angle + float(s[3])
		var joint := hinge - Vector2(cos(tail_angle), sin(tail_angle)) * first_tail * scale
		var tip_angle := tail_angle + float(s[4])
		var tip := joint - Vector2(cos(tip_angle), sin(tip_angle)) * second_tail * scale
		var eel := s.size() > 13 and int(s[13]) != 0
		var segment_width := 0.15 * scale if eel else 7.0
		world.draw_line(hinge, joint, creature_color.darkened(0.15), segment_width, true)
		world.draw_line(joint, tip, creature_color.darkened(0.3), segment_width if eel else 5.0, true)
		if eel:
			world.draw_circle(joint, segment_width * 0.5, creature_color.darkened(0.15))
			world.draw_circle(tip, segment_width * 0.5, creature_color.darkened(0.3))
		if fin_drag > 2.0:
			var normal := Vector2(-sin(tip_angle), cos(tip_angle)) * 0.18 * scale
			var middle := joint.lerp(tip, 0.6)
			world.draw_colored_polygon(PackedVector2Array([joint, middle + normal, tip, middle - normal]), creature_color.darkened(0.25))
		if not eel:
			world.draw_circle(joint, 4, creature_color)
		if s.size() > 12 and int(s[12]) != 0:
			var outline := PackedVector2Array()
			var perpendicular := Vector2(-direction.y, direction.x)
			for q in 11:
				var t := float(q) / 10.0
				var width := 0.06 + 0.17 * sin(PI * t) * (0.6 + 0.4 * t)
				outline.append(position + (direction * (t - 0.5) * head_length + perpendicular * width) * scale)
			for q in range(10, -1, -1):
				var t := float(q) / 10.0
				var width := 0.06 + 0.17 * sin(PI * t) * (0.6 + 0.4 * t)
				outline.append(position + (direction * (t - 0.5) * head_length - perpendicular * width) * scale)
			world.draw_colored_polygon(outline, creature_color)
		else:
			world.draw_line(hinge, position + direction * head_length * 0.5 * scale, creature_color, segment_width if eel else 15.0, true)
			if eel:
				world.draw_circle(position + direction * head_length * 0.5 * scale, segment_width * 0.5, creature_color)
		world.draw_circle(position + direction * head_length * 0.36 * scale, 3, BG)
		var distance := Vector2(float(s[0]) - float(s[5]), float(s[1]) - float(s[6])).length()
		var reached := distance <= 0.1801
		var caption := "Food reached · %.1f s" if reached else "Attempt · %.1f s"
		world.draw_string(font, rect.position + Vector2(14, rect.size.y - 14), caption % (float(s[7]) * 0.05), HORIZONTAL_ALIGNMENT_LEFT, -1, 14, MUTED)

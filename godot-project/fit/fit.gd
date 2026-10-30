extends Control

const NativeLabFactory = preload("res://shared/native_lab_factory.gd")

# Live view of a neural network learning a curve. Every frame the GPU runs a
# batch of training steps (src/demos/function_fit), and this script draws the
# network's prediction over the target plus a loss history. The UI is built in
# code so the scene file stays trivial.

const PANEL_WIDTH := 320
const PLOT_POINTS := 400
const LOSS_HISTORY := 900
const Y_RANGE := 1.25

const BG := Color(0.09, 0.10, 0.12)
const GRID := Color(0.22, 0.24, 0.28)
const TARGET_COLOR := Color(0.55, 0.57, 0.62, 0.55)
const PREDICTION_COLOR := Color(0.25, 0.80, 0.95)
const LOSS_COLOR := Color(0.98, 0.65, 0.25)

var trainer
var xs := PackedFloat32Array()
var target_ys := PackedFloat32Array()
var pred_ys := PackedFloat32Array()
var loss_history: Array[float] = []
var stats := {}
var paused := false

var target_option: OptionButton
var activation_option: OptionButton
var width_spin: SpinBox
var depth_spin: SpinBox
var lr_slider: HSlider
var lr_label: Label
var speed_slider: HSlider
var speed_label: Label
var pause_button: Button
var stats_label: Label
var status_label: Label
var plot: Control
var loss_plot: Control

func _ready() -> void:
	var native := NativeLabFactory.create("NetworkTrainer", ["get_x_min", "get_x_max", "get_target_names", "configure", "is_ready", "train", "predict", "get_error", "target_values", "get_num_params"])
	trainer = native.instance
	_build_ui()
	if trainer == null:
		status_label.text = native.error
		NativeLabFactory.disable_controls(self)
		set_process(false)
		return
	xs.resize(PLOT_POINTS)
	for i in PLOT_POINTS:
		xs[i] = lerpf(trainer.get_x_min(), trainer.get_x_max(), float(i) / (PLOT_POINTS - 1))
	print(NativeLabFactory.device_description())
	_restart()

func _exit_tree() -> void:
	trainer = null

func _unhandled_key_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_cancel"):
		get_viewport().set_input_as_handled()
		DemoLaunch.show_menu()

func _process(_delta: float) -> void:
	if trainer == null:
		return
	if trainer.is_ready() and not paused:
		stats = trainer.train(int(speed_slider.value))
		if stats.has("loss"):
			loss_history.append(maxf(stats["loss"], 1e-9))
			if loss_history.size() > LOSS_HISTORY:
				loss_history.pop_front()
		pred_ys = trainer.predict(xs)
		if not trainer.is_ready():
			status_label.text = trainer.get_error()
	_update_stats()
	plot.queue_redraw()
	loss_plot.queue_redraw()

func _restart() -> void:
	if trainer == null:
		return
	loss_history.clear()
	stats = {}
	var ok: bool = trainer.configure(target_option.selected, int(width_spin.value), int(depth_spin.value),
			activation_option.get_selected_id(), _learning_rate())
	target_ys = trainer.target_values(xs)
	pred_ys = trainer.predict(xs) if ok else PackedFloat32Array()
	status_label.text = ("%d trainable parameters" % trainer.get_num_params()) if ok else trainer.get_error()

func _learning_rate() -> float:
	return pow(10.0, lr_slider.value)

# --- UI ---------------------------------------------------------------------

func _build_ui() -> void:
	var root := HBoxContainer.new()
	root.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(root)

	var panel := PanelContainer.new()
	panel.custom_minimum_size.x = PANEL_WIDTH
	root.add_child(panel)
	var margin := MarginContainer.new()
	for side in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 14)
	panel.add_child(margin)
	var controls := VBoxContainer.new()
	controls.add_theme_constant_override("separation", 8)
	margin.add_child(controls)
	var back := Button.new()
	back.name = "BackToDemos"
	back.text = "Pond menu"
	back.custom_minimum_size.y = 44
	back.pressed.connect(DemoLaunch.show_menu)
	controls.add_child(back)

	controls.add_child(_heading("Neural network, trained live on the GPU"))
	controls.add_child(_note("Hand-written CUDA: forward, backprop and Adam. No ML framework."))

	target_option = OptionButton.new()
	if trainer != null:
		for target_name in trainer.get_target_names():
			target_option.add_item(target_name)
	if target_option.item_count > 1:
		target_option.select(1)   # Wiggle: takes a few seconds, so the learning is visible
	target_option.item_selected.connect(func(_i): _restart())
	controls.add_child(_labeled("Target function", target_option))

	activation_option = OptionButton.new()
	activation_option.add_item("tanh", 1)
	activation_option.add_item("ReLU", 2)
	activation_option.add_item("none (linear model)", 0)
	activation_option.item_selected.connect(func(_i): _restart())
	controls.add_child(_labeled("Hidden activation", activation_option))

	width_spin = _spin(1, 256, 32)
	controls.add_child(_labeled("Hidden width", width_spin))
	depth_spin = _spin(1, 6, 2)
	controls.add_child(_labeled("Hidden layers", depth_spin))

	lr_slider = _slider(-4.0, -1.0, -2.0, 0.05)
	lr_label = Label.new()
	lr_slider.value_changed.connect(func(_v): _update_slider_labels())
	lr_slider.drag_ended.connect(func(_changed): _restart())
	controls.add_child(_labeled("Learning rate (log scale, restarts)", lr_slider))
	controls.add_child(lr_label)

	speed_slider = _slider(1, 200, 5, 1)
	speed_label = Label.new()
	speed_slider.value_changed.connect(func(_v): _update_slider_labels())
	controls.add_child(_labeled("Training steps per frame", speed_slider))
	controls.add_child(speed_label)

	var buttons := HBoxContainer.new()
	var restart_button := Button.new()
	restart_button.text = "Restart"
	restart_button.pressed.connect(_restart)
	buttons.add_child(restart_button)
	pause_button = Button.new()
	pause_button.text = "Pause"
	pause_button.pressed.connect(_toggle_pause)
	buttons.add_child(pause_button)
	controls.add_child(buttons)

	controls.add_child(HSeparator.new())
	status_label = _note("")
	controls.add_child(status_label)

	var charts := VBoxContainer.new()
	charts.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	root.add_child(charts)
	plot = Control.new()
	plot.size_flags_vertical = Control.SIZE_EXPAND_FILL
	plot.size_flags_stretch_ratio = 3.0
	plot.draw.connect(_draw_curve_plot)
	charts.add_child(plot)
	stats_label = Label.new()
	stats_label.position = Vector2(24, 44)
	stats_label.add_theme_color_override("font_color", Color(0.85, 0.87, 0.9))
	plot.add_child(stats_label)
	loss_plot = Control.new()
	loss_plot.size_flags_vertical = Control.SIZE_EXPAND_FILL
	loss_plot.size_flags_stretch_ratio = 1.0
	loss_plot.draw.connect(_draw_loss_plot)
	charts.add_child(loss_plot)

	_update_slider_labels()

func _toggle_pause() -> void:
	if trainer == null:
		return
	paused = not paused
	pause_button.text = "Resume" if paused else "Pause"

func _update_slider_labels() -> void:
	lr_label.text = "  = %.4f" % _learning_rate()
	speed_label.text = "  = %d steps per frame" % int(speed_slider.value)

func _update_stats() -> void:
	if not stats.has("step"):
		stats_label.text = ""
		return
	var fps := Engine.get_frames_per_second()
	stats_label.text = "Step  %d\nLoss (MSE)  %s\nGPU time / step  %.3f ms\nSteps / second  %d\nFrame rate  %d fps" % [
			stats["step"], _scientific(stats["loss"]), stats["ms_per_step"], int(speed_slider.value * fps), fps]

# GDScript's % formatting has no %e.
func _scientific(v: float) -> String:
	if v <= 0.0:
		return "0"
	var exponent := floori(log(v) / log(10.0))
	return "%.2fe%d" % [v / pow(10.0, exponent), exponent]

func _heading(text: String) -> Label:
	var l := Label.new()
	l.text = text
	l.add_theme_font_size_override("font_size", 18)
	l.autowrap_mode = TextServer.AUTOWRAP_WORD
	return l

func _note(text: String) -> Label:
	var l := Label.new()
	l.text = text
	l.modulate = Color(1, 1, 1, 0.65)
	l.autowrap_mode = TextServer.AUTOWRAP_WORD
	return l

func _labeled(text: String, control: Control) -> VBoxContainer:
	var box := VBoxContainer.new()
	var l := Label.new()
	l.text = text
	box.add_child(l)
	box.add_child(control)
	return box

func _spin(lo: float, hi: float, value: float) -> SpinBox:
	var s := SpinBox.new()
	s.min_value = lo
	s.max_value = hi
	s.value = value
	s.value_changed.connect(func(_v): _restart())
	return s

func _slider(lo: float, hi: float, value: float, step: float) -> HSlider:
	var s := HSlider.new()
	s.min_value = lo
	s.max_value = hi
	s.step = step
	s.value = value
	return s

# --- Drawing ----------------------------------------------------------------

func _draw_curve_plot() -> void:
	var r := Rect2(Vector2.ZERO, plot.size).grow(-16)
	plot.draw_rect(Rect2(Vector2.ZERO, plot.size), BG)
	for i in 5:
		var y := lerpf(r.position.y, r.end.y, i / 4.0)
		plot.draw_line(Vector2(r.position.x, y), Vector2(r.end.x, y), GRID)
	plot.draw_line(Vector2(r.get_center().x, r.position.y), Vector2(r.get_center().x, r.end.y), GRID)
	_draw_series(plot, r, target_ys, TARGET_COLOR, 9.0)
	_draw_series(plot, r, pred_ys, PREDICTION_COLOR, 2.5)
	var font := ThemeDB.fallback_font
	plot.draw_string(font, r.position + Vector2(8, 20), "target", HORIZONTAL_ALIGNMENT_LEFT, -1, 16, TARGET_COLOR)
	plot.draw_string(font, r.position + Vector2(72, 20), "network prediction", HORIZONTAL_ALIGNMENT_LEFT, -1, 16, PREDICTION_COLOR)

func _draw_series(canvas: Control, r: Rect2, ys: PackedFloat32Array, color: Color, width: float) -> void:
	if ys.size() != xs.size() or ys.is_empty():
		return
	var pts := PackedVector2Array()
	pts.resize(ys.size())
	for i in ys.size():
		var y := clampf(ys[i], -Y_RANGE, Y_RANGE)
		pts[i] = Vector2(lerpf(r.position.x, r.end.x, float(i) / (ys.size() - 1)),
				remap(y, -Y_RANGE, Y_RANGE, r.end.y, r.position.y))
	canvas.draw_polyline(pts, color, width, true)

func _draw_loss_plot() -> void:
	var r := Rect2(Vector2.ZERO, loss_plot.size).grow(-16)
	loss_plot.draw_rect(Rect2(Vector2.ZERO, loss_plot.size), BG.darkened(0.2))
	var font := ThemeDB.fallback_font
	loss_plot.draw_string(font, r.position + Vector2(8, 18), "loss (log scale, last %d frames)" % LOSS_HISTORY,
			HORIZONTAL_ALIGNMENT_LEFT, -1, 14, LOSS_COLOR)
	if loss_history.size() < 2:
		return
	var lo := 1e9
	var hi := -1e9
	for v in loss_history:
		lo = minf(lo, log(v))
		hi = maxf(hi, log(v))
	if hi - lo < 1e-3:
		hi = lo + 1e-3
	var pts := PackedVector2Array()
	pts.resize(loss_history.size())
	for i in loss_history.size():
		pts[i] = Vector2(lerpf(r.position.x, r.end.x, float(i) / (LOSS_HISTORY - 1)),
				remap(log(loss_history[i]), lo, hi, r.end.y, r.position.y + 24))
	loss_plot.draw_polyline(pts, LOSS_COLOR, 2.0, true)
	var label_color := Color(LOSS_COLOR, 0.7)
	loss_plot.draw_string(font, Vector2(r.end.x - 80, r.position.y + 34), _scientific(exp(hi)),
			HORIZONTAL_ALIGNMENT_LEFT, -1, 13, label_color)
	loss_plot.draw_string(font, Vector2(r.end.x - 80, r.end.y), _scientific(exp(lo)),
			HORIZONTAL_ALIGNMENT_LEFT, -1, 13, label_color)

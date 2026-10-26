extends Control

# PPO learning CartPole, live. Training runs on a GPU worker thread inside the
# extension (src/demos/ppo_session); this scene steps a dozen showcase carts in
# real time with the newest policy and draws the learning curve. At full speed
# training solves CartPole in about a third of a second, so the default speed
# is throttled to a few updates per second to make the learning watchable.

const PANEL_WIDTH := 320
const COLS := 4
const ROWS := 3
const SOLVED := 475.0
const MAX_LENGTH := 500.0
const X_LIMIT := 2.4
const ENV_CHOICES := [256, 1024, 4096]

const BG := Color(0.09, 0.10, 0.12)
const CELL := Color(0.12, 0.13, 0.16)
const TRACK := Color(0.35, 0.37, 0.42)
const CART := Color(0.80, 0.82, 0.86)
const POLE_OK := Color(0.30, 0.85, 0.55)
const POLE_FAIL := Color(0.95, 0.35, 0.30)
const CURVE := Color(0.98, 0.65, 0.25)
const GRID := Color(0.22, 0.24, 0.28)

var lab := PpoLab.new()
var run_seed := 1
var carts := PackedFloat32Array()
var curve := PackedVector2Array()
var stats := {}
var paused := false
var watch_mode := false
var layout: BoxContainer
var panel_scroll: ScrollContainer

var envs_option: OptionButton
var speed_slider: HSlider
var speed_label: Label
var full_speed: CheckBox
var greedy: CheckBox
var pause_button: Button
var stats_label: Label
var status_label: Label
var cart_view: Control
var curve_view: Control

func _ready() -> void:
	Engine.physics_ticks_per_second = 50   # CartPole's own time step is 20 ms: real time
	watch_mode = DemoLaunch.watching("cartpole")
	_build_ui()
	resized.connect(_fit_layout)
	_fit_layout()
	if watch_mode:
		var weights := DemoLaunch.read_policy("cartpole")
		var ok := not weights.is_empty() and lab.watch_policy(weights)
		status_label.text = "Saved policy · seed 1 · 20 training updates" if ok else (DemoLaunch.last_error if not DemoLaunch.last_error.is_empty() else lab.get_error())
	else:
		print(CudaDevice.new().describe())
		_restart()

func _exit_tree() -> void:
	lab.stop()   # join the training thread before the scene goes away

func _unhandled_key_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_cancel"):
		get_viewport().set_input_as_handled()
		DemoLaunch.show_menu()

func _physics_process(_delta: float) -> void:
	carts = lab.step_showcase(greedy.button_pressed)
	cart_view.queue_redraw()

func _process(_delta: float) -> void:
	stats = lab.get_stats()
	curve = lab.get_learning_curve()
	var err: String = lab.get_error()
	if not err.is_empty():
		status_label.text = err
	elif lab.is_finished():
		status_label.text = "Training finished (all updates done)."
	_update_stats()
	curve_view.queue_redraw()

func _restart() -> void:
	paused = true
	carts.clear()
	curve.clear()
	stats.clear()
	pause_button.text = "Start learning"
	var ups := 0.0 if full_speed.button_pressed else pow(10.0, speed_slider.value)
	var ok := lab.start(ENV_CHOICES[envs_option.selected], run_seed, COLS * ROWS, ups)
	status_label.text = ("Untrained · seed %d" % run_seed) if ok else lab.get_error()
	_apply_speed()

func _apply_speed() -> void:
	var ups := 0.0 if full_speed.button_pressed else pow(10.0, speed_slider.value)
	lab.set_updates_per_second(ups)
	speed_label.text = "  = full speed" if full_speed.button_pressed else "  = %.1f updates per second" % ups

func _restart_with_new_seed() -> void:
	run_seed += 1
	_restart()

func _toggle_pause() -> void:
	paused = not paused
	lab.set_paused(paused)
	pause_button.text = ("Start learning" if stats.get("update", 0) == 0 else "Resume") if paused else "Pause"

# --- UI ---------------------------------------------------------------------

func _build_ui() -> void:
	var root := BoxContainer.new()
	layout = root
	root.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(root)

	panel_scroll = ScrollContainer.new()
	panel_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	root.add_child(panel_scroll)
	var panel := PanelContainer.new()
	panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	panel.size_flags_vertical = Control.SIZE_EXPAND_FILL
	panel_scroll.add_child(panel)
	var margin := MarginContainer.new()
	for side in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 14)
	panel.add_child(margin)
	var controls := VBoxContainer.new()
	controls.add_theme_constant_override("separation", 8)
	margin.add_child(controls)
	var back := Button.new()
	back.name = "BackToDemos"
	back.text = "Back to demos"
	back.custom_minimum_size.y = 44
	back.pressed.connect(DemoLaunch.show_menu)
	controls.add_child(back)

	var title := Label.new()
	title.text = "CartPole" if watch_mode else "PPO learning CartPole, live"
	title.add_theme_font_size_override("font_size", 18)
	controls.add_child(title)
	if not watch_mode:
		controls.add_child(_note("Rollouts, GAE, the PPO loss, backprop and Adam all run on the GPU as hand-written CUDA."))
	var outer_controls := controls
	var training_options := VBoxContainer.new()
	training_options.add_theme_constant_override("separation", 8)
	training_options.visible = not watch_mode
	controls.add_child(training_options)
	controls = training_options

	speed_slider = HSlider.new()
	speed_slider.min_value = log(0.5) / log(10.0)
	speed_slider.max_value = log(50.0) / log(10.0)
	speed_slider.step = 0.01
	speed_slider.value = log(2.0) / log(10.0)
	speed_slider.value_changed.connect(func(_v): _apply_speed())
	controls.add_child(_labeled("Training speed (log scale)", speed_slider))
	speed_label = Label.new()
	controls.add_child(speed_label)
	full_speed = CheckBox.new()
	full_speed.text = "Full speed (solves in ~0.4 s)"
	full_speed.toggled.connect(func(_on): _apply_speed())
	controls.add_child(full_speed)

	envs_option = OptionButton.new()
	for n in ENV_CHOICES:
		envs_option.add_item("%d environments" % n)
	envs_option.select(1)
	envs_option.item_selected.connect(func(_i): _restart())
	controls.add_child(_labeled("Parallel environments (restarts)", envs_option))

	greedy = CheckBox.new()
	greedy.text = "Showcase: greedy actions"
	greedy.button_pressed = true
	controls.add_child(greedy)

	var buttons := HBoxContainer.new()
	var restart := Button.new()
	restart.text = "Restart (new seed)"
	restart.pressed.connect(_restart_with_new_seed)
	buttons.add_child(restart)
	pause_button = Button.new()
	pause_button.text = "Pause"
	pause_button.pressed.connect(_toggle_pause)
	buttons.add_child(pause_button)
	controls.add_child(buttons)
	controls = outer_controls
	var reset_carts := Button.new()
	reset_carts.text = "Reset showcase carts"
	reset_carts.pressed.connect(func(): lab.reset_showcase())
	controls.add_child(reset_carts)

	controls.add_child(HSeparator.new())
	stats_label = Label.new()
	stats_label.visible = not watch_mode
	controls.add_child(stats_label)
	status_label = _note("")
	controls.add_child(status_label)

	var right := VBoxContainer.new()
	right.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	root.add_child(right)
	cart_view = Control.new()
	cart_view.size_flags_vertical = Control.SIZE_EXPAND_FILL
	cart_view.size_flags_stretch_ratio = 3.0
	cart_view.draw.connect(_draw_carts)
	right.add_child(cart_view)
	curve_view = Control.new()
	curve_view.visible = not watch_mode
	curve_view.size_flags_vertical = Control.SIZE_EXPAND_FILL
	curve_view.size_flags_stretch_ratio = 2.0
	curve_view.draw.connect(_draw_curve)
	right.add_child(curve_view)

func _update_stats() -> void:
	if watch_mode:
		return
	pause_button.disabled = not stats.get("policy_ready", false)
	if not stats.has("update"):
		stats_label.text = "Waiting for the first update..."
		return
	stats_label.text = "Update  %d\nEnv steps  %s\nMean episode length  %.0f / 500\nPolicy entropy  %.3f\nApprox KL  %.4f\nClip fraction  %.3f\nLearning rate  %s\nGPU throughput  %.2f M steps/s" % [
			stats["update"], _thousands(stats["env_steps"]), stats["mean_episode_length"], stats["entropy"],
			stats["approx_kl"], stats["clip_fraction"], _scientific(stats["learning_rate"]),
			stats["env_steps_per_second"] / 1e6]

func _fit_layout() -> void:
	var compact := size.x < 850 or size.y > size.x * 1.2
	layout.vertical = compact
	panel_scroll.custom_minimum_size = Vector2(0, 168 if watch_mode else 272) if compact else Vector2(PANEL_WIDTH, 0)
	panel_scroll.size_flags_horizontal = Control.SIZE_EXPAND_FILL if compact else Control.SIZE_FILL
	panel_scroll.size_flags_vertical = Control.SIZE_FILL if compact else Control.SIZE_EXPAND_FILL

# GDScript's % formatting has no %e.
func _scientific(v: float) -> String:
	if v <= 0.0:
		return "0"
	var exponent := floori(log(v) / log(10.0))
	return "%.1fe%d" % [v / pow(10.0, exponent), exponent]

func _thousands(n: int) -> String:
	var s := str(n)
	var out := ""
	for i in s.length():
		if i > 0 and (s.length() - i) % 3 == 0:
			out += ","
		out += s[i]
	return out

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

# --- Drawing ----------------------------------------------------------------

func _draw_carts() -> void:
	cart_view.draw_rect(Rect2(Vector2.ZERO, cart_view.size), BG)
	var font := ThemeDB.fallback_font
	var cell := Vector2(cart_view.size.x / COLS, cart_view.size.y / ROWS)
	for i in COLS * ROWS:
		var r := Rect2(Vector2(i % COLS, i / COLS) * cell, cell).grow(-6)
		cart_view.draw_rect(r, CELL)
		if carts.size() < (i + 1) * 4:
			continue
		var x: float = carts[i * 4 + 0]
		var theta: float = carts[i * 4 + 1]
		var steps: float = carts[i * 4 + 2]
		var failed: bool = carts[i * 4 + 3] > 0.5
		var track_y := r.position.y + r.size.y * 0.78
		var left := r.position.x + 18
		var right := r.end.x - 18
		cart_view.draw_line(Vector2(left, track_y), Vector2(right, track_y), TRACK, 2.0)
		for edge in [left, right]:
			cart_view.draw_line(Vector2(edge, track_y - 8), Vector2(edge, track_y + 8), TRACK, 2.0)
		var cx := remap(x, -X_LIMIT, X_LIMIT, left, right)
		var cart_size := Vector2(34, 16)
		var cart_rect := Rect2(Vector2(cx, track_y) - Vector2(cart_size.x / 2, cart_size.y), cart_size)
		cart_view.draw_rect(cart_rect, CART)
		var pivot := Vector2(cx, cart_rect.position.y)
		var pole_len := r.size.y * 0.55
		var tip := pivot + Vector2(sin(theta), -cos(theta)) * pole_len
		cart_view.draw_line(pivot, tip, POLE_FAIL if failed else POLE_OK, 5.0, true)
		cart_view.draw_circle(pivot, 3.5, CART.darkened(0.3))
		cart_view.draw_string(font, r.position + Vector2(8, 18), "%d steps" % int(steps),
				HORIZONTAL_ALIGNMENT_LEFT, -1, 13, Color(1, 1, 1, 0.6))

func _draw_curve() -> void:
	var r := Rect2(Vector2.ZERO, curve_view.size).grow(-16)
	curve_view.draw_rect(Rect2(Vector2.ZERO, curve_view.size), BG.darkened(0.2))
	var font := ThemeDB.fallback_font
	var plot := Rect2(r.position + Vector2(40, 28), r.size - Vector2(48, 44))
	for v in [0.0, 250.0, 500.0]:
		var y := remap(v, 0.0, MAX_LENGTH, plot.end.y, plot.position.y)
		curve_view.draw_line(Vector2(plot.position.x, y), Vector2(plot.end.x, y), GRID)
		curve_view.draw_string(font, Vector2(r.position.x, y + 5), "%d" % int(v), HORIZONTAL_ALIGNMENT_LEFT, -1, 12, GRID.lightened(0.4))
	var solved_y := remap(SOLVED, 0.0, MAX_LENGTH, plot.end.y, plot.position.y)
	curve_view.draw_dashed_line(Vector2(plot.position.x, solved_y), Vector2(plot.end.x, solved_y), POLE_OK, 1.5, 6.0)
	curve_view.draw_string(font, Vector2(plot.position.x + 6, solved_y + 16), "solved (475)", HORIZONTAL_ALIGNMENT_LEFT, -1, 12, POLE_OK)
	curve_view.draw_string(font, r.position + Vector2(40, 16), "mean episode length vs. environment steps",
			HORIZONTAL_ALIGNMENT_LEFT, -1, 14, CURVE)
	if curve.size() < 2:
		return
	var max_steps := maxf(curve[curve.size() - 1].x, 1.0e6)
	var pts := PackedVector2Array()
	for p in curve:
		pts.append(Vector2(remap(p.x, 0.0, max_steps, plot.position.x, plot.end.x),
				remap(p.y, 0.0, MAX_LENGTH, plot.end.y, plot.position.y)))
	curve_view.draw_polyline(pts, CURVE, 2.5, true)
	curve_view.draw_string(font, Vector2(plot.end.x - 90, plot.end.y + 16), "%.1fM steps" % (max_steps / 1e6),
			HORIZONTAL_ALIGNMENT_LEFT, -1, 12, GRID.lightened(0.4))

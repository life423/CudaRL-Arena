extends Control

# PPO with a Gaussian policy learning to drive a car around a track, live.
# Training runs on a GPU worker thread inside the extension (src/rl/racing_ppo);
# this scene drives a few showcase cars on the CPU with the newest policy, draws
# the track with each car's raycasts, and plots laps per episode. The physics
# step is the environment's own step (50 ms), so the cars drive in real time.

const PANEL_WIDTH := 340
const SHOWCASE_CARS := 8
const ENV_CHOICES := [512, 2048, 8192]
const TIME_LIMIT_LAPS := 4.0   # 50 s at top speed (15 m/s) around a lap of about 188 m
const MAX_LAPS := 4.5
const TRACK_EXTENT := 44.0     # metres from the origin that the track view must show
const CAR_LENGTH := 4.0
const CAR_WIDTH := 2.0
const STEP_HZ := 20           # racing::kDt = 0.05 s
const TARGET_LAPS := 1.0

const BG := Color(0.09, 0.10, 0.12)
const GRASS := Color(0.12, 0.16, 0.13)
const TARMAC := Color(0.22, 0.23, 0.26)
const WALL := Color(0.85, 0.86, 0.90)
const START_LINE := Color(0.98, 0.98, 0.98)
const CRASH := Color(0.95, 0.35, 0.30)
const RAY_NEAR := Color(0.95, 0.35, 0.30)
const RAY_FAR := Color(0.30, 0.85, 0.55)
const CURVE := Color(0.98, 0.65, 0.25)
const GRID := Color(0.22, 0.24, 0.28)

var lab := RacingLab.new()
var run_seed := 1
var cars := PackedFloat32Array()
var car_stride := 16
var track_outer := PackedVector2Array()
var track_inner := PackedVector2Array()
var curve := PackedVector2Array()
var stats := {}
var paused := false

var envs_option: OptionButton
var speed_slider: HSlider
var speed_label: Label
var full_speed: CheckBox
var pause_button: Button
var stats_label: Label
var status_label: Label
var track_view: Control
var curve_view: Control

func _ready() -> void:
	Engine.physics_ticks_per_second = STEP_HZ
	car_stride = lab.get_car_stride()
	track_outer = lab.get_track_outer()
	track_inner = lab.get_track_inner()
	_build_ui()
	print(CudaDevice.new().describe())
	_restart()

func _exit_tree() -> void:
	lab.stop()   # join the training thread before the scene goes away

func _physics_process(_delta: float) -> void:
	cars = lab.step_showcase()
	track_view.queue_redraw()

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
	paused = false
	pause_button.text = "Pause"
	var ok := lab.start(ENV_CHOICES[envs_option.selected], run_seed, SHOWCASE_CARS)
	status_label.text = ("Training with seed %d" % run_seed) if ok else lab.get_error()
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
	pause_button.text = "Resume" if paused else "Pause"

# --- UI ---------------------------------------------------------------------

func _build_ui() -> void:
	var root := HBoxContainer.new()
	root.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(root)

	# The panel scrolls, so the row keeps the window's height on small screens.
	var scroll := ScrollContainer.new()
	scroll.custom_minimum_size.x = PANEL_WIDTH
	root.add_child(scroll)
	var panel := PanelContainer.new()
	panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(panel)
	var margin := MarginContainer.new()
	for side in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 14)
	panel.add_child(margin)
	var controls := VBoxContainer.new()
	controls.add_theme_constant_override("separation", 8)
	margin.add_child(controls)

	var title := Label.new()
	title.text = "PPO driving a track, live"
	title.add_theme_font_size_override("font_size", 18)
	controls.add_child(title)
	controls.add_child(_note("A Gaussian policy outputs steering and throttle from 9 raycasts and its speed. Rollouts, GAE, the loss, backprop and Adam all run on the GPU as hand-written CUDA."))

	speed_slider = HSlider.new()
	speed_slider.min_value = log(0.5) / log(10.0)
	speed_slider.max_value = log(50.0) / log(10.0)
	speed_slider.step = 0.01
	speed_slider.value = log(5.0) / log(10.0)
	speed_slider.value_changed.connect(func(_v): _apply_speed())
	controls.add_child(_labeled("Training speed (log scale)", speed_slider))
	speed_label = Label.new()
	controls.add_child(speed_label)
	full_speed = CheckBox.new()
	full_speed.text = "Full speed (plateau in about 3 s)"
	full_speed.toggled.connect(func(_on): _apply_speed())
	controls.add_child(full_speed)

	envs_option = OptionButton.new()
	for n in ENV_CHOICES:
		envs_option.add_item("%d cars" % n)
	envs_option.select(1)
	envs_option.item_selected.connect(func(_i): _restart())
	controls.add_child(_labeled("Parallel cars in training (restarts)", envs_option))

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
	var reset_cars := Button.new()
	reset_cars.text = "Reset showcase cars"
	reset_cars.pressed.connect(func(): lab.reset_showcase())
	controls.add_child(reset_cars)

	controls.add_child(HSeparator.new())
	stats_label = Label.new()
	controls.add_child(stats_label)
	status_label = _note("")
	controls.add_child(status_label)

	var right := VBoxContainer.new()
	right.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	root.add_child(right)
	track_view = Control.new()
	track_view.size_flags_vertical = Control.SIZE_EXPAND_FILL
	track_view.size_flags_stretch_ratio = 3.0
	track_view.draw.connect(_draw_track)
	right.add_child(track_view)
	curve_view = Control.new()
	curve_view.size_flags_vertical = Control.SIZE_EXPAND_FILL
	curve_view.size_flags_stretch_ratio = 2.0
	curve_view.draw.connect(_draw_curve)
	right.add_child(curve_view)

func _update_stats() -> void:
	if not stats.has("update"):
		stats_label.text = "Waiting for the first update..."
		return
	stats_label.text = "Update  %d\nEnv steps  %s\nLaps per episode  %.2f\nCrashed episodes  %.0f%%\nAction std dev  %.3f\nEntropy  %.2f\nApprox KL  %.4f\nClip fraction  %.3f\nLearning rate  %s\nGPU throughput  %.2f M steps/s" % [
			stats["update"], _thousands(stats["env_steps"]), stats["mean_laps"], 100.0 * stats["crash_fraction"],
			stats["mean_std"], stats["entropy"], stats["approx_kl"], stats["clip_fraction"],
			_scientific(stats["learning_rate"]), stats["env_steps_per_second"] / 1e6]

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

# Track coordinates have y up; the screen has y down, so flip y.
func _to_screen(p: Vector2, origin: Vector2, px_per_m: float) -> Vector2:
	return origin + Vector2(p.x, -p.y) * px_per_m

func _draw_track() -> void:
	var area := track_view.size
	track_view.draw_rect(Rect2(Vector2.ZERO, area), GRASS)
	var px_per_m := minf(area.x, area.y) / (2.0 * TRACK_EXTENT)
	var origin := area * 0.5
	var outer := PackedVector2Array()
	var inner := PackedVector2Array()
	for p in track_outer:
		outer.append(_to_screen(p, origin, px_per_m))
	for p in track_inner:
		inner.append(_to_screen(p, origin, px_per_m))
	# The tarmac is the outer region with the inner region cut out.
	track_view.draw_colored_polygon(outer, TARMAC)
	track_view.draw_colored_polygon(inner, GRASS)
	track_view.draw_polyline(outer, WALL, 2.0, true)
	track_view.draw_polyline(inner, WALL, 2.0, true)
	# The start line sits on the centreline at angle zero.
	var start_a := _to_screen(track_inner[0], origin, px_per_m)
	var start_b := _to_screen(track_outer[0], origin, px_per_m)
	track_view.draw_line(start_a, start_b, START_LINE, 3.0)

	var n := cars.size() / car_stride
	for i in n:
		_draw_car(i, origin, px_per_m)

	var font := ThemeDB.fallback_font
	track_view.draw_string(font, Vector2(12, 22), "Showcase cars, driven by the newest policy (mean action)",
			HORIZONTAL_ALIGNMENT_LEFT, -1, 13, Color(1, 1, 1, 0.6))

func _draw_car(i: int, origin: Vector2, px_per_m: float) -> void:
	var base := i * car_stride
	var pos := Vector2(cars[base + 0], cars[base + 1])
	var heading: float = cars[base + 2]
	var crashed: bool = cars[base + 6] > 0.5
	var color := Color.from_hsv(float(i) / float(SHOWCASE_CARS), 0.55, 0.95)
	var dir := Vector2(cos(heading), sin(heading))
	var side := Vector2(-dir.y, dir.x)

	# Raycasts: ray r sits at heading + (r - 4) * PI / 8, from -90 to +90 degrees.
	var rays := car_stride - 7
	for r in rays:
		var angle := heading + (float(r) - 0.5 * float(rays - 1)) * PI / float(rays - 1)
		var ray_dir := Vector2(cos(angle), sin(angle))
		var dist: float = cars[base + 7 + r]
		var near := clampf(dist / 40.0, 0.0, 1.0)
		var a := _to_screen(pos, origin, px_per_m)
		var b := _to_screen(pos + ray_dir * dist, origin, px_per_m)
		track_view.draw_line(a, b, RAY_NEAR.lerp(RAY_FAR, near).darkened(0.1), 1.0, true)

	var corners := PackedVector2Array()
	for c in [Vector2(1, 1), Vector2(1, -1), Vector2(-1, -1), Vector2(-1, 1)]:
		var local: Vector2 = dir * (c.x * CAR_LENGTH * 0.5) + side * (c.y * CAR_WIDTH * 0.5)
		corners.append(_to_screen(pos + local, origin, px_per_m))
	var body := CRASH if crashed else color
	track_view.draw_colored_polygon(corners, body)
	# The nose marks the heading.
	var nose := _to_screen(pos + dir * CAR_LENGTH * 0.5, origin, px_per_m)
	track_view.draw_circle(nose, 0.5 * px_per_m, Color.WHITE)

func _draw_curve() -> void:
	var r := Rect2(Vector2.ZERO, curve_view.size).grow(-16)
	curve_view.draw_rect(Rect2(Vector2.ZERO, curve_view.size), BG.darkened(0.2))
	var font := ThemeDB.fallback_font
	var plot := Rect2(r.position + Vector2(40, 28), r.size - Vector2(48, 44))
	for v in [0.0, 1.0, 2.0, 3.0, 4.0]:
		var y := remap(v, 0.0, MAX_LAPS, plot.end.y, plot.position.y)
		curve_view.draw_line(Vector2(plot.position.x, y), Vector2(plot.end.x, y), GRID)
		curve_view.draw_string(font, Vector2(r.position.x, y + 5), "%d" % int(v), HORIZONTAL_ALIGNMENT_LEFT, -1, 12, GRID.lightened(0.4))
	var limit_y := remap(TIME_LIMIT_LAPS, 0.0, MAX_LAPS, plot.end.y, plot.position.y)
	curve_view.draw_dashed_line(Vector2(plot.position.x, limit_y), Vector2(plot.end.x, limit_y), RAY_FAR, 1.5, 6.0)
	curve_view.draw_string(font, Vector2(plot.position.x + 6, limit_y + 16),
			"time limit at top speed (about 4 laps)", HORIZONTAL_ALIGNMENT_LEFT, -1, 12, RAY_FAR)
	curve_view.draw_string(font, r.position + Vector2(40, 16), "laps per episode vs. environment steps",
			HORIZONTAL_ALIGNMENT_LEFT, -1, 14, CURVE)
	if curve.size() < 2:
		return
	var max_steps := maxf(curve[curve.size() - 1].x, 1.0e6)
	var pts := PackedVector2Array()
	for p in curve:
		pts.append(Vector2(remap(p.x, 0.0, max_steps, plot.position.x, plot.end.x),
				remap(p.y, 0.0, MAX_LAPS, plot.end.y, plot.position.y)))
	curve_view.draw_polyline(pts, CURVE, 2.5, true)
	curve_view.draw_string(font, Vector2(plot.end.x - 90, plot.end.y + 16), "%.1fM steps" % (max_steps / 1e6),
			HORIZONTAL_ALIGNMENT_LEFT, -1, 12, GRID.lightened(0.4))

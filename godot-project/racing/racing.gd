extends Control

# PPO with a Gaussian policy learning to drive a car around a track, live.
# Training runs on a GPU worker thread inside the extension (src/rl/racing_ppo);
# this scene drives a few showcase cars on the CPU with the newest policy, draws
# the track with each car's raycasts, and plots laps per episode. The physics
# step is the environment's own step (50 ms), so the cars drive in real time.

const PANEL_WIDTH := 288
const SPACE := 8
const TYPE_CAPTION := 12
const TYPE_BODY := 14
const TYPE_SECTION := 16
const TYPE_TITLE := 24
const SURFACE := Color("1b2028")
const SURFACE_RAISED := Color("272f3a")
const BORDER := Color("394454")
const TEXT := Color("edf1f7")
const TEXT_MUTED := Color("abb8c9")
const SHOWCASE_CARS := 8
const ENV_CHOICES := [512, 2048, 8192]
const MAX_LAPS := 2.0
const CAR_LENGTH := 4.0
const CAR_WIDTH := 2.0
const STEP_HZ := 20           # racing::kDt = 0.05 s

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
var car_stride := 25
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
var show_sensors := false
var inspected_car := 0
var layout: BoxContainer
var panel_scroll: ScrollContainer
var previous_speeds := PackedFloat32Array()
var braking := PackedByteArray()
var evaluation_started := false
var untrained_view := false
var baseline_checkbox: CheckBox
var side_panel: PanelContainer
var phase_label: Label
var seed_label: Label
var metric_values := {}
var training_readout: VBoxContainer
var training_summary: Label
var settings_button: Button
var settings_panel: VBoxContainer
var diagnostics_label: Label
var inspect_option: OptionButton
var reset_button: Button

func _ready() -> void:
	Engine.physics_ticks_per_second = STEP_HZ
	car_stride = lab.get_car_stride()
	track_outer = lab.get_track_outer()
	track_inner = lab.get_track_inner()
	_build_ui()
	resized.connect(_fit_layout)
	_fit_layout()
	print(CudaDevice.new().describe())
	_restart()

func _exit_tree() -> void:
	lab.stop()   # join the training thread before the scene goes away

func _physics_process(_delta: float) -> void:
	cars = lab.step_showcase()
	braking.resize(SHOWCASE_CARS)
	for i in SHOWCASE_CARS:
		if cars.size() >= (i + 1) * car_stride:
			var speed: float = cars[i * car_stride + 3]
			braking[i] = 1 if previous_speeds.size() > i and speed < previous_speeds[i] - 0.08 else 0
	previous_speeds.resize(SHOWCASE_CARS)
	for i in SHOWCASE_CARS:
		if cars.size() >= (i + 1) * car_stride:
			previous_speeds[i] = cars[i * car_stride + 3]
	track_view.queue_redraw()

func _process(_delta: float) -> void:
	stats = lab.get_stats()
	curve = lab.get_learning_curve()
	var err: String = lab.get_error()
	if not err.is_empty():
		status_label.text = err
		status_label.visible = true
	elif lab.is_finished():
		if not evaluation_started and not untrained_view:
			lab.reset_showcase()
			evaluation_started = true
	_update_stats()
	curve_view.queue_redraw()

func _restart() -> void:
	evaluation_started = false
	untrained_view = false
	baseline_checkbox.set_pressed_no_signal(false)
	cars.clear()
	stats.clear()
	curve.clear()
	previous_speeds.clear()
	paused = true
	pause_button.text = "Start learning"
	var ups := 0.0 if full_speed.button_pressed else pow(10.0, speed_slider.value)
	var ok := lab.start(ENV_CHOICES[envs_option.selected], run_seed, SHOWCASE_CARS, ups)
	status_label.text = "" if ok else lab.get_error()
	status_label.visible = not ok
	_apply_speed()

func _apply_speed() -> void:
	var ups := 0.0 if full_speed.button_pressed else pow(10.0, speed_slider.value)
	lab.set_updates_per_second(ups)
	speed_slider.editable = not full_speed.button_pressed
	speed_label.text = "Maximum" if full_speed.button_pressed else "%.1f updates / second" % ups

func _restart_with_new_seed() -> void:
	run_seed += 1
	_restart()

func _toggle_pause() -> void:
	if lab.is_finished():
		return
	paused = not paused
	lab.set_paused(paused)
	pause_button.text = ("Start learning" if stats.get("update", 0) == 0 else "Resume training") if paused else "Pause training"

# --- UI ---------------------------------------------------------------------

func _surface(color: Color, outlined := false) -> StyleBoxFlat:
	var style := StyleBoxFlat.new()
	style.bg_color = color
	style.set_corner_radius_all(6)
	style.content_margin_left = 12
	style.content_margin_right = 12
	style.content_margin_top = 8
	style.content_margin_bottom = 8
	if outlined:
		style.set_border_width_all(1)
		style.border_color = BORDER
	return style

func _configure_theme() -> void:
	var ui_theme := Theme.new()
	ui_theme.default_font_size = TYPE_BODY
	for kind in ["Label", "Button", "CheckBox", "OptionButton"]:
		ui_theme.set_color("font_color", kind, TEXT)
		ui_theme.set_color("font_hover_color", kind, TEXT)
		ui_theme.set_color("font_pressed_color", kind, TEXT)
		ui_theme.set_color("font_focus_color", kind, TEXT)
		ui_theme.set_color("font_disabled_color", kind, TEXT_MUTED)
	for kind in ["Button", "OptionButton"]:
		ui_theme.set_stylebox("normal", kind, _surface(SURFACE_RAISED, true))
		ui_theme.set_stylebox("hover", kind, _surface(Color("333e4c"), true))
		ui_theme.set_stylebox("pressed", kind, _surface(Color("202833"), true))
		ui_theme.set_stylebox("disabled", kind, _surface(SURFACE, true))
		var focus := _surface(Color.TRANSPARENT)
		focus.draw_center = false
		focus.set_border_width_all(2)
		focus.border_color = CURVE
		ui_theme.set_stylebox("focus", kind, focus)
	for state in ["normal", "hover", "pressed"]:
		var check_style := _surface(Color.TRANSPARENT if state == "normal" else SURFACE_RAISED)
		check_style.content_margin_left = 0
		check_style.content_margin_right = 0
		ui_theme.set_stylebox(state, "CheckBox", check_style)
	ui_theme.set_constant("h_separation", "CheckBox", SPACE)
	for checked in [false, true]:
		var svg := '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"><rect x="2" y="2" width="16" height="16" rx="3" fill="%s" stroke="#abb8c9" stroke-width="1.5"/>%s</svg>' % ["#faa640" if checked else "none", '<path d="M6 10l3 3 5-6" fill="none" stroke="#151a21" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>' if checked else ""]
		var icon := Image.new()
		icon.load_svg_from_string(svg)
		ui_theme.set_icon("checked" if checked else "unchecked", "CheckBox", ImageTexture.create_from_image(icon))
	theme = ui_theme
	RenderingServer.set_default_clear_color(BG)

func _text(value: String, font_size := TYPE_BODY, color := TEXT) -> Label:
	var label := Label.new()
	label.text = value
	label.add_theme_font_size_override("font_size", font_size)
	label.add_theme_color_override("font_color", color)
	return label

func _column(gap := SPACE) -> VBoxContainer:
	var column := VBoxContainer.new()
	column.add_theme_constant_override("separation", gap)
	return column

func _button(value: String, action: Callable) -> Button:
	var button := Button.new()
	button.text = value
	button.custom_minimum_size.y = 44
	button.pressed.connect(action)
	return button

func _divider() -> ColorRect:
	var line := ColorRect.new()
	line.color = BORDER
	line.custom_minimum_size.y = 1
	line.mouse_filter = Control.MOUSE_FILTER_IGNORE
	return line

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, size), BG)

func _build_ui() -> void:
	_configure_theme()
	layout = BoxContainer.new()
	layout.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	layout.add_theme_constant_override("separation", 1)
	add_child(layout)

	panel_scroll = ScrollContainer.new()
	panel_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	panel_scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	layout.add_child(panel_scroll)
	side_panel = PanelContainer.new()
	side_panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	side_panel.size_flags_vertical = Control.SIZE_EXPAND_FILL
	var panel_style := _surface(SURFACE)
	panel_style.set_corner_radius_all(0)
	panel_style.content_margin_left = 20
	panel_style.content_margin_right = 20
	panel_style.content_margin_top = 20
	panel_style.content_margin_bottom = 20
	side_panel.add_theme_stylebox_override("panel", panel_style)
	panel_scroll.add_child(side_panel)
	var controls := _column(12)
	side_panel.add_child(controls)

	var header := _column(4)
	var title_row := HBoxContainer.new()
	var title := _text("Race Lab", TYPE_TITLE)
	title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	title_row.add_child(title)
	seed_label = _text("Seed 1", TYPE_CAPTION, TEXT_MUTED)
	seed_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	title_row.add_child(seed_label)
	header.add_child(title_row)
	phase_label = _text("Preparing policy…", TYPE_CAPTION, TEXT_MUTED)
	header.add_child(phase_label)
	controls.add_child(header)

	pause_button = _button("Start learning", _toggle_pause)
	for state in ["normal", "hover", "pressed"]:
		var fill := CURVE.lightened(0.08) if state == "hover" else CURVE
		if state == "pressed":
			fill = CURVE.darkened(0.1)
		pause_button.add_theme_stylebox_override(state, _surface(fill))
	for state in ["font_color", "font_hover_color", "font_pressed_color", "font_focus_color"]:
		pause_button.add_theme_color_override(state, Color("151a21"))
	var primary_focus := _surface(Color.TRANSPARENT)
	primary_focus.draw_center = false
	primary_focus.set_border_width_all(2)
	primary_focus.border_color = TEXT
	primary_focus.expand_margin_left = 3
	primary_focus.expand_margin_right = 3
	primary_focus.expand_margin_top = 3
	primary_focus.expand_margin_bottom = 3
	pause_button.add_theme_stylebox_override("focus", primary_focus)
	controls.add_child(pause_button)
	var race_actions := HBoxContainer.new()
	race_actions.add_theme_constant_override("separation", SPACE)
	var restart := _button("New seed", _restart_with_new_seed)
	restart.tooltip_text = "Restart learning from scratch with a new random seed."
	restart.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	race_actions.add_child(restart)
	reset_button = _button("Reset cars", func(): lab.reset_showcase())
	reset_button.tooltip_text = "Reset the visible race and its results. Keep the model and training progress."
	reset_button.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	race_actions.add_child(reset_button)
	controls.add_child(race_actions)

	baseline_checkbox = CheckBox.new()
	baseline_checkbox.text = "Show untrained policy"
	baseline_checkbox.custom_minimum_size.y = 44
	baseline_checkbox.tooltip_text = "Compare against this seed's frozen update-zero weights. Training continues; switching resets the visible race."
	baseline_checkbox.toggled.connect(func(on):
		untrained_view = on
		evaluation_started = false
		lab.set_untrained(on)
		cars.clear()
		previous_speeds.clear())
	controls.add_child(baseline_checkbox)
	controls.add_child(_divider())

	var results_header := HBoxContainer.new()
	var results_title := _text("Race results", TYPE_SECTION)
	results_title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	results_header.add_child(results_title)
	results_header.add_child(_text("Since reset", TYPE_CAPTION, TEXT_MUTED))
	controls.add_child(results_header)
	var metrics := GridContainer.new()
	metrics.columns = 2
	metrics.add_theme_constant_override("h_separation", 16)
	metrics.add_theme_constant_override("v_separation", 16)
	for spec in [["laps", "Clean laps"], ["best", "Best lap"], ["contacts", "Car contacts"], ["walls", "Wall hits"]]:
		var cell := _column(4)
		cell.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		cell.add_child(_text(spec[1], TYPE_CAPTION, TEXT_MUTED))
		var value := _text("—" if spec[0] == "best" else "0", TYPE_TITLE)
		metric_values[spec[0]] = value
		cell.add_child(value)
		metrics.add_child(cell)
	controls.add_child(metrics)

	training_readout = _column(4)
	training_summary = _text("", TYPE_BODY)
	training_readout.add_child(training_summary)
	stats_label = _text("", TYPE_CAPTION, TEXT_MUTED)
	training_readout.add_child(stats_label)
	controls.add_child(training_readout)
	controls.add_child(_divider())

	var inspection := HBoxContainer.new()
	inspection.add_theme_constant_override("separation", SPACE)
	inspect_option = OptionButton.new()
	inspect_option.custom_minimum_size = Vector2(96, 44)
	inspect_option.tooltip_text = "Choose a car to follow in the speed readout and sensor view."
	for i in SHOWCASE_CARS:
		inspect_option.add_item("Car %d" % (i + 1))
	inspect_option.item_selected.connect(func(i): inspected_car = i)
	inspection.add_child(inspect_option)
	var sensors := CheckBox.new()
	sensors.text = "Show sensors"
	sensors.custom_minimum_size.y = 44
	sensors.toggled.connect(func(on): show_sensors = on)
	inspection.add_child(sensors)
	controls.add_child(inspection)

	settings_button = _button("Training settings ▸", func():
		settings_panel.visible = settings_button.button_pressed
		settings_button.text = "Training settings ▾" if settings_panel.visible else "Training settings ▸")
	settings_button.toggle_mode = true
	settings_button.alignment = HORIZONTAL_ALIGNMENT_LEFT
	controls.add_child(settings_button)
	settings_panel = _column(12)
	settings_panel.visible = false
	controls.add_child(settings_panel)

	var speed_box := _column(4)
	speed_box.add_child(_text("Training speed", TYPE_CAPTION, TEXT_MUTED))
	speed_label = _text("2.0 updates / second")
	speed_box.add_child(speed_label)
	speed_slider = HSlider.new()
	speed_slider.custom_minimum_size.y = 44
	speed_slider.min_value = log(0.5) / log(10.0)
	speed_slider.max_value = log(50.0) / log(10.0)
	speed_slider.step = 0.01
	speed_slider.value = log(2.0) / log(10.0)
	speed_slider.value_changed.connect(func(_v): _apply_speed())
	speed_box.add_child(speed_slider)
	settings_panel.add_child(speed_box)
	full_speed = CheckBox.new()
	full_speed.text = "Maximum training speed"
	full_speed.custom_minimum_size.y = 44
	full_speed.tooltip_text = "Remove the display throttle. Learning may become too fast to observe."
	full_speed.toggled.connect(func(_on): _apply_speed())
	settings_panel.add_child(full_speed)

	var batch_box := _column(4)
	batch_box.add_child(_text("Training batch", TYPE_CAPTION, TEXT_MUTED))
	envs_option = OptionButton.new()
	envs_option.custom_minimum_size.y = 44
	for n in ENV_CHOICES:
		envs_option.add_item("%d races / %s cars" % [n / 8, _thousands(n)])
	envs_option.select(1)
	envs_option.item_selected.connect(func(_i): _restart())
	batch_box.add_child(envs_option)
	var batch_hint := _note("Changing the batch restarts learning.")
	batch_box.add_child(batch_hint)
	settings_panel.add_child(batch_box)
	diagnostics_label = _note("")
	settings_panel.add_child(diagnostics_label)

	status_label = _note("")
	status_label.add_theme_color_override("font_color", Color("ffb2a9"))
	status_label.visible = false
	controls.add_child(status_label)

	var right := _column(1)
	right.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	right.size_flags_vertical = Control.SIZE_EXPAND_FILL
	layout.add_child(right)
	track_view = Control.new()
	track_view.clip_contents = true
	track_view.size_flags_vertical = Control.SIZE_EXPAND_FILL
	track_view.draw.connect(_draw_track)
	right.add_child(track_view)
	curve_view = Control.new()
	curve_view.clip_contents = true
	curve_view.size_flags_vertical = Control.SIZE_FILL
	curve_view.draw.connect(_draw_curve)
	right.add_child(curve_view)

func _update_stats() -> void:
	var ready: bool = stats.get("policy_ready", false)
	var update: int = stats.get("update", 0)
	var finished := lab.is_finished()
	var error := lab.get_error()
	seed_label.text = "Seed %d" % run_seed
	pause_button.disabled = not ready or finished or not error.is_empty()
	reset_button.disabled = not ready
	baseline_checkbox.disabled = not ready
	phase_label.add_theme_color_override("font_color", TEXT_MUTED)
	if not error.is_empty():
		phase_label.text = "Training unavailable"
		phase_label.add_theme_color_override("font_color", Color("ffb2a9"))
	elif not ready:
		phase_label.text = "Preparing policy…"
	elif untrained_view:
		phase_label.text = "Untrained reference · update 0"
	elif update == 0:
		phase_label.text = "Untrained · update 0"
	elif finished:
		phase_label.text = "Training complete · update %d" % update
	elif paused:
		phase_label.text = "Paused · update %d" % update
	else:
		phase_label.text = "Learning · update %d" % update
	pause_button.text = "Training complete" if finished else ("Start learning" if update == 0 and paused else ("Resume learning" if paused else "Pause learning"))

	metric_values["laps"].text = str(stats.get("showcase_clean_laps", 0))
	var best: float = stats.get("showcase_best_lap", 0.0)
	metric_values["best"].text = "—" if best <= 0 else "%.2f s" % best
	metric_values["contacts"].text = str(stats.get("showcase_contacts", 0))
	metric_values["walls"].text = str(stats.get("showcase_walls", 0))
	training_readout.visible = update > 0
	training_summary.text = "%.2f laps / training episode" % stats.get("mean_laps", 0.0)
	stats_label.text = "GPU · %.2fM car steps/s active" % (stats.get("env_steps_per_second", 0.0) / 1e6)
	diagnostics_label.text = "Update %d · %s car steps\n%.1f ms / update\n%d clean training laps this update" % [update, _thousands(stats.get("env_steps", 0)), stats.get("update_ms", 0.0), stats.get("clean_laps", 0)]

func _fit_layout() -> void:
	if layout == null:
		return
	var compact := size.x < 850 or size.y > size.x * 1.2
	layout.vertical = compact
	panel_scroll.custom_minimum_size = Vector2(0, 272) if compact else Vector2(PANEL_WIDTH, 0)
	panel_scroll.size_flags_horizontal = Control.SIZE_EXPAND_FILL if compact else Control.SIZE_FILL
	panel_scroll.size_flags_vertical = Control.SIZE_FILL if compact else Control.SIZE_EXPAND_FILL
	curve_view.custom_minimum_size.y = 128 if compact else clampf(size.y * 0.22, 144, 220)
	queue_redraw()

func _thousands(n: int) -> String:
	var s := str(n)
	var out := ""
	for i in s.length():
		if i > 0 and (s.length() - i) % 3 == 0:
			out += ","
		out += s[i]
	return out

func _note(text: String) -> Label:
	var l := _text(text, TYPE_CAPTION, TEXT_MUTED)
	l.autowrap_mode = TextServer.AUTOWRAP_WORD
	return l

# --- Drawing ----------------------------------------------------------------

# Track coordinates have y up; the screen has y down, so flip y.
func _to_screen(p: Vector2, origin: Vector2, px_per_m: float) -> Vector2:
	return origin + Vector2(p.x, -p.y) * px_per_m

func _draw_track() -> void:
	var area := track_view.size
	track_view.draw_rect(Rect2(Vector2.ZERO, area), GRASS)
	var px_per_m := minf((area.x - 32.0) / 100.0, (area.y - 72.0) / 84.0)
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
	var centre := PackedVector2Array()
	for k in outer.size():
		centre.append((outer[k] + inner[k]) * 0.5)
	for k in range(0, centre.size() - 1, 2):
		track_view.draw_line(centre[k], centre[k + 1], Color(0.65, 0.66, 0.58, 0.4), 1.0, true)
	# The start line sits on the centreline at angle zero.
	var start_a := _to_screen(track_inner[0], origin, px_per_m)
	var start_b := _to_screen(track_outer[0], origin, px_per_m)
	track_view.draw_line(start_a, start_b, START_LINE, 3.0)

	var n := cars.size() / car_stride
	for i in n:
		_draw_car(i, origin, px_per_m)

	var font := ThemeDB.fallback_font
	var caption := "8-car race · trained policy" if evaluation_started else "8-car race · learning"
	if untrained_view or stats.get("update", 0) == 0:
		caption = "8-car race · untrained policy"
	track_view.draw_string(font, Vector2(12, 22), caption,
			HORIZONTAL_ALIGNMENT_LEFT, area.x - 24, 13, Color(0.82, 0.84, 0.87))
	if cars.size() >= (inspected_car + 1) * car_stride:
		track_view.draw_string(font, Vector2(12, 42), "Car %d · %.0f km/h" % [inspected_car + 1, cars[inspected_car * car_stride + 3] * 3.6],
			HORIZONTAL_ALIGNMENT_LEFT, area.x - 24, 13, Color(0.82, 0.84, 0.87))

func _draw_car(i: int, origin: Vector2, px_per_m: float) -> void:
	var base := i * car_stride
	var pos := Vector2(cars[base + 0], cars[base + 1])
	var heading: float = cars[base + 2]
	var crashed: bool = cars[base + 6] > 0.5
	var color := Color.from_hsv(float(i) / float(SHOWCASE_CARS), 0.55, 0.95)
	var dir := Vector2(cos(heading), sin(heading))
	var side := Vector2(-dir.y, dir.x)

	# Raycasts: ray r sits at heading + (r - 4) * PI / 8, from -90 to +90 degrees.
	var rays := 9 if show_sensors and i == inspected_car else 0
	for r in rays:
		var angle := heading + (float(r) - 0.5 * float(rays - 1)) * PI / float(rays - 1)
		var ray_dir := Vector2(cos(angle), sin(angle))
		var dist: float = minf(cars[base + 7 + r], cars[base + 16 + r])
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
	var rear := _to_screen(pos - dir * CAR_LENGTH * 0.45, origin, px_per_m)
	if braking.size() > i and braking[i] > 0:
		track_view.draw_line(rear - side * px_per_m * 0.65, rear + side * px_per_m * 0.65, Color(1.0, 0.12, 0.07), 3.0, true)
	if i == inspected_car:
		track_view.draw_arc(_to_screen(pos, origin, px_per_m), 3.0 * px_per_m, 0, TAU, 24, Color(1, 1, 1, 0.7), 1.0, true)
	track_view.draw_string(ThemeDB.fallback_font, _to_screen(pos, origin, px_per_m) + Vector2(7, -7), str(i + 1), HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color.WHITE)
	# The nose marks the heading.
	var nose := _to_screen(pos + dir * CAR_LENGTH * 0.5, origin, px_per_m)
	track_view.draw_circle(nose, 0.5 * px_per_m, Color.WHITE)

func _draw_curve() -> void:
	var r := Rect2(Vector2.ZERO, curve_view.size).grow(-16)
	curve_view.draw_rect(Rect2(Vector2.ZERO, curve_view.size), BG.darkened(0.2))
	var font := ThemeDB.fallback_font
	var plot := Rect2(r.position + Vector2(40, 28), r.size - Vector2(48, 44))
	curve_view.draw_string(font, r.position + Vector2(40, 16), "Learning progress · average laps",
			HORIZONTAL_ALIGNMENT_LEFT, r.size.x - 48, TYPE_BODY, TEXT)
	if curve.is_empty():
		curve_view.draw_string(font, r.position + Vector2(40, 56), "Start learning to build the curve.",
				HORIZONTAL_ALIGNMENT_LEFT, r.size.x - 48, TYPE_BODY, TEXT_MUTED)
		return
	var chart_max := MAX_LAPS
	for p in curve:
		chart_max = maxf(chart_max, ceilf(p.y * 2.0) / 2.0)
	for tick in 5:
		var v := chart_max * float(tick) / 4.0
		var y := remap(v, 0.0, chart_max, plot.end.y, plot.position.y)
		curve_view.draw_line(Vector2(plot.position.x, y), Vector2(plot.end.x, y), GRID)
		curve_view.draw_string(font, Vector2(r.position.x, y + 5), "%.1f" % v, HORIZONTAL_ALIGNMENT_LEFT, -1, 12, GRID.lightened(0.4))
	if curve.size() < 2:
		return
	var max_steps := maxf(curve[curve.size() - 1].x, 1.0e6)
	var pts := PackedVector2Array()
	for p in curve:
		pts.append(Vector2(remap(p.x, 0.0, max_steps, plot.position.x, plot.end.x),
				remap(p.y, 0.0, chart_max, plot.end.y, plot.position.y)))
	curve_view.draw_polyline(pts, CURVE, 2.5, true)
	curve_view.draw_string(font, Vector2(plot.end.x - 90, plot.end.y + 16), "%.1fM steps" % (max_steps / 1e6),
			HORIZONTAL_ALIGNMENT_LEFT, -1, 12, GRID.lightened(0.4))

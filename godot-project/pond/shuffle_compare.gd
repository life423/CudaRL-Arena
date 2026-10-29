extends Control

const Pond = preload("res://pond/pond.tscn")
const PondCamera = preload("res://pond/pond_camera.gd")
const REFERENCE_PATH := "res://pond/policies/shuffle_reference.json"
const CHECKPOINT_PATH := "user://pond-policies/shuffle-v1/best.json"
const TEXT := Color("eef4e9")
const MUTED := Color("b8cbc4")
const ACCENT := Color("faa640")
var reference_pond
var learner_pond
var panels: Array[VBoxContainer] = []
var viewports: Array[SubViewport] = []
var headings: Array[Label] = []
var metrics: Array[Label] = []
var views: HBoxContainer
var tabs: TabBar
var pause_button: Button
var watch_button: Button
var maximum: CheckBox
var status: Label
var curve
var old_tick_rate := 60
var reference_score := 0.0
var reference_update := 0
var last_points := -1
var paused := false
var controls: HFlowContainer
var page_scroll: ScrollContainer
var training_save: Label
var camera_selector: OptionButton
var tab_row: HBoxContainer

func label(value: String, font_size := 14, color := TEXT) -> Label:
	var item := Label.new()
	item.text = value
	item.add_theme_font_size_override("font_size", font_size)
	item.add_theme_color_override("font_color", color)
	item.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	return item

func button(value: String) -> Button:
	var item := Button.new()
	item.text = value
	item.custom_minimum_size = Vector2(0, 44)
	return item

func _ready() -> void:
	old_tick_rate = Engine.physics_ticks_per_second
	Engine.physics_ticks_per_second = 20
	get_window().title = "CudaRL · Shuffled-input comparison"
	RenderingServer.set_default_clear_color(Color("0b2826"))
	var theme := Theme.new()
	theme.default_font_size = 14
	for kind in ["Button", "CheckBox", "TabBar", "OptionButton"]:
		theme.set_color("font_color", kind, TEXT)
	var normal := StyleBoxFlat.new()
	normal.bg_color = Color("263631")
	normal.border_color = Color("50645c")
	normal.set_border_width_all(1)
	normal.set_corner_radius_all(5)
	normal.content_margin_left = 12
	normal.content_margin_right = 12
	var hover := normal.duplicate() as StyleBoxFlat
	hover.bg_color = Color("354a42")
	var focus := normal.duplicate() as StyleBoxFlat
	focus.bg_color = Color.TRANSPARENT
	focus.border_color = ACCENT
	focus.set_border_width_all(2)
	for kind in ["Button", "CheckBox", "TabBar", "OptionButton"]:
		theme.set_stylebox("normal", kind, normal)
		theme.set_stylebox("hover", kind, hover)
		theme.set_stylebox("pressed", kind, hover)
		theme.set_stylebox("focus", kind, focus)
	self.theme = theme
	var margin := MarginContainer.new()
	margin.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	for side in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 16)
	add_child(margin)
	page_scroll = ScrollContainer.new()
	page_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	margin.add_child(page_scroll)
	var column := VBoxContainer.new()
	column.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	column.size_flags_vertical = Control.SIZE_EXPAND_FILL
	column.add_theme_constant_override("separation", 8)
	page_scroll.add_child(column)
	column.add_child(label("Food-order experiment", 25))
	status = label("Same PPO, network and physics · 1,024 background swimmers", 14, MUTED)
	column.add_child(status)
	training_save = label("Restoring training…", 12, MUTED)
	column.add_child(training_save)
	controls = HFlowContainer.new()
	controls.add_theme_constant_override("h_separation", 8)
	controls.add_theme_constant_override("v_separation", 4)
	column.add_child(controls)
	var back := button("Pond menu")
	back.name = "BackToDemos"
	back.pressed.connect(DemoLaunch.show_menu)
	controls.add_child(back)
	pause_button = button("Pause learning")
	pause_button.pressed.connect(_pause)
	controls.add_child(pause_button)
	watch_button = button("Watch candidate best")
	watch_button.toggle_mode = true
	watch_button.tooltip_text = "Switch the right pond to its saved best. Background learning continues; body and food stay in place."
	watch_button.toggled.connect(func(enabled: bool):
		if learner_pond != null:
			learner_pond._watch_best(enabled))
	controls.add_child(watch_button)
	maximum = CheckBox.new()
	maximum.text = "Maximum speed"
	maximum.custom_minimum_size.y = 44
	maximum.button_pressed = true
	maximum.toggled.connect(func(enabled: bool):
		learner_pond.maximum_speed.button_pressed = enabled)
	controls.add_child(maximum)
	camera_selector = PondCamera.selector(true)
	camera_selector.item_selected.connect(func(index: int):
		reference_pond.set_camera_close(index == 0)
		learner_pond.set_camera_close(index == 0))
	controls.add_child(camera_selector)
	tabs = TabBar.new()
	tabs.add_tab("Saved best")
	tabs.add_tab("Shuffled learner")
	tabs.current_tab = 1
	tabs.custom_minimum_size.y = 44
	tabs.tab_changed.connect(func(_index: int): _fit())
	tab_row = HBoxContainer.new()
	tab_row.add_theme_constant_override("separation", 4)
	column.add_child(tab_row)
	tabs.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	tab_row.add_child(tabs)
	views = HBoxContainer.new()
	views.custom_minimum_size.y = 240
	views.size_flags_vertical = Control.SIZE_EXPAND_FILL
	views.add_theme_constant_override("separation", 12)
	column.add_child(views)
	for index in 2:
		var panel := VBoxContainer.new()
		panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		views.add_child(panel)
		panels.append(panel)
		var heading := label("Saved best" if index == 0 else "Shuffled learner", 20)
		headings.append(heading)
		panel.add_child(heading)
		var metric := label("Preparing…", 14, MUTED)
		metrics.append(metric)
		panel.add_child(metric)
		var container := SubViewportContainer.new()
		container.stretch = true
		container.size_flags_vertical = Control.SIZE_EXPAND_FILL
		container.mouse_filter = Control.MOUSE_FILTER_IGNORE
		panel.add_child(container)
		var viewport := SubViewport.new()
		viewport.size = Vector2i(640, 360)
		viewport.own_world_3d = true
		viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
		viewports.append(viewport)
		container.add_child(viewport)
		var pond = Pond.instantiate()
		pond.embedded_view = true
		pond.live_learning = index == 1
		pond.playback_checkpoint = REFERENCE_PATH if index == 0 else ""
		pond.shuffle_food_inputs = index == 1
		pond.best_policy_path = CHECKPOINT_PATH
		pond.training_updates_per_second = 0
		viewport.add_child(pond)
		if index == 0:
			reference_pond = pond
		else:
			learner_pond = pond
	curve = preload("res://pond/shuffle_curve.gd").new()
	column.add_child(curve)
	var note := label("64 fixed one-minute trials. The left policy stays frozen; the right pond resumes its saved training.", 12, MUTED)
	column.add_child(note)
	resized.connect(_fit)
	_fit()
	back.grab_focus.call_deferred()

func _fit() -> void:
	if tabs == null:
		return
	var narrow := size.x < 900
	tabs.visible = narrow
	tab_row.visible = narrow
	var selector_parent: Node = tab_row if narrow else controls
	if camera_selector.get_parent() != selector_parent:
		camera_selector.reparent(selector_parent)
	camera_selector.custom_minimum_size.x = 124 if narrow else 164
	camera_selector.set_item_text(0, "Close-up" if narrow else "Follow tadpole")
	camera_selector.set_item_text(1, "Overview" if narrow else "Pond overview")
	for i in panels.size():
		panels[i].visible = not narrow or tabs.current_tab == i
		viewports[i].render_target_update_mode = SubViewport.UPDATE_ALWAYS if panels[i].visible else SubViewport.UPDATE_DISABLED
	if curve != null:
		curve.custom_minimum_size.y = 142 if size.y < 600 else 172

func _pause() -> void:
	paused = not paused
	learner_pond.user_paused = paused
	learner_pond.warmup_timer.stop()
	learner_pond.lab.set_paused(paused)
	pause_button.text = "Resume learning" if paused else "Pause learning"

func _process(_delta: float) -> void:
	if learner_pond == null:
		return
	var error: String = learner_pond.lab.get_error()
	if error.is_empty():
		error = reference_pond.lab.get_error()
	if not error.is_empty():
		status.text = error
		status.add_theme_color_override("font_color", Color("ffb2a9"))
		return
	if not learner_pond.ready_policy or not reference_pond.ready_policy:
		return
	var baseline: Dictionary = reference_pond.lab.get_stats()
	var stats: Dictionary = learner_pond.lab.get_stats()
	if not stats.training_ready:
		status.text = "Restoring training…"
		return
	reference_score = baseline.best_food_per_minute
	reference_update = int(baseline.best_update)
	headings[0].text = "Saved best · update %s" % reference_pond._format_count(reference_update)
	metrics[0].text = "Fixed score %.2f / min · eaten %d" % [reference_score, int(reference_pond.current.eaten)]
	var view := "best %d" % int(stats.best_update) if stats.watching_best else "latest"
	headings[1].text = "Shuffled · %s · update %s" % [view, learner_pond._format_count(int(stats.update))]
	metrics[1].text = "Current %.2f · best %.2f / min · eaten %d" % [stats.current_food_per_minute, stats.best_food_per_minute, int(learner_pond.current.eaten)]
	watch_button.disabled = not stats.best_ready
	watch_button.set_pressed_no_signal(stats.watching_best)
	watch_button.text = "Watch latest learner" if stats.watching_best else "Watch candidate best"
	status.text = "Same PPO, network and physics · 1,024 swimmers · %s" % ("paused" if paused else "learning")
	training_save.text = learner_pond.training_save.text
	training_save.tooltip_text = learner_pond.training_save.tooltip_text
	if stats.resumed_training and str(stats.training_save_error).is_empty():
		training_save.text = "Resumed from %s · saved at %s" % [learner_pond._format_count(int(stats.training_loaded_update)), learner_pond._format_count(int(stats.training_saved_update))]
	if not str(stats.checkpoint_error).is_empty():
		status.text = stats.checkpoint_error
	elif stats.best_saved:
		status.text += " · candidate best saved"
	if learner_pond.curve_view.points.size() != last_points:
		last_points = learner_pond.curve_view.points.size()
		curve.points = learner_pond.curve_view.points
		curve.reference = reference_score
		curve.queue_redraw()

func _unhandled_key_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_ESCAPE:
		get_viewport().set_input_as_handled()
		DemoLaunch.show_menu()

func _exit_tree() -> void:
	Engine.physics_ticks_per_second = old_tick_rate

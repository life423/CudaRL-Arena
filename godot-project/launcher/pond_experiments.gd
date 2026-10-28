extends AcceptDialog

const TEXT := Color("f1f2ee")
const MUTED := Color("b4bac2")
const ACCENT := Color("faa640")
var actions: Array[Button] = []
var message: Label
var scroll: ScrollContainer

static func style_dialog(dialog: AcceptDialog) -> void:
	var palette := Theme.new()
	var panel := StyleBoxFlat.new()
	panel.bg_color = Color("171c21")
	panel.set_content_margin_all(16)
	palette.set_stylebox("panel", "AcceptDialog", panel)
	palette.set_constant("buttons_min_height", "AcceptDialog", 44)
	palette.set_constant("buttons_min_width", "AcceptDialog", 112)
	var border := StyleBoxFlat.new()
	border.bg_color = Color("202930")
	border.border_color = Color("36414b")
	border.set_border_width_all(1)
	border.expand_margin_top = 36
	palette.set_stylebox("embedded_border", "Window", border)
	palette.set_stylebox("embedded_unfocused_border", "Window", border)
	palette.set_color("title_color", "Window", TEXT)
	for state in ["normal", "hover", "pressed"]:
		var button := StyleBoxFlat.new()
		button.bg_color = Color("26313a").lightened(0.08 if state == "hover" else 0)
		button.set_corner_radius_all(5)
		palette.set_stylebox(state, "Button", button)
	var focus := StyleBoxFlat.new()
	focus.bg_color = Color.TRANSPARENT
	focus.border_color = ACCENT
	focus.set_border_width_all(2)
	palette.set_stylebox("focus", "Button", focus)
	for state in ["font_color", "font_hover_color", "font_pressed_color", "font_focus_color"]:
		palette.set_color(state, "Button", TEXT)
	palette.set_color("font_color", "Label", MUTED)
	dialog.theme = palette
	dialog.visibility_changed.connect(func():
		var host := dialog.get_parent() as Control
		if host == null:
			return
		var backdrop := host.get_node_or_null("ModalBackdrop") as ColorRect
		if backdrop == null:
			backdrop = ColorRect.new()
			backdrop.name = "ModalBackdrop"
			backdrop.color = Color(0, 0, 0, 0.55)
			backdrop.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
			backdrop.z_index = 100
			host.add_child(backdrop)
		backdrop.visible = dialog.visible)

func _ready() -> void:
	name = "PondExperimentsDialog"
	title = "Pond experiments"
	ok_button_text = "Close"
	style_dialog(self)
	get_ok_button().custom_minimum_size.y = 44
	scroll = ScrollContainer.new()
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	add_child(scroll)
	var body := VBoxContainer.new()
	body.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	body.add_theme_constant_override("separation", 20)
	scroll.add_child(body)
	var options := []
	if FileAccess.file_exists("res://../docs/experiments/swimmer/gentle-currents/bank-recovery/selected-policy.bin"):
		options.append(["Latest swimmer", "Live simulation · saved PPO agent. Includes calm water, gentle current and an untrained comparison.", "Open live pond", "pond_live"])
	options.append(["Original swimmer", "Live simulation · earlier saved agent, preserved for comparison.", "Watch original", "deep_water"])
	if not ProjectSettings.get_setting("cudarl/portable_player", false):
		options.append(["Shallow-water learning", "Live PPO training · resumes the shallow pond's separate training save.", "Watch training", "pond"])
		if FileAccess.file_exists("res://../build/pond-current-study/diagnosis/replay.json"):
			options.append(["Current stall recording", "Recorded stronger-current diagnosis. Playback and seeking only.", "Play recording", "pond_current_replay"])
		if FileAccess.file_exists("res://../build/pond-reliability-study/bank-preview/replay.json"):
			options.append(["Recovery comparison recording", "Recorded original and bank-trained agents from the same starting state.", "Play recording", "pond_recovery"])
	for option: Array in options:
		var row := VBoxContainer.new()
		row.add_theme_constant_override("separation", 8)
		body.add_child(row)
		var heading := Label.new()
		heading.text = option[0]
		heading.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		heading.add_theme_font_size_override("font_size", 20)
		heading.add_theme_color_override("font_color", TEXT)
		row.add_child(heading)
		var description := Label.new()
		description.text = option[1]
		description.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		description.add_theme_font_size_override("font_size", 15)
		description.add_theme_color_override("font_color", MUTED)
		row.add_child(description)
		var button := Button.new()
		button.name = "Open_" + option[3]
		button.text = option[2]
		button.custom_minimum_size.y = 44
		button.add_theme_font_size_override("font_size", 16)
		for state in ["normal", "hover", "pressed"]:
			var style := StyleBoxFlat.new()
			style.bg_color = Color("26313a").lightened(0.08 if state == "hover" else 0)
			style.set_corner_radius_all(5)
			button.add_theme_stylebox_override(state, style)
		var focus := StyleBoxFlat.new()
		focus.bg_color = Color.TRANSPARENT
		focus.border_color = ACCENT
		focus.set_border_width_all(2)
		button.add_theme_stylebox_override("focus", focus)
		for state in ["font_color", "font_hover_color", "font_pressed_color", "font_focus_color"]:
			button.add_theme_color_override(state, TEXT)
		button.pressed.connect(func(): _open_demo(option[3]))
		row.add_child(button)
		actions.append(button)
	message = Label.new()
	message.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	message.add_theme_color_override("font_color", Color("ffb2a9"))
	message.hide()
	body.add_child(message)
	get_parent().resized.connect(_fit)
	visibility_changed.connect(func():
		if visible:
			scroll.scroll_vertical = 0
			actions[0].grab_focus.call_deferred())

func open() -> void:
	popup_centered(_dimensions())

func _dimensions() -> Vector2i:
	var available: Vector2 = get_parent().size
	return Vector2i(mini(610, int(available.x) - 24), mini(640, int(available.y) - 80))

func _fit() -> void:
	if visible:
		size = _dimensions()
		position = (Vector2i(get_parent().size) - size) / 2

func _open_demo(demo: String) -> void:
	var error := DemoLaunch.open_demo(demo, demo != "pond")
	if error != OK:
		message.text = "Could not open this experiment. Its scene may be unavailable."
		message.show()
	else:
		hide()

extends Control

const Preview = preload("res://launcher/thumbnail.gd")
const BG := Color("171c21")
const TEXT := Color("f1f2ee")
const MUTED := Color("b4bac2")
const ACCENT := Color("faa640")
const BORDER := Color("36414b")

var margin: MarginContainer
var title_row: HBoxContainer
var title_labels: Array[Label] = []
var rows: Array[Dictionary] = []
var buttons: Array[Button] = []
var status: Label
var opening := false
var content: VBoxContainer

func _ready() -> void:
	RenderingServer.set_default_clear_color(BG)
	_build()
	resized.connect(_fit_layout)
	_fit_layout()
	buttons[0].grab_focus.call_deferred()

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, size), BG)

func _label(value: String, font_size: int, color := TEXT) -> Label:
	var label := Label.new()
	label.text = value
	label.add_theme_font_size_override("font_size", font_size)
	label.add_theme_color_override("font_color", color)
	return label

func _style(fill: Color, border: Color, thickness := 1) -> StyleBoxFlat:
	var style := StyleBoxFlat.new()
	style.bg_color = fill
	style.border_color = border
	style.set_border_width_all(thickness)
	style.set_corner_radius_all(6)
	style.content_margin_left = 20
	style.content_margin_right = 20
	return style

func _button(value: String, demo: String, watch: bool, primary := false) -> Button:
	var button := Button.new()
	button.name = "%s_%s" % [demo, "watch" if watch else "train"]
	button.text = value
	button.custom_minimum_size = Vector2(156 if watch else 188, 48)
	button.add_theme_font_size_override("font_size", 16)
	var filled := watch or primary
	for state in ["normal", "hover", "pressed", "disabled"]:
		var fill := ACCENT if filled else BG
		if state == "hover":
			fill = ACCENT.lightened(0.1) if filled else Color("2b3034")
		elif state == "pressed":
			fill = ACCENT.darkened(0.12) if filled else Color("343b43")
		elif state == "disabled":
			fill = Color("343b43")
		button.add_theme_stylebox_override(state, _style(fill, ACCENT))
	for state in ["font_color", "font_hover_color", "font_pressed_color", "font_focus_color"]:
		button.add_theme_color_override(state, Color("151a21") if filled else ACCENT)
	button.add_theme_color_override("font_disabled_color", MUTED)
	var focus := _style(Color.TRANSPARENT, TEXT, 2)
	focus.draw_center = false
	button.add_theme_stylebox_override("focus", focus)
	button.tooltip_text = "Play the saved policy without training." if watch else "Open a fresh, untrained run."
	if demo == "pond":
		button.tooltip_text = "Open the three-food pond and watch 1,024 background swimmers learn from scratch."
	button.pressed.connect(func(): _open(demo, watch))
	buttons.append(button)
	return button

func _build() -> void:
	margin = MarginContainer.new()
	margin.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(margin)
	var scroll := ScrollContainer.new()
	scroll.name = "DemoScroll"
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	margin.add_child(scroll)
	content = VBoxContainer.new()
	content.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	content.add_theme_constant_override("separation", 24)
	scroll.add_child(content)
	var header := VBoxContainer.new()
	header.add_theme_constant_override("separation", 8)
	title_row = HBoxContainer.new()
	title_row.add_theme_constant_override("separation", 0)
	var bold := SystemFont.new()
	bold.font_names = PackedStringArray(["Segoe UI"])
	bold.font_weight = 700
	for word in ["CudaRL", " Arena"]:
		var label := _label(word, 48, ACCENT if word == "CudaRL" else TEXT)
		label.add_theme_font_override("font", bold)
		title_row.add_child(label)
		title_labels.append(label)
	header.add_child(title_row)
	header.add_child(_label("Choose a demonstration", 22, MUTED))
	content.add_child(header)
	for spec in [["cartpole", "CartPole", "Learn to balance a pole."], ["racing", "Racing", "Eight agents on a shared circuit."], ["pond", "Tadpole Pond", "Learn to forage in a three-food pond."]]:
		var row := BoxContainer.new()
		row.name = spec[0] + "_row"
		row.add_theme_constant_override("separation", 28)
		content.add_child(row)
		var preview := Preview.new()
		preview.demo = spec[0]
		row.add_child(preview)
		var details := VBoxContainer.new()
		details.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		details.alignment = BoxContainer.ALIGNMENT_CENTER
		details.add_theme_constant_override("separation", 8)
		row.add_child(details)
		var heading := _label(spec[1], 30)
		heading.add_theme_font_override("font", bold)
		details.add_child(heading)
		var caption := _label(spec[2], 18, MUTED)
		caption.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		details.add_child(caption)
		var actions := BoxContainer.new()
		actions.size_flags_vertical = Control.SIZE_SHRINK_CENTER
		actions.alignment = BoxContainer.ALIGNMENT_CENTER
		actions.add_theme_constant_override("separation", 12)
		row.add_child(actions)
		if spec[0] == "pond":
			actions.add_child(_button("Watch it learn", spec[0], false, true))
			actions.add_child(_button("Compare shuffled", "pond_shuffle", false))
		else:
			actions.add_child(_button("Watch trained", spec[0], true))
			actions.add_child(_button("Train from scratch", spec[0], false))
		rows.append({"row": row, "preview": preview, "details": details, "actions": actions})
		var line := ColorRect.new()
		line.color = BORDER
		line.custom_minimum_size.y = 1
		line.mouse_filter = Control.MOUSE_FILTER_IGNORE
		content.add_child(line)
	status = _label("", 16, Color("ffb2a9"))
	status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	status.visible = false
	content.add_child(status)

func _fit_layout() -> void:
	var padding := 48 if size.x >= 1440 else (32 if size.x >= 700 else 24)
	for side in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, padding)
	for label in title_labels:
		label.add_theme_font_size_override("font_size", 52 if size.x >= 1440 else (44 if size.x >= 700 else 30))
	var width := size.x - padding * 2
	var wide := width >= 1000
	content.add_theme_constant_override("separation", 16 if wide else 24)
	var preview_height := clampf((size.y - padding * 2 - 240) / 3.0, 112, 184)
	for data in rows:
		var row: BoxContainer = data.row
		row.vertical = not wide
		row.add_theme_constant_override("separation", 28 if wide else 16)
		var preview: Control = data.preview
		preview.custom_minimum_size = Vector2(clampf(width * 0.39, 320, 560), preview_height) if wide else Vector2(0, clampf(width / 2.8, 130, 216))
		preview.size_flags_horizontal = Control.SIZE_FILL if wide else Control.SIZE_EXPAND_FILL
		var details: VBoxContainer = data.details
		details.custom_minimum_size.x = 180 if wide else 0
		var actions: BoxContainer = data.actions
		actions.vertical = width < 420
		actions.size_flags_horizontal = Control.SIZE_FILL if wide else Control.SIZE_EXPAND_FILL
		for button: Button in actions.get_children():
			button.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	queue_redraw()

func _open(demo: String, watch: bool) -> void:
	if opening:
		return
	opening = true
	for button in buttons:
		button.disabled = true
	status.text = "Opening %s…" % {"cartpole": "CartPole", "racing": "Racing", "pond": "Tadpole Pond"}.get(demo, demo)
	status.add_theme_color_override("font_color", MUTED)
	status.visible = true
	await get_tree().process_frame
	var error := DemoLaunch.open_demo(demo, watch)
	if error != OK:
		status.text = "Could not open the demonstration. Please try again."
		status.add_theme_color_override("font_color", Color("ffb2a9"))
		opening = false
		for button in buttons:
			button.disabled = false

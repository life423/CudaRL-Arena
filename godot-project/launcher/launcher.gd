extends Control

const Preview = preload("res://launcher/thumbnail.gd")
const PondExperiments = preload("res://launcher/pond_experiments.gd")
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
var guide: AcceptDialog
var guide_button: Button
var header: VBoxContainer
var header_spacer: Control
var experiments: AcceptDialog

func _ready() -> void:
	get_window().title = "Tadpole Pond · CudaRL"
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
		button.tooltip_text = "Watch the three-food shallow pond learn. Saved training resumes automatically."
	elif demo in ["deep_water", "pond_live", "pond_feeding"]:
		button.tooltip_text = "Open a live physical simulation controlled by the saved neural networks."
		if demo == "pond_feeding":
			button.tooltip_text = "Drop sinking food while native CUDA PPO practices and validates improved swimming."
	if demo not in ["guide", "headless"]:
		if demo != "pond_experiments":
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
	header = VBoxContainer.new()
	header.add_theme_constant_override("separation", 8)
	title_row = HBoxContainer.new()
	title_row.add_theme_constant_override("separation", 0)
	var bold := SystemFont.new()
	bold.font_names = PackedStringArray(["Segoe UI"])
	bold.font_weight = 700
	for word in ["Tadpole", " Pond"]:
		var label := _label(word, 48, ACCENT if word == "Tadpole" else TEXT)
		label.add_theme_font_override("font", bold)
		title_row.add_child(label)
		title_labels.append(label)
	header.add_child(title_row)
	var subtitle := _label("Learned control in a changing world", 20, MUTED)
	subtitle.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	header.add_child(subtitle)
	content.add_child(header)
	var portable: bool = ProjectSettings.get_setting("cudarl/portable_player", false)
	guide_button = _button("About this build" if portable else "Help & training", "guide", false)
	# This control opens help, not a demonstration.
	guide_button.pressed.connect(_show_guide)
	guide_button.tooltip_text = "How to watch, save, resume and train faster without rendering."
	header_spacer = Control.new()
	header_spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	title_row.add_child(header_spacer)
	title_row.add_child(guide_button)
	# Keep legacy scenes, policies and routing intact; the product entry is the pond.
	var demos := [["pond", "Feed your swimmer", "Drop food into the water and watch it sink and drift. The creature chooses its own targets while native PPO practices in the background."]]
	for spec in demos:
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
		details.add_child(actions)
		if spec[0] == "pond":
			if portable:
				heading.text = "Watch your swimmer"
				caption.text = "The saved trained creature, with CUDA or CPU playback. This portable build does not run live learning."
				actions.add_child(_button("Open pond", "deep_water", false, true))
			else:
				var live_demo := "pond_feeding" if FileAccess.file_exists("res://../docs/experiments/swimmer/gentle-currents/bank-recovery/selected-policy.bin") else "deep_water"
				if live_demo == "pond_feeding" and (not ClassDB.class_exists("CudaDevice") or not ClassDB.instantiate("CudaDevice").is_available()):
					live_demo = "pond_live"
					heading.text = "Watch your swimmer"
					caption.text = "The saved trained creature runs with CPU fallback. Sinking-food learning requires a working CUDA device."
				actions.add_child(_button("Open pond", live_demo, true, true))
				var more := _button("Experiments", "pond_experiments", false)
				more.tooltip_text = "Separate learning labs, older agents and clearly labeled recordings."
				more.pressed.connect(_show_experiments)
				actions.add_child(more)
		else:
			actions.add_child(_button("Watch trained", spec[0], true))
			actions.add_child(_button("Train from scratch", spec[0], false))
		rows.append({"row": row, "preview": preview, "details": details, "actions": actions})
	status = _label("", 16, Color("ffb2a9"))
	status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	status.visible = false
	content.add_child(status)
	# Follow the demo actions in keyboard order, despite the guide's header position.
	buttons[-1].focus_next = guide_button.get_path()
	guide_button.focus_previous = buttons[-1].get_path()
	guide_button.focus_next = buttons[0].get_path()
	buttons[0].focus_previous = guide_button.get_path()

func _fit_layout() -> void:
	var guide_parent: Node = title_row if size.x >= 700 else header
	if guide_button.get_parent() != guide_parent:
		guide_button.reparent(guide_parent)
	header_spacer.visible = size.x >= 700
	buttons[-1].focus_next = guide_button.get_path()
	guide_button.focus_previous = buttons[-1].get_path()
	guide_button.focus_next = buttons[0].get_path()
	buttons[0].focus_previous = guide_button.get_path()
	var padding := 48 if size.x >= 1440 else (32 if size.x >= 700 else 24)
	for side in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, padding)
	for label in title_labels:
		label.add_theme_font_size_override("font_size", 52 if size.x >= 1440 else (44 if size.x >= 700 else 30))
	var width := size.x - padding * 2
	var wide := width >= 1000
	content.add_theme_constant_override("separation", 16 if wide else 24)
	var preview_height := clampf(size.y - padding * 2 - 200, 280, 440)
	for data in rows:
		var row: BoxContainer = data.row
		row.vertical = not wide
		row.add_theme_constant_override("separation", 28 if wide else 16)
		var preview: Control = data.preview
		preview.custom_minimum_size = Vector2(width * 0.56, preview_height) if wide else Vector2(0, clampf(width / 1.65, 180, 360))
		preview.size_flags_horizontal = Control.SIZE_FILL if wide else Control.SIZE_EXPAND_FILL
		var details: VBoxContainer = data.details
		details.custom_minimum_size.x = 180 if wide else 0
		var actions: BoxContainer = data.actions
		actions.vertical = width < 420 or (wide and width * 0.44 - 28 < 356)
		actions.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		for button: Button in actions.get_children():
			button.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	queue_redraw()
	if guide and guide.visible:
		guide.size = Vector2i(mini(650, int(size.x) - 24), mini(640, int(size.y) - 80))
		guide.position = (Vector2i(size) - guide.size) / 2

func _headless_command() -> String:
	var bundled := ProjectSettings.globalize_path("res://pond/policies/deep_selector.bin")
	if FileAccess.file_exists(bundled):
		var runner := ProjectSettings.globalize_path("res://../build/Release/pond_selector_train.exe")
		var folder := ProjectSettings.globalize_path("user://pond-policies/deep-pond-selector-v1")
		var command := "& '%s' --motor-policy '%s' --checkpoint '%s/training.bin' --policy '%s/candidate-best.bin' --updates 0 --recovery" % [runner.replace("'", "''"), bundled.replace("'", "''"), folder.replace("'", "''"), folder.replace("'", "''")]
		var fixture := ProjectSettings.globalize_path("res://pond/policies/deep_selector_recovery.bin")
		if FileAccess.file_exists(fixture):
			command += " --fixture '%s'" % fixture.replace("'", "''")
		return command
	var executable := ProjectSettings.globalize_path("res://../build/Release/deep_swimmer_train.exe")
	var checkpoint := ProjectSettings.globalize_path("user://pond-policies/deep-pond-v1/training.bin")
	return "& '%s' --pond --checkpoint '%s'" % [executable.replace("'", "''"), checkpoint.replace("'", "''")]

func _show_experiments() -> void:
	if experiments == null:
		experiments = PondExperiments.new()
		add_child(experiments)
	experiments.open()

func _start_headless() -> void:
	var executable := ProjectSettings.globalize_path("res://../build/Release/pond_selector_train.exe" if FileAccess.file_exists("res://pond/policies/deep_selector.bin") else "res://../build/Release/deep_swimmer_train.exe")
	if not FileAccess.file_exists(executable):
		status.text = "The native headless runner is missing. Build the project first."
		status.visible = true
		return
	var pid := OS.create_process("powershell.exe", PackedStringArray(["-NoExit", "-Command", _headless_command()]), true)
	status.text = "Headless console opened. Ctrl+C saves and stops. Deep water shows the validated controller." if pid > 0 else "Could not open the training console. Copy the command from the guide."
	status.visible = true
	if guide:
		guide.hide()

func _show_guide() -> void:
	if ProjectSettings.get_setting("cudarl/portable_player", false):
		if guide == null:
			guide = AcceptDialog.new()
			guide.title = "Portable pond"
			guide.dialog_autowrap = true
			PondExperiments.style_dialog(guide)
			guide.dialog_text = "The trained AI and pond physics prefer CUDA when its device-and-kernel check passes, with CPU fallback. Graphics use your available graphics hardware.\n\nUse Closer view or Pond overview to change the camera.\n\nThis build plays the saved tadpole controller. Physical growth and new movement skills are future stages."
			add_child(guide)
		guide.popup_centered(Vector2i(mini(610, int(size.x) - 24), mini(320, int(size.y) - 80)))
		return
	if guide == null:
		guide = AcceptDialog.new()
		guide.title = "Help & training"
		guide.ok_button_text = "Done"
		PondExperiments.style_dialog(guide)
		add_child(guide)
		var scroll := ScrollContainer.new()
		scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
		guide.add_child(scroll)
		var body := VBoxContainer.new()
		body.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		body.add_theme_constant_override("separation", 12)
		scroll.add_child(body)
		for item in [
			["Start in the pond", "Open pond offers sinking-food interaction with concurrent native CUDA PPO. Click or tap open water to feed; the creature makes its own choices. GPU learning is available in the NVIDIA developer build. Experiments preserves the frozen trained reference and its CPU fallback."],
			["Interact and learn", "Current applies steady 0.5 or 0.75 cm/s flow. Pellets keep sinking when flow is off. Background swimmers practice your drop locations and water conditions; improved motor weights are adopted only after comparison checks. The display separates PPO updates from adopted updates. The original selector and swimming source remain preserved."],
			["Save and resume feeding", "The feeding pond autosaves every 30 seconds and on exit: learner, optimizer, practice worlds and visible pond. Leaving the pond stops learning; reopen it to resume. Pause learning keeps the visible swimmer running. Your original policies and older training saves are preserved."],
			["Original headless tools", "The console below runs the preserved original food-selector task, not sinking-food learning. It resumes its own full save; Ctrl+C saves and stops. The feeding learner runs while its pond is open and uses a separate checkpoint. Recordings in Experiments are labeled and do not train."]
		]:
			body.add_child(_label(item[0], 20))
			var paragraph := _label(item[1], 15, MUTED)
			paragraph.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
			body.add_child(paragraph)
		var run := _button("Open original selector console", "headless", false, true)
		run.tooltip_text = "Runs the original selector task, using its separate checkpoint; does not train the sinking-food pond."
		run.pressed.connect(_start_headless)
		body.add_child(run)
		var copy := Button.new()
		copy.text = "Copy PowerShell command"
		copy.custom_minimum_size.y = 44
		copy.pressed.connect(func(): DisplayServer.clipboard_set(_headless_command()); copy.text = "Command copied")
		body.add_child(copy)
	guide.popup_centered(Vector2i(mini(650, int(size.x) - 24), mini(640, int(size.y) - 80)))

func _open(demo: String, watch: bool) -> void:
	if opening:
		return
	opening = true
	for button in buttons:
		button.disabled = true
	status.text = "Opening %s…" % {"cartpole": "CartPole", "racing": "Racing", "pond": "Shallow-water learning", "pond_live": "Live pond", "deep_water": "Original swimmer"}.get(demo, demo)
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

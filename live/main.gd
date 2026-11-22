extends Node3D
## One frontend and one native world. No UI physics, replay actor or optimizer.
const PondWorld = preload("res://pond_world.gd")
const TEXT := Color("e8f3ee")
const MUTED := Color("b1c9c7")
const MINT := Color("a1e2c5")
var preview: RefCounted
var snapshot: Dictionary = {}
var metadata: Dictionary = {}
var metadata_issue := "Assessment metadata is unavailable. No success claim is made."
var policy_path := "res://research_data/learned-amplitude-v1.bin"
var metadata_path := "res://research_data/viewer.json"
var world: PondWorld
var loaded := false
var running := false
var accumulator := 0.0
var physical_substeps := 0
var error := ""
var last_action := ""
var canvas: Control
var brand: Label
var status: Label
var details_button: Button
var rail: PanelContainer
var play_button: Button
var restart_button: Button
var task: OptionButton
var measurements: HBoxContainer
var speed: Label
var displacement: Label
var depth: Label
var target_caption: Label
var notice: Label
var scrim: ColorRect
var drawer: PanelContainer
var drawer_scroll: ScrollContainer
var drawer_text: RichTextLabel
var close_button: Button
var measurement_toggle: CheckBox
var show_measurements := true
var capture_path := ""

func label(value: String, size := 15, color := TEXT) -> Label:
	var result := Label.new()
	result.text = value
	result.add_theme_font_size_override("font_size",size)
	result.add_theme_color_override("font_color",color)
	result.mouse_filter = Control.MOUSE_FILTER_IGNORE
	return result

func panel(color: Color, padding := 12) -> StyleBoxFlat:
	var result := StyleBoxFlat.new()
	result.bg_color = color
	result.set_corner_radius_all(10)
	result.content_margin_left = padding
	result.content_margin_right = padding
	result.content_margin_top = padding
	result.content_margin_bottom = padding
	return result

func button(value: String, action: Callable, primary := false) -> Button:
	var result := Button.new()
	result.text = value
	result.custom_minimum_size = Vector2(76,48)
	result.focus_mode = Control.FOCUS_ALL
	result.mouse_default_cursor_shape = Control.CURSOR_POINTING_HAND
	result.add_theme_font_size_override("font_size",15)
	result.add_theme_color_override("font_color",Color("173b3b") if primary else TEXT)
	result.add_theme_color_override("font_hover_color",Color("173b3b") if primary else TEXT)
	result.add_theme_color_override("font_pressed_color",Color("173b3b") if primary else TEXT)
	result.add_theme_color_override("font_focus_color",Color("173b3b") if primary else TEXT)
	result.add_theme_color_override("font_disabled_color",MUTED)
	result.add_theme_stylebox_override("normal",panel(MINT if primary else Color("244750"),10))
	result.add_theme_stylebox_override("hover",panel(Color("b8efda") if primary else Color("31565f"),10))
	result.add_theme_stylebox_override("pressed",panel(Color("7fc4a6") if primary else Color("3d646a"),10))
	result.add_theme_stylebox_override("disabled",panel(Color("24434b"),10))
	var focus := panel(Color.TRANSPARENT,10)
	focus.set_border_width_all(2)
	focus.border_color = TEXT
	focus.expand_margin_left = 3
	focus.expand_margin_right = 3
	focus.expand_margin_top = 3
	focus.expand_margin_bottom = 3
	result.add_theme_stylebox_override("focus",focus)
	result.pressed.connect(action)
	return result

func _ready() -> void:
	get_window().title = "Tadpole Pond"
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--pond-capture="): capture_path = argument.trim_prefix("--pond-capture=")
		if argument.begins_with("--pond-policy="): policy_path = argument.trim_prefix("--pond-policy=")
		if argument.begins_with("--pond-metadata="): metadata_path = argument.trim_prefix("--pond-metadata=")
	_read_metadata()
	_load_native()
	_bind_metadata()
	world = PondWorld.new()
	if preview != null: world.geometry = preview.call("get_geometry")
	add_child(world)
	_build_interface()
	get_window().size_changed.connect(_layout)
	_layout()
	_refresh()
	play_button.grab_focus()
	if not capture_path.is_empty(): _capture.call_deferred()

func _read_metadata() -> void:
	if not FileAccess.file_exists(metadata_path): return
	var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(metadata_path))
	if parsed is Dictionary and int(parsed.get("schema",0))==1:
		var declared: String=str(parsed.get("policy",{}).get("package_sha256","")).to_lower()
		var actual:=FileAccess.get_sha256(policy_path).to_lower() if FileAccess.file_exists(policy_path) else ""
		if declared.is_empty() or declared!=actual:
			metadata_issue="Assessment metadata does not match this saved policy. No success claim is made."
			return
		metadata = parsed

func _bind_metadata() -> void:
	if metadata.is_empty(): return
	var policy: Dictionary=metadata.get("policy",{})
	if not loaded or int(policy.get("holding_updates",-1))!=int(snapshot.get("holding_updates",-2)) or int(policy.get("package_version",-1))!=int(snapshot.get("package_version",-2)) or str(policy.get("policy_method",""))!=str(snapshot.get("policy_method","unknown")):
		metadata.clear()
		metadata_issue="Assessment metadata does not match the loaded policy generation. No success claim is made."

func _load_native() -> void:
	if not ClassDB.class_exists("HoldLabPreview"):
		GDExtensionManager.load_extension("res://addons/frozen_swimmer/frozen_swimmer.gdextension")
	if not ClassDB.class_exists("HoldLabPreview"):
		error = "The native pond module is unavailable. Rebuild the app to continue."
		return
	preview = ClassDB.instantiate("HoldLabPreview") as RefCounted
	if preview == null:
		error = "The native pond module could not start."
		return
	if not FileAccess.file_exists(policy_path):
		error = "The saved policy is missing. No simulation has started."
		return
	loaded = bool(preview.call("load_controller",ProjectSettings.globalize_path(policy_path)))
	if loaded: snapshot = preview.call("get_state")
	else: error = "This saved policy could not load. " + str(preview.call("get_error"))

func _build_interface() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	canvas = Control.new()
	canvas.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	canvas.mouse_filter = Control.MOUSE_FILTER_IGNORE
	layer.add_child(canvas)
	brand = label("Tadpole Pond",22)
	canvas.add_child(brand)
	status = label("Paused · live, saved policy",13,MUTED)
	status.tooltip_text = "Native live physics using saved weights. No training occurs in this view."
	canvas.add_child(status)
	details_button = button("Details",_open_details)
	canvas.add_child(details_button)
	rail = PanelContainer.new()
	rail.add_theme_stylebox_override("panel",panel(Color(.07,.17,.20,.94)))
	canvas.add_child(rail)
	var column := VBoxContainer.new()
	column.add_theme_constant_override("separation",12)
	rail.add_child(column)
	measurements = HBoxContainer.new()
	measurements.add_theme_constant_override("separation",16)
	column.add_child(measurements)
	for value in ["0.00 mm/s","0.0 mm away","0.00 mm off depth"]:
		var reading := label(value,13,MUTED)
		reading.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		reading.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		measurements.add_child(reading)
		if speed == null: speed = reading
		elif displacement == null: displacement = reading
		else: depth = reading
	var controls := HBoxContainer.new()
	controls.add_theme_constant_override("separation",8)
	column.add_child(controls)
	play_button = button("Play",_play_pause,true)
	play_button.tooltip_text = "Play or pause native physics · Space"
	controls.add_child(play_button)
	restart_button = button("Restart",_restart)
	restart_button.tooltip_text = "Restart this task from its original physical state"
	controls.add_child(restart_button)
	task = OptionButton.new()
	task.add_item("Hold test",0)
	task.add_item("Swim test",1)
	task.custom_minimum_size = Vector2(100,48)
	task.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	task.add_theme_font_size_override("font_size",15)
	task.tooltip_text = "Choose the movement request. Changing it restarts the task."
	task.item_selected.connect(func(_index: int): _restart())
	controls.add_child(task)
	target_caption = label("Target",12,MINT)
	target_caption.add_theme_stylebox_override("normal",panel(Color("15363f"),5))
	canvas.add_child(target_caption)
	notice = label("",15,TEXT)
	notice.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	notice.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	canvas.add_child(notice)
	_build_details()

func _build_details() -> void:
	scrim = ColorRect.new()
	scrim.color = Color(0,.03,.04,.55)
	scrim.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	scrim.visible = false
	scrim.mouse_filter = Control.MOUSE_FILTER_STOP
	canvas.add_child(scrim)
	scrim.gui_input.connect(func(event: InputEvent):
		if event is InputEventMouseButton and event.pressed: _close_details())
	drawer = PanelContainer.new()
	drawer.add_theme_stylebox_override("panel",panel(Color("17373f"),20))
	drawer.visible = false
	canvas.add_child(drawer)
	var column := VBoxContainer.new()
	column.add_theme_constant_override("separation",16)
	drawer.add_child(column)
	var top := HBoxContainer.new()
	column.add_child(top)
	var heading := label("About this swimmer",20)
	heading.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	top.add_child(heading)
	close_button = button("Close",_close_details)
	top.add_child(close_button)
	drawer_scroll = ScrollContainer.new()
	drawer_scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	drawer_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	column.add_child(drawer_scroll)
	drawer_text = RichTextLabel.new()
	drawer_text.bbcode_enabled = true
	drawer_text.fit_content = true
	drawer_text.selection_enabled = true
	drawer_text.focus_mode = Control.FOCUS_ALL
	drawer_text.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	drawer_text.add_theme_font_size_override("normal_font_size",15)
	drawer_text.add_theme_color_override("default_color",TEXT)
	drawer_scroll.add_child(drawer_text)
	measurement_toggle = CheckBox.new()
	measurement_toggle.text = "Show measurements"
	measurement_toggle.button_pressed = true
	measurement_toggle.custom_minimum_size.y = 48
	measurement_toggle.toggled.connect(func(value: bool): show_measurements = value; _layout())
	column.add_child(measurement_toggle)

func _layout() -> void:
	if canvas == null: return
	var size := get_viewport().get_visible_rect().size
	var inset := 16.0 if size.x<650 or size.y<500 else 24.0
	brand.position = Vector2(inset,18)
	status.position = Vector2(inset,48)
	details_button.position = Vector2(size.x-inset-88,16)
	details_button.size = Vector2(88,48)
	measurements.visible = show_measurements
	var rail_height := 108.0 if show_measurements else 72.0
	var rail_width := minf(600,size.x-inset*2)
	rail.position = Vector2((size.x-rail_width)/2,size.y-inset-rail_height)
	rail.size = Vector2(rail_width,rail_height)
	for reading in [speed,displacement,depth]: reading.add_theme_font_size_override("font_size",12 if size.x<500 else 13)
	world.viewport_area = Rect2(inset,88,size.x-inset*2,maxf(100,rail.position.y-104))
	world.fit_view()
	notice.position = Vector2(inset,rail.position.y-76)
	notice.size = Vector2(size.x-inset*2,60)
	var drawer_width := minf(440,size.x-inset*2)
	drawer.position = Vector2(size.x-inset-drawer_width,inset)
	drawer.size = Vector2(drawer_width,size.y-inset*2)
	_refresh_target()

func _play_pause() -> void:
	if not loaded: return
	last_action = ""
	if running: preview.call("pause")
	else: preview.call("start")
	snapshot = preview.call("get_state")
	running = not bool(snapshot.get("paused",true))
	accumulator = 0
	_refresh()

func _restart() -> void:
	if not loaded: return
	preview.call("pause")
	running = false
	accumulator = 0
	if not bool(preview.call("reset",0,task.selected==0)): error = str(preview.call("get_error"))
	snapshot = preview.call("get_state")
	last_action = "Hold test restarted" if task.selected==0 else "Swim test restarted"
	_refresh()

func _process(delta: float) -> void:
	if not loaded or not running: return
	accumulator += delta
	var controls := 0
	while accumulator>=.05 and controls<4 and running:
		var before := int(snapshot.get("substeps",0))
		var advanced := bool(preview.call("step"))
		snapshot = preview.call("get_state")
		physical_substeps += maxi(0,int(snapshot.get("substeps",0))-before)
		running = advanced and not bool(snapshot.get("paused",true))
		accumulator -= .05
		controls += 1
	_refresh()

func _refresh() -> void:
	if play_button == null: return
	play_button.disabled = not loaded or bool(snapshot.get("finished",false)) or not str(snapshot.get("error","")).is_empty()
	restart_button.disabled = not loaded
	task.disabled = not loaded
	play_button.text = "Pause" if running else "Play"
	if loaded:
		var state := "Running" if running else ("Test complete" if bool(snapshot.get("finished",false)) else "Paused")
		status.text = (last_action+" · saved policy") if not running and not last_action.is_empty() else state+" · live, saved policy"
		world.show_state(snapshot)
		speed.text = "%.2f mm/s" % (float(snapshot.get("speed_m_s",0))*1000)
		var holding := bool(snapshot.get("holding",true))
		var amount := float(snapshot.get("anchor_error_m",0) if holding else snapshot.get("forward_m",0))*1000
		displacement.text = ("%.1f mm away" % amount) if holding else ("%.1f mm swum" % amount)
		depth.text = "%.2f mm off depth" % (float(snapshot.get("depth_m",0))*1000)
		error = str(snapshot.get("error",error))
	else: status.text = "Preview unavailable"
	notice.text = error
	notice.visible = not error.is_empty()
	_refresh_target()

func _refresh_target() -> void:
	if target_caption == null or world == null: return
	target_caption.visible = loaded and bool(snapshot.get("holding",true)) and show_measurements
	if target_caption.visible:
		target_caption.text = world.target_caption()
		target_caption.position = world.target_caption_position()

func _open_details() -> void:
	if preview != null:
		preview.call("pause")
		if loaded: snapshot = preview.call("get_state")
	running = false
	accumulator = 0
	_refresh()
	var result: Dictionary = metadata.get("result",{})
	var policy: Dictionary = metadata.get("policy",{})
	drawer_text.text = "[b]A saved policy in a physical world[/b]\nThe creature moves through native C++ physics. This view uses saved weights; it does not train or update them.\n\n[b]Latest assessment[/b]\n"
	if result.is_empty(): drawer_text.append_text(metadata_issue+"\n")
	else: drawer_text.append_text("%s\nHolding: %d / %d\nSwimming: %d / 4 short tests\nLong windows: %d / 30; %d / 28 prior passes retained\n\n%s\n" % [str(result.get("outcome","See source evidence")),int(result.get("hold_passes",0)),int(result.get("hold_total",6)),int(result.get("short_passes",0)),int(result.get("long_passes",0)),int(result.get("long_retained",0)),str(result.get("method",""))])
	if not str(result.get("validation_note","")).is_empty(): drawer_text.append_text("\n[b]Additional validation[/b]\n"+str(result.validation_note)+"\n")
	drawer_text.append_text("\n[b]Scope[/b]\nBoth tests start from rest. Changing the task restarts its physical state; it does not ask the swimmer to brake. One four-output motor. Holding, braking, currents and recovery are separate skills. This research preview does not replace the working pond.\n")
	if not policy.is_empty(): drawer_text.append_text("\n[b]Saved generation[/b]\n%s\n%d lesson updates\n" % [str(policy.get("family","Saved research motor")),int(policy.get("holding_updates",0))])
	var history: Array = metadata.get("history",[])
	if not history.is_empty():
		drawer_text.append_text("\n[b]Earlier results[/b]\n")
		for entry in history: drawer_text.append_text("%s — %s\n" % [str(entry.get("name","Earlier study")),str(entry.get("outcome","See evidence"))])
	drawer_text.append_text("\n[b]Evidence source[/b]\n"+str(result.get("source_run","Unavailable")))
	scrim.visible = true
	drawer.visible = true
	for control in [play_button,restart_button,task,details_button]: control.focus_mode=Control.FOCUS_NONE
	close_button.grab_focus()

func _close_details() -> void:
	scrim.visible = false
	drawer.visible = false
	for control in [play_button,restart_button,task,details_button]: control.focus_mode=Control.FOCUS_ALL
	details_button.grab_focus()

func _unhandled_key_input(event: InputEvent) -> void:
	if not event is InputEventKey or not event.pressed or event.echo: return
	if event.keycode==KEY_ESCAPE and drawer.visible:
		_close_details()
		get_viewport().set_input_as_handled()
	elif event.keycode==KEY_SPACE and not drawer.visible:
		if get_viewport().gui_get_focus_owner() is BaseButton: return
		_play_pause()
		get_viewport().set_input_as_handled()

func _capture() -> void:
	await get_tree().process_frame
	await RenderingServer.frame_post_draw
	var code := get_viewport().get_texture().get_image().save_png(capture_path)
	print(JSON.stringify({"pond_capture":code==OK,"loaded":loaded,"paused":not running,"physical_substeps":physical_substeps,"controls":int(snapshot.get("controls",0))}))
	get_tree().quit(0 if code==OK else 3)

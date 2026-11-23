extends Node3D
## One frontend and one native world. No UI physics, replay actor or optimizer.
const PondWorld = preload("res://pond_world.gd")
const ContinuousStudy = preload("res://continuous_study.gd")
const RequestTurnStudy = preload("res://request_turn_study.gd")
const TEXT := Color("243e37")
const MUTED := Color("566b61")
const MINT := Color("2d6757")
const PAPER := Color("f7f5ed")
const SURFACE := Color("fdfbf5")
var preview: RefCounted
var snapshot: Dictionary = {}
var metadata: Dictionary = {}
var metadata_issue := "Assessment metadata is unavailable. No success claim is made."
var policy_path := "res://research_data/learned-amplitude-v1.bin"
var metadata_path := "res://research_data/viewer.json"
var continuous_metadata_path := "res://research_data/continuous-pond-v1.json"
var continuous_available := false
var current_study_path := "res://research_data/current-study.json"
var current_study: Dictionary = {}
var current_study_issue := ""
var world: PondWorld
var loaded := false
var running := false
var accumulator := 0.0
var physical_substeps := 0
var error := ""
var last_action := ""
var canvas: Control
var identity_panel: PanelContainer
var brand: Label
var status: Label
var details_button: Button
var view_button: Button
var pond_overview := true
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
var show_measurements := false
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
	result.add_theme_color_override("font_color",Color("ffffff") if primary else TEXT)
	result.add_theme_color_override("font_hover_color",Color("ffffff") if primary else TEXT)
	result.add_theme_color_override("font_pressed_color",Color("ffffff") if primary else TEXT)
	result.add_theme_color_override("font_focus_color",Color("ffffff") if primary else TEXT)
	result.add_theme_color_override("font_disabled_color",MUTED)
	result.add_theme_stylebox_override("normal",panel(MINT if primary else Color("e8ede2"),10))
	result.add_theme_stylebox_override("hover",panel(Color("397660") if primary else Color("dce7d6"),10))
	result.add_theme_stylebox_override("pressed",panel(Color("245443") if primary else Color("cfdcc8"),10))
	result.add_theme_stylebox_override("disabled",panel(Color("e3e8dd"),10))
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
	_select_continuous_package()
	_read_metadata()
	_load_native()
	_bind_metadata()
	_read_current_study()
	world = PondWorld.new()
	if preview != null: world.geometry = preview.call("get_geometry")
	add_child(world)
	world.set_overview(pond_overview)
	_build_interface()
	if continuous_available:
		if bool(preview.call("reset_free_swim",0)): snapshot = preview.call("get_state")
	elif loaded and int(snapshot.get("gait_updates",0))>0:
		task.select(1)
		if bool(preview.call("reset",0,false)): snapshot = preview.call("get_state")
	get_window().size_changed.connect(_layout)
	_layout()
	_refresh()
	play_button.grab_focus()
	if not capture_path.is_empty(): _capture.call_deferred()

func _select_continuous_package() -> void:
	# Explicit developer overrides keep their own policy and assessment binding.
	if policy_path!="res://research_data/learned-amplitude-v1.bin" or metadata_path!="res://research_data/viewer.json": return
	var candidate := _dictionary_file(continuous_metadata_path)
	var declared: Variant=candidate.get("policy",{})
	if int(candidate.get("schema",0))!=1 or not declared is Dictionary: return
	if declared.get("qualified",false)!=true: return
	var saved_path := "res://research_data/continuous-pond-v1.bin"
	if not FileAccess.file_exists(saved_path): return
	if str(declared.get("package_sha256","")).to_lower()!=FileAccess.get_sha256(saved_path).to_lower(): return
	var assessment := _bound_study_file(candidate.get("assessment",{}))
	var proof := _bound_study_file(candidate.get("verification",{}))
	if assessment.get("qualified",false)!=true or proof.get("passed",false)!=true or proof.get("qualified",false)!=true: return
	if str(proof.get("assessment_sha256","")).to_lower()!=str(candidate.get("assessment",{}).get("sha256","")).to_lower(): return
	policy_path=saved_path
	metadata_path=continuous_metadata_path

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

func _dictionary_file(path: String) -> Dictionary:
	if not FileAccess.file_exists(path): return {}
	var parser := JSON.new()
	if parser.parse(FileAccess.get_file_as_string(path))!=OK: return {}
	return parser.data if parser.data is Dictionary else {}

func _valid_study_count(value: Variant) -> bool:
	if not (value is int or value is float): return false
	var number := float(value)
	return number>=0 and number<=1000000000000.0 and floorf(number)==number

func _bound_study_file(record: Variant) -> Dictionary:
	if not record is Dictionary: return {}
	var path: Variant=record.get("path",null)
	var digest: Variant=record.get("sha256",null)
	if not path is String or not digest is String or digest.length()!=64: return {}
	if not FileAccess.file_exists(path) or FileAccess.get_sha256(path).to_lower()!=digest.to_lower(): return {}
	return _dictionary_file(path)

func _load_current_study(path: String) -> Dictionary:
	if not FileAccess.file_exists(path): return {}
	var invalid := {"issue":"The current-study result is unavailable or does not match its saved evidence."}
	var declared := _dictionary_file(path)
	if not _valid_study_count(declared.get("schema",null)) or int(declared.schema)!=1: return invalid
	var source := _bound_study_file(declared.get("source",null))
	if not source.get("closed",null) is bool or not bool(source.closed) or not source.get("qualified",null) is bool: return invalid
	for field in ["controls","substeps","PPO_updates","actor_steps"]:
		if not _valid_study_count(source.get(field,null)): return invalid
	if float(source.substeps)>10*float(source.controls): return invalid
	var explicit_current_study: bool=source.get("study",null) is String and str(source.study) in ["current_swim_v1","current_swim_v2"]
	var case_limits := {"baseline_current_passes":16,"final_current_passes":16,"withheld_passes":16,"endurance_passes":2}
	for field in case_limits:
		var count: Variant=source.get(field,null)
		if count==null:
			if not explicit_current_study or bool(source.qualified): return invalid
		elif not _valid_study_count(count) or float(count)>float(case_limits[field]): return invalid
	if not source.get("fault",null) is String or not source.get("gate",null) is String: return invalid
	if bool(source.qualified) and not str(source.fault).is_empty(): return invalid
	if str(source.get("phase",""))=="physics_gate_failed":
		if bool(source.qualified) or int(source.PPO_updates)!=0 or int(source.actor_steps)!=0: return invalid
		for field in ["training_started","current_suites_run"]:
			if not source.get(field,null) is bool or bool(source[field]): return invalid
	var suite_fields := {"withheld_suites_run":"withheld_passes","endurance_suites_run":"endurance_passes"}
	for suite in suite_fields:
		if source.has(suite):
			if not source[suite] is bool: return invalid
			if not bool(source[suite]) and (source.get(suite_fields[suite],null)!=null or bool(source.qualified)): return invalid
	var verified := false
	var proof := _bound_study_file(declared.get("verification",null))
	var inputs: Variant=proof.get("input_sha256",{})
	var source_identity: String
	if declared.source.has("verification_input"):
		var alias: Variant=declared.source.verification_input
		if not alias is String or alias.strip_edges().is_empty(): return invalid
		# This is the original audit's literal identity, not a file to open.
		# Do not globalize an old Windows path on a different operating system.
		source_identity=alias.replace("\\","/").simplify_path().to_lower()
	else:
		source_identity=ProjectSettings.globalize_path(str(declared.source.path)).replace("\\","/").simplify_path().to_lower()
	if proof.get("passed",null) is bool and bool(proof.passed) and inputs is Dictionary:
		for bound_path in inputs:
			if str(bound_path).replace("\\","/").simplify_path().to_lower()==source_identity:
				verified=str(inputs[bound_path]).to_lower()==str(declared.source.sha256).to_lower()
	return {"result":source,"verified":verified,"source_path":str(declared.source.path)}

func _read_current_study() -> void:
	var checked := _load_current_study(current_study_path)
	current_study=checked if checked.has("result") else {}
	current_study_issue=str(checked.get("issue",""))

func _append_current_study() -> void:
	_read_current_study()
	if current_study.is_empty() and current_study_issue.is_empty(): return
	drawer_text.append_text("\n[b]Current swimming study[/b]\n")
	if current_study.is_empty(): drawer_text.append_text(current_study_issue+"\n")
	else:
		var result: Dictionary=current_study.result
		var physics_gate_failed := str(result.get("phase",""))=="physics_gate_failed"
		if physics_gate_failed:
			drawer_text.append_text("Physics check stopped · No PPO updates.\nCurrent swimming has not yet been assessed.\n")
		elif bool(result.qualified) and bool(current_study.verified):
			drawer_text.append_text("Current swimming passed this study's independent validation.\n")
		elif bool(result.qualified):
			drawer_text.append_text("Study closed; independent verification is unavailable.\n")
		elif not str(result.fault).is_empty():
			drawer_text.append_text("Study stopped at a safety or data check. Current swimming is not qualified.\n")
		else: drawer_text.append_text("Study complete. Current swimming is not ready.\n")
		if not physics_gate_failed: drawer_text.append_text("Native PPO updates: %d\n" % int(result.PPO_updates))
		if _valid_study_count(result.get("final_current_passes",null)):
			if _valid_study_count(result.get("baseline_current_passes",null)):
				drawer_text.append_text("Current cases: %d → %d / 16\n" % [int(result.baseline_current_passes),int(result.final_current_passes)])
			else: drawer_text.append_text("Current cases: %d / 16\n" % int(result.final_current_passes))
		if _valid_study_count(result.get("withheld_passes",null)): drawer_text.append_text("Withheld cases: %d / 16\n" % int(result.withheld_passes))
		if _valid_study_count(result.get("endurance_passes",null)): drawer_text.append_text("Endurance cases: %d / 2\n" % int(result.endurance_passes))
		if bool(current_study.verified): drawer_text.append_text("Saved evidence independently checked.\n")
	drawer_text.append_text("This preview still uses the saved calm-water policy. No current policy is adopted here.\n")

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
	if loaded:
		snapshot = preview.call("get_state")
		continuous_available=bool(snapshot.get("has_continuous",false))
	else: error = "This saved policy could not load. " + str(preview.call("get_error"))

func _build_interface() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	canvas = Control.new()
	canvas.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	canvas.mouse_filter = Control.MOUSE_FILTER_IGNORE
	layer.add_child(canvas)
	identity_panel = PanelContainer.new()
	identity_panel.add_theme_stylebox_override("panel",panel(SURFACE,12))
	identity_panel.mouse_filter = Control.MOUSE_FILTER_IGNORE
	canvas.add_child(identity_panel)
	var identity := VBoxContainer.new()
	identity.add_theme_constant_override("separation",0)
	identity.mouse_filter = Control.MOUSE_FILTER_IGNORE
	identity_panel.add_child(identity)
	brand = label("Tadpole Pond",22)
	identity.add_child(brand)
	status = label("Paused · live, saved policy",13,MUTED)
	status.clip_text = true
	status.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
	status.tooltip_text = "Native live physics using saved weights. No training occurs in this view."
	identity.add_child(status)
	details_button = button("Details",_open_details)
	canvas.add_child(details_button)
	view_button = button("Side view" if pond_overview else "Aerial view",_toggle_pond_view)
	view_button.tooltip_text = "Change the camera angle · Scroll, pinch or + / − to zoom"
	canvas.add_child(view_button)
	rail = PanelContainer.new()
	rail.add_theme_stylebox_override("panel",panel(Color(.992,.984,.961,.97)))
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
	if continuous_available:
		task.add_item("Free swim",2)
		task.add_item("Rest",0)
	else:
		task.add_item("Rest",0)
		task.add_item("Swim",1)
	task.custom_minimum_size = Vector2(100,48)
	task.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	task.add_theme_font_size_override("font_size",15)
	for state in ["normal","hover","pressed"]:
		task.add_theme_stylebox_override(state,panel(Color("e8ede2"),10))
		task.add_theme_color_override("font_"+state+"_color" if state!="normal" else "font_color",TEXT)
	var task_focus := panel(Color.TRANSPARENT,10)
	task_focus.set_border_width_all(2)
	task_focus.border_color = MINT
	task.add_theme_stylebox_override("focus",task_focus)
	task.add_theme_color_override("font_focus_color",TEXT)
	task.add_theme_color_override("font_disabled_color",MUTED)
	task.add_theme_color_override("icon_normal_color",TEXT)
	task.add_theme_color_override("icon_hover_color",TEXT)
	var choice_menu := task.get_popup()
	choice_menu.add_theme_stylebox_override("panel",panel(SURFACE,8))
	choice_menu.add_theme_stylebox_override("hover",panel(Color("dce7d6"),8))
	choice_menu.add_theme_color_override("font_color",TEXT)
	choice_menu.add_theme_color_override("font_hover_color",TEXT)
	task.tooltip_text = "Choose the movement request. Changing it restarts the task."
	task.item_selected.connect(func(_index: int): _restart())
	controls.add_child(task)
	target_caption = label("Target",12,MINT)
	target_caption.add_theme_stylebox_override("normal",panel(SURFACE,5))
	canvas.add_child(target_caption)
	notice = label("",15,TEXT)
	notice.add_theme_stylebox_override("normal",panel(SURFACE,12))
	notice.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	notice.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	canvas.add_child(notice)
	_build_details()

func _build_details() -> void:
	scrim = ColorRect.new()
	scrim.color = Color(.1,.18,.14,.28)
	scrim.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	scrim.visible = false
	scrim.mouse_filter = Control.MOUSE_FILTER_STOP
	canvas.add_child(scrim)
	scrim.gui_input.connect(func(event: InputEvent):
		if event is InputEventMouseButton and event.pressed: _close_details())
	drawer = PanelContainer.new()
	drawer.add_theme_stylebox_override("panel",panel(SURFACE,20))
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
	measurement_toggle.add_theme_color_override("font_color",TEXT)
	measurement_toggle.add_theme_color_override("font_hover_color",TEXT)
	measurement_toggle.button_pressed = show_measurements
	measurement_toggle.custom_minimum_size.y = 48
	measurement_toggle.toggled.connect(func(value: bool): show_measurements = value; _layout())
	column.add_child(measurement_toggle)

func _layout() -> void:
	if canvas == null: return
	var size := get_viewport().get_visible_rect().size
	var inset := 16.0 if size.x<650 or size.y<500 else 24.0
	identity_panel.position = Vector2(inset,12)
	var identity_height := maxf(68,identity_panel.get_combined_minimum_size().y)
	identity_panel.size = Vector2(minf(300,size.x-inset*2-96),identity_height)
	details_button.position = Vector2(size.x-inset-88,16)
	details_button.size = Vector2(88,48)
	view_button.position = Vector2(size.x-inset-112,12+identity_height+8) if size.x<600 else Vector2(size.x-inset-208,16)
	view_button.size = Vector2(112,48)
	measurements.visible = show_measurements
	var rail_height := 108.0 if show_measurements else 72.0
	var rail_width := minf(600,size.x-inset*2)
	rail.position = Vector2((size.x-rail_width)/2,size.y-inset-rail_height)
	rail.size = Vector2(rail_width,rail_height)
	for reading in [speed,displacement,depth]: reading.add_theme_font_size_override("font_size",12 if size.x<500 else 13)
	var world_top := view_button.position.y+64 if size.x<600 else maxf(96,12+identity_height+12)
	world.viewport_area = Rect2(inset,world_top,size.x-inset*2,maxf(80,rail.position.y-world_top-16))
	world.fit_view()
	notice.position = Vector2(inset,rail.position.y-112)
	notice.size = Vector2(size.x-inset*2,96)
	var drawer_width := minf(440,size.x-inset*2)
	drawer.position = Vector2(size.x-inset-drawer_width,inset)
	drawer.size = Vector2(drawer_width,size.y-inset*2)
	_refresh_target()

func _toggle_pond_view() -> void:
	pond_overview = not pond_overview
	view_button.text = "Side view" if pond_overview else "Aerial view"
	world.set_overview(pond_overview)
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
	var choice := task.get_selected_id()
	var reset_ok := bool(preview.call("reset_free_swim",0)) if choice==2 else bool(preview.call("reset",0,choice==0))
	if not reset_ok: error = str(preview.call("get_error"))
	snapshot = preview.call("get_state")
	last_action = "Rest restarted" if choice==0 else ("Free swim restarted" if choice==2 else "Swim restarted")
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
		var state := ("Resting" if bool(snapshot.get("holding",true)) else "Swimming") if running else ("Session complete" if bool(snapshot.get("finished",false)) else "Paused")
		if not str(snapshot.get("error","")).is_empty(): state="Stopped"
		status.text = (last_action+" · saved policy") if not running and not last_action.is_empty() else state+" · live, saved policy"
		world.show_state(snapshot)
		speed.text = "%.2f mm/s" % (float(snapshot.get("speed_m_s",0))*1000)
		var holding := bool(snapshot.get("holding",true))
		var amount := float(snapshot.get("anchor_error_m",0) if holding else snapshot.get("forward_m",0))*1000
		displacement.text = ("%.1f mm away" % amount) if holding else ("%.1f mm swum" % amount)
		if bool(snapshot.get("continuous",false)):
			var elapsed := int(float(snapshot.get("time_s",0)))
			displacement.text = "%d:%02d swimming" % [floori(elapsed/60.0),elapsed%60]
		depth.text = "%.2f mm off depth" % (float(snapshot.get("depth_m",0))*1000)
		error = str(snapshot.get("error",error))
	else: status.text = "Preview unavailable"
	notice.text = error
	notice.visible = not error.is_empty()
	_refresh_target()

func _refresh_target() -> void:
	if target_caption == null or world == null: return
	target_caption.visible = loaded and bool(snapshot.get("holding",true)) and show_measurements and not pond_overview
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
	drawer_text.append_text(RequestTurnStudy.text("res://research_data/request-turn-study.json"))
	drawer_text.append_text(ContinuousStudy.text("res://research_data/continuous-study.json"))
	_append_current_study()
	drawer_text.append_text("\n[b]Your pond[/b]\nThe aerial camera stays above the creature; Side view lowers the angle. Scroll, pinch or use + / − to zoom in and out. The body, water and boundaries come from the native simulation.\n\n[b]Movement scope[/b]\n")
	if continuous_available:
		drawer_text.append_text("Free swim has no session timer or food target. The saved PPO motor steers using its physical state and distance from the bank. It receives no scripted route, bounce or rescue. Pause leaves the world in place.\n\n"+str(result.get("continuous_validation_note","See the saved continuous-swimming assessment."))+"\n\n")
	else:
		drawer_text.append_text("Rest and Swim use bounded tests and pause when complete.\n\n")
	drawer_text.append_text("Changing the request or choosing Restart starts a new physical state; it does not demonstrate braking. This view does not run learning updates. Current swimming remains separate research.\n")
	if not policy.is_empty(): drawer_text.append_text("\n[b]Saved generation[/b]\n%s\n%d lesson updates\n" % [str(policy.get("family","Saved research motor")),int(policy.get("holding_updates",0))])
	var history: Array = metadata.get("history",[])
	if not history.is_empty():
		drawer_text.append_text("\n[b]Earlier results[/b]\n")
		for entry in history: drawer_text.append_text("%s — %s\n" % [str(entry.get("name","Earlier study")),str(entry.get("outcome","See evidence"))])
	drawer_text.append_text("\n[b]Evidence source[/b]\n"+str(result.get("source_run","Unavailable")))
	scrim.visible = true
	drawer.visible = true
	for control in [play_button,restart_button,task,details_button,view_button]: control.focus_mode=Control.FOCUS_NONE
	close_button.grab_focus()

func _close_details() -> void:
	scrim.visible = false
	drawer.visible = false
	for control in [play_button,restart_button,task,details_button,view_button]: control.focus_mode=Control.FOCUS_ALL
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
	elif not drawer.visible and event.keycode in [KEY_PLUS,KEY_EQUAL,KEY_KP_ADD,KEY_MINUS,KEY_KP_SUBTRACT]:
		world.zoom_by(1.0/1.16 if event.keycode in [KEY_PLUS,KEY_EQUAL,KEY_KP_ADD] else 1.16)
		_refresh_target()
		get_viewport().set_input_as_handled()

func _unhandled_input(event: InputEvent) -> void:
	if drawer.visible or world==null: return
	var factor := 0.0
	if event is InputEventMouseButton and event.pressed and world.viewport_area.has_point(event.position):
		if event.button_index==MOUSE_BUTTON_WHEEL_UP: factor=1.0/1.16
		elif event.button_index==MOUSE_BUTTON_WHEEL_DOWN: factor=1.16
	elif event is InputEventMagnifyGesture and event.factor>0 and world.viewport_area.has_point(event.position):
		factor=1.0/event.factor
	if factor>0:
		world.zoom_by(factor)
		_refresh_target()
		get_viewport().set_input_as_handled()

func _capture() -> void:
	await get_tree().process_frame
	await RenderingServer.frame_post_draw
	var code := get_viewport().get_texture().get_image().save_png(capture_path)
	print(JSON.stringify({"pond_capture":code==OK,"loaded":loaded,"paused":not running,"physical_substeps":physical_substeps,"controls":int(snapshot.get("controls",0))}))
	get_tree().quit(0 if code==OK else 3)

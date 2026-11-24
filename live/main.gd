extends Node3D
## One frontend and one native world. No UI physics, replay actor or optimizer.
const PondWorld = preload("res://pond_world.gd")
const TEXT := Color("243e37")
const MUTED := Color("566b61")
const MINT := Color("2d6757")
const PAPER := Color("f7f5ed")
const SURFACE := Color("fdfbf5")
const VERIFICATION_ERROR := Color("8a3c2e")
const SHORELINE_BINDING_PATH := "res://research_data/shoreline-runtime.json"
const SHORELINE_CPU_PATH := "res://research_data/shoreline-cpu-v4.json"
const SHORELINE_GPU_PATH := "res://research_data/shoreline-gpu-v4.json"
const SHORELINE_UPDATE_PATH := "res://research_data/shoreline-update.json"
const CANDIDATE_STUDY_PATH := "res://research_data/candidate-study.json"
const STUDY_EVIDENCE_PREFIX := "res://research_data/studies/"
var NATIVE_DLL_PATH := "res://addons/frozen_swimmer/bin/frozen_swimmer.dll"
var preview: RefCounted
var snapshot: Dictionary = {}
var metadata: Dictionary = {}
var metadata_issue := "Assessment metadata is unavailable. No success claim is made."
var policy_path := "res://research_data/learned-amplitude-v1.bin"
var metadata_path := "res://research_data/viewer.json"
var shoreline_binding: Dictionary = {}
var shoreline_proof: Dictionary = {}
var shoreline_qualified := false
var cuda_qualified := false
var shoreline_issue := "The shoreline verification files are unavailable. No qualification claim is made."
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
var steering: VBoxContainer
var steering_status: Label
var turn_left: Button
var turn_right: Button
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
	_read_metadata()
	_load_native()
	_bind_metadata()
	_read_shoreline_status()
	world = PondWorld.new()
	if preview != null:
		world.geometry = preview.call("get_geometry")
		world.geometry_source = preview
	add_child(world)
	if world.landscape!=null and not world.landscape.geometry_ready:
		error=str(world.landscape.geometry_issue)
		loaded=false
	world.set_overview(pond_overview)
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

func _dictionary_file(path: String) -> Dictionary:
	if not FileAccess.file_exists(path): return {}
	var parser := JSON.new()
	if parser.parse(FileAccess.get_file_as_string(path))!=OK: return {}
	return parser.data if parser.data is Dictionary else {}

func _shoreline_validation_issue(binding: Dictionary, proof: Dictionary, dll_hash: String, cpu_hash: String, descriptor: Dictionary) -> String:
	if int(binding.get("schema",0))!=1 or int(binding.get("runtime_version",0))!=4 or int(binding.get("query_arithmetic_version",0))!=2:
		return "The shoreline verification has an incompatible version. Reopen the updated app."
	if int(descriptor.get("runtime_version",0))!=4 or int(descriptor.get("query_arithmetic_version",0))!=2 or int(descriptor.get("collision_envelope_version",0))!=2 or descriptor.get("used_by_contact",false)!=true or descriptor.get("active_motion_domain","")!="shared_pond_terrain":
		return "The shoreline update is still being installed. This app is using the previous runtime."
	var declared_dll := str(binding.get("dll_sha256","")).to_lower()
	if dll_hash.length()!=64 or declared_dll!=dll_hash.to_lower():
		return "This native build has not been matched to the recorded shoreline verification."
	if binding.get("dll_path","")=="res://addons/frozen_swimmer/bin/frozen_swimmer-steering-v1.dll":
		var source: Variant=binding.get("bridge_equivalence",{})
		if not source is Dictionary or source.get("path","")!="res://research_data/live-steering-check.json" or source.get("sha256","")!=FileAccess.get_sha256(str(source.get("path",""))):
			return "The live steering check is missing or has changed. Verification cannot be confirmed."
		var steering_proof := _dictionary_file(str(source.path))
		if steering_proof.get("passed",false)!=true or steering_proof.get("DLL_sha256","")!=dll_hash or steering_proof.get("original_physics_sources_unchanged",false)!=true or int(steering_proof.get("turn_cases_passed",0))!=4 or int(steering_proof.get("total_CPU_controls",0))!=6080 or int(steering_proof.get("optimizer_calls",-1))!=0 or steering_proof.get("candidate_loaded",true)!=false:
			return "The live steering evidence does not match this original-motor build."
	return _cpu_proof_validation_issue(binding,proof,cpu_hash)

func _cpu_proof_validation_issue(binding: Dictionary, proof: Dictionary, cpu_hash: String) -> String:
	var source: Variant=binding.get("cpu_source",{})
	if not source is Dictionary or source.get("path","")!=SHORELINE_CPU_PATH:
		return "The shoreline verification is missing its CPU evidence binding."
	if cpu_hash.length()!=64 or str(source.get("sha256","")).to_lower()!=cpu_hash.to_lower():
		return "The shoreline evidence has changed or is missing. Verification cannot be confirmed."
	if typeof(proof.get("passed",null))!=TYPE_BOOL or proof.get("passed",false)!=true or typeof(proof.get("preflight_only",null))!=TYPE_BOOL or proof.get("preflight_only",true)!=false:
		return "The shoreline evidence does not report a completed physical pass."
	if proof.get("protocol_id","")!="shoreline-product-v4-2026-11-23":
		return "The shoreline evidence belongs to a different physical check."
	var required := {"runtime_version":4,"query_arithmetic_version":2,"collision_envelope_version":2,"planned_cases":36,"attempted_cases":36,"completed_cases":36,"failed_cases":0,"skipped_cases":0,"attempted_controls":68000,"accepted_controls":68000,"attempted_substeps":688000,"retention_old_accepted_controls":30400,"retention_new_accepted_controls":30400,"bank_accepted_controls":1600,"half_step_accepted_controls":800,"outward_accepted_controls":4800,"optimizer_steps":0}
	for key in required:
		var value: Variant=proof.get(key,null)
		if not (value is int or value is float) or not is_finite(float(value)) or float(value)!=float(required[key]):
			return "The shoreline evidence is incomplete or has different case counts."
	if typeof(proof.get("autonomous_navigation_qualified",null))!=TYPE_BOOL or proof.get("autonomous_navigation_qualified",true)!=false:
		return "The shoreline evidence mixes physical verification with an unsupported navigation claim."
	return ""

func _shoreline_update_issue(update: Dictionary) -> String:
	if int(update.get("schema",0))==1 and int(update.get("runtime_version",0))==4:
		if update.get("status","")=="failed":
			return "The corrected shoreline check did not pass (%d of 36 cases completed). The previous runtime remains installed." % int(update.get("completed_cases",0))
		if update.get("status","")=="passed":
			return "The corrected shoreline check passed. This running app still uses the previous runtime; reopen the updated app."
	return "The corrected shoreline qualification is pending. This app is using the previous runtime."

func _cuda_verification_valid(binding: Dictionary, proof: Dictionary, gpu_hash: String) -> bool:
	var source: Variant=binding.get("gpu_source",{})
	if not source is Dictionary or source.get("path","")!=SHORELINE_GPU_PATH or gpu_hash.length()!=64 or source.get("sha256","")!=gpu_hash:
		return false
	if source.get("cpu_sha256","")!=str(binding.get("cpu_source",{}).get("sha256","")):
		return false
	if proof.get("passed",false)!=true or proof.get("preflight_only",true)!=false or proof.get("CUDA_substep_readback_complete",false)!=true or proof.get("protocol_id","")!="shoreline-product-v4-cuda-2026-11-23":
		return false
	var required := {"runtime_version":4,"query_arithmetic_version":2,"controls_completed_per_case":128,"CPU_attempted_controls":1024,"CUDA_attempted_controls":1024,"CPU_attempted_substeps":10240,"CUDA_attempted_substeps":10240,"optimizer_steps":0,"PPO_updates":0,"GPU_NN_forward_calls":0}
	for key in required:
		var value: Variant=proof.get(key,null)
		if not (value is int or value is float) or not is_finite(float(value)) or float(value)!=float(required[key]): return false
	return proof.get("autonomous_navigation_qualified",true)==false

func _recorded_runtime_verdicts() -> String:
	var binding := _dictionary_file(SHORELINE_BINDING_PATH)
	var cpu_hash := FileAccess.get_sha256(SHORELINE_CPU_PATH) if FileAccess.file_exists(SHORELINE_CPU_PATH) else ""
	if not _cpu_proof_validation_issue(binding,_dictionary_file(SHORELINE_CPU_PATH),cpu_hash).is_empty(): return "Recorded shoreline evidence is unavailable.\n"
	var text := "Recorded CPU physical checks: 36 / 36 passed.\n"
	var gpu_hash := FileAccess.get_sha256(SHORELINE_GPU_PATH) if FileAccess.file_exists(SHORELINE_GPU_PATH) else ""
	text += "Recorded CUDA agreement: 8 / 8 matched cases.\n" if _cuda_verification_valid(binding,_dictionary_file(SHORELINE_GPU_PATH),gpu_hash) else "Recorded CUDA agreement is unavailable.\n"
	return text

func _candidate_study_status(status_binding: Dictionary = {}) -> String:
	var unavailable := "[color=#%s]Candidate assessment is unavailable. The original motor remains selected.[/color]" % VERIFICATION_ERROR.to_html(false)
	# Historical results are independent of the fingerprint of a local rebuild.
	var binding := status_binding if not status_binding.is_empty() else _dictionary_file(SHORELINE_BINDING_PATH)
	var source: Variant=binding.get("candidate_source",{})
	if not source is Dictionary or source.get("path","")!=CANDIDATE_STUDY_PATH or not FileAccess.file_exists(CANDIDATE_STUDY_PATH): return unavailable
	if source.get("sha256","")!=FileAccess.get_sha256(CANDIDATE_STUDY_PATH): return unavailable
	var receipt := _dictionary_file(CANDIDATE_STUDY_PATH)
	var updates := int(receipt.get("completed_updates",-1))
	if int(receipt.get("schema",0))!=2 or receipt.get("status","") not in ["stopped","completed"] or updates<0 or updates>32 or receipt.get("candidate_adopted",true)!=false: return unavailable
	var label := str(receipt.get("study_label",""))
	if label.length()>48 or label.contains("[") or label.contains("]") or label.contains("\n") or label.contains("\r"): return unavailable
	var study_prefix := label+"\n" if not label.is_empty() else ""
	if receipt.get("playing_package_sha256","")!=FileAccess.get_sha256(policy_path): return unavailable
	var references: Variant=receipt.get("sources",{})
	if not references is Dictionary: return unavailable
	for name in ["checkpoint","training","evaluation","movement","cases"]:
		if not references.has(name): return unavailable
	for name in references:
		var reference: Variant=references.get(name,{})
		if not reference is Dictionary: return unavailable
		var path := str(reference.get("path",""))
		var digest := str(reference.get("sha256",""))
		if not path.begins_with(STUDY_EVIDENCE_PREFIX) or path.contains("..") or digest.length()!=64 or not FileAccess.file_exists(path) or FileAccess.get_sha256(path)!=digest: return unavailable
	var checkpoint := _dictionary_file(str(references.checkpoint.path))
	var assessment := _dictionary_file(str(references.evaluation.path))
	var training := _dictionary_file(str(references.training.path))
	if checkpoint.get("raw_checkpoint_in_app",true)!=false or str(checkpoint.get("file_sha256","")).length()!=64 or int(checkpoint.get("completed_updates",-1))!=updates: return unavailable
	if int(assessment.get("update",-1))!=updates or int(training.get("completed_updates",-1))!=updates or assessment.get("preflight_only",true)!=false: return unavailable
	var prefix := "Recorded candidate stopped after %d PPO updates" % updates
	match str(receipt.get("reason","")):
		"long_swim_lane_drift":
			var movement := _dictionary_file(str(references.movement.path))
			var drifted := false
			for item in movement.get("cases",[]):
				if item is Dictionary and item.get("suite","")=="long" and float(item.get("maximum_lane_m",0))>.002: drifted=true
			if not drifted or not str(assessment.get("reason","")).begins_with("retention_failed_long") or assessment.get("retention_passed",true)!=false: return unavailable
			return study_prefix+prefix+": long-swim drift exceeded the 2 mm lane."
		"rest_retention_failed":
			if assessment.get("retention_passed",true)!=false or not str(assessment.get("reason","")).contains("hold"): return unavailable
			return study_prefix+prefix+": resting checks regressed."
		"long_swim_retention_failed":
			if assessment.get("retention_passed",true)!=false or not str(assessment.get("reason","")).begins_with("retention_failed_long"): return unavailable
			return study_prefix+prefix+": long-swim retention failed."
		"physical_check_failed":
			if assessment.get("hard_failure",false)!=true: return unavailable
			return study_prefix+prefix+": a physical check failed."
		"retention_failed":
			if assessment.get("retention_passed",true)!=false: return unavailable
			return study_prefix+prefix+": retained movement checks failed."
		"turning_unqualified":
			if assessment.get("retention_passed",false)!=true or int(assessment.get("retention_cases_passed",0))!=20: return unavailable
			return study_prefix+"Recorded candidate completed %d PPO updates; swimming and resting retained. Turns: %d / %d passed." % [updates,int(assessment.get("turn_cases_passed",0)),int(assessment.get("turn_cases_total",16))]
		"requested_turns_qualified":
			if assessment.get("retention_passed",false)!=true or assessment.get("turning_qualified",false)!=true: return unavailable
			return study_prefix+"Recorded candidate passed swimming, resting and requested-turn checks after %d PPO updates. Original 336 remains selected." % updates
	return unavailable

func _read_shoreline_status() -> void:
	shoreline_binding.clear()
	shoreline_proof.clear()
	shoreline_qualified=false
	cuda_qualified=false
	shoreline_issue="The shoreline verification files are unavailable. No qualification claim is made."
	if preview==null: return
	var descriptor: Dictionary=preview.call("get_pond_geometry_descriptor")
	if int(snapshot.get("runtime_version",0))!=4 or int(descriptor.get("runtime_version",0))!=4 or int(descriptor.get("query_arithmetic_version",0))!=2:
		shoreline_issue=_shoreline_update_issue(_dictionary_file(SHORELINE_UPDATE_PATH))
		return
	var binding := _dictionary_file(SHORELINE_BINDING_PATH)
	if binding.is_empty(): return
	var declared_path := str(binding.get("dll_path",NATIVE_DLL_PATH))
	if declared_path in ["res://addons/frozen_swimmer/bin/frozen_swimmer.dll","res://addons/frozen_swimmer/bin/frozen_swimmer-steering-v1.dll"]:
		NATIVE_DLL_PATH=declared_path
	var proof := _dictionary_file(SHORELINE_CPU_PATH)
	var dll_hash := FileAccess.get_sha256(NATIVE_DLL_PATH) if FileAccess.file_exists(NATIVE_DLL_PATH) else ""
	var cpu_hash := FileAccess.get_sha256(SHORELINE_CPU_PATH) if FileAccess.file_exists(SHORELINE_CPU_PATH) else ""
	shoreline_issue=_shoreline_validation_issue(binding,proof,dll_hash,cpu_hash,descriptor)
	if not shoreline_issue.is_empty(): return
	shoreline_binding=binding
	shoreline_proof=proof
	shoreline_qualified=true
	var gpu_hash := FileAccess.get_sha256(SHORELINE_GPU_PATH) if FileAccess.file_exists(SHORELINE_GPU_PATH) else ""
	cuda_qualified=_cuda_verification_valid(binding,_dictionary_file(SHORELINE_GPU_PATH),gpu_hash)

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
	if not preview.has_method("reset_untimed_swim") or not preview.has_method("get_pond_geometry_descriptor") or not preview.has_method("sample_pond_outline") or not preview.has_method("sample_pond_surface"):
		error = "The app components do not match. Reopen the updated Tadpole Pond app."
		return
	if not FileAccess.file_exists(policy_path):
		error = "The saved policy is missing. No simulation has started."
		return
	loaded = bool(preview.call("load_controller",ProjectSettings.globalize_path(policy_path)))
	if loaded:
		snapshot = preview.call("get_state")
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
	task.tooltip_text = "Selecting Rest or Swim restarts the creature. Swim continues until paused or a physical check stops it."
	task.item_selected.connect(func(_index: int): _restart())
	controls.add_child(task)
	target_caption = label("Target",12,MINT)
	target_caption.add_theme_stylebox_override("normal",panel(SURFACE,5))
	canvas.add_child(target_caption)
	notice = label("",15,TEXT)
	notice.add_theme_stylebox_override("normal",panel(SURFACE,12))
	notice.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	notice.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	notice.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
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
	drawer_scroll.follow_focus = true
	column.add_child(drawer_scroll)
	var content := VBoxContainer.new()
	content.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	content.add_theme_constant_override("separation",16)
	drawer_scroll.add_child(content)
	steering = VBoxContainer.new()
	steering.add_theme_constant_override("separation",8)
	content.add_child(steering)
	steering.add_child(label("Test the saved motor",16))
	var turns := HBoxContainer.new()
	turns.add_theme_constant_override("separation",8)
	steering.add_child(turns)
	turn_left = button("Turn left 5°",func(): _request_turn(1))
	turn_right = button("Turn right 5°",func(): _request_turn(-1))
	for control in [turn_left,turn_right]:
		control.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		var focus := control.get_theme_stylebox("focus") as StyleBoxFlat
		focus.expand_margin_left = 0
		focus.expand_margin_right = 0
		turns.add_child(control)
	steering_status = label("",14,MUTED)
	steering_status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	steering.add_child(steering_status)
	drawer_text = RichTextLabel.new()
	drawer_text.bbcode_enabled = true
	drawer_text.fit_content = true
	drawer_text.selection_enabled = true
	drawer_text.focus_mode = Control.FOCUS_ALL
	drawer_text.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	drawer_text.add_theme_font_size_override("normal_font_size",15)
	drawer_text.add_theme_color_override("default_color",TEXT)
	content.add_child(drawer_text)
	measurement_toggle = CheckBox.new()
	measurement_toggle.text = "Show measurements"
	for state in ["font_color","font_hover_color","font_pressed_color","font_hover_pressed_color","font_focus_color"]:
		measurement_toggle.add_theme_color_override(state,TEXT)
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
	notice.size = Vector2(size.x-inset*2,48)
	var notice_height := maxf(48,notice.get_combined_minimum_size().y)
	notice.size.y = notice_height
	notice.position = Vector2(inset,rail.position.y-notice_height-16)
	var world_bottom := notice.position.y-16 if notice.visible else rail.position.y-16
	world.viewport_area = Rect2(inset,world_top,size.x-inset*2,maxf(80,world_bottom-world_top))
	world.fit_view()
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
	var reset_ok := false
	if choice==1: reset_ok=bool(preview.call("reset_untimed_swim",0))
	else: reset_ok=bool(preview.call("reset",0,true))
	if not reset_ok: error = str(preview.call("get_error"))
	snapshot = preview.call("get_state")
	last_action = "Rest restarted" if choice==0 else "Swim restarted"
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

func _request_turn(direction: int) -> void:
	if not loaded or not preview.has_method("request_swim_turn") or bool(snapshot.get("holding",true)): return
	if not bool(preview.call("request_swim_turn",direction)): return
	snapshot=preview.call("get_state")
	_refresh()

func _refresh() -> void:
	if play_button == null: return
	play_button.disabled = not loaded or bool(snapshot.get("finished",false)) or not str(snapshot.get("error","")).is_empty()
	restart_button.disabled = not loaded
	task.disabled = not loaded
	steering.visible = loaded and preview.has_method("request_swim_turn") and not bool(snapshot.get("holding",true))
	if steering.visible:
		var stopped := not str(snapshot.get("error","")).is_empty() or bool(snapshot.get("finished",false))
		turn_left.disabled = stopped
		turn_right.disabled = stopped
		if stopped:
			steering_status.text = "Restart Swim to test a heading."
		elif bool(snapshot.get("heading_only_request",false)):
			steering_status.text = "Requested turn %+.0f°. Close Details and press Play." % float(snapshot.get("requested_turn_degrees",0))
		else:
			steering_status.text = "Choose a turn, then close Details and press Play."
	play_button.text = "Pause" if running else "Play"
	if loaded:
		var state := ("Resting" if bool(snapshot.get("holding",true)) else "Swimming") if running else ("Session complete" if bool(snapshot.get("finished",false)) else "Paused")
		if not str(snapshot.get("error","")).is_empty(): state="Stopped"
		status.text = (last_action+" · saved policy") if state!="Stopped" and not running and not last_action.is_empty() else state+" · live, saved policy"
		if state!="Stopped" and bool(snapshot.get("heading_only_request",false)):
			status.text=("Live" if running else "Paused")+" · requested turn %+.0f°" % float(snapshot.get("requested_turn_degrees",0))
		world.show_state(snapshot)
		speed.text = "%.2f mm/s" % (float(snapshot.get("speed_m_s",0))*1000)
		var holding := bool(snapshot.get("holding",true))
		var amount := float(snapshot.get("anchor_error_m",0) if holding else snapshot.get("forward_m",0))*1000
		displacement.text = ("%.1f mm away" % amount) if holding else ("%.1f mm swum" % amount)
		if bool(snapshot.get("untimed_swim",false)):
			var elapsed := int(float(snapshot.get("time_s",0)))
			displacement.text = "%d:%02d · %.1f mm" % [floori(elapsed/60.0),elapsed%60,amount]
			displacement.tooltip_text = "Elapsed swimming time · forward distance travelled"
		else:
			displacement.tooltip_text = "Distance from the rest anchor" if holding else "Forward distance travelled"
		depth.text = "%.2f mm off depth" % (float(snapshot.get("depth_m",0))*1000)
		error = str(snapshot.get("error",error))
	else: status.text = "Preview unavailable"
	var previous_notice_visible := notice.visible
	var previous_notice_text := notice.text
	notice.text = error
	notice.visible = not error.is_empty()
	if notice.visible!=previous_notice_visible or (notice.visible and notice.text!=previous_notice_text): _layout()
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
	running=false
	accumulator=0
	_refresh()
	_read_shoreline_status()
	var descriptor: Dictionary=preview.call("get_pond_geometry_descriptor") if preview!=null else {}
	var terrain_contact := bool(descriptor.get("used_by_contact",false))
	_append_shoreline_details(terrain_contact)
	drawer_text.append_text("\n[b]The saved creature[/b]\nPlaying original motor 336.\n%s\nThis view does not train its saved weights.\n" % _candidate_study_status())
	drawer_text.append_text("\n[b]Controls[/b]\nSwim has no session timer. Rest is a 64-second check. Opening Details pauses the world. Restart or selecting Rest/Swim creates a new starting state.\nScroll, pinch or use + / − to zoom. Side view changes the camera angle.\n")
	var result: Dictionary=metadata.get("result",{})
	drawer_text.append_text("\n[b]Saved motor reference assessment[/b]\n")
	if result.is_empty(): drawer_text.append_text(metadata_issue+"\n")
	else:
		drawer_text.append_text("%s\nHolding: %d / %d · Short swimming: %d / 4\nThese are the preserved reference tests, not qualification of new shoreline behavior.\n" % [str(result.get("outcome","See source evidence")),int(result.get("hold_passes",0)),int(result.get("hold_total",6)),int(result.get("short_passes",0))])
	drawer_text.append_text("\n[b]Saved generation[/b]\nMotor 336 · %d holding updates\n" % int(snapshot.get("holding_updates",0)))
	drawer_scroll.scroll_vertical=0
	scrim.visible=true
	drawer.visible=true
	for control in [play_button,restart_button,task,details_button,view_button]: control.focus_mode=Control.FOCUS_NONE
	close_button.grab_focus()

func _append_shoreline_details(terrain_contact: bool) -> void:
	drawer_text.clear()
	drawer_text.append_text("[b]Pond world[/b]\n")
	if shoreline_qualified:
		drawer_text.append_text("Shared shoreline verified. The smaller legacy limit has been removed.\nThe visible bed and native contact use the same terrain. Original swimming and holding checks passed with the saved motor.\n")
		drawer_text.append_text("CPU physical checks: 36 / 36 passed.\n")
		drawer_text.append_text("CUDA agreement: 8 / 8 matched cases.\n" if cuda_qualified else "[color=#%s]CUDA verification is unavailable.[/color]\n" % VERIFICATION_ERROR.to_html(false))
	else:
		drawer_text.append_text("[color=#%s][b]Shoreline verification unavailable[/b]\n%s[/color]\n" % [VERIFICATION_ERROR.to_html(false),shoreline_issue])
		drawer_text.append_text(_recorded_runtime_verdicts())
		if not terrain_contact:
			drawer_text.append_text("The shoreline upgrade is in progress. The visible bed uses native geometry; the installed app is awaiting the matching shoreline runtime.\n")
	drawer_text.append_text(_current_physics_text(snapshot))
	drawer_text.append_text("\nOriginal saved motor. Learned pond navigation is still unqualified. Physical contact is not a learned turn. Water exit is outside this submerged model.\n")

func _current_physics_text(state: Dictionary) -> String:
	var text := "\n[b]Right now[/b]\n"
	var lower := float(state.get("whole_body_clearance_lower_m",NAN))
	var upper := float(state.get("whole_body_clearance_upper_m",NAN))
	if int(state.get("runtime_version",0))==4 and int(state.get("query_arithmetic_version",0))==2 and bool(state.get("whole_body_query_certified",false)) and is_finite(lower) and is_finite(upper) and lower<=upper:
		var contact := "Clear of the bed now" if lower>=0 else "Bed contact now" if upper<0 else "Possible bed contact now"
		text += "%s\nCertified whole-body clearance: %.3f to %.3f mm\n" % [contact,lower*1000,upper*1000]
	else:
		text += "[color=#%s]Current bed clearance is unavailable.[/color]\n" % VERIFICATION_ERROR.to_html(false)
	var horizontal := float(state.get("horizontal_speed_m_s",NAN))
	var velocity: Variant=state.get("terminal_velocity",null)
	if is_finite(horizontal) and velocity is Vector3 and velocity.is_finite():
		var requested := Vector2(velocity.x,velocity.z).length()*1000
		text += "Horizontal COM speed: %.2f mm/s · Requested: %.2f mm/s\n" % [horizontal*1000,requested]
	var tilt := float(state.get("tilt_deg",NAN))
	var depth_error := float(state.get("depth_m",NAN))
	if is_finite(tilt) and is_finite(depth_error):
		text += "Tilt: %.1f° · Depth error: %.2f mm\n" % [tilt,depth_error*1000]
	return text

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

extends SceneTree
## Static UI checks. Disable the view's process loop before exercising Play,
## so start/pause can be tested while native controls/substeps remain exactly zero.
var view: Node
var failures: Array = []
var checks: Array = []
var output := ""
var requested_size := Vector2i(1280,720)
var missing_policy := false
var mismatch := false

func _initialize() -> void:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--test-output="): output=argument.trim_prefix("--test-output=")
		if argument.begins_with("--test-size="):
			var parts := argument.trim_prefix("--test-size=").split("x")
			requested_size=Vector2i(int(parts[0]),int(parts[1]))
		if argument=="--test-missing-policy": missing_policy=true
		if argument=="--test-metadata-mismatch": mismatch=true
	root.size=requested_size
	view=load("res://main.tscn").instantiate()
	root.add_child.call_deferred(view)
	_run.call_deferred()

func check(message: String, good: bool) -> void:
	checks.append({"check":message,"passed":good})
	if not good: failures.append(message)

func settle() -> void:
	await process_frame
	await process_frame

func click(control: Control, touch := false) -> void:
	var at:=control.get_global_rect().get_center()
	for pressed in [true,false]:
		if touch:
			var event:=InputEventScreenTouch.new()
			event.index=0
			event.position=at
			event.pressed=pressed
			Input.parse_input_event(event)
		else:
			var event:=InputEventMouseButton.new()
			event.button_index=MOUSE_BUTTON_LEFT
			event.position=at
			event.global_position=at
			event.pressed=pressed
			Input.parse_input_event(event)
		await process_frame

func key(code: Key) -> void:
	for pressed in [true,false]:
		var event:=InputEventKey.new()
		event.keycode=code
		event.physical_keycode=code
		event.pressed=pressed
		Input.parse_input_event(event)
		await process_frame

func contrast(a: Color, b: Color) -> float:
	var first:=a.srgb_to_linear()
	var second:=b.srgb_to_linear()
	var la:=.2126*first.r+.7152*first.g+.0722*first.b
	var lb:=.2126*second.r+.7152*second.g+.0722*second.b
	return (maxf(la,lb)+.05)/(minf(la,lb)+.05)

func _run() -> void:
	await settle()
	view.set_process(false)
	check("Native stepping is disabled for the UI test",not view.is_processing())
	check("View starts paused",not view.running)
	check("No controls or substeps at startup",int(view.snapshot.get("controls",0))==0 and view.physical_substeps==0)
	for control in [view.details_button,view.play_button,view.restart_button,view.task]:
		var box: Rect2=control.get_global_rect()
		check("44px target: "+control.text,box.size.x>=44 and box.size.y>=44)
		check("Control fits viewport: "+control.text,box.position.x>=0 and box.end.x<=root.size.x+1 and box.position.y>=0 and box.end.y<=root.size.y+1)
	for state in ["normal","hover","pressed"]:
		var background: Color=view.play_button.get_theme_stylebox(state).bg_color
		var foreground: Color=view.play_button.get_theme_color("font_focus_color" if state=="normal" else "font_"+state+"_color")
		check("Primary text contrast: "+state,contrast(foreground,background)>=4.5)
	if missing_policy:
		check("Missing policy is clearly reported",not view.loaded and view.notice.visible and view.notice.text.contains("missing"))
		check("Unavailable controls disabled",view.play_button.disabled and view.restart_button.disabled and view.task.disabled)
	else:
		check("Actual saved policy loaded",view.loaded)
		check("Live/frozen state is explicit",view.status.text.contains("live") and view.status.text.contains("saved policy"))
		check("Creature remains visible",view.world.viewport_area.has_point(view.world.camera.unproject_position(view.world.links[0].position)))
		await click(view.play_button)
		check("Mouse Play changes state without stepping",view.running and view.physical_substeps==0)
		view.play_button.grab_focus()
		await key(KEY_SPACE)
		check("Keyboard pauses",not view.running)
		await key(KEY_TAB)
		check("Keyboard focus advances",root.gui_get_focus_owner()!=null)
		ProjectSettings.set_setting("input_devices/pointing/emulate_mouse_from_touch",true)
		await click(view.restart_button,true)
		check("Touch restart stays paused",not view.running and int(view.snapshot.get("controls",-1))==0)
		view.task.select(view.task.get_item_index(2 if view.continuous_available else 1))
		view._restart()
		check("Swim request resets without stepping",not bool(view.snapshot.get("holding",true)) and view.physical_substeps==0)
		if view.continuous_available:
			check("Free swim is explicit and has no completed timer",view.task.get_selected_id()==2 and bool(view.snapshot.get("continuous",false)) and not bool(view.snapshot.get("finished",true)))
		view.task.select(view.task.get_item_index(0))
		view._restart()
	await click(view.details_button)
	await settle()
	check("Details opens and preserves paused state",view.drawer.visible and view.scrim.visible and not view.running)
	check("Focus cannot enter controls behind Details",view.play_button.focus_mode==Control.FOCUS_NONE and view.task.focus_mode==Control.FOCUS_NONE)
	var disclosed: String=view.drawer_text.get_parsed_text()
	check("Details contains actual source and method",disclosed.contains("Latest assessment") and disclosed.contains("Evidence source"))
	if mismatch: check("Mismatched package cannot claim assessment success",view.metadata.is_empty() and disclosed.contains("does not match") and not disclosed.contains("Calm hold verified"))
	var drawer_box: Rect2=view.drawer.get_global_rect()
	check("Details fits viewport",drawer_box.position.x>=0 and drawer_box.end.x<=root.size.x+1 and drawer_box.position.y>=0 and drawer_box.end.y<=root.size.y+1)
	if not output.is_empty():
		await RenderingServer.frame_post_draw
		root.get_texture().get_image().save_png(output.get_basename()+"-details.png")
	view.measurement_toggle.button_pressed=false
	await settle()
	check("Measurements can be hidden",not view.measurements.visible and not view.target_caption.visible)
	check("Compact rail stays inside viewport",view.rail.get_global_rect().end.y<=root.size.y+1)
	view.measurement_toggle.button_pressed=true
	await key(KEY_ESCAPE)
	check("Escape closes Details",not view.drawer.visible and not view.scrim.visible)
	check("Focus returns to Details",root.gui_get_focus_owner()==view.details_button)
	check("Entire UI test takes zero native controls",view.physical_substeps==0 and int(view.snapshot.get("substeps",0))==0)
	var result: Dictionary={"passed":failures.is_empty(),"checks":checks,"failures":failures,"size":[root.size.x,root.size.y],"physical_substeps":view.physical_substeps}
	if not output.is_empty():
		var file:=FileAccess.open(output,FileAccess.WRITE)
		if file: file.store_string(JSON.stringify(result,"\t"))
	print(JSON.stringify(result))
	quit(0 if failures.is_empty() else 3)

extends SceneTree
## Presentation checks only. The process loop is disabled before changing modes;
## no step() call or native physics advance is allowed in this test.
var view: Node
var output := ""
var requested_size := Vector2i(1280,800)
var failures: Array[String]=[]
var checks: Array[Dictionary]=[]

func check(label: String, passed: bool) -> void:
	checks.append({"check":label,"passed":passed})
	if not passed: failures.append(label)

func _initialize() -> void:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--test-output="): output=argument.trim_prefix("--test-output=")
		if argument.begins_with("--test-size="):
			var parts:=argument.trim_prefix("--test-size=").split("x")
			requested_size=Vector2i(int(parts[0]),int(parts[1]))
	root.size=requested_size
	view=load("res://main.tscn").instantiate()
	root.add_child.call_deferred(view)
	_run.call_deferred()

func settle() -> void:
	await process_frame
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
	# The engine applies project window defaults after _initialize(). Resize
	# the started window, then let layout settle before asserting its size.
	root.size=requested_size
	await settle()
	view._layout()
	await settle()
	check("View processing stays disabled",not view.is_processing())
	check("Requested viewport rendered",root.size==requested_size)
	check("Saved calm policy is loaded",view.loaded and view.policy_path.ends_with("learned-amplitude-v1.bin"))
	check("View begins paused without physics",not view.running and int(view.snapshot.get("controls",-1))==0 and view.physical_substeps==0)
	check("Unqualified autonomous mode remains unavailable",not view.continuous_available and view.task.get_item_index(2)==-1)
	view.task.select(view.task.get_item_index(1))
	view._restart()
	var actual: Dictionary=view.preview.call("get_state")
	check("Swim is explicitly untimed",bool(actual.get("untimed_swim",false)) and not bool(actual.get("holding",true)))
	check("Untimed Swim has no completion deadline",not bool(actual.get("finished",true)) and int(actual.get("maximum_controls",-1))==0 and float(actual.get("duration_s",-1))==0)
	check("Untimed Swim remains distinct from qualified Free swim",not bool(actual.get("continuous",true)))
	check("Swim change stays paused without stepping",not view.running and int(actual.get("controls",-1))==0 and int(actual.get("substeps",-1))==0)
	check("Selector explains the bank stop",view.task.tooltip_text.contains("bank") and view.task.tooltip_text.contains("paused"))
	# Synthetic presentation values exercise long elapsed/distance formatting;
	# the native world itself stays at its initial zero-control state.
	view.snapshot=actual.duplicate(true)
	view.snapshot["time_s"]=65.4
	view.snapshot["forward_m"]=1.2345
	view.last_action=""
	view.show_measurements=true
	view.measurement_toggle.set_pressed_no_signal(true)
	view._layout()
	view._refresh()
	await settle()
	check("Elapsed time and distance share one reading",view.displacement.text=="1:05 · 1234.5 mm")
	check("Distance reading explains both values",view.displacement.tooltip_text.contains("time") and view.displacement.tooltip_text.contains("distance"))
	check("Untimed paused state is not Session complete",view.status.text.begins_with("Paused") and not view.play_button.disabled)
	var rail_box: Rect2=view.rail.get_global_rect()
	check("Measurement rail fits the viewport",rail_box.position.x>=0 and rail_box.position.y>=0 and rail_box.end.x<=root.size.x+1 and rail_box.end.y<=root.size.y+1)
	for control in [view.play_button,view.restart_button,view.task]:
		var box: Rect2=control.get_global_rect()
		check("44px usable control: "+control.text,box.size.x>=44 and box.size.y>=44)
		check("Control fits measurement rail: "+control.text,rail_box.encloses(box))
	for reading in [view.speed,view.displacement,view.depth]:
		check("Measurement fits rail: "+reading.text,rail_box.encloses(reading.get_global_rect()))
	if not output.is_empty():
		await RenderingServer.frame_post_draw
		check("Swim measurements screenshot saved",root.get_texture().get_image().save_png(output.get_basename()+"-swim.png")==OK)
	view.snapshot=actual
	view._refresh()
	view._open_details()
	await settle()
	var text: String=view.drawer_text.get_parsed_text()
	check("Details explains untimed Swim and preserved Rest limit",text.contains("Swim has no session timer") and text.contains("Rest remains a 64-second check"))
	check("Details retains the turning limitation",text.contains("Reliable turns around the pond are still unverified"))
	check("Details retains no-learning disclosure",text.contains("does not train or update"))
	check("Details retains the actual closed comparison",text.contains("Conclusion: comparison blocked") and text.contains("0 / 88"))
	check("Details is modal with visible keyboard focus",view.drawer.visible and root.gui_get_focus_owner()==view.close_button and view.play_button.focus_mode==Control.FOCUS_NONE)
	for state in ["font_color","font_hover_color","font_pressed_color","font_hover_pressed_color","font_focus_color"]:
		check("Measurements checkbox text contrast: "+state,contrast(view.measurement_toggle.get_theme_color(state),view.SURFACE)>=4.5)
	var drawer_box: Rect2=view.drawer.get_global_rect()
	check("Details fits viewport",drawer_box.position.x>=0 and drawer_box.position.y>=0 and drawer_box.end.x<=root.size.x+1 and drawer_box.end.y<=root.size.y+1)
	if not output.is_empty():
		await RenderingServer.frame_post_draw
		check("Movement scope screenshot saved",root.get_texture().get_image().save_png(output.get_basename()+"-details.png")==OK)
	view._close_details()
	check("Close restores Details focus",not view.drawer.visible and root.gui_get_focus_owner()==view.details_button)
	view.task.select(view.task.get_item_index(0))
	view._restart()
	actual=view.preview.call("get_state")
	check("Rest retains its bounded check",bool(actual.get("holding",false)) and not bool(actual.get("untimed_swim",true)) and int(actual.get("maximum_controls",0))==1280 and float(actual.get("duration_s",0))==64)
	check("Rest measurement has no swimming clock",view.displacement.text.ends_with("mm away") and not view.displacement.text.contains(":"))
	check("All presentation checks used zero native controls",view.physical_substeps==0 and int(actual.get("controls",-1))==0 and int(actual.get("substeps",-1))==0)
	var report: Dictionary={"passed":failures.is_empty(),"checks":checks,"failures":failures,"size":[root.size.x,root.size.y],"synthetic_display_values":true,"physical_controls":0,"physical_substeps":0,"optimizer_steps":0}
	if not output.is_empty(): FileAccess.open(output,FileAccess.WRITE).store_string(JSON.stringify(report,"  "))
	print(JSON.stringify(report))
	quit(0 if failures.is_empty() else 3)

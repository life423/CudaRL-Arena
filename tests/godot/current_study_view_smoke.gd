extends SceneTree
## Render the actual closed-study disclosure. Never start native playback.
var failures: Array[String] = []
var checks: Array[Dictionary] = []
var output := ""
var requested_size := Vector2i(1280,800)
var view: Node

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

func check(message: String, good: bool) -> void:
	checks.append({"check":message,"passed":good})
	if not good: failures.append(message)

func settle() -> void:
	await process_frame
	await process_frame

func _run() -> void:
	await settle()
	view.set_process(false)
	check("Actual viewport matches requested size",root.size==requested_size)
	check("Saved calm policy remains loaded and paused",view.loaded and not view.running and view.policy_path.ends_with("learned-amplitude-v1.bin"))
	check("No physical controls at startup",view.physical_substeps==0 and int(view.snapshot.get("controls",-1))==0)
	check("Actual closed current-study evidence is available",view.current_study.has("result"))
	check("Disclosure is bound to independently checked evidence",bool(view.current_study.get("verified",false)))
	var study: Dictionary=view.current_study.get("result",{})
	check("Study completed its bounded PPO budget",str(study.get("phase",""))=="bounded_updates_exhausted" and int(study.get("PPO_updates",-1))==32 and study.get("current_suites_run",false)==true)
	check("Measured current cases improved without qualifying",int(study.get("baseline_current_passes",-1))==4 and int(study.get("final_current_passes",-1))==8 and study.get("qualified",true)==false)
	check("Unrun validation case results remain null",study.get("withheld_passes","missing")==null and study.get("endurance_passes","missing")==null and study.get("withheld_suites_run",true)==false and study.get("endurance_suites_run",true)==false)
	view._open_details()
	await settle()
	check("Details pauses without stepping",view.drawer.visible and not view.running and view.physical_substeps==0)
	check("Focus stays within Details",root.gui_get_focus_owner()==view.close_button and view.play_button.focus_mode==Control.FOCUS_NONE)
	var text: String=view.drawer_text.get_parsed_text()
	check("Current study outcome is explicit",text.contains("Current swimming study") and text.contains("Current swimming is not ready") and text.contains("Native PPO updates: 32") and text.contains("4 → 8 / 16"))
	check("Unrun behavior suites are not displayed as zero passes",not text.contains("Withheld cases:") and not text.contains("Endurance cases:"))
	check("Active preview remains identified as calm saved policy",text.contains("This preview still uses the saved calm-water policy"))
	var box: Rect2=view.drawer.get_global_rect()
	check("Details fits viewport",box.position.x>=0 and box.position.y>=0 and box.end.x<=root.size.x+1 and box.end.y<=root.size.y+1)
	check("Disclosure has no horizontal scroll",view.drawer_scroll.horizontal_scroll_mode==ScrollContainer.SCROLL_MODE_DISABLED)
	var title_index:=text.find("Current swimming study")
	if title_index>=0:
		var line: int=view.drawer_text.get_character_line(title_index)
		view.drawer_scroll.scroll_vertical=int(maxf(0,view.drawer_text.get_line_offset(line)-8))
	await settle()
	if not output.is_empty():
		await RenderingServer.frame_post_draw
		check("Actual disclosure screenshot is saved",root.get_texture().get_image().save_png(output.get_basename()+".png")==OK)
	view._close_details()
	check("Closing Details restores keyboard focus",not view.drawer.visible and root.gui_get_focus_owner()==view.details_button)
	check("Entire rendered check uses zero physical substeps",view.physical_substeps==0 and int(view.preview.call("get_state").get("substeps",-1))==0)
	var result: Dictionary={"passed":failures.is_empty(),"checks":checks,"failures":failures,"size":[root.size.x,root.size.y],
		"physical_controls":0,"physical_substeps":view.physical_substeps,"study_phase":str(study.get("phase","")),"PPO_updates":int(study.get("PPO_updates",-1))}
	if not output.is_empty(): FileAccess.open(output,FileAccess.WRITE).store_string(JSON.stringify(result,"  "))
	print(JSON.stringify(result))
	quit(0 if failures.is_empty() else 3)

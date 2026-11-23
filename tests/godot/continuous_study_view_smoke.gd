extends SceneTree
## Render the actual rejected study disclosure without starting physics.
var view: Node
var output := ""
var requested_size := Vector2i(1280,800)
var failures: Array[String] = []
var checks: Array[Dictionary] = []

func check(message: String, good: bool) -> void:
	checks.append({"check":message,"passed":good})
	if not good: failures.append(message)

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

func _run() -> void:
	await process_frame
	await process_frame
	view.set_process(false)
	check("Actual viewport matches requested size",root.size==requested_size)
	check("The qualified calm swimmer remains active",view.loaded and view.policy_path.ends_with("learned-amplitude-v1.bin"))
	check("Rejected free-swim actor is unavailable",not view.continuous_available and view.task.get_item_index(2)==-1)
	check("The app starts paused without physical work",not view.running and int(view.snapshot.get("controls",-1))==0)
	view._open_details()
	await process_frame
	await process_frame
	var text: String=view.drawer_text.get_parsed_text()
	check("Latest failed study is explicit",text.contains("Free swimming study") and text.contains("Continuous swimming is not ready") and text.contains("Starting cases: 0 / 16"))
	check("Actual learning and failure are distinguished",text.contains("Native PPO updates: 64") and text.contains("slowed and stalled"))
	check("Unrun validation is not presented as zero passes",text.contains("Thirty-minute validation was not run") and not text.contains("Thirty-minute cases: 0"))
	check("Retained previous skills are shown",text.contains("all 14 holding cases passed again"))
	check("Details stays modal and scrollable",view.drawer.visible and view.play_button.focus_mode==Control.FOCUS_NONE and root.gui_get_focus_owner()==view.close_button and view.drawer_scroll.horizontal_scroll_mode==ScrollContainer.SCROLL_MODE_DISABLED)
	var box: Rect2=view.drawer.get_global_rect()
	check("Drawer fits the viewport",box.position.x>=0 and box.position.y>=0 and box.end.x<=root.size.x+1 and box.end.y<=root.size.y+1)
	var section:=text.find("Free swimming study")
	if section>=0:
		var line: int=view.drawer_text.get_character_line(section)
		view.drawer_scroll.scroll_vertical=int(maxf(0,view.drawer_text.get_line_offset(line)-8))
	await process_frame
	if not output.is_empty():
		await RenderingServer.frame_post_draw
		check("Actual disclosure screenshot saved",root.get_texture().get_image().save_png(output.get_basename()+".png")==OK)
	view._close_details()
	check("Closing restores focus",not view.drawer.visible and root.gui_get_focus_owner()==view.details_button)
	check("Entire check used zero native controls",view.physical_substeps==0 and int(view.preview.call("get_state").get("substeps",-1))==0)
	var report := {"passed":failures.is_empty(),"checks":checks,"failures":failures,"size":[root.size.x,root.size.y],"physical_controls":0,"physical_substeps":0}
	if not output.is_empty(): FileAccess.open(output,FileAccess.WRITE).store_string(JSON.stringify(report,"  "))
	print(JSON.stringify(report))
	quit(0 if failures.is_empty() else 3)

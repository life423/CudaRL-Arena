extends SceneTree
## Render actual comparison disclosure without advancing the native world.
const Study = preload("res://interface_comparison_launch_stop_v2.gd")
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

func _run() -> void:
	await process_frame
	await process_frame
	view.set_process(false)
	check("Requested viewport is rendered",root.size==requested_size)
	check("Qualified calm saved swimmer stays active",view.loaded and view.policy_path.ends_with("learned-amplitude-v1.bin"))
	check("Comparison adds no autonomous mode",not view.continuous_available and view.task.get_item_index(2)==-1)
	check("The view starts paused with zero controls",not view.running and int(view.snapshot.get("controls",-1))==0)
	view._open_details()
	await process_frame
	await process_frame
	var checked: Dictionary=Study.read("res://research_data/interface-comparison-v2-launch-stop.json")
	check("Actual comparison is byte-bound and independently verified",checked.get("verified",false)==true)
	var text: String=view.drawer_text.get_parsed_text()
	check("Comparison is visible before older motor assessment",text.contains("Request interface comparison") and text.find("Request interface comparison")<text.find("Saved motor assessment"))
	check("No new learning or continuous ability is claimed",text.contains("No learning updates") and text.contains("Continuous pond swimming is still unqualified"))
	check("A launch failure is not presented as an interface result",checked.get("launch_stop",false) and text.contains("Conclusion: comparison blocked"))
	check("Zero run and unassessed retention stay explicit",text.contains("Physical controls: 0") and text.contains("0 / 88") and text.contains("were not run") and text.contains("No interface preference"))
	check("Earlier unrun endurance remains explicit",text.contains("Thirty-minute validation was not run"))
	check("Details is modal and has keyboard focus",view.drawer.visible and view.play_button.focus_mode==Control.FOCUS_NONE and root.gui_get_focus_owner()==view.close_button)
	check("Details prevents horizontal scrolling",view.drawer_scroll.horizontal_scroll_mode==ScrollContainer.SCROLL_MODE_DISABLED)
	var box: Rect2=view.drawer.get_global_rect()
	check("Details fits the viewport",box.position.x>=0 and box.position.y>=0 and box.end.x<=root.size.x+1 and box.end.y<=root.size.y+1)
	await process_frame
	await process_frame
	if not output.is_empty():
		await RenderingServer.frame_post_draw
		check("Actual comparison screenshot saved",root.get_texture().get_image().save_png(output.get_basename()+".png")==OK)
	view._close_details()
	check("Closing restores Details button focus",not view.drawer.visible and root.gui_get_focus_owner()==view.details_button)
	check("Whole view check used zero native substeps",view.physical_substeps==0 and int(view.preview.call("get_state").get("substeps",-1))==0)
	var report := {"passed":failures.is_empty(),"checks":checks,"failures":failures,"size":[root.size.x,root.size.y],"physical_controls":0,"physical_substeps":0,"optimizer_steps":0}
	if not output.is_empty(): FileAccess.open(output,FileAccess.WRITE).store_string(JSON.stringify(report,"  "))
	print(JSON.stringify(report))
	quit(0 if failures.is_empty() else 3)

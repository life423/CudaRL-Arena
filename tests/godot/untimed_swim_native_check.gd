extends SceneTree
## One bounded product check: original frozen motor, no learning or research run.
const CAP := 800
var output := ""
var failures: Array[String]=[]
var checks: Array[Dictionary]=[]
var trace: Array[Dictionary]=[]
var attempts := 0
var reference: Dictionary={}
var maximum_pose_error := 0.0
var pause_verified := false
var started := 0

func check(label: String, good: bool) -> void:
	checks.append({"check":label,"passed":good})
	if not good: failures.append(label)

func _initialize() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--test-output="): output=arg.trim_prefix("--test-output=")
	call_deferred("_run")

func _run() -> void:
	if output.is_empty() or FileAccess.file_exists(output) or FileAccess.file_exists(output+".started.json"):
		push_error("One-shot check needs a new explicit output path")
		quit(2)
		return
	FileAccess.open(output+".started.json",FileAccess.WRITE).store_string(JSON.stringify({"attempt_cap":CAP,"substep_cap":CAP*10,"training_updates":0,"optimizer_steps":0}))
	started=Time.get_ticks_msec()
	_load_reference()
	if not ClassDB.class_exists("HoldLabPreview"):
		GDExtensionManager.load_extension("res://addons/frozen_swimmer/frozen_swimmer.gdextension")
	check("Native class loads",ClassDB.class_exists("HoldLabPreview"))
	if not failures.is_empty():
		_finish({})
		return
	var preview: RefCounted=ClassDB.instantiate("HoldLabPreview")
	check("Saved motor loads",bool(preview.call("load_controller","res://research_data/learned-amplitude-v1.bin")))
	check("Legacy finite Swim API remains available",bool(preview.call("reset",0,false)))
	var state: Dictionary=preview.call("get_state")
	check("Legacy Swim keeps original eight-second horizon",int(state.maximum_controls)==160 and float(state.duration_s)==8 and not bool(state.untimed_swim))
	check("Rest resets without physics",bool(preview.call("reset",0,true)))
	state=preview.call("get_state")
	check("Rest keeps original 64-second horizon",int(state.maximum_controls)==1280 and float(state.duration_s)==64 and int(state.controls)==0 and int(state.substeps)==0)
	check("Explicit untimed Swim resets",bool(preview.call("reset_untimed_swim",0)))
	state=preview.call("get_state")
	check("Untimed metadata has no deadline",bool(state.untimed_swim) and not bool(state.continuous) and not bool(state.finished) and int(state.maximum_controls)==0 and float(state.duration_s)==0)
	trace.append(state.duplicate(true))
	var origin: Variant=state.request_origin
	var heading: Variant=state.request_heading
	var target: Variant=state.request_position
	var terminal: Variant=state.terminal_velocity
	check("Swim starts",bool(preview.call("start")))
	if not failures.is_empty():
		_finish(state)
		return
	while attempts<CAP:
		if Time.get_ticks_msec()-started>50000:
			check("Within 50-second collection reserve",false)
			break
		attempts+=1
		var advanced: bool=preview.call("step")
		state=preview.call("get_state")
		trace.append(state.duplicate(true))
		check("Attempt accounting "+str(attempts),int(state.attempted_controls)==attempts and int(state.substeps)<=attempts*10)
		check("Request persists "+str(attempts),state.request_origin==origin and state.request_heading==heading and state.request_position==target and state.terminal_velocity==terminal)
		if not advanced:
			break
		check("Continuous clocks "+str(attempts),int(state.controls)==attempts and int(state.world_controls)==attempts and int(state.task_controls)==attempts and int(state.substeps)==attempts*10 and absf(float(state.time_s)-attempts*.05)<.00001)
		check("No timed completion "+str(attempts),not bool(state.finished) and not bool(state.paused) and not bool(state.training))
		if attempts<=160:
			check("Original pose available "+str(attempts),reference.has(attempts))
			if reference.has(attempts):
				var wanted: Array=reference[attempts]
				var keys := ["head_x","head_y","head_z","qw","qx","qy","qz","j0","j1","j2","j3"]
				for k in keys.size(): maximum_pose_error=maxf(maximum_pose_error,absf(float(state[keys[k]])-float(wanted[k])))
		if attempts==170:
			preview.call("pause")
			var paused: Dictionary=preview.call("get_state")
			var expected: Dictionary=state.duplicate(true)
			expected.paused=true
			check("Pause changes no movement state",paused==expected and preview.call("get_state")==paused)
			check("Resume succeeds",bool(preview.call("start")))
			pause_verified=preview.call("get_state")==state
			check("Resume changes no movement state",pause_verified)
	preview.call("pause")
	state=preview.call("get_state")
	check("Passed former eight-second boundary",int(state.controls)>170 and trace.size()>172)
	check("Original first160 poses retain fidelity",maximum_pose_error<=.000002)
	check("Pause/resume exercised",pause_verified)
	check("Physical counts stay bounded",attempts<=CAP and int(state.substeps)<=CAP*10)
	check("Existing pond boundary stops safely",str(state.error)=="The creature reached the edge of the supported pond area" and bool(state.pond_domain_valid) and bool(state.failure_recorded) and bool(state.paused) and not bool(state.finished))
	_finish(state)

func _load_reference() -> void:
	var file:=FileAccess.open(ProjectSettings.globalize_path("res://../reference-poses.csv"),FileAccess.READ)
	check("Saved reference opens",file!=null)
	if file==null: return
	var header:=file.get_csv_line()
	while not file.eof_reached():
		var row:=file.get_csv_line()
		if row.size()!=header.size() or row[0]!="short" or row[1]!="0": continue
		var values: Array=[]
		for key in ["position_x_model","position_y_model","position_z_model","quat_w","quat_x","quat_y","quat_z","angle0","angle1","angle2","angle3"]:
			values.append(float(row[header.find(key)]))
		reference[int(row[2])]=values

func _finish(state: Dictionary) -> void:
	var result := {"passed":failures.is_empty(),"checks":checks,"failures":failures,"attempted_controls":attempts,"accepted_controls":int(state.get("controls",0)),"attempted_substeps":int(state.get("substeps",0)),"accepted_substeps":int(state.get("controls",0))*10,"maximum_original_pose_error":maximum_pose_error,"pause_resume_verified":pause_verified,"elapsed_process_seconds":(Time.get_ticks_msec()-started)/1000.0,"final_state":state,"trace":trace,"training_updates":0,"optimizer_steps":0,"research_interface_cases":0}
	FileAccess.open(output,FileAccess.WRITE).store_string(JSON.stringify(result,"\t"))
	print(JSON.stringify({"passed":result.passed,"failures":failures,"attempted_controls":attempts,"accepted_controls":result.accepted_controls,"attempted_substeps":result.attempted_substeps,"maximum_original_pose_error":maximum_pose_error,"pause_resume_verified":pause_verified,"elapsed_process_seconds":result.elapsed_process_seconds,"distance_m":state.get("forward_m",0),"stop_reason":state.get("error",""),"training_updates":0,"optimizer_steps":0}))
	quit(0 if failures.is_empty() else 3)

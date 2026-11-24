extends SceneTree
## One bounded actual-DLL comparison against the native reference. No optimizer.
var source := ""
var output := ""
var policy := ""
var controls := 0
var substeps := 0
var maximum_difference := 0.0
var failures: Array[String]=[]

func values(state: Dictionary) -> Array:
	var body: Dictionary=state.body
	var v: Vector3=body.position
	var q: Quaternion=body.orientation
	var result: Array=[v.x,v.y,v.z,q.x,q.y,q.z,q.w]
	for j in 4: result.append(body["j%d" % j])
	result.append_array(body.velocity)
	result.append_array(state.held_targets)
	result.append_array(state.reference_rates)
	result.append_array(state.previous_torques)
	result.append_array(state.last_action)
	result.append_array([body.work_J,body.travel_model,state.time_s,state.whole_body_clearance_lower_m,state.whole_body_clearance_upper_m])
	return result

func _initialize() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--reference="): source=arg.trim_prefix("--reference=")
		if arg.begins_with("--result="): output=arg.trim_prefix("--result=")
		if arg.begins_with("--policy="): policy=arg.trim_prefix("--policy=")
	var begun := Time.get_ticks_msec()
	var extension := GDExtensionManager.load_extension("res://steering.gdextension")
	if extension!=GDExtensionManager.LOAD_STATUS_OK and extension!=GDExtensionManager.LOAD_STATUS_ALREADY_LOADED:
		failures.append("Installed extension could not load")
	if not ClassDB.class_exists("HoldLabPreview"):
		_finish();return
	var preview: RefCounted=ClassDB.instantiate("HoldLabPreview")
	if not preview.call("load_controller",policy):
		failures.append("Original actor could not load");_finish();return
	var current_case := -1
	var last_control := 0
	var transitions := 0
	var file:=FileAccess.open(source,FileAccess.READ)
	while file!=null and file.get_position()<file.get_length():
		var line:=file.get_line()
		if line.is_empty(): continue
		var row: Dictionary=JSON.parse_string(line)
		var id:=int(row.case)
		var control:=int(row.control)
		if id!=current_case:
			current_case=id;last_control=0
			var pose := (3 if id/2>=1 else 0) if id<4 else 0
			var reset_ok: bool=preview.call("reset",0,true) if id==4 else preview.call("reset_untimed_swim",pose)
			if not reset_ok: failures.append("Fixture reset failed");break
		if control>0:
			if control!=last_control+1 or controls>=3040 or substeps+10>30400 or Time.get_ticks_msec()-begun>=60000:
				failures.append("Sequence or finite budget failed");break
			if control==41 and id<4:
				var before: Dictionary=preview.call("get_state")
				var direction:=1 if id%2 else -1
				if not preview.call("request_swim_turn",direction): failures.append("Heading request failed");break
				var after: Dictionary=preview.call("get_state")
				if values(before)!=values(after) or before.controls!=after.controls or before.substeps!=after.substeps or before.paused!=after.paused:
					failures.append("Heading request altered body/filter/time state");break
				if not after.heading_only_request or float(after.requested_turn_degrees)!=direction*5:
					failures.append("Heading request was not reported");break
				transitions+=1
			if control==1: preview.call("start")
			controls+=1
			var prior: Dictionary=preview.call("get_state")
			var advanced: bool=preview.call("step")
			var state: Dictionary=preview.call("get_state")
			substeps+=int(state.attempted_substeps)-int(prior.attempted_substeps)
			if not advanced: failures.append("Actual native physical stop: "+str(state.error));break
		var actual: Dictionary=preview.call("get_state")
		var actual_values:=values(actual)
		if actual_values.size()!=row.values.size(): failures.append("Native state shape differs");break
		for j in actual_values.size():
			maximum_difference=maxf(maximum_difference,absf(float(actual_values[j])-float(row.values[j])))
		if maximum_difference>2e-6 or int(actual.controls)!=control or int(actual.substeps)!=control*10:
			failures.append("Installed DLL state differs from CPU reference");break
		var heading: Vector3=actual.request_heading
		if heading.distance_to(Vector3(row.heading[0],row.heading[1],row.heading[2]))>2e-6:
			failures.append("Actual heading differs");break
		last_control=control
	if controls!=3040 or transitions!=4: failures.append("Not all six planned cases completed")
	preview.call("pause")
	_finish()

func _finish() -> void:
	var report: Dictionary={"passed":failures.is_empty(),"failures":failures,"actual_bridge_controls":controls,"actual_bridge_substeps":substeps,"maximum_scalar_difference":maximum_difference,"CPU_reference_tolerance":2e-6,"optimizer_calls":0,"CUDA_kernels":0,"original_motor":336,"candidate_loaded":false}
	var file:=FileAccess.open(output,FileAccess.WRITE)
	if file!=null: file.store_string(JSON.stringify(report,"  "));file.flush()
	print(JSON.stringify(report))
	quit(0 if failures.is_empty() else 2)

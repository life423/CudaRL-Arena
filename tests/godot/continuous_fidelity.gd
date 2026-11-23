extends SceneTree
## Reserved post-qualification native/render capture: at most 200 controls.
## It never supplies an action, target, heading, rotation or recovery.
var view: Node
var output := ""
var checks: Array[Dictionary] = []
var failures: Array[String] = []
var states: Array[Dictionary] = []
var executed := 0

func _initialize() -> void:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--test-output="): output=argument.trim_prefix("--test-output=")
	view=load("res://main.tscn").instantiate()
	root.add_child.call_deferred(view)
	_run.call_deferred()

func check(message: String, good: bool) -> void:
	checks.append({"check":message,"passed":good})
	if not good: failures.append(message)

func vector(value: Vector3) -> Array:
	return [value.x,value.y,value.z]

func rotation(value: Quaternion) -> Array:
	return [value.x,value.y,value.z,value.w]

func _run() -> void:
	await process_frame
	await process_frame
	view.set_process(false)
	check("The saved qualified continuous policy is loaded",view.loaded and view.continuous_available and bool(view.snapshot.get("qualified_candidate",false)))
	check("Capture starts at zero controls",int(view.snapshot.get("controls",-1))==0 and int(view.snapshot.get("substeps",-1))==0)
	check("A nonterminating Free swim request is active",bool(view.snapshot.get("continuous",false)) and int(view.snapshot.get("maximum_controls",-1))==0)
	if failures.is_empty():
		check("Native playback starts",bool(view.preview.call("start")))
	var faithful_mesh := true
	var maximum_position_error := 0.0
	var maximum_orientation_error := 0.0
	var domain_valid := true
	var posture_valid := true
	var counter_valid := true
	if failures.is_empty():
		for control in range(1,201):
			var advanced := bool(view.preview.call("step"))
			var saved: Dictionary=view.preview.call("get_state")
			executed=int(saved.get("attempted_controls",executed))
			view.snapshot=saved
			view.running=not bool(saved.get("paused",true))
			view._refresh()
			counter_valid=counter_valid and int(saved.controls)==control and int(saved.substeps)==control*10
			domain_valid=domain_valid and bool(saved.get("pond_domain_valid",false))
			posture_valid=posture_valid and float(saved.depth_m)<=.001 and float(saved.tilt_deg)<=10
			var links: Array[Dictionary] = []
			for i in range(3):
				var pose: Dictionary=saved.links[i]
				var rendered_position: Vector3=view.world.links[i].position
				var rendered_orientation: Quaternion=view.world.links[i].quaternion
				var native_orientation: Quaternion=pose.orientation
				var position_error: float=rendered_position.distance_to(pose.position)
				# q and -q are the same orientation; Node3D may normalize its basis.
				var orientation_error := absf(1.0-absf(rendered_orientation.normalized().dot(native_orientation.normalized())))
				maximum_position_error=maxf(maximum_position_error,position_error)
				maximum_orientation_error=maxf(maximum_orientation_error,orientation_error)
				faithful_mesh=faithful_mesh and position_error<=.000001 and orientation_error<=.000001
				links.append({"position_model":vector(pose.position),"orientation":rotation(native_orientation),"rendered_position_model":vector(rendered_position),"rendered_orientation":rotation(rendered_orientation)})
			states.append({"control":control,"substeps":saved.substeps,"time_s":saved.time_s,"speed_m_s":saved.speed_m_s,"depth_m":saved.depth_m,"tilt_deg":saved.tilt_deg,"links":links})
			if not advanced or bool(saved.get("paused",true)) or not str(saved.get("error","")).is_empty():
				failures.append("Native free swimming stopped during the reserved capture")
				break
			await process_frame
	check("Rendered links match their native poses within float32 precision",faithful_mesh)
	check("Every native state is within the supported pond",domain_valid)
	check("Depth and body tilt remain within their gates",posture_valid)
	check("Actual control/substep counts agree",counter_valid)
	check("Playback passes the old eight-second limit",executed==200 and int(view.snapshot.get("controls",-1))==200 and not bool(view.snapshot.get("finished",true)) and not bool(view.snapshot.get("paused",true)))
	if view.preview!=null: view.preview.call("pause")
	view.running=false
	view._refresh()
	if not output.is_empty():
		await RenderingServer.frame_post_draw
		check("Actual pond screenshot is saved",root.get_texture().get_image().save_png(output.get_basename()+".png")==OK)
		var result := {"passed":failures.is_empty(),"checks":checks,"failures":failures,"actual_controls":executed,"actual_substeps":int(view.snapshot.get("substeps",0)),"maximum_controls":200,"physical_seconds":view.snapshot.get("time_s",0),"maximum_render_position_error_model":maximum_position_error,"maximum_render_orientation_dot_error":maximum_orientation_error,"policy_sha256":FileAccess.get_sha256(view.policy_path),"checkpoint_sha256":view.snapshot.get("turning_checkpoint_sha256",""),"optimizer_steps":0,"states":states}
		FileAccess.open(output,FileAccess.WRITE).store_string(JSON.stringify(result,"  "))
	print(JSON.stringify({"passed":failures.is_empty(),"actual_controls":executed,"failures":failures}))
	quit(0 if failures.is_empty() else 3)

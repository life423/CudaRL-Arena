extends RefCounted
## Saved diagnostic disclosure only. Never loads a controller or starts physics.
const Saved = preload("res://continuous_study.gd")
const OFFSETS = [0,5,-5,15,-15,30,-30,60,-60,90,-90]

static func identity(value: String) -> String:
	return value.replace("\\","/").simplify_path().to_lower()

static func finite_number(value: Variant) -> bool:
	return (value is int or value is float) and is_finite(float(value))

static func read(path: String) -> Dictionary:
	if not FileAccess.file_exists(path): return {}
	var invalid := {"issue":"The turning-check result could not be verified. The saved pond policy is unchanged."}
	var wrapper := Saved.dictionary_file(path)
	if not Saved.count(wrapper.get("schema",null),1) or int(wrapper.schema)!=1: return invalid
	var result := Saved.bound_file(wrapper.get("source",null))
	var proof := Saved.bound_file(wrapper.get("verification",null))
	if not Saved.count(result.get("schema",null),1) or int(result.schema)!=1 or not Saved.count(proof.get("schema",null),1) or int(proof.schema)!=1: return invalid
	if result.get("study","")!="frozen_requested_turn_probe": return invalid
	for field in ["closed","descriptive_probe_only"]:
		if not result.get(field,null) is bool or not bool(result[field]): return invalid
	for field in ["qualified","runtime_adopted"]:
		if not result.get(field,null) is bool or bool(result[field]): return invalid
	for field in ["PPO_updates","optimizer_steps"]:
		if not Saved.count(result.get(field,null),0): return invalid
	var limits := {"controls":7360,"attempted_substeps":73600,"accepted_substeps":73600,"precheck_completed_cases":4,"probe_completed_cases":22}
	for field in limits:
		if not Saved.count(result.get(field,null),float(limits[field])): return invalid
	if float(result.accepted_substeps)>float(result.attempted_substeps) or float(result.attempted_substeps)>10*float(result.controls): return invalid
	if not result.get("hard_fault",null) is String: return invalid
	if not proof.get("audit_passed",null) is bool or not bool(proof.audit_passed): return invalid
	for field in ["qualified"]:
		if not proof.get(field,null) is bool or bool(proof[field]): return invalid
	if not proof.get("descriptive_only",null) is bool or not bool(proof.descriptive_only): return invalid
	if not proof.get("native_result",null) is Dictionary or proof.native_result!=result: return invalid
	var source_record: Dictionary=wrapper.source
	var alias: Variant=source_record.get("verification_input",null)
	var inputs: Variant=proof.get("input_sha256",null)
	if not alias is String or alias.strip_edges().is_empty() or not inputs is Dictionary: return invalid
	# An original audit identity is a lookup key only. Never open or globalize it.
	var bound := false
	for key in inputs:
		if identity(str(key))==identity(alias):
			bound=str(inputs[key]).to_lower()==str(source_record.sha256).to_lower()
	if not bound: return invalid
	var proof_counts := {"recorded_controls":"controls","recorded_attempted_substeps":"attempted_substeps","recorded_accepted_substeps":"accepted_substeps","matched_precheck_cases":"precheck_completed_cases"}
	for field in proof_counts:
		if not Saved.count(proof.get(field,null),float(limits[proof_counts[field]])) or int(proof[field])!=int(result[proof_counts[field]]): return invalid
	if not proof.get("full_probe_completed",null) is bool: return invalid
	var full: bool=bool(proof.full_probe_completed)
	if full!=(str(result.hard_fault).is_empty() and int(result.probe_completed_cases)==22 and int(result.precheck_completed_cases)==4): return invalid
	if proof.get("audit_status","")!=("verified_descriptive_probe" if str(result.hard_fault).is_empty() else "verified_hard_stop"): return invalid
	var cases: Variant=proof.get("cases",null)
	var unfinished: Variant=proof.get("unfinished_cases",null)
	var unrun: Variant=proof.get("unrun_cases",null)
	if not cases is Array or not unfinished is Array or not unrun is Array or cases.size()!=int(result.probe_completed_cases): return invalid
	var seen: Array[int]=[]
	var late_errors: Array[float]=[]
	var five_speeds: Array[float]=[]
	var five_errors: Array[float]=[]
	var thirty_speeds: Array[float]=[]
	var large_tilt_stops := 0
	var maximum_depth := 0.0
	for item in cases+unfinished:
		if not item is Dictionary or not Saved.count(item.get("case",null),21): return invalid
		var case_id := int(item.case)
		if case_id in seen: return invalid
		seen.append(case_id)
		if not Saved.count(item.get("pose",null),3) or int(item.pose)!=(0 if case_id<11 else 3): return invalid
		if not finite_number(item.get("offset_degrees",null)) or float(item.offset_degrees)!=float(OFFSETS[case_id%11]): return invalid
		if not Saved.count(item.get("controls",null),320) or int(item.controls)<1: return invalid
		if not Saved.count(item.get("behavior_reason",null),5) or int(item.behavior_reason) not in [0,1,4,5]: return invalid
		if not finite_number(item.get("max_depth_m",null)) or float(item.max_depth_m)<0: return invalid
		maximum_depth=maxf(maximum_depth,float(item.max_depth_m))
		var error: Variant=item.get("last4_velocity_course_error_deg",null)
		var vx: Variant=item.get("last4_velocity_x_m_s",null)
		var vz: Variant=item.get("last4_velocity_z_m_s",null)
		if error!=null:
			if not finite_number(error) or absf(float(error))>180 or int(item.controls)!=320 or int(item.get("behavior_reason",-1))!=0: return invalid
			if not finite_number(vx) or not finite_number(vz): return invalid
			var speed := sqrt(float(vx)*float(vx)+float(vz)*float(vz))*1000
			if not is_finite(speed): return invalid
			if case_id%11!=0: late_errors.append(absf(float(error)))
			if absf(float(item.offset_degrees))==5:
				five_speeds.append(speed)
				five_errors.append(absf(float(error)))
			if absf(float(item.offset_degrees))==30: thirty_speeds.append(speed)
		elif vx!=null or vz!=null: return invalid
		if absf(float(item.offset_degrees)) in [60.0,90.0] and int(item.behavior_reason)==5: large_tilt_stops+=1
	for item in unrun:
		if not Saved.count(item,21) or int(item) in seen: return invalid
		seen.append(int(item))
	if seen.size()!=22 or (full and (not unrun.is_empty() or not unfinished.is_empty())): return invalid
	return {"result":result,"verified":true,"full":full,"observed":cases.size()+unfinished.size(),"unrun":unrun.size(),"late_errors":late_errors,"five_speeds":five_speeds,"five_errors":five_errors,"thirty_speeds":thirty_speeds,"large_tilt_stops":large_tilt_stops,"maximum_depth_m":maximum_depth}

static func text(path: String) -> String:
	var checked := read(path)
	if checked.is_empty(): return ""
	var message := "\n[b]Turning check[/b]\n"
	if checked.has("issue"): return message+str(checked.issue)+"\n"
	if bool(checked.full): message+="The frozen swimmer was checked on 22 requested headings.\n"
	else: message+="The check stopped before all requests finished.\n"
	message+="Requests observed: %d / 22 · No learning updates\n" % int(checked.observed)
	var errors: Array=checked.late_errors
	var five: Array=checked.five_speeds
	var five_error: Array=checked.five_errors
	var thirty: Array=checked.thirty_speeds
	if not five.is_empty(): message+="±5° requests: %.2f–%.2f mm/s; %.1f–%.1f° course error (%d final windows).\n" % [float(five.min()),float(five.max()),float(five_error.min()),float(five_error.max()),five.size()]
	if not thirty.is_empty(): message+="±30° requests: %.2f–%.2f mm/s (%d final windows).\n" % [float(thirty.min()),float(thirty.max()),thirty.size()]
	if int(checked.large_tilt_stops)>0: message+="%d requests at ±60° / ±90° stopped at the tilt guard.\n" % int(checked.large_tilt_stops)
	if errors.is_empty(): message+="No changed-heading request has a complete final-window measurement.\n"
	if int(checked.observed)>0: message+="Largest recorded depth drift: %.2f mm.\n" % (float(checked.maximum_depth_m)*1000)
	if int(checked.unrun)>0: message+="%d requests were not run.\n" % int(checked.unrun)
	message+="This diagnostic does not qualify continuous swimming. The saved pond policy is unchanged.\n"
	return message

extends RefCounted
## Saved evidence disclosure only. No actor loading, physics, or optimizer calls.
const Saved = preload("res://continuous_study.gd")
const STUDY = "frozen_interface_comparison_v2"
const ARITHMETIC_PROFILE = "cpu_compatible_no_implicit_fma_v1"
const ARMS = ["lane_following","heading_only"]
const LABELS = {"lane_following":"Lane following","heading_only":"Heading only"}
const LIMITS = {"controls":127680,"attempted_substeps":1276800,"accepted_substeps":1276800,"precheck_completed_cases":4,"probe_completed_cases":88}
const RETAINED_TOTALS = {"short_passes":4,"original_holds":6,"long60_retained":28,"recovery_passes":3,"long300_retained":148,"withheld_holds":6,"endurance_holds":2}
const CONCLUSIONS = ["lane_following_qualifies","heading_only_qualifies","both_qualify","neither_qualifies","comparison_blocked"]

static func identity(value: String) -> String:
	return value.replace("\\","/").simplify_path().to_lower()

static func finite_number(value: Variant) -> bool:
	return (value is int or value is float) and is_finite(float(value))

static func retained_check(value: Variant) -> bool:
	if not value is Dictionary or not value.get("passed",null) is bool: return false
	if not Saved.count(value.get("controls",null),46240) or not Saved.count(value.get("substeps",null),462400): return false
	if int(value.substeps)>10*int(value.controls): return false
	if not value.get("cases",null) is Array or value.cases.size()>25: return false
	var totals: Variant=value.get("totals",null)
	if not totals is Dictionary: return false
	for field in RETAINED_TOTALS:
		if not Saved.count(totals.get(field,null),float(RETAINED_TOTALS[field])): return false
		if bool(value.passed) and int(totals[field])!=int(RETAINED_TOTALS[field]): return false
	if bool(value.passed) and (int(value.controls)!=46240 or int(value.substeps)!=462400 or value.cases.size()!=25): return false
	return true

static func read(path: String) -> Dictionary:
	if not FileAccess.file_exists(path): return {}
	var invalid := {"issue":"The interface comparison could not be verified. The saved pond policy is unchanged."}
	var wrapper := Saved.dictionary_file(path)
	if not Saved.count(wrapper.get("schema",null),1) or int(wrapper.schema)!=1: return invalid
	var result := Saved.bound_file(wrapper.get("source",null))
	var proof := Saved.bound_file(wrapper.get("verification",null))
	for value in [result,proof]:
		if not Saved.count(value.get("schema",null),1) or int(value.schema)!=1: return invalid
		if value.get("study","")!=STUDY: return invalid
		if not Saved.count(value.get("study_version",null),2) or int(value.study_version)!=2: return invalid
		if value.get("arithmetic_profile","")!=ARITHMETIC_PROFILE: return invalid
	for field in ["closed","descriptive_comparison_only"]:
		if not result.get(field,null) is bool or not bool(result[field]): return invalid
	for field in ["qualified","runtime_adopted","training"]:
		if not result.get(field,null) is bool or bool(result[field]): return invalid
	for field in ["PPO_updates","optimizer_steps"]:
		if not Saved.count(result.get(field,null),0): return invalid
	for field in LIMITS:
		if not Saved.count(result.get(field,null),float(LIMITS[field])): return invalid
	if not Saved.count(result.get("cases_started",null),88) or int(result.cases_started)<int(result.probe_completed_cases) or int(result.cases_started)>mini(88,int(result.probe_completed_cases)+1): return invalid
	if float(result.accepted_substeps)>float(result.attempted_substeps) or float(result.attempted_substeps)>10*float(result.controls): return invalid
	if not result.get("hard_fault",null) is String: return invalid
	for field in ["audit_passed","descriptive_only","weights_unchanged"]:
		if not proof.get(field,null) is bool or not bool(proof[field]): return invalid
	if not proof.get("qualified",null) is bool or bool(proof.qualified): return invalid
	if not proof.get("native_result",null) is Dictionary or proof.native_result!=result: return invalid
	var source_record: Dictionary=wrapper.source
	var alias: Variant=source_record.get("verification_input",null)
	var inputs: Variant=proof.get("input_sha256",null)
	if not alias is String or alias.strip_edges().is_empty() or not inputs is Dictionary: return invalid
	var arithmetic_digest: Variant=proof.get("arithmetic_preparation_sha256",null)
	if not arithmetic_digest is String or arithmetic_digest.length()!=64 or not arithmetic_digest.is_valid_hex_number(): return invalid
	var arithmetic_bound := false
	for digest in inputs.values():
		if str(digest).to_lower()==arithmetic_digest.to_lower(): arithmetic_bound=true
	if not arithmetic_bound: return invalid
	var retained: Variant=proof.get("retention",null)
	if not retained is Dictionary or not retained_check(retained.get("cpu",null)) or not retained_check(retained.get("cuda",null)): return invalid
	var retained_both := bool(retained.cpu.passed) and bool(retained.cuda.passed)
	if not proof.get("original_swim_rest_retained",null) is bool or bool(proof.original_swim_rest_retained)!=retained_both: return invalid
	if not result.get("retention_passed",null) is bool or bool(result.retention_passed)!=retained_both: return invalid
	if not result.get("comparison_blocked_reason",null) is String: return invalid
	if int(result.controls)<int(retained.cpu.controls)+int(retained.cuda.controls): return invalid
	if int(result.attempted_substeps)<int(retained.cpu.substeps)+int(retained.cuda.substeps): return invalid
	if proof.get("conclusion",null) not in CONCLUSIONS: return invalid
	# The original audit identity is a lookup key only; never open that path.
	var bound := false
	for key in inputs:
		if identity(str(key))==identity(alias): bound=str(inputs[key]).to_lower()==str(source_record.sha256).to_lower()
	if not bound: return invalid
	var counts := {"recorded_controls":"controls","recorded_attempted_substeps":"attempted_substeps","recorded_accepted_substeps":"accepted_substeps"}
	for field in counts:
		if not Saved.count(proof.get(field,null),float(LIMITS[counts[field]])) or int(proof[field])!=int(result[counts[field]]): return invalid
	if not proof.get("full_comparison_completed",null) is bool: return invalid
	var full := bool(proof.full_comparison_completed)
	if full!=(str(result.hard_fault).is_empty() and int(result.precheck_completed_cases)==4 and int(result.probe_completed_cases)==88): return invalid
	var retention_stop := false
	if not full and str(result.hard_fault).is_empty():
		retention_stop=not retained_both and result.comparison_blocked_reason=="retention_regression" and int(result.precheck_completed_cases)==4 and int(result.cases_started)==0 and int(result.probe_completed_cases)==0
		for backend in ["cpu","cuda"]:
			retention_stop=retention_stop and retained[backend].cases.size()==25 and int(retained[backend].controls)==46240 and int(retained[backend].substeps)==462400
		if not retention_stop: return invalid
	var expected_status := "passed" if full else "verified_retention_stop" if retention_stop else "verified_hard_stop"
	if proof.get("audit_status","")!=expected_status: return invalid
	if not full:
		if proof.conclusion!="comparison_blocked": return invalid
		if not proof.get("cases",null) is Array or proof.cases.size()>int(result.cases_started): return invalid
		return {"verified":true,"full":false,"result":result,"retention":retained,"original_swim_rest_retained":retained_both,"retention_stop":retention_stop,"conclusion":proof.conclusion}
	if not retained_both: return invalid
	var cases: Variant=proof.get("cases",null)
	if not cases is Array or cases.size()!=88: return invalid
	var derived := {}
	var required_passes := {"lane_following":0,"heading_only":0}
	for arm in ARMS: derived[arm]={"cases":0,"completed":0,"zero_passes":0,"nonzero_passes":0,"repeated_return_passes":0,"tilt_stops":0,"depth_stops":0,"bank_stops":0}
	var seen: Array[int]=[]
	for item in cases:
		if not item is Dictionary or not Saved.count(item.get("case",null),87): return invalid
		var case_id := int(item.case)
		if case_id in seen: return invalid
		seen.append(case_id)
		var arm: String=ARMS[case_id%2]
		var scenario := int(case_id/2)%11
		if item.get("arm",null)!=arm or not Saved.count(item.get("scenario",null),10) or int(item.scenario)!=scenario: return invalid
		for field in ["completed","functional_pass"]:
			if not item.get(field,null) is bool: return invalid
		if not Saved.count(item.get("reason",null),5) or int(item.reason) not in [0,1,4,5]: return invalid
		if bool(item.functional_pass) and not bool(item.completed): return invalid
		if bool(item.completed)!=(int(item.reason)==0): return invalid
		derived[arm].cases+=1
		if bool(item.completed): derived[arm].completed+=1
		if bool(item.functional_pass):
			derived[arm]["zero_passes" if scenario==0 else "nonzero_passes"]+=1
			if scenario>=5: derived[arm].repeated_return_passes+=1
			if scenario in [5,6,9,10]: required_passes[arm]+=1
		if int(item.reason)==1: derived[arm].bank_stops+=1
		if int(item.reason)==4: derived[arm].depth_stops+=1
		if int(item.reason)==5: derived[arm].tilt_stops+=1
	var summaries: Variant=proof.get("arm_summary",null)
	if not summaries is Array or summaries.size()!=2: return invalid
	var arms := {}
	for item in summaries:
		if not item is Dictionary or not item.get("arm",null) is String or item.arm not in ARMS or arms.has(item.arm): return invalid
		for field in derived[item.arm]:
			if not Saved.count(item.get(field,null),44) or int(item[field])!=int(derived[item.arm][field]): return invalid
		if not finite_number(item.get("median_late_forward_m_s_incomplete_zero",null)): return invalid
		arms[item.arm]=item
	var selection: Variant=proof.get("selection",null)
	if not selection is Dictionary or selection.get("preferred_arm",null) not in ARMS+["tie"]: return invalid
	var expected_preference := "tie"
	for field in ["zero_passes","repeated_return_passes","nonzero_passes"]:
		if int(arms.lane_following[field])!=int(arms.heading_only[field]):
			expected_preference="lane_following" if int(arms.lane_following[field])>int(arms.heading_only[field]) else "heading_only"
			break
	if expected_preference=="tie":
		var delta := float(arms.lane_following.median_late_forward_m_s_incomplete_zero)-float(arms.heading_only.median_late_forward_m_s_incomplete_zero)
		if absf(delta)>=1e-9: expected_preference="lane_following" if delta>0 else "heading_only"
	if selection.preferred_arm!=expected_preference: return invalid
	var adequate: Variant=selection.get("adequate_arms",null)
	if not adequate is Array or adequate.size()>2: return invalid
	var expected_adequate: Array[String]=[]
	for arm in ARMS:
		if int(arms[arm].zero_passes)==4 and int(required_passes[arm])==16: expected_adequate.append(arm)
	if adequate.size()!=expected_adequate.size(): return invalid
	var seen_adequate: Array[String]=[]
	for arm in adequate:
		if not arm is String or arm not in expected_adequate or arm in seen_adequate: return invalid
		seen_adequate.append(arm)
	var expected_conclusion := "neither_qualifies"
	if expected_adequate.size()==2: expected_conclusion="both_qualify"
	elif expected_adequate.size()==1: expected_conclusion=str(expected_adequate[0])+"_qualifies"
	if proof.conclusion!=expected_conclusion: return invalid
	return {"verified":true,"full":true,"result":result,"arms":arms,"selection":selection,"adequate_arms":adequate,"retention":retained,"original_swim_rest_retained":retained_both,"conclusion":proof.conclusion}

static func text(path: String) -> String:
	var checked := read(path)
	if checked.is_empty(): return ""
	var message := "\n[b]Request interface comparison[/b]\n"
	if checked.has("issue"): return message+str(checked.issue)+"\n"
	message+="Same saved weights · No learning updates\n"
	message+="Rerun after the arithmetic-consistency correction.\n"
	if bool(checked.original_swim_rest_retained):
		message+="Original swimming, recovery and resting checks passed on the corrected CPU and CUDA runtimes.\n"
	else:
		for backend in ["cpu","cuda"]:
			var retained: Dictionary=checked.retention[backend]
			var outcome := "passed" if bool(retained.passed) else "did not retain every required skill" if retained.cases.size()==25 else "not run" if retained.cases.is_empty() else "stopped before completion"
			message+="Original-skill checks on %s: %s.\n" % [str(backend).to_upper(),outcome]
	if not bool(checked.full):
		message+="Conclusion: comparison blocked.\n"
		if checked.get("retention_stop",false):
			message+="The CPU / CUDA agreement check passed, but original-skill retention did not. Interface tests were not run.\n"
			message+="Main interface cases attempted: 0 / 88.\n"
			message+="Recorded controls: %d · Accepted physical substeps: %d.\n" % [int(checked.result.controls),int(checked.result.accepted_substeps)]
			return message+"The saved weights are unchanged. No replacement policy was installed. Continuous pond swimming is still unqualified.\n"
		if int(checked.result.cases_started)==0:
			message+="The comparison stopped before either interface was tested.\n"
			if str(checked.result.hard_fault).contains("Matched-command"):
				message+="The CPU / CUDA matched-command agreement check failed.\n"
			else: message+="A prerequisite check failed.\n"
			message+="Main interface cases attempted: 0 / 88.\n"
		else: message+="The comparison stopped with %d / 88 interface cases closed.\n" % int(checked.result.probe_completed_cases)
		message+="Recorded controls: %d · Accepted physical substeps: %d.\n" % [int(checked.result.controls),int(checked.result.accepted_substeps)]
		message+="The saved prefix was independently checked. There is no interface preference or new capability from this stopped result.\n"
		return message+"Continuous pond swimming is still unqualified. The saved motor is unchanged; no policy was installed.\n"
	message+="88 trajectories in 44 lane-following / heading-only pairs.\n"
	message+="The fresh CPU / CUDA agreement check passed.\n"
	var conclusions := {"lane_following_qualifies":"Lane following met the finite interface criteria.","heading_only_qualifies":"Heading only met the finite interface criteria.","both_qualify":"Both interfaces met the finite interface criteria.","neither_qualifies":"Neither interface met the finite interface criteria."}
	message+=str(conclusions[checked.conclusion])+"\n"
	for arm in ARMS:
		var summary: Dictionary=checked.arms[arm]
		message+="%s: %d / 40 changed-course cases met the movement criteria; %d / 24 repeated or return cases.\n" % [LABELS[arm],int(summary.nonzero_passes),int(summary.repeated_return_passes)]
		message+="Full horizons: %d / 44. Unchanged-course controls: %d / 4.\n" % [int(summary.completed),int(summary.zero_passes)]
		message+="Guard stops — bank: %d; depth: %d; tilt: %d.\n" % [int(summary.bank_stops),int(summary.depth_stops),int(summary.tilt_stops)]
	var preferred := str(checked.selection.preferred_arm)
	if preferred=="tie": message+="Neither interface ranked above the other under this comparison's rule.\n"
	else: message+="%s ranked higher under this comparison's rule.\n" % LABELS[preferred]
	if checked.adequate_arms.is_empty(): message+="Neither interface met the required repeated-turn and return criteria.\n"
	else: message+="The required repeated-turn and return criteria were met by %s.\n" % ", ".join(checked.adequate_arms.map(func(arm): return str(LABELS[arm]).to_lower()))
	message+="Continuous pond swimming is still unqualified. The saved motor is unchanged; no policy was installed.\n"
	return message

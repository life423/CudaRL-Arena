extends RefCounted
## A verified launch failure is evidence of no run, never a physical result.
const Saved = preload("res://continuous_study.gd")
const STUDY := "frozen_interface_comparison_v2"
const PROFILE := "cpu_compatible_no_implicit_fma_v1"

static func flag(value: Variant, expected: bool) -> bool:
	return value is bool and value==expected

static func identity(value: String) -> String:
	return value.replace("\\","/").simplify_path().to_lower()

static func bound_input(proof: Dictionary, record: Dictionary) -> bool:
	var inputs: Variant=proof.get("input_sha256",null)
	var alias: Variant=record.get("verification_input",null)
	if not inputs is Dictionary or not alias is String or alias.is_empty(): return false
	var matched := 0
	for path in inputs:
		if identity(str(path))==identity(alias):
			if str(inputs[path]).to_lower()!=str(record.get("sha256","")).to_lower(): return false
			matched+=1
	return matched==1

static func read(path: String) -> Dictionary:
	if not FileAccess.file_exists(path): return {}
	var invalid := {"issue":"The latest comparison launch-stop evidence could not be verified."}
	var wrapper := Saved.dictionary_file(path)
	if not Saved.count(wrapper.get("schema",null),1) or int(wrapper.schema)!=1: return invalid
	var proof := Saved.bound_file(wrapper.get("source",null))
	var process := Saved.bound_file(wrapper.get("process",null))
	var arithmetic := Saved.bound_file(wrapper.get("arithmetic",null))
	var protection := Saved.bound_file(wrapper.get("protection",null))
	for document in [proof,arithmetic]:
		if not Saved.count(document.get("schema",null),1) or int(document.schema)!=1 or document.get("study",null)!=STUDY or not Saved.count(document.get("study_version",null),2) or int(document.study_version)!=2 or document.get("arithmetic_profile",null)!=PROFILE: return invalid
	for field in ["audit_passed","launch_failed_before_dynamics","weights_unchanged","descriptive_only"]:
		if not flag(proof.get(field,null),true): return invalid
	for field in ["qualified","runtime_adopted","full_comparison_completed"]:
		if not flag(proof.get(field,null),false): return invalid
	if proof.get("audit_status",null)!="verified_launch_stop" or proof.get("conclusion",null)!="comparison_blocked": return invalid
	for field in ["recorded_controls","recorded_attempted_substeps","recorded_accepted_substeps","matched_precheck_cases","identical_warmup_pairs","PPO_updates","optimizer_steps"]:
		if not Saved.count(proof.get(field,null),0): return invalid
	if proof.get("native_result","missing")!=null or proof.get("original_swim_rest_retained","missing")!=null: return invalid
	if proof.get("cases",null)!=[] or proof.get("arm_summary",null)!=[]: return invalid
	var unrun: Variant=proof.get("unrun_cases",null)
	if not unrun is Array or unrun.size()!=88: return invalid
	for case_id in range(88):
		if unrun[case_id]!=case_id: return invalid
	var retention: Variant=proof.get("retention",null)
	if not retention is Dictionary: return invalid
	for backend in ["cpu","cuda"]:
		var kept: Variant=retention.get(backend,null)
		if not kept is Dictionary or kept.get("status",null)!="not_run" or kept.get("passed","missing")!=null or kept.get("totals","missing")!=null or kept.get("cases",null)!=[]: return invalid
		if not Saved.count(kept.get("controls",null),0) or not Saved.count(kept.get("substeps",null),0): return invalid
	if not Saved.count(process.get("schema",null),1) or int(process.schema)!=1 or process.get("study",null)!=STUDY or not Saved.count(process.get("exit_code",null),2) or int(process.exit_code)!=2 or not flag(process.get("terminated_at_cap",null),false): return invalid
	if not Saved.count(process.get("PPO_updates",null),0) or not Saved.count(process.get("optimizer_steps",null),0): return invalid
	var seconds: Variant=process.get("seconds",null)
	if not (seconds is int or seconds is float) or not is_finite(float(seconds)) or float(seconds)<=0 or float(seconds)>900: return invalid
	if proof.get("process_execution",null)!=process: return invalid
	if str(proof.get("process_sha256","")).to_lower()!=str(wrapper.process.get("sha256","")).to_lower(): return invalid
	var stop: Variant=proof.get("launch_stop",null)
	if not stop is Dictionary or stop.get("phase",null)!="pre_probe_inventory" or stop.get("native_exit_code",null)!=2 or stop.get("reason",null)!="filesystem_iterator_error": return invalid
	if not stop.get("stderr",null) is String or not str(stop.stderr).contains("recursive_directory_iterator"): return invalid
	if arithmetic.get("phase",null)!="arithmetic-preparation" or not flag(arithmetic.get("passed",null),true): return invalid
	for field in ["physical_controls","physical_substeps","integration_calls","PPO_updates","optimizer_steps"]:
		if not Saved.count(arithmetic.get(field,null),0): return invalid
	if str(proof.get("arithmetic_preparation_sha256","")).to_lower()!=str(wrapper.arithmetic.get("sha256","")).to_lower(): return invalid
	if not bound_input(proof,wrapper.process) or not bound_input(proof,wrapper.arithmetic): return invalid
	if not flag(protection.get("passed",null),true) or protection.get("study",null)!=STUDY or not Saved.count(protection.get("files",null),1000000) or int(protection.files)==0: return invalid
	for field in ["physical_controls","optimizer_steps"]:
		if not Saved.count(protection.get(field,null),0): return invalid
	for pair in [["launch_stop_verification_sha256","source"],["process_execution_sha256","process"],["arithmetic_preparation_sha256","arithmetic"]]:
		if str(protection.get(pair[0],"")).to_lower()!=str(wrapper[pair[1]].get("sha256","")).to_lower(): return invalid
	return {"verified":true,"full":false,"launch_stop":true,"conclusion":"comparison_blocked","proof":proof}

static func text(path: String) -> String:
	var checked := read(path)
	if checked.is_empty(): return ""
	var message := "\n[b]Request interface comparison[/b]\n"
	if checked.has("issue"): return message+str(checked.issue)+"\n"
	message+="Same saved weights · No learning updates\n"
	message+="Conclusion: comparison blocked.\n"
	message+="Fixed-state CPU / CUDA arithmetic checks passed.\n"
	message+="The physical study could not start because of an experiment file-path error.\n"
	message+="Physical controls: 0 · Interface cases attempted: 0 / 88.\n"
	message+="The fresh agreement gate and original swimming / resting checks were not run.\n"
	message+="No interface preference was measured. The authorized attempt is closed.\n"
	return message+"The saved motor is unchanged; no policy was installed. Continuous pond swimming is still unqualified.\n"

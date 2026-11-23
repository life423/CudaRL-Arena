extends SceneTree
## TESTONLY provenance/disclosure fixtures. No native object or simulation exists.
const Study = preload("res://request_turn_study.gd")
const Saved = preload("res://continuous_study.gd")
var checks: Array[Dictionary]=[]
var failures: Array[String]=[]
var output := ""
var serial := 0
const ALIAS = "Z:/TESTONLY-missing-pc/frozen-turn/result.json"

func check(label: String, passed: bool) -> void:
	checks.append({"check":label,"passed":passed})
	if not passed: failures.append(label)

func write_json(path: String, value: Dictionary) -> void:
	FileAccess.open(path,FileAccess.WRITE).store_string(JSON.stringify(value))

func source() -> Dictionary:
	return {"schema":1,"study":"frozen_requested_turn_probe","closed":true,"hard_fault":"","precheck_completed_cases":4,"probe_completed_cases":22,"controls":7360,"attempted_substeps":73600,"accepted_substeps":73600,"PPO_updates":0,"optimizer_steps":0,"qualified":false,"runtime_adopted":false,"descriptive_probe_only":true}

func proof() -> Dictionary:
	var cases: Array=[]
	for id in range(22):
		cases.append({"case":id,"pose":0 if id<11 else 3,"offset_degrees":Study.OFFSETS[id%11],"controls":320,"behavior_reason":0,"max_depth_m":0.0004,"last4_velocity_course_error_deg":float(id%11),"last4_velocity_x_m_s":0.008,"last4_velocity_z_m_s":0.0})
	return {"schema":1,"audit_status":"verified_descriptive_probe","audit_passed":true,"full_probe_completed":true,"qualified":false,"descriptive_only":true,"recorded_controls":7360,"recorded_attempted_substeps":73600,"recorded_accepted_substeps":73600,"matched_precheck_cases":4,"cases":cases,"unfinished_cases":[],"unrun_cases":[]}

func fixture(native: Dictionary, verified: Dictionary, bad_binding: bool=false) -> String:
	serial+=1
	var stem := "res://test-fixtures/TESTONLY-request-turn-%d" % serial
	write_json(stem+"-result.json",native)
	var digest := FileAccess.get_sha256(stem+"-result.json")
	verified=verified.duplicate(true)
	verified.native_result=native.duplicate(true)
	verified.input_sha256={ALIAS:"0".repeat(64) if bad_binding else digest}
	write_json(stem+"-proof.json",verified)
	write_json(stem+".json",{"schema":1,"source":{"path":stem+"-result.json","sha256":digest,"verification_input":ALIAS},"verification":{"path":stem+"-proof.json","sha256":FileAccess.get_sha256(stem+"-proof.json")}})
	return stem+".json"

func _initialize() -> void:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--test-output="): output=argument.trim_prefix("--test-output=")
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path("res://test-fixtures"))
	var valid := fixture(source(),proof())
	check("Bound relocated TESTONLY result is accepted",Study.read(valid).get("verified",false)==true)
	var message := Study.text(valid)
	check("Diagnostic does not claim continuous ability",message.contains("does not qualify continuous swimming") and message.contains("No learning updates"))
	check("Requests are observations, never pass counts",message.contains("Requests observed: 22 / 22") and not message.contains("22 passes"))
	check("Actual bands include speed and course without pass claims",message.contains("±5° requests: 8.00–8.00 mm/s; 1.0–2.0° course error") and message.contains("±30° requests: 8.00–8.00 mm/s"))
	check("Missing optional result makes no claim",Study.read("res://test-fixtures/TESTONLY-absent-turn.json").is_empty())
	check("Proof must bind copied source bytes",Study.read(fixture(source(),proof(),true)).has("issue"))
	for field in ["qualified","runtime_adopted"]:
		var changed := source();changed[field]=true
		check("Diagnostic cannot set "+field,Study.read(fixture(changed,proof())).has("issue"))
	for field in ["PPO_updates","optimizer_steps"]:
		var changed := source();changed[field]=1
		check("Diagnostic cannot claim "+field,Study.read(fixture(changed,proof())).has("issue"))
	var changed := source();changed.closed=false
	check("Open native result is rejected",Study.read(fixture(changed,proof())).has("issue"))
	var verified := proof();verified.audit_passed=false
	check("Failed independent audit is unverified",Study.read(fixture(source(),verified)).has("issue"))
	verified=proof();verified.recorded_controls=7359
	check("Counter disagreement is unverified",Study.read(fixture(source(),verified)).has("issue"))
	verified=proof();verified.cases[1].case=0
	check("Duplicate request identities are rejected",Study.read(fixture(source(),verified)).has("issue"))
	verified=proof();verified.cases.pop_back()
	check("Missing cases cannot claim full completion",Study.read(fixture(source(),verified)).has("issue"))
	var altered: Dictionary=Saved.dictionary_file(valid)
	altered.source.verification_input={"not":"a path identity"}
	write_json("res://test-fixtures/TESTONLY-bad-turn-alias.json",altered)
	check("Malformed provenance alias is rejected",Study.read("res://test-fixtures/TESTONLY-bad-turn-alias.json").has("issue"))
	altered=Saved.dictionary_file(valid);altered.source.sha256="0".repeat(64)
	write_json("res://test-fixtures/TESTONLY-bad-turn-hash.json",altered)
	check("Wrong copied native bytes cannot make claims",Study.read("res://test-fixtures/TESTONLY-bad-turn-hash.json").has("issue"))
	changed=source();changed.hard_fault="TESTONLY stop";changed.probe_completed_cases=1;changed.controls=643;changed.attempted_substeps=6430;changed.accepted_substeps=6430
	verified=proof();verified.audit_status="verified_hard_stop";verified.full_probe_completed=false;verified.recorded_controls=643;verified.recorded_attempted_substeps=6430;verified.recorded_accepted_substeps=6430
	verified.cases=[verified.cases[0]];verified.unfinished_cases=[{"case":1,"pose":0,"offset_degrees":5,"controls":3,"behavior_reason":0,"max_depth_m":0.0051,"last4_velocity_course_error_deg":null,"last4_velocity_x_m_s":null,"last4_velocity_z_m_s":null}];verified.unrun_cases=range(2,22)
	var partial := fixture(changed,verified)
	check("Audited hard stop prefix is readable",Study.read(partial).get("verified",false)==true)
	message=Study.text(partial)
	check("Unrun requests remain explicitly unrun",message.contains("20 requests were not run") and message.contains("Requests observed: 2 / 22"))
	check("Unavailable final windows are not zero errors",message.contains("No changed-heading request has a complete") and not message.contains("error: 0.0"))
	check("Hard stop never claims all requests finished",message.contains("stopped before all requests finished") and not message.contains("checked on 22"))
	var actual_path := "res://research_data/request-turn-study.json"
	if FileAccess.file_exists(actual_path):
		var actual := Study.read(actual_path)
		check("Actual portable diagnostic is independently bound",actual.get("verified",false)==true)
		check("Actual diagnostic cannot adopt a policy",actual.has("result") and actual.result.qualified==false and actual.result.runtime_adopted==false and actual.result.PPO_updates==0)
	var report := {"passed":failures.is_empty(),"checks":checks,"failures":failures,"physical_controls":0,"native_instances":0,"GPU_kernels":0,"optimizer_steps":0}
	if not output.is_empty(): FileAccess.open(output,FileAccess.WRITE).store_string(JSON.stringify(report,"  "))
	print(JSON.stringify(report))
	quit(0 if failures.is_empty() else 3)

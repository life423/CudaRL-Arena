extends SceneTree
## TESTONLY disclosure fixtures. No native instance, physics, CUDA or optimizer.
const Study = preload("res://interface_comparison_study_v2.gd")
const Saved = preload("res://continuous_study.gd")
const ALIAS = "Z:/TESTONLY-other-computer/interface-comparison/result.json"
const ARITHMETIC_ALIAS = "Z:/TESTONLY-other-computer/interface-comparison/arithmetic-proof.json"
const ARITHMETIC_DIGEST = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
var checks: Array[Dictionary]=[]
var failures: Array[String]=[]
var output := ""
var serial := 0

func check(label: String, passed: bool) -> void:
	checks.append({"check":label,"passed":passed})
	if not passed: failures.append(label)

func write_json(path: String, value: Dictionary) -> void:
	FileAccess.open(path,FileAccess.WRITE).store_string(JSON.stringify(value))

func source() -> Dictionary:
	return {"schema":1,"study":"frozen_interface_comparison_v2","study_version":2,"arithmetic_profile":Study.ARITHMETIC_PROFILE,"closed":true,"hard_fault":"","retention_passed":true,"comparison_blocked_reason":"","precheck_completed_cases":4,"cases_started":88,"probe_completed_cases":88,"controls":127680,"attempted_substeps":1276800,"accepted_substeps":1276800,"PPO_updates":0,"optimizer_steps":0,"qualified":false,"runtime_adopted":false,"training":false,"descriptive_comparison_only":true}

func retention(passed: bool=true) -> Dictionary:
	var totals: Dictionary=Study.RETAINED_TOTALS.duplicate(true)
	if not passed:
		for name in totals: totals[name]=0
	var cases: Array=[]
	if passed:
		for id in range(25): cases.append({"TESTONLY":true,"case":id,"passed":true})
	return {"passed":passed,"controls":46240 if passed else 0,"substeps":462400 if passed else 0,"totals":totals,"cases":cases}

func proof() -> Dictionary:
	var cases: Array=[]
	var arms: Array=[]
	for name in Study.ARMS:
		arms.append({"arm":name,"cases":44,"completed":44,"zero_passes":4,"nonzero_passes":4 if name=="lane_following" else 8,"repeated_return_passes":0,"tilt_stops":0,"depth_stops":0,"bank_stops":0,"median_late_forward_m_s_incomplete_zero":0.008})
	for id in range(88):
		var scenario := int(id/2)%11
		cases.append({"arm":Study.ARMS[id%2],"case":id,"scenario":scenario,"completed":true,"reason":0,"functional_pass":scenario<=(1 if id%2==0 else 2)})
	return {"schema":1,"study":Study.STUDY,"study_version":2,"arithmetic_profile":Study.ARITHMETIC_PROFILE,"arithmetic_preparation_sha256":ARITHMETIC_DIGEST,"audit_passed":true,"audit_status":"passed","full_comparison_completed":true,"descriptive_only":true,"qualified":false,"weights_unchanged":true,"retention":{"cpu":retention(),"cuda":retention()},"original_swim_rest_retained":true,"conclusion":"neither_qualifies","recorded_controls":127680,"recorded_attempted_substeps":1276800,"recorded_accepted_substeps":1276800,"cases":cases,"arm_summary":arms,"selection":{"preferred_arm":"heading_only","adequate_arms":[],"status":"relative_preference_only"}}

func fixture(native: Dictionary, verified: Dictionary, bad_binding: bool=false) -> String:
	serial+=1
	var stem := "res://test-fixtures/TESTONLY-interface-comparison-v2-%d" % serial
	write_json(stem+"-result.json",native)
	var digest := FileAccess.get_sha256(stem+"-result.json")
	verified=verified.duplicate(true)
	verified.native_result=native.duplicate(true)
	verified.input_sha256={ALIAS:"0".repeat(64) if bad_binding else digest}
	verified.input_sha256[ARITHMETIC_ALIAS]=ARITHMETIC_DIGEST
	write_json(stem+"-proof.json",verified)
	write_json(stem+".json",{"schema":1,"source":{"path":stem+"-result.json","sha256":digest,"verification_input":ALIAS},"verification":{"path":stem+"-proof.json","sha256":FileAccess.get_sha256(stem+"-proof.json")}})
	return stem+".json"

func qualifying_proof(qualified: Array[String]) -> Dictionary:
	var verified := proof()
	for item in verified.cases:
		item.functional_pass=int(item.scenario)==0 or (str(item.arm) in qualified and int(item.scenario) in [5,6,9,10])
	for arm in verified.arm_summary:
		arm.nonzero_passes=16 if str(arm.arm) in qualified else 0
		arm.repeated_return_passes=16 if str(arm.arm) in qualified else 0
	verified.selection.adequate_arms=qualified
	verified.selection.preferred_arm="tie" if qualified.size()==2 else qualified[0]
	verified.conclusion="both_qualify" if qualified.size()==2 else qualified[0]+"_qualifies"
	return verified

func _initialize() -> void:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--test-output="): output=argument.trim_prefix("--test-output=")
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path("res://test-fixtures"))
	var valid := fixture(source(),proof())
	check("Relocated byte-bound TESTONLY comparison is accepted",Study.read(valid).get("verified",false)==true)
	var message := Study.text(valid)
	check("UI derives distinct per-interface counts",message.contains("Lane following: 4 / 40") and message.contains("Heading only: 8 / 40"))
	check("Relative preference is not navigation qualification",message.contains("Heading only ranked higher") and message.contains("Neither interface met the required repeated-turn") and message.contains("Continuous pond swimming is still unqualified"))
	check("No learning or adoption is claimed",message.contains("No learning updates") and message.contains("no policy was installed"))
	check("The arithmetic rerun is identified without implying new learning",message.contains("arithmetic-consistency correction") and message.contains("Same saved weights"))
	var incompatible := source();incompatible.study_version=1
	check("An old study version cannot masquerade as this rerun",Study.read(fixture(incompatible,proof())).has("issue"))
	incompatible=source();incompatible.arithmetic_profile="TESTONLY-wrong-profile"
	check("A different arithmetic profile is rejected",Study.read(fixture(incompatible,proof())).has("issue"))
	var bad_arithmetic := proof();bad_arithmetic.arithmetic_preparation_sha256="b".repeat(64)
	check("The independent arithmetic preparation must be bound",Study.read(fixture(source(),bad_arithmetic)).has("issue"))
	bad_arithmetic=proof();bad_arithmetic.study_version=1
	check("The independent proof must identify the new study",Study.read(fixture(source(),bad_arithmetic)).has("issue"))
	var bad_retention := proof();bad_retention.retention.cpu.totals.original_holds=5
	check("Saved weights alone cannot substitute for retained resting on the corrected runtime",Study.read(fixture(source(),bad_retention)).has("issue"))
	bad_retention=proof();bad_retention.retention.cuda.passed=false;bad_retention.original_swim_rest_retained=false
	check("A complete comparison requires CPU and CUDA retention",Study.read(fixture(source(),bad_retention)).has("issue"))
	bad_retention=proof();bad_retention.retention.cpu.controls=0
	check("Claimed retention must have its full recorded controls",Study.read(fixture(source(),bad_retention)).has("issue"))
	bad_retention=proof();bad_retention.conclusion="both_qualify"
	check("The explicit conclusion must follow finite interface adequacy",Study.read(fixture(source(),bad_retention)).has("issue"))
	check("Corrected-runtime retention is stated explicitly",message.contains("Original swimming, recovery and resting checks passed on the corrected CPU and CUDA runtimes"))
	var adequate := proof()
	for item in adequate.cases:
		if item.arm=="heading_only": item.functional_pass=int(item.scenario) in [0,5,6,9,10]
	adequate.arm_summary[1].nonzero_passes=16
	adequate.arm_summary[1].repeated_return_passes=16
	adequate.selection.adequate_arms=["heading_only"]
	adequate.conclusion="heading_only_qualifies"
	var adequate_path := fixture(source(),adequate)
	check("Essential repeated and return cases suffice without optional fast turns",Study.read(adequate_path).get("adequate_arms",[])==["heading_only"])
	check("Finite adequacy still cannot qualify pond navigation",Study.text(adequate_path).contains("Continuous pond swimming is still unqualified"))
	var lane_qualified := fixture(source(),qualifying_proof(["lane_following"]))
	check("Lane-following qualification is an explicit finite conclusion",Study.read(lane_qualified).get("conclusion","")=="lane_following_qualifies")
	var both_qualified := fixture(source(),qualifying_proof(["lane_following","heading_only"]))
	check("Both-qualified remains distinct from a ranking tie or navigation success",Study.read(both_qualified).get("conclusion","")=="both_qualify" and Study.text(both_qualified).contains("Both interfaces met the finite interface criteria") and Study.text(both_qualified).contains("Continuous pond swimming is still unqualified"))
	check("Missing optional disclosure makes no claim",Study.read("res://test-fixtures/TESTONLY-absent-interface.json").is_empty())
	check("Proof must bind copied native bytes",Study.read(fixture(source(),proof(),true)).has("issue"))
	for field in ["qualified","runtime_adopted","training"]:
		var changed := source();changed[field]=true
		check("Comparison cannot set "+field,Study.read(fixture(changed,proof())).has("issue"))
	for field in ["PPO_updates","optimizer_steps"]:
		var changed := source();changed[field]=1
		check("Frozen comparison cannot claim "+field,Study.read(fixture(changed,proof())).has("issue"))
	var verified := proof();verified.weights_unchanged=false
	check("Unverified saved weights cannot claim unchanged motor",Study.read(fixture(source(),verified)).has("issue"))
	verified=proof();verified.audit_passed=false
	check("Failed audit cannot make measured claims",Study.read(fixture(source(),verified)).has("issue"))
	verified=proof();verified.recorded_controls=36479
	check("Counter mismatch is rejected",Study.read(fixture(source(),verified)).has("issue"))
	verified=proof();verified.cases[1].case=0
	check("Duplicate case identity is rejected",Study.read(fixture(source(),verified)).has("issue"))
	verified=proof();verified.cases[1].arm="lane_following"
	check("Swapped interface identity is rejected",Study.read(fixture(source(),verified)).has("issue"))
	verified=proof();verified.cases[1].scenario=3
	check("Case scenario disagreement is rejected",Study.read(fixture(source(),verified)).has("issue"))
	verified=proof();verified.cases.pop_back()
	check("Missing cases cannot claim complete comparison",Study.read(fixture(source(),verified)).has("issue"))
	verified=proof();verified.arm_summary[0].nonzero_passes=40
	check("Arm aggregate must agree with individual cases",Study.read(fixture(source(),verified)).has("issue"))
	verified=proof();verified.selection.adequate_arms=["heading_only"]
	check("A relative winner cannot claim full adequacy",Study.read(fixture(source(),verified)).has("issue"))
	verified=proof();verified.selection.preferred_arm="TESTONLY-unknown"
	check("Unknown preference is rejected",Study.read(fixture(source(),verified)).has("issue"))
	verified=proof();verified.selection.preferred_arm="lane_following"
	check("Selection must follow the preregistered ranking",Study.read(fixture(source(),verified)).has("issue"))
	var altered: Dictionary=Saved.dictionary_file(valid);altered.source.sha256="0".repeat(64)
	write_json("res://test-fixtures/TESTONLY-wrong-interface-hash.json",altered)
	check("Changed portable source bytes are rejected",Study.read("res://test-fixtures/TESTONLY-wrong-interface-hash.json").has("issue"))
	var changed := source();changed.closed=false
	check("An open result cannot make final claims",Study.read(fixture(changed,proof())).has("issue"))
	changed=source();changed.hard_fault="TESTONLY hard stop";changed.probe_completed_cases=1;changed.cases_started=2;changed.controls=93980;changed.attempted_substeps=939800;changed.accepted_substeps=939800
	verified=proof();verified.audit_status="verified_hard_stop";verified.full_comparison_completed=false;verified.recorded_controls=93980;verified.recorded_attempted_substeps=939800;verified.recorded_accepted_substeps=939800;verified.conclusion="comparison_blocked"
	verified.cases=[verified.cases[0]]
	var partial := fixture(changed,verified)
	check("Verified stopped prefix is readable",Study.read(partial).get("verified",false)==true)
	message=Study.text(partial)
	check("Stopped prefix cannot rank an interface",message.contains("stopped with 1 / 88") and not message.contains("ranked higher") and not message.contains("88 trajectories in 44"))
	changed=source();changed.hard_fault="Matched-command physics disagreement";changed.retention_passed=false;changed.precheck_completed_cases=0;changed.probe_completed_cases=0;changed.cases_started=0;changed.controls=214;changed.attempted_substeps=2140;changed.accepted_substeps=2140
	verified=proof();verified.audit_status="verified_hard_stop";verified.full_comparison_completed=false;verified.recorded_controls=214;verified.recorded_attempted_substeps=2140;verified.recorded_accepted_substeps=2140;verified.cases=[];verified.arm_summary=[];verified.retention={"cpu":retention(false),"cuda":retention(false)};verified.original_swim_rest_retained=false;verified.conclusion="comparison_blocked"
	var stopped_gate := fixture(changed,verified)
	message=Study.text(stopped_gate)
	check("Verified prerequisite stop is readable",Study.read(stopped_gate).get("verified",false)==true)
	check("Prerequisite failure explicitly leaves every arm untested",message.contains("before either interface was tested") and message.contains("0 / 88") and message.contains("CPU / CUDA") and not message.contains("Lane following:") and not message.contains("ranked higher"))
	check("Recorded gate work remains explicit",message.contains("Recorded controls: 214") and message.contains("substeps: 2140"))
	check("A blocked gate does not claim retained skills were measured",message.contains("Original-skill checks on CPU: not run") and message.contains("Original-skill checks on CUDA: not run") and not message.contains("checks passed on the corrected"))
	var bad_stop := verified.duplicate(true);bad_stop.conclusion="neither_qualifies"
	check("A blocked comparison cannot be relabeled as neither qualifying",Study.read(fixture(changed,bad_stop)).has("issue"))
	verified.audit_passed=false
	check("An unverified prerequisite stop cannot claim an audited result",Study.read(fixture(changed,verified)).has("issue"))
	changed=source();changed.retention_passed=false;changed.comparison_blocked_reason="retention_regression";changed.cases_started=0;changed.probe_completed_cases=0;changed.controls=93480;changed.attempted_substeps=934800;changed.accepted_substeps=934800
	verified=proof();verified.audit_status="verified_retention_stop";verified.full_comparison_completed=false;verified.conclusion="comparison_blocked";verified.recorded_controls=93480;verified.recorded_attempted_substeps=934800;verified.recorded_accepted_substeps=934800;verified.cases=[];verified.arm_summary=[];verified.original_swim_rest_retained=false;verified.retention.cpu.passed=false;verified.retention.cpu.totals.short_passes=3
	var retention_block := fixture(changed,verified)
	check("A completed retention regression is distinguished from a hard physics stop",Study.read(retention_block).get("retention_stop",false)==true)
	message=Study.text(retention_block)
	check("Retention regression blocks interface tests without inventing a winner",message.contains("CPU / CUDA agreement check passed") and message.contains("original-skill retention did not") and message.contains("Conclusion: comparison blocked") and not message.contains("ranked higher"))
	var actual_path := "res://research_data/interface-comparison-v2-study.json"
	if FileAccess.file_exists(actual_path):
		var actual := Study.read(actual_path)
		check("Actual comparison is independently bound",actual.get("verified",false)==true)
		check("Actual comparison cannot install or train a motor",actual.has("result") and actual.result.runtime_adopted==false and actual.result.PPO_updates==0 and actual.result.optimizer_steps==0)
	var report := {"passed":failures.is_empty(),"checks":checks,"failures":failures,"physical_controls":0,"native_instances":0,"GPU_kernels":0,"optimizer_steps":0}
	if not output.is_empty(): FileAccess.open(output,FileAccess.WRITE).store_string(JSON.stringify(report,"  "))
	print(JSON.stringify(report))
	quit(0 if failures.is_empty() else 3)

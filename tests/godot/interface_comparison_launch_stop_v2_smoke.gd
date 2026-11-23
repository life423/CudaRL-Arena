extends SceneTree
## Portable launch-stop evidence checks only; no native instance is created.
const Study = preload("res://interface_comparison_launch_stop_v2.gd")
const Saved = preload("res://continuous_study.gd")
const SOURCE := "res://research_data/interface-comparison-v2-launch-stop.json"
var checks: Array[Dictionary]=[]
var failures: Array[String]=[]
var output := ""
var original: Dictionary

func check(label: String, passed: bool) -> void:
	checks.append({"check":label,"passed":passed})
	if not passed: failures.append(label)

func write_json(path: String, value: Dictionary) -> String:
	FileAccess.open(path,FileAccess.WRITE).store_string(JSON.stringify(value))
	return FileAccess.get_sha256(path)

func fixture(values: Dictionary) -> String:
	var wrapper := original.duplicate(true)
	var proof: Dictionary=values.source
	for key in ["process","arithmetic"]:
		var path: String="user://TESTONLY-launch-stop-"+key+".json"
		var digest := write_json(path,values[key])
		wrapper[key].path=path
		wrapper[key].sha256=digest
		proof.input_sha256[wrapper[key].verification_input]=digest
		if key=="process":
			proof.process_execution=values.process.duplicate(true)
			proof.process_sha256=digest
		else: proof.arithmetic_preparation_sha256=digest
	var source_path := "user://TESTONLY-launch-stop-source.json"
	wrapper.source.path=source_path
	wrapper.source.sha256=write_json(source_path,proof)
	for pair in [["launch_stop_verification_sha256","source"],["process_execution_sha256","process"],["arithmetic_preparation_sha256","arithmetic"]]:
		values.protection[pair[0]]=wrapper[pair[1]].sha256
	var protected_path := "user://TESTONLY-launch-stop-protection.json"
	wrapper.protection.path=protected_path
	wrapper.protection.sha256=write_json(protected_path,values.protection)
	var wrapper_path := "user://TESTONLY-launch-stop.json"
	write_json(wrapper_path,wrapper)
	return wrapper_path

func _initialize() -> void:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--test-output="): output=argument.trim_prefix("--test-output=")
	original=Saved.dictionary_file(SOURCE)
	var actual := Study.read(SOURCE)
	check("Actual launch-stop evidence is byte-bound and independently verified",actual.get("verified",false)==true)
	var text := Study.text(SOURCE)
	check("Arithmetic preparation is distinct from a dynamic agreement gate",text.contains("Fixed-state CPU / CUDA arithmetic checks passed") and text.contains("fresh agreement gate") and text.contains("were not run"))
	check("No physical run or main case is invented",text.contains("Physical controls: 0") and text.contains("0 / 88"))
	check("Retained skills remain unassessed",actual.get("proof",{}).get("original_swim_rest_retained","missing")==null)
	check("No winner, learning, adoption or continuous ability is claimed",text.contains("No interface preference") and text.contains("No learning updates") and text.contains("no policy was installed") and text.contains("Continuous pond swimming is still unqualified"))
	check("The authorized attempt is visibly closed",text.contains("authorized attempt is closed"))
	var documents := {}
	for key in ["source","process","arithmetic","protection"]: documents[key]=Saved.bound_file(original.get(key,{}))
	check("Relocated TESTONLY evidence retains valid provenance",Study.read(fixture(documents.duplicate(true))).get("verified",false)==true)
	var mutations := [
		["source","audit_passed",false],["source","weights_unchanged",false],
		["source","conclusion","neither_qualifies"],["source","native_result",{}],
		["source","original_swim_rest_retained",false],["source","recorded_controls",1],
		["source","PPO_updates",1],["source","qualified",true],["source","runtime_adopted",true],
		["source","full_comparison_completed",true],["source","unrun_cases",[]],
		["process","exit_code",0],["process","terminated_at_cap",true],
		["arithmetic","passed",false],["arithmetic","integration_calls",1],
		["arithmetic","arithmetic_profile","other"],["protection","passed",false]]
	for change in mutations:
		var changed := documents.duplicate(true)
		changed[change[0]][change[1]]=change[2]
		check("Reject changed "+str(change[0])+"."+str(change[1]),Study.read(fixture(changed)).has("issue"))
	var changed := documents.duplicate(true)
	changed.source.retention.cuda.passed=false
	check("Unrun retention cannot be presented as a measured failure",Study.read(fixture(changed)).has("issue"))
	var path := fixture(documents.duplicate(true))
	var wrapper := Saved.dictionary_file(path)
	wrapper.source.sha256="0".repeat(64)
	write_json(path,wrapper)
	check("Substituted portable source bytes are rejected",Study.read(path).has("issue"))
	check("Absent optional disclosure is silent",Study.read("user://TESTONLY-no-such-launch-stop.json").is_empty())
	var report := {"passed":failures.is_empty(),"checks":checks,"failures":failures,"physical_controls":0,"physical_substeps":0,"native_instances":0,"GPU_kernels":0,"optimizer_steps":0}
	if not output.is_empty(): FileAccess.open(output,FileAccess.WRITE).store_string(JSON.stringify(report,"  "))
	print(JSON.stringify(report))
	quit(0 if failures.is_empty() else 3)

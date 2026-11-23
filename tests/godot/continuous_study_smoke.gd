extends SceneTree
## Disclosure/provenance tests only; no native object or physical world exists.
const Study = preload("res://continuous_study.gd")
var checks: Array[Dictionary] = []
var failures: Array[String] = []
var output := ""
var wrapper: Dictionary
var result: Dictionary
var proof: Dictionary

func check(message: String, good: bool) -> void:
	checks.append({"check":message,"passed":good})
	if not good: failures.append(message)

func write_dictionary(path: String, value: Dictionary) -> void:
	FileAccess.open(path,FileAccess.WRITE).store_string(JSON.stringify(value))

func fixture(name: String, source_changes: Dictionary={}, proof_changes: Dictionary={}) -> String:
	var declared := wrapper.duplicate(true)
	var source := result.duplicate(true)
	var verified := proof.duplicate(true)
	for key in source_changes: source[key]=source_changes[key]
	for key in proof_changes: verified[key]=proof_changes[key]
	var source_path := "res://test-fixtures/TESTONLY-"+name+"-result.json"
	write_dictionary(source_path,source)
	declared.source={"path":source_path,"sha256":FileAccess.get_sha256(source_path)}
	if not proof_changes.has("assessment_sha256"): verified.assessment_sha256=declared.source.sha256
	var proof_path := "res://test-fixtures/TESTONLY-"+name+"-proof.json"
	write_dictionary(proof_path,verified)
	declared.verification={"path":proof_path,"sha256":FileAccess.get_sha256(proof_path)}
	var path := "res://test-fixtures/TESTONLY-"+name+".json"
	write_dictionary(path,declared)
	return path

func _initialize() -> void:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--test-output="): output=argument.trim_prefix("--test-output=")
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path("res://test-fixtures"))
	wrapper=Study.dictionary_file("res://research_data/continuous-study.json")
	result=Study.bound_file(wrapper.get("source",{}))
	proof=Study.bound_file(wrapper.get("verification",{}))
	var actual: Dictionary=Study.read("res://research_data/continuous-study.json")
	check("Actual audited rejection is readable",actual.has("result") and bool(actual.get("verified",false)))
	check("Native result is rejected after 64 updates",result.get("qualified",true)==false and result.get("PPO_updates",-1)==64 and result.get("selection_passes",-1)==0)
	var text: String=Study.text("res://research_data/continuous-study.json")
	check("Unrun long validation is explicit",text.contains("Thirty-minute validation was not run") and not text.contains("Thirty-minute cases: 0"))
	check("Slowing is not presented as success",text.contains("slowed and stalled") and text.contains("not ready"))
	check("Earlier retained skills are reported",text.contains("all 14 holding cases passed again"))
	check("Missing optional study makes no claim",Study.read("res://test-fixtures/absent.json").is_empty())
	var changed := wrapper.duplicate(true)
	changed.source.sha256="0".repeat(64)
	write_dictionary("res://test-fixtures/TESTONLY-bad-source.json",changed)
	check("Wrong source bytes cannot make a claim",Study.read("res://test-fixtures/TESTONLY-bad-source.json").has("issue"))
	changed=wrapper.duplicate(true)
	changed.verification.sha256="0".repeat(64)
	write_dictionary("res://test-fixtures/TESTONLY-bad-proof.json",changed)
	check("Wrong proof bytes cannot make a claim",Study.read("res://test-fixtures/TESTONLY-bad-proof.json").has("issue"))
	check("Proof must bind this exact assessment",Study.read(fixture("wrong-binding",{},{"assessment_sha256":"0".repeat(64)})).has("issue"))
	check("Proof and source qualification must agree",Study.read(fixture("wrong-qualification",{},{"qualified":true})).has("issue"))
	check("Impossible update count is rejected",Study.read(fixture("too-many-updates",{"PPO_updates":65})).has("issue"))
	check("Unclosed data cannot claim completion",Study.read(fixture("not-closed",{"closed":false})).has("issue"))
	check("Different study schemas are rejected",Study.read(fixture("other-study",{"study":"current_swim_v2"})).has("issue"))
	check("Partial retention cannot claim all retained",not Study.text(fixture("partial-retention",{"recovery_passes":2})).contains("all 14 holding cases passed again"))
	var report := {"passed":failures.is_empty(),"checks":checks,"failures":failures,"physical_controls":0,"optimizer_steps":0}
	if not output.is_empty(): FileAccess.open(output,FileAccess.WRITE).store_string(JSON.stringify(report,"  "))
	print(JSON.stringify(report))
	quit(0 if failures.is_empty() else 3)

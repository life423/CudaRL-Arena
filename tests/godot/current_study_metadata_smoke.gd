extends SceneTree
## Parser-only checks. No scene _ready, native object, rendering or physical step.
## Fixtures are synthetic and stay under ignored build/test-only-fixtures.
var failures: Array[String] = []
var checks: Array[Dictionary] = []
var output := ""

func _initialize() -> void:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--test-output="): output=argument.trim_prefix("--test-output=")
	_run.call_deferred()

func check(message: String, passed: bool) -> void:
	checks.append({"check":message,"passed":passed})
	if not passed: failures.append(message)

func write_json(path: String, value: Variant) -> void:
	var file:=FileAccess.open(path,FileAccess.WRITE)
	if file==null:
		failures.append("Cannot write test-only fixture")
		return
	file.store_string(JSON.stringify(value))

func bound(path: String) -> Dictionary:
	return {"path":path,"sha256":FileAccess.get_sha256(path)}

func _run() -> void:
	if output.is_empty():
		push_error("Use --test-output with an ignored build directory")
		quit(2)
		return
	var fixture_dir:=output.get_base_dir().path_join("test-only-fixtures")
	DirAccess.make_dir_recursive_absolute(fixture_dir)
	var source_path:=fixture_dir.path_join("TEST-ONLY-native-result.json")
	var metadata_path:=fixture_dir.path_join("TEST-ONLY-current-study.json")
	var proof_path:=fixture_dir.path_join("TEST-ONLY-verification.json")
	var view: Node=load("res://main.gd").new()
	check("Parser checks never instantiate native physics",view.preview==null and view.physical_substeps==0)
	var source: Dictionary={"test_fixture":true,"closed":true,"qualified":false,"fault":"","gate":"TEST_ONLY_closed_unqualified",
		"controls":160,"substeps":1600,"PPO_updates":0,"actor_steps":0,
		"baseline_current_passes":0,"final_current_passes":0,"withheld_passes":0,"endurance_passes":0}
	write_json(source_path,source)
	var declared: Dictionary={"schema":1,"test_fixture":true,"source":bound(source_path)}
	write_json(metadata_path,declared)
	var result: Dictionary=view.call("_load_current_study",metadata_path)
	check("Valid closed native result is available",result.has("result") and not bool(result.get("verified",true)))
	var missing: Dictionary=view.call("_load_current_study",fixture_dir.path_join("not-created.json"))
	check("Missing metadata adds no study status",missing.is_empty())
	var bad:=declared.duplicate(true)
	bad.source.sha256="0".repeat(64)
	write_json(metadata_path,bad)
	result=view.call("_load_current_study",metadata_path)
	check("Hash mismatch cannot expose a result",not result.has("result") and result.has("issue"))
	var malformed:=FileAccess.open(metadata_path,FileAccess.WRITE)
	malformed.store_string("{TEST-ONLY malformed JSON")
	malformed=null
	result=view.call("_load_current_study",metadata_path)
	check("Malformed metadata is rejected without a success claim",not result.has("result") and result.has("issue"))
	bad=declared.duplicate(true)
	bad.schema={"not":"a schema number"}
	write_json(metadata_path,bad)
	result=view.call("_load_current_study",metadata_path)
	check("Malformed schema type is safely rejected",not result.has("result"))
	source.closed=false
	write_json(source_path,source)
	declared.source=bound(source_path)
	write_json(metadata_path,declared)
	result=view.call("_load_current_study",metadata_path)
	check("Open results cannot masquerade as a closed study",not result.has("result"))
	source.closed=true
	source.qualified=true
	source.gate="TEST_ONLY_qualified"
	write_json(source_path,source)
	declared.source=bound(source_path)
	write_json(metadata_path,declared)
	result=view.call("_load_current_study",metadata_path)
	check("Native qualification alone is not independent verification",result.has("result") and not bool(result.get("verified",true)))
	write_json(proof_path,{"test_fixture":true,"passed":true,"input_sha256":{source_path:FileAccess.get_sha256(source_path)}})
	declared.verification=bound(proof_path)
	write_json(metadata_path,declared)
	result=view.call("_load_current_study",metadata_path)
	check("Verification binds the exact native result",result.has("result") and bool(result.get("verified",false)))
	var relocated_path:=fixture_dir.path_join("TEST-ONLY-relocated-result.json")
	write_json(relocated_path,source)
	var audit_identity := "Z:\\TEST-ONLY-ARCHIVE\\closed-study\\result.json"
	write_json(proof_path,{"test_fixture":true,"passed":true,"input_sha256":{audit_identity:FileAccess.get_sha256(source_path)}})
	var relocated:=declared.duplicate(true)
	relocated.source=bound(relocated_path)
	relocated.source.verification_input="z:/test-only-archive/closed-study/result.json"
	relocated.verification=bound(proof_path)
	write_json(metadata_path,relocated)
	result=view.call("_load_current_study",metadata_path)
	check("A relocated exact-byte result retains its original audit identity",result.has("result") and bool(result.get("verified",false)))
	var changed_source:=source.duplicate(true)
	changed_source.controls=159
	changed_source.substeps=1590
	write_json(relocated_path,changed_source)
	relocated.source=bound(relocated_path)
	relocated.source.verification_input=audit_identity
	write_json(metadata_path,relocated)
	result=view.call("_load_current_study",metadata_path)
	check("An audit alias cannot verify a different result SHA",result.has("result") and not bool(result.get("verified",true)))
	relocated.source.verification_input={"not":"a literal identity"}
	write_json(metadata_path,relocated)
	result=view.call("_load_current_study",metadata_path)
	check("A malformed audit identity fails closed",not result.has("result"))
	write_json(proof_path,{"test_fixture":true,"passed":true,"input_sha256":{source_path:"0".repeat(64)}})
	declared.verification=bound(proof_path)
	write_json(metadata_path,declared)
	result=view.call("_load_current_study",metadata_path)
	check("Unbound verification cannot qualify the result",result.has("result") and not bool(result.get("verified",true)))
	source.erase("baseline_current_passes")
	source.erase("final_current_passes")
	write_json(source_path,source)
	declared.source=bound(source_path)
	write_json(metadata_path,declared)
	result=view.call("_load_current_study",metadata_path)
	check("A different study family cannot claim current swimming",not result.has("result"))
	source={"test_fixture":true,"study":"current_swim_v1","phase":"physics_gate_failed","closed":true,"qualified":false,
		"fault":"TEST_ONLY_physics_comparison","gate":"TEST_ONLY_physics_gate_failed","controls":2634,"substeps":26340,
		"PPO_updates":0,"actor_steps":0,"training_started":false,"current_suites_run":false,
		"baseline_current_passes":null,"final_current_passes":null,"withheld_passes":null,"endurance_passes":null}
	write_json(source_path,source)
	declared={"schema":1,"test_fixture":true,"source":bound(source_path)}
	write_json(metadata_path,declared)
	result=view.call("_load_current_study",metadata_path)
	check("A closed failed physics gate is distinct from behavior testing",result.has("result") and result.result.final_current_passes==null)
	source.training_started={"not":"a boolean"}
	write_json(source_path,source)
	declared.source=bound(source_path)
	write_json(metadata_path,declared)
	result=view.call("_load_current_study",metadata_path)
	check("Malformed phase flags cannot invent a stopped pretraining study",not result.has("result"))
	source.training_started=false
	write_json(source_path,source)
	declared.source=bound(source_path)
	write_json(metadata_path,declared)
	var disclosure:=RichTextLabel.new()
	disclosure.bbcode_enabled=true
	view.drawer_text=disclosure
	view.current_study_path=metadata_path
	view.call("_append_current_study")
	var disclosure_text:=disclosure.get_parsed_text()
	check("Failed gate does not invent zero of sixteen behavior results",disclosure_text.contains("Physics check stopped") and disclosure_text.contains("No PPO updates") and not disclosure_text.contains("/ 16"))
	source={"test_fixture":true,"study":"current_swim_v2","phase":"bounded_updates_exhausted","closed":true,"qualified":false,
		"fault":"","gate":"TEST_ONLY_closed_32_updates","controls":89566320,"substeps":895663200,
		"PPO_updates":32,"actor_steps":344,"training_started":true,"current_suites_run":true,
		"baseline_current_passes":4,"final_current_passes":8,"withheld_passes":null,"endurance_passes":null,
		"withheld_suites_run":false,"endurance_suites_run":false}
	write_json(source_path,source)
	declared={"schema":1,"test_fixture":true,"source":bound(source_path)}
	write_json(metadata_path,declared)
	result=view.call("_load_current_study",metadata_path)
	check("Partial current learning is distinct from unrun validation",result.has("result") and result.result.final_current_passes==8 and result.result.withheld_passes==null)
	disclosure.clear()
	view.call("_append_current_study")
	disclosure_text=disclosure.get_parsed_text()
	check("Partial study shows baseline and final result",disclosure_text.contains("4 → 8 / 16") and disclosure_text.contains("Native PPO updates: 32"))
	check("Partial study does not show unrun tests as failures",not disclosure_text.contains("Withheld cases:") and not disclosure_text.contains("Endurance cases:"))
	check("Partial study cannot claim current readiness",disclosure_text.contains("Current swimming is not ready") and disclosure_text.contains("saved calm-water policy"))
	source.withheld_passes=0
	write_json(source_path,source)
	declared.source=bound(source_path)
	write_json(metadata_path,declared)
	result=view.call("_load_current_study",metadata_path)
	check("An unrun suite cannot have an invented zero-pass score",not result.has("result"))
	source.withheld_passes=null
	source.endurance_suites_run="false"
	write_json(source_path,source)
	declared.source=bound(source_path)
	write_json(metadata_path,declared)
	result=view.call("_load_current_study",metadata_path)
	check("Malformed suite state cannot expose a result",not result.has("result"))
	source.endurance_suites_run=false
	source.qualified=true
	write_json(source_path,source)
	declared.source=bound(source_path)
	write_json(metadata_path,declared)
	result=view.call("_load_current_study",metadata_path)
	check("Unrun validation cannot qualify current swimming",not result.has("result"))
	disclosure.free()
	check("All parser checks used zero native controls",view.preview==null and view.physical_substeps==0)
	view.free()
	var report: Dictionary={"passed":failures.is_empty(),"checks":checks,"failures":failures,
		"physical_controls":0,"physical_substeps":0,"native_physics_instances":0,"fixtures":"Synthetic test-only fixtures"}
	write_json(output,report)
	print(JSON.stringify(report))
	quit(0 if failures.is_empty() else 3)

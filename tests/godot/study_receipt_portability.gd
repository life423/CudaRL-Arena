extends SceneTree
## Recorded evidence must remain readable without a DLL or private build folder.
## The frontend is never attached to the tree: no world, actor or optimizer runs.
var output := ""

func _initialize() -> void:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--receipt-output="): output=argument.trim_prefix("--receipt-output=")
	var view: Node=load("res://main.gd").new()
	var receipt: Dictionary=view._dictionary_file(view.CANDIDATE_STUDY_PATH)
	var portable := int(receipt.get("schema",0))==2
	for reference in receipt.get("sources",{}).values():
		portable=portable and reference is Dictionary and str(reference.get("path","")).begins_with(view.STUDY_EVIDENCE_PREFIX) and not str(reference.get("path","")).contains("..")
	var statement: String=view._candidate_study_status()
	var recorded: String=view._recorded_runtime_verdicts()
	var bad_binding: Dictionary=view._dictionary_file(view.SHORELINE_BINDING_PATH).duplicate(true)
	bad_binding.candidate_source.sha256="0".repeat(64)
	var checks := {
		"only_portable_evidence_paths":portable,
		"recorded_candidate_visible":statement.contains("Recorded candidate") and not statement.contains("unavailable"),
		"actual_update_count_visible":statement.contains("%d PPO updates" % int(receipt.get("completed_updates",-1))),
		"recorded_CPU_and_CUDA_visible":recorded.contains("Recorded CPU physical checks: 36 / 36 passed") and recorded.contains("Recorded CUDA agreement: 8 / 8 matched cases"),
		"changed_receipt_rejected":view._candidate_study_status(bad_binding).contains("Candidate assessment is unavailable"),
		"original_package_present":FileAccess.get_sha256(view.policy_path)=="4b6400c4db65873bd346c3e0c4fed384e3a77f9e5174d0dcea5d74073b94616d",
		"no_native_module_required":not FileAccess.file_exists(view.NATIVE_DLL_PATH),
		"no_world_or_actor_started":not view.loaded and view.preview==null and view.physical_substeps==0
	}
	var passed := true
	for value in checks.values(): passed=passed and bool(value)
	var report := {"schema":1,"passed":passed,"checks":checks,"study_id":receipt.get("study_id",""),"statement":statement,"recorded_verdicts":recorded,"native_DLL_present":FileAccess.file_exists(view.NATIVE_DLL_PATH),"physical_controls":0,"physical_substeps":0,"optimizer_steps":0,"CUDA_kernels":0,"limitation":"A portable record of measured results and artifact digests; this does not execute or independently requalify a candidate or a local native build."}
	if not output.is_empty():
		var file:=FileAccess.open(output,FileAccess.WRITE)
		if file!=null: file.store_string(JSON.stringify(report,"  ")); file.flush()
	view.free()
	print(JSON.stringify(report))
	quit(0 if passed else 2)

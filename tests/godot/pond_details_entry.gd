extends SceneTree

var result_path := ""
var expected_study := ""

func _initialize() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--entry-result="): result_path=arg.trim_prefix("--entry-result=")
		if arg.begins_with("--expected-study="): expected_study=arg.trim_prefix("--expected-study=")
	call_deferred("check_entry")

func check_entry() -> void:
	var view=load("res://main.tscn").instantiate()
	root.add_child(view)
	view.set_process(false)
	await process_frame
	view.task.select(1)
	view._restart()
	var hidden_on_pond: bool = not view.steering.is_visible_in_tree()
	view.details_button.pressed.emit()
	await process_frame
	var statement: String = view._candidate_study_status()
	var published: Dictionary = view._dictionary_file(view.CANDIDATE_STUDY_PATH)
	var assessment: Dictionary = view._dictionary_file(published.sources.evaluation.path)
	var body: Dictionary = view.snapshot.body.duplicate(true)
	view.turn_left.pressed.emit()
	var checks := {
		"real_project_loaded": view.loaded and view.shoreline_qualified and view.cuda_qualified,
		"original_motor": FileAccess.get_sha256(view.policy_path)=="4b6400c4db65873bd346c3e0c4fed384e3a77f9e5174d0dcea5d74073b94616d",
		"expected_study": not expected_study.is_empty() and published.study_id==expected_study and statement.begins_with(str(published.study_label)),
		"matched_assessment": int(assessment.update)==int(published.completed_updates) and view.drawer_text.get_parsed_text().contains(statement),
		"heading_test_only_in_Swim_Details": hidden_on_pond and view.drawer.is_ancestor_of(view.turn_left) and view.turn_right.is_visible_in_tree(),
		"heading_confirmed": view.steering_status.text.contains("+5°") and int(view.snapshot.requested_turn_degrees)==5,
		"body_held": view.snapshot.body==body and view.physical_substeps==0 and int(view.snapshot.controls)==0,
		"research_candidate_separate": not published.candidate_adopted
	}
	var ok := true
	for passed in checks.values(): ok=ok and bool(passed)
	var receipt := {"passed":ok,"checks":checks,"active_motor_336_sha256":FileAccess.get_sha256(view.policy_path),"actual_Godot_project":ProjectSettings.globalize_path("res://"),"recorded_study_id":published.study_id,"assessment_updates":assessment.update,"turns_passed":assessment.turn_cases_passed,"long_retained":assessment.retained_parent_long_window_passes,"rests":assessment.rest_cases_passed,"short_swims":assessment.short_swim_cases_passed,"details_text":statement,"optimizer_calls":0,"physical_controls":view.snapshot.controls}
	if not result_path.is_empty():
		var f:=FileAccess.open(result_path,FileAccess.WRITE)
		f.store_string(JSON.stringify(receipt,"  "))
		f.close()
	print(JSON.stringify(receipt))
	quit(0 if ok else 2)
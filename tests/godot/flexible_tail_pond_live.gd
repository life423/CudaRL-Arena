extends SceneTree

func _initialize() -> void:
	call_deferred("_run")

func _run() -> void:
	var trace_path := ""
	var report_path := ""
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--trace="): trace_path = arg.trim_prefix("--trace=")
		if arg.begins_with("--report="): report_path = arg.trim_prefix("--report=")
	var live: Object = ClassDB.instantiate(&"FlexibleTailLive")
	if live == null:
		push_error("Native live extension unavailable")
		quit(1)
		return
	var data := ProjectSettings.globalize_path("res://live_data/")
	if not live.load_autonomous(data + "fitted-gait.bin", data + "exploration-motor-88.bin", data + "exploration-chooser-88.bin", data + "straight-evaluation-starts.bin"):
		push_error(live.get_error())
		quit(1)
		return
	var trace := FileAccess.open(trace_path, FileAccess.READ)
	if trace == null:
		push_error("Expected native qualification trace")
		quit(1)
		return
	trace.get_csv_line()
	var controls := 0
	var compared := 0
	var max_com_error := 0.0
	var max_speed_error := 0.0
	var decisions := 0
	while not trace.eof_reached():
		var row := trace.get_csv_line()
		if row.size() < 18 or row[12] != "macro": continue
		var expected_control := int(row[0])
		while controls < expected_control:
			if not bool(live.step_control()):
				push_error("Godot runtime stopped: " + String(live.get_error()))
				quit(1)
				return
			controls += 4
		var state: Dictionary = live.snapshot()
		var expected := Vector3(float(row[2]), float(row[3]), float(row[4]))
		var actual: Vector3 = state.com_m
		max_com_error = maxf(max_com_error, actual.distance_to(expected))
		max_speed_error = maxf(max_speed_error, absf(float(state.requested_speed_mm_s) - float(row[14])))
		decisions = int(state.chooser_decisions)
		compared += 1
	var state: Dictionary = live.snapshot()
	var passed := compared > 1000 and controls <= 15000 and max_com_error <= 0.000001 and max_speed_error <= 0.0001 and int(state.fault) == 0
	var report := {"pass": passed, "compared_states": compared, "physics_controls": controls, "physics_substeps": controls * 16, "max_COM_error_m": max_com_error, "max_requested_speed_error_mm_s": max_speed_error, "chooser_decisions": decisions, "resets_during_run": 0, "optimizer_updates": 0, "final_state": {"time_s": state.time_s, "speed_mm_s": state.rolling_speed_mm_s, "visited_cells": state.visited_cells, "fault": state.fault}}
	if not report_path.is_empty():
		FileAccess.open(report_path, FileAccess.WRITE).store_string(JSON.stringify(report, "\t"))
	print(JSON.stringify(report))
	quit(0 if passed else 1)

extends SceneTree

var failures: Array[String] = []
var attempted := 0
var completed := 0
var observed_substeps := 0
var max_pose_error := 0.0
var report_path := ""

func _initialize() -> void:
	call_deferred("_run")

func check(ok: bool, message: String) -> void:
	if not ok:
		failures.append(message)

func _run() -> void:
	report_path = ProjectSettings.globalize_path("res://../build/ui-native-v2-2026-11-22/verification.json")
	if FileAccess.file_exists(report_path):
		push_error("Native UI verification already attempted; do not rerun")
		quit(2)
		return
	FileAccess.open(report_path, FileAccess.WRITE).store_string('{"status":"started","controls_cap":80,"substeps_cap":800}')
	if not ClassDB.class_exists("HoldLabPreview"):
		GDExtensionManager.load_extension("res://addons/frozen_swimmer/frozen_swimmer.gdextension")
	check(ClassDB.class_exists("HoldLabPreview"), "Native HoldLabPreview class missing")
	if not failures.is_empty():
		_finish()
		return
	var preview: RefCounted = ClassDB.instantiate("HoldLabPreview")
	check(bool(preview.call("load_controller", "res://research_data/learned-amplitude-v1.bin")), "Research checkpoint load failed")
	var state: Dictionary = preview.call("get_state")
	check(state.get("loaded", false) and state.get("paused", false), "Load must leave preview paused")
	check(not state.get("training", true) and int(state.get("controls", -1)) == 0, "Load must not train or step")
	check(int(state.get("holding_updates", 0)) == 7 and state.get("qualified_candidate", false), "Wrong latest research metadata")
	check(state.get("policy_method", "") == "learned_persistent_amplitude" and int(state.get("package_version", 0)) == 2, "Wrong versioned learned method")
	check(absf(float(state.get("deterministic_amplitude", 0)) - 0.26211655139923096) < 0.0000002, "Wrong learned amplitude")
	if not failures.is_empty():
		_finish()
		return
	var poses := _read_poses()
	for holding in [true, false]:
		check(bool(preview.call("reset", 0, holding)), "Native explicit reset failed")
		check(bool(preview.call("start")), "Native preview start failed")
		var category: String = "hold" if holding else "short"
		for k in range(1, 41):
			attempted += 1
			var advanced: bool = preview.call("step")
			state = preview.call("get_state")
			check(advanced, "Native physical step failed: " + str(state.get("error", "")))
			if not advanced:
				observed_substeps += int(state.get("substeps", 0))
				preview.call("pause")
				_finish()
				return
			completed += 1
			var key := category + ":" + str(k)
			check(poses.has(key), "Missing recorded comparison pose: " + key)
			if poses.has(key):
				var expected: Array = poses[key]
				var keys := ["head_x", "head_y", "head_z", "qw", "qx", "qy", "qz", "j0", "j1", "j2", "j3"]
				for i in keys.size():
					max_pose_error = maxf(max_pose_error, absf(float(state[keys[i]]) - float(expected[i])))
		preview.call("pause")
		state = preview.call("get_state")
		observed_substeps += int(state.get("substeps", 0))
		check(state.get("paused", false), "Pause did not stop playback")
	check(max_pose_error <= 0.000002, "Live and recorded native poses differ")
	check(attempted == 80 and completed == 80 and observed_substeps == 800, "Native viewer control accounting mismatch")
	var original := FileAccess.get_file_as_bytes("res://research_data/learned-amplitude-v1.bin")
	original[original.size() - 1] = original[original.size() - 1] ^ 1
	var bad_path := ProjectSettings.globalize_path("res://../build/ui-native-v2-2026-11-22/bad-checksum.bin")
	FileAccess.open(bad_path, FileAccess.WRITE).store_buffer(original)
	check(not bool(preview.call("load_controller", bad_path)), "Corrupt checkpoint unexpectedly accepted")
	state = preview.call("get_state")
	check(not state.get("loaded", true) and state.get("paused", false), "Failed package load left old policy live")
	check(not bool(preview.call("load_controller", "res://research_data/does-not-exist.bin")), "Missing checkpoint unexpectedly accepted")
	_finish()

func _read_poses() -> Dictionary:
	var path := ProjectSettings.globalize_path("res://../build/ppo/learned-amplitude-2026-11-22-v1/training-started/assessment-7/all-cpu-poses.csv")
	var file := FileAccess.open(path, FileAccess.READ)
	var result: Dictionary = {}
	check(file != null, "Native comparison poses unavailable")
	if file == null:
		return result
	var header := file.get_csv_line()
	while not file.eof_reached():
		var row := file.get_csv_line()
		if row.size() != header.size() or row[1] != "0" or row[0] not in ["hold", "short"]:
			continue
		var values: Array = []
		for name in ["position_x_model", "position_y_model", "position_z_model", "quat_w", "quat_x", "quat_y", "quat_z", "angle0", "angle1", "angle2", "angle3"]:
			values.append(float(row[header.find(name)]))
		result[row[0] + ":" + row[2]] = values
	return result

func _finish() -> void:
	var result := {"passed": failures.is_empty(), "failures": failures,
		"attempted_controls": attempted, "completed_controls": completed,
		"physical_substeps": observed_substeps, "maximum_pose_error_model_units": max_pose_error,
		"training_updates": 0, "optimizer_steps": 0,
		"scope": "Native viewer fidelity, separate from the closed research budget"}
	FileAccess.open(report_path, FileAccess.WRITE).store_string(JSON.stringify(result, "  "))
	print(JSON.stringify(result))
	quit(0 if failures.is_empty() else 1)

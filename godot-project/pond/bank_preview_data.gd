extends "res://pond/current_replay_data.gd"

func _init() -> void:
	expected_case_count = 2

func open(path: String) -> bool:
	var loaded := super.open(path)
	if not loaded:
		error = "Recovery recording is missing or invalid. Generate it with the documented bank-preview recording command."
	return loaded

func get_stats() -> Dictionary:
	var stats := super.get_stats()
	if not data.is_empty():
		var item: Dictionary = data.cases[case_index]
		for key in ["loaded_update", "saved_update", "selector_update", "update", "best_update"]:
			stats[key] = int(item.selector_update)
		stats.policy_update = int(item.motor_update)
	stats.compute_detail = "Recorded CUDA policy comparison. Both policies are frozen; this viewer does not train."
	return stats

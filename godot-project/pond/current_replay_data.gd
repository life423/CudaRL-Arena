extends RefCounted

var data: Dictionary = {}
var case_index := 0
var index := 0
var error := ""
var paused := false
var expected_case_count := 10

func open(path: String) -> bool:
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		error = "Replay is missing. Run the documented frozen-policy current diagnosis first."
		return false
	var parsed: Variant = JSON.parse_string(file.get_as_text())
	if not parsed is Dictionary or parsed.get("version", 0) != 1 or parsed.get("cases", []).size() != expected_case_count:
		error = "Unsupported current replay."
		return false
	data = parsed
	return true

func frames() -> Array:
	return data.cases[case_index].frames if not data.is_empty() else []

func select_case(value: int) -> void:
	case_index = value
	index = 0

func seek(seconds: float) -> void:
	var origin: float = frames()[0].time
	index = clampi(roundi((seconds - origin) * 20), 0, frames().size() - 1)

func get_state() -> Dictionary:
	if data.is_empty():
		return {}
	var raw: Dictionary = frames()[index]
	var result := raw.duplicate()
	result.position = _vector(raw.position)
	result.orientation = _quaternion(raw.orientation)
	result.foods = []
	for food in raw.foods:
		result.foods.append(_vector(food))
	result.food = result.foods[0]
	result.links = []
	for link in raw.links:
		result.links.append({"position": _vector(link.position), "orientation": _quaternion(link.orientation)})
	result.elapsed = raw.time
	result.seconds_since_food = raw.since_food
	result.depth_cm = -result.position.y * 4
	result.selected_food = int(raw.target)
	result.ready = true
	return result

func _vector(value: Array) -> Vector3:
	return Vector3(value[0], value[1], value[2])

func _quaternion(value: Array) -> Quaternion:
	return Quaternion(value[0], value[1], value[2], value[3])

func step() -> Dictionary:
	if not paused:
		index = mini(index + 1, frames().size() - 1)
		if index == frames().size() - 1:
			paused = true
	return get_state()

func get_stats() -> Dictionary:
	return {"playback_mode": true, "validated_controller": true, "showcase_untrained": false,
		"training_ready": not data.is_empty(), "paused": true, "resumed": false, "loaded_update": 250,
		"saved_update": 250, "transferred": false, "food_count": 3, "policy_inputs": 33, "watching_best": true,
		"save_error": "", "policy_update": 64350, "selector_update": 250, "decision_steps": 20, "update": 250,
		"env_steps": 0, "update_ms": 0.0, "training_seconds": 0.0, "background_swimmers": 0,
		"current_food_per_minute": 0.0, "best_food_per_minute": 0.0, "best_update": 250,
		"compute_backend": "Recorded CUDA", "compute_detail": "Recorded from the original 64-world GPU evaluator; this viewer does not train or simulate."}

func get_evaluation_history(_from := 0) -> Array:
	return []
func get_error() -> String:
	return error
func stop() -> void:
	pass
func set_paused(value: bool) -> void:
	paused = value
func set_watch_best(_value: bool) -> void:
	pass
func set_updates_per_second(_value: float) -> void:
	pass
func set_untrained(_value: bool) -> bool:
	return false

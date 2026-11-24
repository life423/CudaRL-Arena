extends SceneTree

var motor: Object

func _initialize() -> void:
	call_deferred("_run")

func fail(message: String) -> void:
	push_error(message)
	quit(1)

func drive_through_turn(direction: int, count: int) -> bool:
	if not motor.call("request_turn", direction):
		return false
	var phase: String = "turn_left" if direction > 0 else "turn_right"
	var started := false
	var last_time := float(motor.call("snapshot")["time_s"])
	for control in range(1150):
		if not motor.call("step_control"):
			return false
		var row: Dictionary = motor.call("snapshot")
		if float(row["time_s"]) <= last_time or float(row["clearance_mm"]) <= 0:
			return false
		last_time = float(row["time_s"])
		if row["motion"] == phase and int(row["turn_count"]) == count:
			started = true
		if started and row["motion"] == "swim":
			print("direction=%d turn_count=%d time_s=%.2f speed_mm_s=%.2f tilt_deg=%.4f depth_error_mm=%.4f" % [direction, count, last_time, row["rolling_speed_mm_s"], row["tilt_deg"], row["depth_error_mm"]])
			return float(row["rolling_speed_mm_s"]) >= 6 and float(row["tilt_deg"]) <= 10 and float(row["depth_error_mm"]) <= 1
	return false

func _run() -> void:
	var scene: Node = load("res://flexible_tail_preview.tscn").instantiate()
	root.add_child(scene)
	motor = scene.get("live_motor")
	if motor == null or float(motor.call("snapshot")["time_s"]) != 0.0:
		fail("Expected the project to load its live frozen motor at rest.")
		return
	if not drive_through_turn(1, 1) or not drive_through_turn(-1, 2):
		fail("Live Godot motor did not complete both user-request transitions.")
		return
	print("Verified two accepted requests on one live Godot body. No resets or training.")
	quit()


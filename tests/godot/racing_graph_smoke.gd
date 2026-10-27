extends SceneTree

var lab: Object
var started_ms := 0

func _initialize() -> void:
	Engine.max_fps = 120
	started_ms = Time.get_ticks_msec()
	lab = ClassDB.instantiate("RacingLab")
	if lab == null or not lab.start(128, 1, 8, 0.0):
		push_error("RacingLab startup failed")
		quit(1)
		return
	lab.set_paused(false)

func _process(_delta: float) -> bool:
	if lab == null:
		return false
	var error: String = lab.get_error()
	if not error.is_empty():
		push_error(error)
		lab.stop()
		quit(1)
		return false
	var stats: Dictionary = lab.get_stats()
	if stats.get("update", 0) >= 12:
		var cars: PackedFloat32Array = lab.step_showcase()
		var good: bool = cars.size() == 8 * lab.get_car_stride()
		print("GRAPH_WORKER_SMOKE updates=%d cars=%d error=%s" % [stats.get("update", 0), cars.size() / lab.get_car_stride(), error])
		lab.stop()
		lab = null
		quit(0 if good else 1)
	elif Time.get_ticks_msec() - started_ms > 15000:
		push_error("RacingLab graph worker timed out")
		lab.stop()
		quit(1)
	return false

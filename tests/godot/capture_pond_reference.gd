extends SceneTree
func _initialize() -> void:
	_run.call_deferred()
func _run() -> void:
	var lab := DeepPondLab.new()
	if not lab.watch_controller("res://pond/policies/deep_selector.bin"):
		push_error(lab.get_error())
		quit(1)
		return
	# Capture before rebuilding the extension, to pin the prior CPU controller.
	if lab.get_stats().has("compute_backend"):
		push_error("Reference capture requires the pre-portability extension")
		quit(1)
		return
	var csv := FileAccess.open("res://../build/portable-checks/reference.csv", FileAccess.WRITE)
	csv.store_line("tick,x,y,z,qx,qy,qz,qw,eaten,target")
	for tick in 1200:
		var state: Dictionary = lab.step()
		if tick < 80:
			var p: Vector3 = state.position
			var q: Quaternion = state.orientation
			csv.store_line("%d,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%d,%d" % [tick + 1, p.x, p.y, p.z, q.x, q.y, q.z, q.w, state.eaten, state.selected_food])
	print("LEGACY_CPU_REFERENCE foods=%d steps=1200" % lab.get_state().eaten)
	lab.stop()
	quit()

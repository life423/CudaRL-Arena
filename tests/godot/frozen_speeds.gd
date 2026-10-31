extends SceneTree
# Opt-in bridge verification of the existing accepted cases, never training.
# Ten four-second swims: exactly 800 controls / 8,000 native substeps maximum.
func _initialize() -> void:
	_verify.call_deferred()

func _verify() -> void:
	var started := Time.get_ticks_msec()
	GDExtensionManager.load_extension("res://addons/frozen_swimmer/frozen_swimmer.gdextension")
	var native = ClassDB.instantiate("FrozenSwimmer")
	var controls := 0
	var substeps := 0
	var results: Array = []
	if native == null or not native.load_controller():
		push_error("Accepted actor failed to load")
		quit(3)
		return
	var reference: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://../assets/steady-speed-v1/evaluation.json"))
	for speed_index in range(5):
		var speed_sum := 0.0
		for pose in [0,3]:
			if not native.set_speed(float(speed_index+4)/1000) or not native.reset(pose) or not native.start():
				push_error("Could not initialize constant request")
				quit(3)
				return
			for tick in range(80):
				if Time.get_ticks_msec()-started > 20000 or controls >= 800 or substeps+10 > 8000:
					native.pause()
					print(JSON.stringify({"error":"Verification budget exhausted","controls":controls,"substeps":substeps}))
					quit(3)
					return
				var advanced: bool = native.step()
				controls += 1
				substeps += 10
				if not advanced:
					push_error(native.get_error())
					quit(3)
					return
			var state: Dictionary = native.get_state()
			if not bool(state.passed) or not bool(state.paused) or not bool(state.finished) or int(state.substeps) != 800:
				push_error("Frozen bridge case failed acceptance or auto-pause")
				quit(3)
				return
			speed_sum += float(state.steady_speed_m_s)*1000
		var measured := speed_sum/2
		if absf(measured-float(reference.after[speed_index].speed_mm_s)) > 0.001:
			push_error("Bridge speed differs from accepted native recording")
			quit(3)
			return
		results.append({"request_mm_s":speed_index+4,"speed_mm_s":measured,"cases_passed":2})
	print(JSON.stringify({"passed":10,"controls":controls,"physical_substeps":substeps,"training":false,"results":results,"wall_seconds":float(Time.get_ticks_msec()-started)/1000}))
	quit()

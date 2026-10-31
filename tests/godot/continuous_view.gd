extends SceneTree
# One opt-in 8mm/s viewing smoke: <=600 controls / 6000 attempted substeps,
# 30 simulated seconds and 20 native wall seconds. No retries or training.
func _initialize() -> void:
	_verify.call_deferred()

func _verify() -> void:
	var started := Time.get_ticks_msec()
	GDExtensionManager.load_extension("res://addons/frozen_swimmer/frozen_swimmer.gdextension")
	var native = ClassDB.instantiate("FrozenSwimmer")
	if native == null or not native.load_controller():
		push_error("Continuous smoke actor load failed")
		quit(3)
		return
	native.set_continuous(true)
	if not native.start():
		quit(3)
		return
	var first_benchmark := -1.0
	var saw_reset := false
	for tick in range(600):
		var before: Dictionary = native.get_state()
		if Time.get_ticks_msec()-started > 20000 or int(before.substeps)+10 > 6000:
			native.pause()
			print(JSON.stringify({"error":"Continuous smoke budget exhausted","state":before}))
			quit(3)
			return
		var advanced: bool = native.step()
		var state: Dictionary = native.get_state()
		if not advanced or bool(state.paused) or bool(state.finished):
			print(JSON.stringify({"error":"Continuous viewing stopped","controls":state.controls,"substeps":state.substeps,"reason":state.error}))
			quit(3)
			return
		if tick == 79:
			if not bool(state.benchmark_complete) or not bool(state.passed):
				quit(3)
				return
			first_benchmark = float(state.steady_speed_m_s)
		if int(state.loops) == 0 and tick > 79 and (float(state.segment_time_s) != 4.0 or float(state.steady_speed_m_s) != first_benchmark):
			push_error("Longer viewing altered the frozen four-second benchmark")
			quit(3)
			return
		if int(state.loops) > 0: saw_reset = true
	var final: Dictionary = native.get_state()
	native.pause()
	if native.step() or not saw_reset or int(final.substeps) != 6000 or int(final.controls) != 600:
		push_error("Pause, explicit edge loop or accounting failed")
		quit(3)
		return
	if not native.set_speed(0.004) or not bool(native.get_state().paused) or int(native.get_state().controls) != 0:
		push_error("Speed selection did not reset and pause")
		quit(3)
		return
	print(JSON.stringify({"continuous_smoke_passed":true,"controls":final.controls,"physical_substeps":final.substeps,"edge_resets":final.loops,"simulated_seconds":final.time_s,"first_four_second_steady_mm_s":first_benchmark*1000,"wall_seconds":float(Time.get_ticks_msec()-started)/1000,"training":false}))
	quit()

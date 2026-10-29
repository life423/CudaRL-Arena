extends SceneTree

func _initialize() -> void:
	run.call_deferred()

func run() -> void:
	var source := "res://../build/arch-current/preserved-20261029104613/arch-v1-training.bin"
	var output := ProjectSettings.globalize_path("res://../build/arch-current/frozen-probe")
	var cove := "--cove" in OS.get_cmdline_user_args()
	if cove:
		source = "res://../build/cove-navigation/preserved-20261029113418/arch-v2-training.bin"
		output = ProjectSettings.globalize_path("res://../build/cove-navigation/frozen-probe")
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--arch-v1-source="):
			source = argument.trim_prefix("--arch-v1-source=")
	DirAccess.make_dir_recursive_absolute(output)
	var source_hash := FileAccess.get_sha256(source)
	var rows := []
	var failures := 0
	for speed in [0.0, 0.5, 0.75]:
		var copied := output.path_join("probe-%d-%d.bin" % [OS.get_process_id(), int(speed * 100)])
		if not cove and DirAccess.copy_absolute(ProjectSettings.globalize_path(source), copied) != OK:
			push_error("Could not protect frozen probe source")
			quit(1)
			return
		var lab = DeepPondLab.new()
		var started: bool = lab.start_cove_learning(source, copied, 1024, "cuda") if cove else lab.start_arch_learning("res://../build/arch-current/missing-source.bin", copied, 1024, "cuda")
		if not started:
			push_error(lab.get_error())
			quit(1)
			return
		lab.set_paused(true)
		if not lab.set_steady_current(speed):
			push_error(lab.get_error())
			quit(1)
			return
		var start: Dictionary = lab.get_state()
		var actor_update: int = lab.get_stats().adopted_update
		var start_stats: Dictionary = lab.get_stats()
		var anchor: Vector3 = start.position
		var still := 0
		var max_still := 0
		var last_food := 0.0
		var max_gap := 0.0
		var last_eaten: int = start.eaten
		var final: Dictionary = start
		var finite := true
		var visited_cove := false
		var cove_foods := 0
		for tick in 1200:
			var previous: Dictionary = final
			final = lab.step()
			finite = finite and (final.position as Vector3).is_finite()
			var p: Vector3 = final.position
			visited_cove = visited_cove or p.x * p.x / (6.9 * 6.9) + p.z * p.z / (5.25 * 5.25) > 1.0
			if int(final.eaten) > int(previous.eaten):
				for index in previous.pellet_ids.size():
					if previous.pellet_ids[index] not in final.pellet_ids:
						var food: Vector3 = previous.foods[index]
						if food.x * food.x / (6.9 * 6.9) + food.z * food.z / (5.25 * 5.25) > 1.0:
							cove_foods += 1
			var time := (tick + 1) * 0.05
			if int(final.eaten) != last_eaten:
				max_gap = maxf(max_gap, time - last_food)
				last_food = time
				last_eaten = final.eaten
			if (final.position as Vector3).distance_to(anchor) > 0.1:
				anchor = final.position
				still = 0
			else:
				still += 1
				max_still = maxi(max_still, still)
			if tick % 100 == 0:
				await process_frame
		max_gap = maxf(max_gap, 60.0 - last_food)
		var final_stats: Dictionary = lab.get_stats()
		var passed := finite and max_still < 100 and int(final.eaten) > int(start.eaten) and int(final_stats.adopted_update) == actor_update
		failures += 0 if passed else 1
		rows.append({"speed_cm_s":speed,"seconds":60,"foods":int(final.eaten)-int(start.eaten),"cove_visited":visited_cove,"cove_foods_collected":cove_foods,"max_immobility_seconds":max_still*0.05,"max_food_gap_seconds":max_gap,"contact_steps":int(final_stats.arch_contacts)-int(start_stats.arch_contacts),"passages":int(final_stats.arch_passages)-int(start_stats.arch_passages),"actor_update":actor_update,"finite":finite,"passed":passed})
		lab.stop()
	var unchanged := FileAccess.get_sha256(source) == source_hash
	if not unchanged:
		failures += 1
	var record := {"cove_mode":cove,"source_sha256":source_hash,"source_unchanged":unchanged,"frozen_inference":true,"training_paused":true,"failures":failures,"cases":rows}
	var file := FileAccess.open(output.path_join("summary.json"), FileAccess.WRITE)
	file.store_string(JSON.stringify(record, "\t"))
	file.close()
	print("ARCH_CURRENT_FROZEN_PROBE ",JSON.stringify(record))
	quit(0 if failures == 0 else 1)

extends SceneTree

var checks := 0
var failures := 0

func check(ok: bool, message: String) -> void:
	checks += 1
	if not ok:
		failures += 1
		push_error(message)

func _initialize() -> void:
	call_deferred("run")

func run() -> void:
	var base := ProjectSettings.globalize_path("res://../build/predator-shelter/")
	var folder := base.path_join("readiness-smoke")
	DirAccess.make_dir_recursive_absolute(folder)
	var source := base.path_join("source-boulder.bin")
	var motor := base.path_join("motor-v2-training.bin")
	var parent_hash := FileAccess.get_sha256(source)
	var motor_hash := FileAccess.get_sha256(motor)
	check(not parent_hash.is_empty() and not motor_hash.is_empty(), "Protected research sources must exist")
	check(DirAccess.copy_absolute(motor, folder.path_join("motor-training.bin")) == OK, "Copy only the private motor trial")
	var node: Node = load("res://pond/predator_pond.tscn").instantiate()
	node.source_boulder_checkpoint = source
	node.predator_checkpoint = folder.path_join("training.bin")
	node.population = 16
	root.add_child(node)
	await process_frame
	check(not node.lab.get_error().is_empty(), "The failed holding trial must not be deployed")
	check("18 of 24" in node.lab.get_error(), "Reject on real motor readiness rather than an unrelated startup failure")
	check(node.predator_button.disabled, "Unavailable predator action is disabled")
	check(node.pause_button.disabled, "An absent learner cannot pretend to be running")
	check(node.title_status.text == "Shelter practice is not ready", "Readable readiness feedback")
	check(node.lab.get_stats().get("learning_active", false) == false, "No background goal PPO started with the unready body")
	check(FileAccess.get_sha256(source) == parent_hash, "Complete original boulder source unchanged")
	check(FileAccess.get_sha256(motor) == motor_hash, "Failed trial remains unchanged")
	check(not FileAccess.file_exists(folder.path_join("training.bin")), "Do not create a misleading approved shelter save")
	node.queue_free()
	await process_frame
	print("PREDATOR_READINESS_SMOKE %d/%d passed" % [checks - failures, checks])
	quit(1 if failures else 0)

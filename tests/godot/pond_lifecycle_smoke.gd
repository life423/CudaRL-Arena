extends SceneTree

# Exercise the real feeding UI/lifecycle with a fake session. No native backend,
# physics, learning or save is started, including in the error cases.
class FakeSession extends RefCounted:
	var starts := 0
	var stops := 0
	var accept_start := true
	var error := ""
	var last_current := 0.0
	var start_arguments: Array = []
	func start_feeding_learning(policy: String, checkpoint: String, population: int, backend: String) -> bool:
		starts += 1
		start_arguments = [policy, checkpoint, population, backend]
		error = "The test save is already open." if not accept_start else ""
		return accept_start
	func get_error() -> String:
		return error
	func get_state() -> Dictionary:
		return {}
	func get_stats() -> Dictionary:
		return {"feeding_mode": true, "loaded_update": 73}
	func set_steady_current(speed: float, _drift: bool) -> bool:
		last_current = speed
		return true
	func stop() -> void:
		stops += 1

var checks := 0
var failures := 0

func check(ok: bool, message: String) -> void:
	checks += 1
	if not ok:
		failures += 1
		push_error(message)

func _initialize() -> void:
	run.call_deferred()

func check_failed_ui(view: Control) -> void:
	check(not view.startup_succeeded and not view.is_processing() and not view.is_physics_processing(), "A failed session must stop both update loops")
	check(not view.viewport_container.visible and not view.curve.visible, "Failure must show an explanation rather than a frozen empty pond")
	check(not (view.find_child("BackToDemos", true, false) as Button).disabled, "The player must be able to return to the menu")
	check((view.find_child("DropFoodAtCenter", true, false) as Button).disabled and view.pause_button.disabled and view.auto_feed_button.disabled, "Actions requiring a live session must be disabled")
	check(not view.startup_error.is_empty() and view.title_status.text == "Pond unavailable", "Failure needs a specific visible reason")

func run() -> void:
	await process_frame
	# Autoload globals are available only after project startup. Loading the
	# probe here tests that real context, rather than standalone script parsing.
	var probe: Script = load("res://../tests/godot/pond_lifecycle_probe.gd")
	if probe == null or not probe.can_instantiate():
		push_error("Feeding lifecycle probe could not compile")
		quit(1)
		return
	var factory = load("res://shared/native_lab_factory.gd")
	var missing: Dictionary = factory.create("MissingPondModuleForLifecycleCheck")
	check(missing.instance == null and not missing.error.is_empty(), "Missing native classes must fail without parse-time construction")
	var incompatible: Dictionary = factory.create("Node")
	check(incompatible.instance == null and incompatible.error.contains("lifetime"), "Node ownership must not be silently cast to a session")
	var missing_methods: Array[String] = ["missing_session_method"]
	var outdated: Dictionary = factory.create("Resource", missing_methods)
	check(outdated.instance == null and outdated.error.contains("missing_session_method"), "Required API must be checked before construction")
	var available_methods: Array[String] = ["get_instance_id"]
	var available: Dictionary = factory.create("Resource", available_methods)
	check(available.instance is RefCounted and available.error.is_empty(), "Compatible modules must still construct")

	for failure in ["none", "missing", "prepare", "start"]:
		var session := FakeSession.new()
		session.accept_start = failure != "start"
		var view: Control = probe.new()
		view.session = session
		view.missing_module = failure == "missing"
		view.accept_prepare = failure != "prepare"
		view.size = Vector2(1280, 720)
		root.add_child(view)
		if failure == "none":
			check(view.calls == ["create", "prepare", "ui", "world", "start", "after", "poll"], "One scene entry must create, start and poll once, in order")
			check(view.startup_succeeded and session.starts == 1, "A valid session must start once")
			check(view.stats.get("loaded_update") == 73, "Mode setup must receive the restored native stats")
			check(session.start_arguments[1] == "user://pond-policies/feeding-learning-v1/training.bin", "Feeding save namespace must remain unchanged")
			check(view.current_switch.toggled.get_connections().size() == 1 and view.current_strength.item_selected.get_connections().size() == 1, "Current controls must connect once")
			view.current_switch.button_pressed = true
			check(is_equal_approx(session.last_current, 0.5), "The native session, not rendering, must apply current")
		else:
			check_failed_ui(view)
			check(not view.calls.has("after") and not view.calls.has("poll"), "Failed startup must not enter normal mode updates")
			check(session.starts == (1 if failure == "start" else 0), "Failed preparation must never open a learner")
		view.queue_free()
		await process_frame
		check(session.stops == (0 if failure == "missing" else 1), "Each owned session must stop exactly once on exit")
	print("POND_LIFECYCLE_SMOKE checks=%d failures=%d native_sessions=0 physical_steps=0 training_updates=0 save_operations=0" % [checks, failures])
	quit(1 if failures else 0)

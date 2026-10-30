extends "res://pond/feeding_pond.gd"

var session: RefCounted
var calls: Array[String] = []
var missing_module := false
var accept_prepare := true

func _create_lab() -> RefCounted:
	calls.append("create")
	if missing_module:
		return _create_native_lab("MissingPondModuleForLifecycleCheck")
	return session

func _prepare_lab() -> bool:
	calls.append("prepare")
	if not accept_prepare:
		startup_error = "The test habitat geometry is unavailable."
	return accept_prepare

func _build_ui() -> void:
	calls.append("ui")
	super._build_ui()

func _build_world() -> void:
	calls.append("world")
	super._build_world()

func _start_lab() -> bool:
	calls.append("start")
	return super._start_lab()

func _after_lab_started() -> void:
	calls.append("after")
	super._after_lab_started()

func _poll() -> void:
	calls.append("poll")

func _process(_delta: float) -> void:
	pass

func _physics_process(_delta: float) -> void:
	pass

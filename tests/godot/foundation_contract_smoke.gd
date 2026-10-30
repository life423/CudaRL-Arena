extends SceneTree

# Frozen API stubs only: no native world construction, physical steps or files.
class StubView:
	extends "res://main.gd"
	var backend: Variant
	func _create_lab() -> Variant:
		return backend

class StaleBackend:
	extends RefCounted
	func get_state() -> Dictionary:
		return {}

class Backend:
	extends RefCounted
	var metadata: Variant = {
		"version": 1, "actuator_count": 4,
		"bounds_min_model": Vector3(-6, -5, -4), "bounds_max_model": Vector3(8, 1, 3),
		"joint_limit_rad": 0.8, "torque_limit_uNm": 7.0,
		"control_seconds": 0.04, "substep_seconds": 0.01
	}
	var snapshot: Variant = {
		"finite": true, "unsafe": false, "paused": true, "learned": false,
		"contact_gate_passed": false, "pending_commands_clipped": false, "physics_version": 2, "mode": 1,
		"elapsed": 0.0, "steps": 0, "substeps": 0, "current_x": 0.0, "current_vector_m_s": Vector3.ZERO,
		"position_cm": Vector3.ZERO, "heading_degrees": 0.0,
		"head_speed_cm_s": 0.0, "com_speed_cm_s": 0.0, "floor_gap_mm": 1.0,
		"floor_spring_force_mN": 0.0, "contact_samples": 0, "first_unsafe_substep": -1,
		"commands": PackedFloat32Array([0, 0, 0, 0]),
		"last_raw_commands": PackedFloat32Array([0, 0, 0, 0]),
		"actual_torques_uNm": PackedFloat32Array([0, 0, 0, 0]),
		"joint_angles_rad": PackedFloat32Array([0, 0, 0, 0]),
		"joint_rates_rad_s": PackedFloat32Array([0, 0, 0, 0]), "links": []
	}
	var commands_calls := 0
	var current_calls := 0
	var save_queries := 0
	var save_calls := 0
	var load_calls := 0
	var step_calls := 0
	var reset_keeps_unsafe := false
	var error := ""
	func _init() -> void:
		for index in range(3):
			snapshot.links.append({"position": Vector3(index, 0, 0), "orientation": Quaternion.IDENTITY, "length_model": 0.5, "half_height_model": 0.16, "half_width_model": 0.2})
	func get_config() -> Variant:
		return metadata
	func get_state() -> Variant:
		return snapshot
	func set_commands(values: PackedFloat32Array) -> bool:
		commands_calls += 1
		snapshot.commands = values
		return true
	func set_mode(value: int) -> bool:
		snapshot.mode = value
		snapshot.paused = true
		snapshot.commands = PackedFloat32Array([0, 0, 0, 0])
		return true
	func set_current(value: float) -> bool:
		current_calls += 1
		snapshot.current_x = value
		snapshot.current_vector_m_s = Vector3(value, 0, 0)
		return true
	func set_paused(value: bool) -> void:
		error = ""
		if not value and snapshot.unsafe:
			error = "Unsafe frozen diagnostic fixture."
			return
		snapshot.paused = value
	func step() -> Dictionary:
		step_calls += 1
		return snapshot
	func single_step() -> Dictionary:
		step_calls += 1
		return snapshot
	func reset_world() -> void:
		snapshot.paused = true
		snapshot.unsafe = reset_keeps_unsafe
	func save_state() -> bool:
		save_calls += 1
		return true
	func load_state() -> bool:
		load_calls += 1
		snapshot.paused = true
		return true
	func has_saved_state() -> bool:
		save_queries += 1
		return true
	func get_error() -> String:
		return error

var failures: Array[String] = []
var checks := 0

func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures.append(message)
		push_error(message)

func _initialize() -> void:
	call_deferred("run")

func create_view(backend: Variant) -> Control:
	var view := StubView.new()
	view.backend = backend
	root.add_child(view)
	view.set_physics_process(false)
	view.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	return view

func rejected(backend: Variant, text: String) -> void:
	var view = create_view(backend)
	check(view.lab == null and view.state.is_empty(), "Rejected startup must have no active backend: " + text)
	check(view.camera == null and view.creature == null and not view.viewport_container.visible, "Rejected startup must not display a synthetic physical world: " + text)
	check(view.run_button.disabled and view.step_button.disabled and view.save_button.disabled and view.load_button.disabled, "Rejected startup must disable actions: " + text)
	check(text in view.status.text, "Rejected startup must explain the incompatibility: " + text)
	view.free()

func run() -> void:
	# Missing extension / null construction share the same explicit factory failure.
	rejected(null, "unavailable")
	rejected(StaleBackend.new(), "missing get_config")
	var bad_config := Backend.new()
	bad_config.metadata.version = 9
	rejected(bad_config, "configuration version")
	var bad_bounds := Backend.new()
	bad_bounds.metadata.bounds_max_model = bad_bounds.metadata.bounds_min_model
	rejected(bad_bounds, "positive volume")
	var bad_state := Backend.new()
	bad_state.snapshot.commands = PackedFloat32Array([0, 0, 0])
	rejected(bad_state, "actuator array commands")
	var missing_field := Backend.new()
	missing_field.snapshot.erase("floor_spring_force_mN")
	rejected(missing_field, "floor_spring_force_mN")
	var wrong_type := Backend.new()
	wrong_type.snapshot.position_cm = "not a vector"
	rejected(wrong_type, "physical position")
	var invalid_backend := Backend.new()
	invalid_backend.snapshot.finite = false
	invalid_backend.snapshot.links = []
	invalid_backend.snapshot.position_cm = Vector3(NAN, 0, 0)
	invalid_backend.snapshot.actual_torques_uNm[0] = NAN
	var invalid_view = create_view(invalid_backend)
	check(invalid_view.lab == invalid_backend and invalid_view.camera != null, "A flagged nonfinite physical result must not be misclassified as an outdated API.")
	check(not invalid_view.reset_button.disabled and invalid_view.run_button.disabled and invalid_view.step_button.disabled, "A flagged physical result must retain Reset while disabling movement.")
	check("invalid" in invalid_view.status.text and not invalid_view.creature.visible, "Invalid physical geometry must be hidden with authoritative failure feedback.")
	invalid_view.free()
	var backend := Backend.new()
	var view = create_view(backend)
	await process_frame
	await process_frame
	check(view.lab == backend and view.camera != null, "A complete frozen contract must initialize the view.")
	check(backend.commands_calls == 0, "Startup snapshot synchronization must not submit commands.")
	check(view.config == backend.metadata, "The view must use the backend's versioned configuration.")
	var floor: MeshInstance3D = view.viewport.find_child("TankFloor", true, false)
	var lower: Vector3 = backend.metadata.bounds_min_model
	var upper: Vector3 = backend.metadata.bounds_max_model
	check(floor != null, "The physical view must expose its tank floor.")
	if floor == null:
		view.free()
		quit(1)
		return
	var floor_mesh := floor.mesh as BoxMesh
	check(floor_mesh != null and is_equal_approx(floor_mesh.size.x, upper.x - lower.x) and is_equal_approx(floor_mesh.size.z, upper.z - lower.z), "Visible floor dimensions must come from native tank bounds.")
	check(floor.position.is_equal_approx(Vector3((lower.x + upper.x) * 0.5, lower.y - 0.02, (lower.z + upper.z) * 0.5)), "The floor must honor asymmetric native bounds.")
	var frame := Rect2(Vector2.ZERO, Vector2(view.viewport.size))
	for x in [lower.x, upper.x]:
		for y in [lower.y, upper.y]:
			for z in [lower.z, upper.z]:
				var corner := Vector3(x, y, z)
				check(not view.camera.is_position_behind(corner) and frame.has_point(view.camera.unproject_position(corner)), "The fixed camera must frame every native tank corner.")
	backend.snapshot.commands = PackedFloat32Array([0.25, -0.5, 0, 0])
	view._capture_state(backend.get_state())
	view._sync_controls()
	check(backend.commands_calls == 0 and backend.snapshot.commands[0] == 0.25, "Snapshot synchronization must remain read-only.")
	check(view.slider_values[0].text == "0.20 rad", "Target labels must use the configured joint-angle scale.")
	backend.snapshot.mode = 0
	view._capture_state(backend.get_state())
	view._sync_controls()
	check(view.slider_values[0].text == "1.8 µN·m", "Torque labels must use the configured physical torque scale.")
	var queries := backend.save_queries
	view._apply_state()
	view._apply_state()
	check(backend.save_queries == queries, "Pose refresh must not poll the filesystem.")
	# This invokes only the frozen stub, never native dynamics.
	view._toggle_run()
	view._physics_process(0.039)
	check(backend.step_calls == 0, "The configured 40ms interval must not tick at39ms.")
	view._physics_process(0.0012)
	check(backend.step_calls == 1, "The scheduler must use the configured40ms interval, not a hardcoded50ms.")
	view._toggle_run()
	backend.snapshot.current_x = 0.004
	backend.snapshot.current_vector_m_s = Vector3(0.004, 0, 0)
	view._load()
	check(view.current.item_count == 4 and view.current.get_item_text(3) == "0.4 cm/s (saved)" and view.current.is_item_disabled(3), "A continuous saved speed must have truthful read-only custom readback.")
	check(backend.current_calls == 0 and is_equal_approx(backend.snapshot.current_x, 0.004), "Custom readback must not mutate the native water current.")
	view.current.select(1)
	view.current.item_selected.emit(1)
	check(backend.current_calls == 1 and is_equal_approx(backend.snapshot.current_x, 0.005), "Selecting a normal preset must submit exactly one configured current command.")
	check(view.current.item_count == 3 and view.current_speeds.size() == 3, "A normal snapshot must remove obsolete custom readback.")
	backend.snapshot.current_x = 0.0
	backend.snapshot.current_vector_m_s = Vector3(0, 0, 0.004)
	view._load()
	check("Z 0.4" in view.current.get_item_text(3) and view.current.is_item_disabled(3), "Saved non-X flow must be displayed from its full vector, not labeled calm.")
	check(backend.current_calls == 1 and backend.snapshot.current_vector_m_s == Vector3(0, 0, 0.004), "Non-X readback must not replace the saved vector with a +X command.")
	backend.snapshot.last_raw_commands = PackedFloat32Array([2, -2, 0.25, 0])
	backend.snapshot.commands = PackedFloat32Array([1, -1, 0.25, 0])
	backend.snapshot.pending_commands_clipped = true
	view._load()
	check("Next inputs use the actuator bounds" in view.status.text, "Derived bounded next inputs must be explained on load.")
	check(backend.snapshot.last_raw_commands == PackedFloat32Array([2, -2, 0.25, 0]) and backend.commands_calls == 0, "Loaded original raw metadata must remain unchanged; snapshot sync must not submit the derived input.")
	backend.snapshot.unsafe = true
	view._load()
	check(backend.load_calls == 4 and backend.save_calls == 0, "Loaded-state checks must use the stub only, with no save writes.")
	check(view.run_button.disabled and view.step_button.disabled, "An unsafe loaded state must disable movement.")
	check("excessive-contact" in view.status.text and not "loaded and paused" in view.status.text, "Loading must not overwrite the safety failure with a success message.")
	check(backend.commands_calls == 0, "Loading must not submit slider commands back to the backend.")
	view._toggle_run()
	check("excessive-contact" in view.status.text and backend.snapshot.paused, "A refused Run must preserve the safety explanation.")
	backend.reset_keeps_unsafe = true
	view._reset()
	check("excessive-contact" in view.status.text, "An unsafe reset result must not receive a success message.")
	check(backend.step_calls == 1 and backend.save_calls == 0, "Only the one frozen scheduler callback may occur; no native steps or saves.")
	view.free()
	print("Foundation contract smoke: %d checks, %d failures; frozen stubs only, zero integration, zero files." % [checks, failures.size()])
	quit(0 if failures.is_empty() else 1)

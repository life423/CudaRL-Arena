extends SceneTree

# Run only in a disposable project with its own user-data directory. This tests
# actual native/file/UI boundaries while the physical world remains paused.
const SAVE := "user://foundation/diagnostic-state.bin"
var checks := 0
var failures := 0

func check(ok: bool, message: String) -> void:
	checks += 1
	if not ok:
		failures += 1
		push_error(message)

func _initialize() -> void:
	run.call_deferred()

func run() -> void:
	var expected := ""
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--audit-user-dir="):
			expected = argument.trim_prefix("--audit-user-dir=")
	if expected.is_empty() or not bool(ProjectSettings.get_setting("application/config/use_custom_user_dir", false)) or str(ProjectSettings.get_setting("application/config/custom_user_dir_name", "")) != expected or not OS.get_user_data_dir().replace("\\", "/").ends_with("/" + expected):
		push_error("Refusing persistence check outside the explicitly isolated audit user directory.")
		quit(1)
		return
	if FileAccess.file_exists(SAVE):
		push_error("Refusing to replace a pre-existing diagnostic state.")
		quit(1)
		return
	var packed: PackedScene = load("res://main.tscn")
	var scene = packed.instantiate()
	root.add_child(scene)
	scene.set_physics_process(false)
	await process_frame
	check(scene.lab != null and not scene.run_button.disabled, "Actual native session must initialize")
	if scene.lab == null:
		quit(1)
		return
	var initial: Dictionary = scene.lab.get_state()
	check(initial.paused and initial.steps == 0 and initial.substeps == 0, "Persistence starts with no dynamics")
	scene.mode.select(1)
	scene.mode.item_selected.emit(1)
	scene.current.select(1)
	scene.current.item_selected.emit(1)
	scene.sliders[0].value = 0.25
	scene.save_button.pressed.emit()
	check(FileAccess.file_exists(SAVE), "The Save button must write the isolated diagnostic file")
	check(scene.status.text.contains("saved"), "Successful saving needs visible feedback")
	var original := FileAccess.get_file_as_bytes(SAVE)
	check(not original.is_empty() and not scene.load_button.disabled, "A saved state must be loadable")
	scene.current.select(2)
	scene.current.item_selected.emit(2)
	scene.sliders[0].value = 0.5
	scene.load_button.pressed.emit()
	var restored: Dictionary = scene.lab.get_state()
	check(restored.paused and restored.steps == 0 and restored.substeps == 0, "Loading must not advance or resume dynamics")
	check(is_equal_approx(float(restored.current_x), 0.005) and restored.mode == 0, "Actual saved mode and water must be restored")
	check(restored.commands == PackedFloat32Array([0, 0, 0, 0]), "Physical saves restore last-applied input, excluding pending slider edits")
	check(scene.sliders[0].value == 0 and scene.current.selected == 1, "Visible controls must reflect the loaded state")
	check(restored.position_cm == initial.position_cm, "Save/load must preserve the physical pose")
	# Force a real file-write failure, then check that the existing file survives.
	var temporary := ProjectSettings.globalize_path(SAVE) + ".tmp"
	check(DirAccess.make_dir_recursive_absolute(temporary) == OK, "Prepare isolated write-failure fixture")
	scene.save_button.pressed.emit()
	check(scene.status.text.contains("Could not save"), "Native write errors must reach the UI")
	check(FileAccess.get_file_as_bytes(SAVE) == original, "Failed saving must preserve previous file bytes")
	check(DirAccess.remove_absolute(temporary) == OK, "Remove isolated failure fixture")
	# Corrupt only this test-owned file. A failed load must preserve current state.
	scene.sliders[0].value = 0.2
	var before: Dictionary = scene.lab.get_state()
	var corrupt := original.duplicate()
	corrupt[0] ^= 1
	var file := FileAccess.open(SAVE, FileAccess.WRITE)
	check(file != null, "Open test-owned corruption fixture")
	if file != null:
		file.store_buffer(corrupt)
		file.close()
	scene.load_button.pressed.emit()
	check(scene.status.text.contains("Could not load"), "Invalid file errors must reach the UI")
	check(scene.lab.get_state() == before, "Failed native load must be transactional")
	check(scene.lab.get_state().steps == 0 and scene.lab.get_state().substeps == 0, "All persistence checks must consume zero physical steps")
	check(DirAccess.remove_absolute(ProjectSettings.globalize_path(SAVE)) == OK, "Remove test-owned state file")
	scene.queue_free()
	await process_frame
	print("FOUNDATION_PERSISTENCE_SMOKE checks=%d failures=%d physics_steps=0 personal_saves_touched=0" % [checks, failures])
	quit(1 if failures else 0)

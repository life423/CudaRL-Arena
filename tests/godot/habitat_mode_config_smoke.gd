extends SceneTree

# Configuration/presentation checks only: no native lab, learner or world step.
class FakeHost extends Control:
	var selected_current := false
	var selected_strength := -1
	func _set_current(enabled: bool) -> void:
		selected_current = enabled
	func _select_strength(index: int) -> void:
		selected_strength = index

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
	var modes = load("res://pond/habitat_mode.gd")
	var expected := {
		"arch": ["start_arch_learning", "source_feeding_checkpoint", "arch_checkpoint"],
		"cove": ["start_cove_learning", "source_arch_checkpoint", "cove_checkpoint"],
		"boulder": ["start_boulder_learning", "source_cove_checkpoint", "boulder_checkpoint"],
		"overhang": ["watch_overhang", "source_boulder_checkpoint", ""]
	}
	for kind in expected:
		var path := "res://pond/modes/" + str(kind) + ".gd"
		var script: Script = load(path)
		check(script != null and script.can_instantiate(), "Candidate mode must parse: " + path)
		if not script or not script.can_instantiate():
			continue
		check(script.get_base_script().resource_path == "res://pond/habitat_pond_view.gd", "Habitat modes must be siblings on the common view")
		var view: Control = script.new()
		check(view._habitat_kind() == kind, "Candidate mode identity changed")
		var mode: Dictionary = modes.describe(kind)
		check(mode.start == expected[kind][0] and mode.source == expected[kind][1] and mode.destination == expected[kind][2], "Mode native/save mapping changed")
		check(view.get(mode.source).begins_with("user://pond-policies/"), "Mode source save namespace changed")
		if not mode.frozen:
			check(view.get(mode.destination).begins_with("user://pond-policies/"), "Mode destination save namespace changed")
		check(view.lab == null and view.state.is_empty(), "Creating a mode script must not create or start a native session")
		view.free()
	var host := FakeHost.new()
	root.add_child(host)
	var controls := VBoxContainer.new()
	host.add_child(controls)
	var camera := Button.new()
	camera.text = "Camera"
	controls.add_child(camera)
	var current = load("res://pond/pond_current_controls.gd")
	var ui: Dictionary = current.build(host, controls, camera)
	check(ui.panel.name == "WaterControls" and ui.switch.name == "GentleCurrent" and ui.strength.name == "CurrentStrength", "Named current-widget compatibility changed")
	check(ui.switch.custom_minimum_size.y >= 44 and ui.strength.custom_minimum_size.y >= 44, "Current widgets need usable action targets")
	check(ui.strength.item_count == 2 and ui.strength.get_item_text(0) == "0.5 cm/s" and ui.strength.get_item_text(1) == "0.75 cm/s", "Supported current strengths changed")
	ui.switch.button_pressed = true
	ui.strength.select(1)
	ui.strength.item_selected.emit(1)
	check(host.selected_current and host.selected_strength == 1, "Current widget signals must reach explicit host callbacks")
	var render = load("res://pond/habitat_render.gd")
	var outline: PackedVector3Array = render.outline({"shore_outline": PackedVector3Array([Vector3.ZERO, Vector3.RIGHT, Vector3.FORWARD, Vector3.ZERO])})
	check(outline.size() == 3, "Native closed shore outline must remove exactly its duplicate endpoint")
	var world := Node3D.new()
	root.add_child(world)
	var geometry := {"major_radius": 1.35, "tube_radius": 0.5, "yaw": 0.4, "center": Vector3(0, -2.25, 0), "boulder_radius": 0.8, "boulder_center": Vector3(2, -2, 1)}
	var pieces: Dictionary = render.solids(world, geometry, StandardMaterial3D.new())
	check(pieces.arch.mesh is TorusMesh and is_equal_approx(pieces.arch.mesh.inner_radius, 0.85) and is_equal_approx(pieces.arch.mesh.outer_radius, 1.85), "Arch visual dimensions changed")
	check(pieces.arch.position == geometry.center and pieces.boulder.position == geometry.boulder_center, "Solid render centers changed")
	check(pieces.boulder.mesh is SphereMesh and is_equal_approx(pieces.boulder.mesh.radius, 0.8) and is_equal_approx(pieces.boulder.mesh.height, 1.6), "Boulder visual shape changed")
	world.queue_free()
	host.queue_free()
	print("HABITAT_MODE_CONFIG_SMOKE checks=%d failures=%d native_sessions=0 physical_steps=0 training_updates=0" % [checks, failures])
	quit(1 if failures else 0)

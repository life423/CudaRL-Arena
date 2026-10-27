extends Node

const SCENES := {
	"cartpole": "res://cartpole/cartpole.tscn",
	"racing": "res://racing/racing.tscn",
	"pond": "res://pond/pond.tscn",
}

var active_demo := ""
var mode := "train"
var last_error := ""

func watching(demo: String) -> bool:
	return active_demo == demo and mode == "watch"

func open_demo(demo: String, watch: bool) -> Error:
	if not SCENES.has(demo) or (demo == "pond" and watch):
		return ERR_INVALID_PARAMETER
	active_demo = demo
	mode = "watch" if watch else "train"
	return get_tree().change_scene_to_file(SCENES[demo])

func show_menu() -> void:
	active_demo = ""
	mode = "train"
	get_tree().change_scene_to_file("res://launcher/launcher.tscn")

func read_policy(demo: String) -> PackedFloat32Array:
	last_error = ""
	if not SCENES.has(demo):
		last_error = "Unknown demonstration."
		return PackedFloat32Array()
	var file := FileAccess.open("res://launcher/policies/%s.json" % demo, FileAccess.READ)
	if file == null:
		last_error = "The saved %s policy could not be opened." % demo
		return PackedFloat32Array()
	var parsed: Variant = JSON.parse_string(file.get_as_text())
	if not parsed is Dictionary:
		last_error = "The saved policy is not valid JSON."
		return PackedFloat32Array()
	var data: Dictionary = parsed
	if data.get("version", 0) != 1 or data.get("demo", "") != demo or data.get("inputs", 0) != (4 if demo == "cartpole" else 30) or data.get("hidden", 0) != 64 or data.get("outputs", 0) != 2:
		last_error = "The saved policy does not match this demonstration."
		return PackedFloat32Array()
	var parameters: Variant = data.get("parameters", null)
	if not parameters is Array:
		last_error = "The saved policy has no model weights."
		return PackedFloat32Array()
	for value: Variant in parameters:
		if not (value is float or value is int):
			last_error = "The saved policy contains an invalid weight."
			return PackedFloat32Array()
	return PackedFloat32Array(parameters)

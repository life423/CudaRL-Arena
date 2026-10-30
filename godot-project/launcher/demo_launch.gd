extends Node

const SCENES := {
	"pond": "res://pond/pond.tscn",
	"pond_shuffle": "res://pond/shuffle_compare.tscn",
	"deep_water": "res://pond/deep_pond.tscn",
	"pond_live": "res://pond/latest_agent.tscn",
	"pond_feeding": "res://pond/feeding_pond.tscn",
	"pond_boulder": "res://pond/boulder_pond.tscn",
	"pond_overhang": "res://pond/overhang_pond.tscn",
	"pond_current_replay": "res://pond/current_replay.tscn",
	"pond_recovery": "res://pond/bank_preview.tscn",
}

var active_demo := ""
var mode := "train"
var last_error := ""

static func select_primary_pond(release_build: bool, cuda_ready: bool, has_save: bool, cpu_requested := false) -> String:
	return "pond_boulder" if not release_build and not cpu_requested and cuda_ready and has_save else "pond_feeding"

func primary_pond() -> String:
	# The local habitat work depends on an inherited full learner and CUDA.
	# Consumer exports retain their existing portable feeding experience.
	var release_build := bool(ProjectSettings.get_setting("cudarl/release_build", false))
	var cpu_requested := "--pond-backend=cpu" in OS.get_cmdline_user_args()
	var has_save := FileAccess.file_exists("user://pond-policies/boulder-navigation-v1/training.bin") or FileAccess.file_exists("user://pond-policies/cove-navigation-v1/training.bin")
	if release_build or cpu_requested or not has_save:
		return "pond_feeding"
	var cuda_ready := false
	if ClassDB.class_exists("CudaDevice") and ClassDB.class_has_method("DeepPondLab", "start_boulder_learning"):
		var device: Object = ClassDB.instantiate("CudaDevice")
		cuda_ready = device != null and device.is_available()
	return select_primary_pond(release_build, cuda_ready, has_save, cpu_requested)

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

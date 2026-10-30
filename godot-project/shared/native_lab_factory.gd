extends RefCounted

# Optional native extensions must fail at runtime with a useful message, not
# prevent unrelated scripts/scenes from parsing in the editor or CPU release.
static func create(type_name: String, required_methods: Array[String] = []) -> Dictionary:
	if not ClassDB.class_exists(type_name) or not ClassDB.can_instantiate(type_name):
		return {"instance": null, "error": "The %s simulation module is unavailable. Restart the app after rebuilding or reinstalling its native extension." % type_name}
	if not ClassDB.is_parent_class(type_name, "RefCounted"):
		return {"instance": null, "error": "The %s simulation module has an incompatible lifetime." % type_name}
	for method in required_methods:
		if not ClassDB.class_has_method(type_name, method):
			return {"instance": null, "error": "The %s simulation module is out of date (missing %s). Restart after rebuilding the matching extension." % [type_name, method]}
	var instance: RefCounted = ClassDB.instantiate(type_name) as RefCounted
	if instance == null:
		return {"instance": null, "error": "The %s simulation module could not be created." % type_name}
	return {"instance": instance, "error": ""}

static func device_description() -> String:
	var result := create("CudaDevice", ["describe"])
	return result.instance.describe() if result.instance != null else "CUDA device diagnostics are unavailable in this build."

static func disable_controls(root: Node, keep_names: Array[String] = ["BackToDemos"]) -> void:
	for child: Node in root.get_children():
		if child is BaseButton and str(child.name) not in keep_names:
			child.disabled = true
		elif child is Slider or child is SpinBox:
			child.editable = false
		disable_controls(child, keep_names)

extends "res://pond/boulder_pond.gd"

# Loaded at runtime by the standalone visual QA entry, after autoload setup.
# It builds the actual native ground/water without a learner or checkpoint.
func _load_habitat_geometry() -> Dictionary:
	var geometry: Dictionary = lab.get_overhang_geometry()
	native_outline = geometry.get("shore_outline", PackedVector3Array())
	if native_outline.size() > 1 and native_outline[0].is_equal_approx(native_outline[-1]):
		native_outline.resize(native_outline.size() - 1)
	return geometry

func _start_lab() -> bool:
	return false

func _poll() -> void:
	pass

func _physics_process(_delta: float) -> void:
	pass

func _process(_delta: float) -> void:
	pass

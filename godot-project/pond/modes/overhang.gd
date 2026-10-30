extends "res://pond/habitat_pond_view.gd"

@export var source_boulder_checkpoint := "user://pond-policies/boulder-navigation-v1/training.bin"

func _habitat_kind() -> String:
	return "overhang"

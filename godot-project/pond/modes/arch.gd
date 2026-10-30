extends "res://pond/habitat_pond_view.gd"

# Candidate entrypoint. The existing arch_pond.gd and route remain preserved
# until the shared implementation passes behavior and native-session checks.
func _habitat_kind() -> String:
	return "arch"

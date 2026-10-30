extends RefCounted

# Shared controls only. The owning session applies current through native code.
# Rendering and button state never supply body forces or learning updates.
static func build(host: Control, controls: VBoxContainer, camera_button: Button) -> Dictionary:
	var panel := VBoxContainer.new()
	panel.name = "WaterControls"
	panel.add_theme_constant_override("separation", 0)
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	panel.add_child(row)
	var toggle := CheckButton.new()
	toggle.name = "GentleCurrent"
	toggle.text = "Current"
	toggle.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	toggle.custom_minimum_size.y = 44
	toggle.add_theme_font_size_override("font_size", 15)
	toggle.add_theme_color_override("font_color", Color("eef4e9"))
	toggle.add_theme_stylebox_override("focus", camera_button.get_theme_stylebox("focus"))
	toggle.toggled.connect(Callable(host, "_set_current"))
	row.add_child(toggle)
	var strength := OptionButton.new()
	strength.name = "CurrentStrength"
	strength.custom_minimum_size = Vector2(112, 44)
	strength.add_theme_font_size_override("font_size", 14)
	for style in ["normal", "hover", "pressed", "focus"]:
		strength.add_theme_stylebox_override(style, camera_button.get_theme_stylebox(style))
	for color in ["font_color", "font_hover_color", "font_pressed_color", "font_focus_color"]:
		strength.add_theme_color_override(color, Color("eef4e9"))
	strength.add_item("0.5 cm/s")
	strength.add_item("0.75 cm/s")
	strength.item_selected.connect(Callable(host, "_select_strength"))
	row.add_child(strength)
	var status := Label.new()
	status.text = "Calm water"
	status.add_theme_font_size_override("font_size", 13)
	status.add_theme_color_override("font_color", Color("b8cbc4"))
	status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	panel.add_child(status)
	controls.add_child(panel)
	controls.move_child(panel, mini(3, controls.get_child_count() - 1))
	return {"panel": panel, "switch": toggle, "strength": strength, "status": status}

extends Camera3D

# A steady world-facing camera: physical turns never orbit the spectator. The
# subject center comes from anatomy, not a guessed food target or policy intent.
var close_up := true
var subject := Vector3(0, 0.25, 0)
var anchor := Vector3(0, 0.25, 0)
var subject_ready := false
var dimensions := Vector2(1280, 720)
const FOLLOW_DIRECTION := Vector3(0, 0.64, 0.77)
const OVERVIEW_POSITION := Vector3(0, 8.3, 8.7)
const OVERVIEW_TARGET := Vector3(0, 0.1, 0)

static func selector(initial_close_up: bool) -> OptionButton:
	var control := OptionButton.new()
	control.name = "CameraView"
	control.custom_minimum_size = Vector2(164, 44)
	control.add_item("Follow tadpole")
	control.add_item("Pond overview")
	control.select(0 if initial_close_up else 1)
	control.tooltip_text = "Camera view: follow the whole tadpole closely, or keep a stationary overview of the pond. Training continues in either view."
	for state in ["font_color", "font_hover_color", "font_pressed_color", "font_focus_color"]:
		control.add_theme_color_override(state, Color("eef4e9"))
	var popup := control.get_popup()
	popup.add_theme_constant_override("v_separation", 28)
	popup.add_theme_color_override("font_color", Color("eef4e9"))
	popup.add_theme_color_override("font_hover_color", Color("eef4e9"))
	var surface := StyleBoxFlat.new()
	surface.bg_color = Color("263631")
	surface.border_color = Color("50645c")
	surface.set_border_width_all(1)
	surface.set_corner_radius_all(5)
	surface.content_margin_left = 8
	surface.content_margin_right = 8
	popup.add_theme_stylebox_override("panel", surface)
	var hover := StyleBoxFlat.new()
	hover.bg_color = Color("354a42")
	hover.set_corner_radius_all(3)
	popup.add_theme_stylebox_override("hover", hover)
	return control

func set_close_up(enabled: bool) -> void:
	close_up = enabled
	anchor = subject
	_place()

func fit(view_size: Vector2) -> void:
	dimensions = view_size
	_place()

func reset_subject() -> void:
	subject_ready = false

func follow(center: Vector3, delta: float) -> void:
	subject = center
	if not subject_ready:
		anchor = center
		subject_ready = true
	elif close_up:
		# Frame-rate independent translation damping; no bobbing, camera shake,
		# automatic orbit or food-triggered zoom. Overview has no camera motion.
		anchor = anchor.lerp(center, 1.0 - exp(-8.0 * maxf(0, delta)))
		if anchor.distance_to(center) > 0.30:
			anchor = center + (anchor - center).normalized() * 0.30
	if close_up:
		_place()

func _place() -> void:
	var aspect := maxf(0.1, dimensions.x / maxf(1, dimensions.y))
	if close_up:
		projection = Camera3D.PROJECTION_PERSPECTIVE
		keep_aspect = Camera3D.KEEP_HEIGHT
		fov = 42.0
		# Fit the complete 1.8-unit body, membrane and a bounded follow lag,
		# including narrow/tall viewports. Tail bends never pump the zoom.
		var half_vertical := deg_to_rad(fov * 0.5)
		var half_horizontal := atan(tan(half_vertical) * aspect)
		var distance := maxf(3.6, 1.40 / sin(minf(half_vertical, half_horizontal)))
		position = anchor + FOLLOW_DIRECTION.normalized() * distance
		look_at(anchor, Vector3.UP)
	else:
		projection = Camera3D.PROJECTION_ORTHOGONAL
		keep_aspect = Camera3D.KEEP_HEIGHT
		size = maxf(8.2, 12.5 / aspect)
		position = OVERVIEW_POSITION
		look_at(OVERVIEW_TARGET, Vector3.UP)

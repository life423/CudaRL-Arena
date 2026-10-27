extends Control

const TEXT := Color("eef4e9")
const MUTED := Color("b8cbc4")
const ACCENT := Color("faa640")
const REFERENCE := Color("7cdbb5")
var points := PackedVector2Array()
var reference := 13.828125

func _ready() -> void:
	custom_minimum_size.y = 172
	resized.connect(queue_redraw)
	mouse_filter = Control.MOUSE_FILTER_IGNORE

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, size), Color("101c1b"))
	var font := ThemeDB.fallback_font
	draw_string(font, Vector2(16, 23), "Matched evaluation · food / minute", HORIZONTAL_ALIGNMENT_LEFT, size.x - 32, 15, TEXT)
	var plot := Rect2(42, 44, maxf(1, size.x - 62), maxf(1, size.y - 96))
	var ceiling := maxf(15, ceilf(reference / 5) * 5)
	for point in points:
		ceiling = maxf(ceiling, ceilf(point.y / 5) * 5)
	for i in 4:
		var y := plot.end.y - plot.size.y * float(i) / 3
		draw_line(Vector2(plot.position.x, y), Vector2(plot.end.x, y), Color("344742"))
		draw_string(font, Vector2(8, y + 4), "%.0f" % (ceiling * i / 3), HORIZONTAL_ALIGNMENT_LEFT, 30, 11, MUTED)
	var ref_y := plot.end.y - reference / ceiling * plot.size.y
	draw_dashed_line(Vector2(plot.position.x, ref_y), Vector2(plot.end.x, ref_y), REFERENCE, 2, 6)
	var max_update := maxf(25, points[-1].x if not points.is_empty() else 25)
	var min_update := points[0].x if not points.is_empty() else 0.0
	var span := maxf(25, max_update - min_update)
	var line := PackedVector2Array()
	for point in points:
		line.append(Vector2(plot.position.x + (point.x - min_update) / span * plot.size.x, plot.end.y - point.y / ceiling * plot.size.y))
	if line.size() >= 2:
		draw_polyline(line, ACCENT, 2, true)
	draw_string(font, Vector2(plot.position.x, size.y - 17), "Shuffled learner", HORIZONTAL_ALIGNMENT_LEFT, -1, 12, ACCENT)
	draw_string(font, Vector2(plot.position.x + 122, size.y - 17), "Saved best (dashed)", HORIZONTAL_ALIGNMENT_LEFT, -1, 12, REFERENCE)
	draw_string(font, Vector2(plot.end.x - 90, size.y - 35), "%d updates" % int(max_update), HORIZONTAL_ALIGNMENT_RIGHT, 90, 11, MUTED)

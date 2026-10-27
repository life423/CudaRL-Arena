extends Control

const TEXT := Color("eef4e9")
const MUTED := Color("b8cbc4")
const GRID := Color("344742")
const ACCENT := Color("faa640")
var points := PackedVector2Array()
var empty_message := "Measuring the untrained policy…"

func _ready() -> void:
	custom_minimum_size.y = 168
	mouse_filter = Control.MOUSE_FILTER_PASS
	resized.connect(queue_redraw)

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, size), Color("101c1b"))
	var font := ThemeDB.fallback_font
	draw_string(font, Vector2(20, 25), "Learning progress", HORIZONTAL_ALIGNMENT_LEFT, -1, 16, TEXT)
	draw_string(font, Vector2(20, 44), "Food / minute · 64 fixed 60 s trials", HORIZONTAL_ALIGNMENT_LEFT, -1, 12, MUTED)
	if not points.is_empty():
		var latest := "%.1f / min" % points[-1].y
		draw_string(font, Vector2(size.x - 20 - font.get_string_size(latest, HORIZONTAL_ALIGNMENT_LEFT, -1, 20).x, 27), latest, HORIZONTAL_ALIGNMENT_LEFT, -1, 20, ACCENT)
	var plot := Rect2(44, 62, maxf(1, size.x - 64), maxf(1, size.y - 88))
	var chart_max := 10.0
	for point in points:
		chart_max = maxf(chart_max, ceilf(point.y / 10.0) * 10.0)
	for value in [0.0, chart_max / 2.0, chart_max]:
		var y: float = plot.end.y - plot.size.y * float(value) / chart_max
		draw_line(Vector2(plot.position.x, y), Vector2(plot.end.x, y), GRID)
		draw_string(font, Vector2(10, y + 4), str(value), HORIZONTAL_ALIGNMENT_LEFT, -1, 11, MUTED)
	if points.is_empty():
		draw_string(font, Vector2(plot.position.x + 8, plot.get_center().y + 4), empty_message, HORIZONTAL_ALIGNMENT_LEFT, plot.size.x - 8, 12, MUTED)
	else:
		var max_update := maxf(25, points[-1].x)
		var line := PackedVector2Array()
		for point in points:
			line.append(Vector2(plot.position.x + plot.size.x * point.x / max_update, plot.end.y - plot.size.y * point.y / chart_max))
		if line.size() > 1:
			draw_polyline(line, ACCENT, 2.0, true)
		draw_circle(line[-1], 3.0, ACCENT, true, -1, true)
		draw_string(font, Vector2(plot.end.x - 64, size.y - 6), str(int(points[-1].x)), HORIZONTAL_ALIGNMENT_RIGHT, 64, 11, MUTED)
	draw_string(font, Vector2(plot.position.x, size.y - 6), "Training update", HORIZONTAL_ALIGNMENT_LEFT, -1, 11, MUTED)

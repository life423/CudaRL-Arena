extends "res://pond/pond_curve.gd"

# Same chart language as the pond, with seconds instead of food-rate units.
func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, size), Color("101c1b"))
	var font := ThemeDB.fallback_font
	draw_string(font, Vector2(20, 25), plot_title, HORIZONTAL_ALIGNMENT_LEFT, -1, 16, TEXT)
	draw_string(font, Vector2(20, 45), status_caption, HORIZONTAL_ALIGNMENT_LEFT, maxf(1, size.x - 40), 12, MUTED)
	var value_text := "Current hold  %.1f s" % current_score if current_score >= 0 else "Current hold  —"
	draw_string(font, Vector2(20, 72), value_text, HORIZONTAL_ALIGNMENT_LEFT, -1, 16, ACCENT)
	if show_best_score and best_score >= 0:
		var best_text := "Best  %.1f s" % best_score
		var best_width := font.get_string_size(best_text, HORIZONTAL_ALIGNMENT_LEFT, -1, 16).x
		draw_string(font, Vector2(size.x - 20 - best_width, 72), best_text, HORIZONTAL_ALIGNMENT_LEFT, -1, 16, BEST)
	var compact := size.y < 170
	if not compact:
		draw_string(font, Vector2(20, 91), plot_caption, HORIZONTAL_ALIGNMENT_LEFT, maxf(1, size.x - 40), 12, MUTED)
	var plot_top := 88 if compact else 108
	var plot := Rect2(44, plot_top, maxf(1, size.x - 64), maxf(1, size.y - plot_top - 24))
	var chart_max := 10.0
	for point in points:
		chart_max = maxf(chart_max, ceilf(point.y / 10.0) * 10.0)
	for point in best_points:
		chart_max = maxf(chart_max, ceilf(point.y / 10.0) * 10.0)
	var ticks := [0.0, chart_max] if plot.size.y < 40 else [0.0, chart_max / 2.0, chart_max]
	for tick in ticks:
		var y: float = plot.end.y - plot.size.y * float(tick) / chart_max
		draw_line(Vector2(plot.position.x, y), Vector2(plot.end.x, y), GRID)
		draw_string(font, Vector2(8, y + 4), "%.0f s" % tick, HORIZONTAL_ALIGNMENT_LEFT, -1, 10, MUTED)
	if points.is_empty():
		draw_string(font, Vector2(plot.position.x + 8, plot.get_center().y + 4), empty_message, HORIZONTAL_ALIGNMENT_LEFT, maxf(1, plot.size.x - 8), 12, MUTED)
	else:
		var minimum := points[0].x
		var span := maxf(25, points[-1].x - minimum)
		var best_line := PackedVector2Array()
		for point in best_points:
			best_line.append(Vector2(plot.position.x + plot.size.x * (point.x - minimum) / span, plot.end.y - plot.size.y * point.y / chart_max))
		if best_line.size() > 1:
			draw_polyline(best_line, BEST, 1.5, true)
		var line := PackedVector2Array()
		for point in points:
			line.append(Vector2(plot.position.x + plot.size.x * (point.x - minimum) / span, plot.end.y - plot.size.y * point.y / chart_max))
		if line.size() > 1:
			draw_polyline(line, ACCENT, 2.0, true)
		draw_circle(line[-1], 3.0, ACCENT, true, -1, true)
	draw_string(font, Vector2(plot.position.x, size.y - 6), axis_label, HORIZONTAL_ALIGNMENT_LEFT, -1, 11, MUTED)

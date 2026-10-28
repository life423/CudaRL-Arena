extends Control

const TEXT := Color("eef4e9")
const MUTED := Color("b8cbc4")
const GRID := Color("344742")
const ACCENT := Color("faa640")
const BEST := Color("7cdbb5")
var points := PackedVector2Array()
var best_points := PackedVector2Array()
var empty_message := "Measuring the untrained policy…"
var current_score := -1.0
var best_score := -1.0
var best_update := -1
var best_saved := false
var live_mode := true
var save_error := ""
var watch_button: Button
var plot_title := "Learning progress"
var score_label := "Current"
var plot_caption := "Food / minute · 64 fixed 60 s trials"
var axis_label := "Training update"
var status_caption := ""
var show_best_score := true

func _ready() -> void:
	custom_minimum_size.y = 212
	mouse_filter = Control.MOUSE_FILTER_PASS
	resized.connect(queue_redraw)
	watch_button = Button.new()
	watch_button.text = "Watch best"
	watch_button.toggle_mode = true
	watch_button.disabled = true
	watch_button.custom_minimum_size = Vector2(116, 44)
	watch_button.set_anchors_and_offsets_preset(Control.PRESET_TOP_RIGHT)
	watch_button.offset_left = -132
	watch_button.offset_right = -16
	watch_button.offset_top = 8
	watch_button.offset_bottom = 52
	watch_button.visible = live_mode
	watch_button.tooltip_text = "Show the best evaluated policy while background learning continues. Switching keeps the tadpole and food in place."
	add_child(watch_button)

func update_status(stats: Dictionary) -> void:
	var score: float = stats.current_food_per_minute if stats.evaluation_update >= 0 else -1.0
	var best: float = stats.best_food_per_minute if stats.best_ready else -1.0
	var changed: bool = score != current_score or best != best_score or best_update != stats.best_update or best_saved != stats.best_saved or save_error != stats.checkpoint_error
	current_score = score
	best_score = best
	best_update = int(stats.best_update)
	best_saved = stats.best_saved
	save_error = stats.checkpoint_error
	watch_button.disabled = not stats.best_ready
	watch_button.set_pressed_no_signal(stats.watching_best)
	watch_button.text = "Watch current" if stats.watching_best else "Watch best"
	tooltip_text = "Current is the latest evaluation, measured every 25 updates on 64 fixed one-minute ponds. Best is the strongest matching evaluation saved across runs."
	if not str(stats.checkpoint_path).is_empty():
		tooltip_text += "\nSaved policy: " + str(stats.checkpoint_path)
	if not save_error.is_empty():
		tooltip_text += "\n" + save_error
	if changed:
		queue_redraw()

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, size), Color("101c1b"))
	var font := ThemeDB.fallback_font
	draw_string(font, Vector2(20, 25), plot_title, HORIZONTAL_ALIGNMENT_LEFT, -1, 16, TEXT)
	var saved_text := "Measuring baseline…"
	if best_score >= 0:
		saved_text = "Best saved · update %d" % best_update if best_saved else "Best this run"
	if not save_error.is_empty():
		saved_text = "Best policy save failed"
	if not status_caption.is_empty():
		saved_text = status_caption
	draw_string(font, Vector2(20, 45), saved_text, HORIZONTAL_ALIGNMENT_LEFT, maxf(1, size.x - 164), 12, MUTED if save_error.is_empty() else Color("ffb2a9"))
	var current_text := "%s  %.1f / min" % [score_label, current_score] if current_score >= 0 else "%s  —" % score_label
	var best_text := "Best  %.1f / min" % best_score if best_score >= 0 else "Best  —"
	draw_string(font, Vector2(20, 72), current_text, HORIZONTAL_ALIGNMENT_LEFT, -1, 16, ACCENT)
	if show_best_score:
		draw_string(font, Vector2(size.x - 20 - font.get_string_size(best_text, HORIZONTAL_ALIGNMENT_LEFT, -1, 16).x, 72), best_text, HORIZONTAL_ALIGNMENT_LEFT, -1, 16, BEST)
	draw_string(font, Vector2(20, 91), plot_caption, HORIZONTAL_ALIGNMENT_LEFT, -1, 12, MUTED)
	var plot := Rect2(44, 108, maxf(1, size.x - 64), maxf(1, size.y - 132))
	var chart_max := 10.0
	for point in points:
		chart_max = maxf(chart_max, ceilf(point.y / 10.0) * 10.0)
	for point in best_points:
		chart_max = maxf(chart_max, ceilf(point.y / 10.0) * 10.0)
	for value in [0.0, chart_max / 2.0, chart_max]:
		var y: float = plot.end.y - plot.size.y * float(value) / chart_max
		draw_line(Vector2(plot.position.x, y), Vector2(plot.end.x, y), GRID)
		draw_string(font, Vector2(10, y + 4), str(value), HORIZONTAL_ALIGNMENT_LEFT, -1, 11, MUTED)
	if points.is_empty():
		draw_string(font, Vector2(plot.position.x + 8, plot.get_center().y + 4), empty_message, HORIZONTAL_ALIGNMENT_LEFT, plot.size.x - 8, 12, MUTED)
	else:
		var max_update := maxf(25, points[-1].x)
		var min_update := points[0].x
		var span := maxf(25, max_update - min_update)
		for i in best_points.size():
			var point: Vector2 = best_points[i]
			var end := Vector2(plot.position.x + plot.size.x * (point.x - min_update) / span, plot.end.y - plot.size.y * point.y / chart_max)
			if i == 0:
				draw_circle(end, 2.0, BEST)
			else:
				var previous: Vector2 = best_points[i - 1]
				var start := Vector2(plot.position.x + plot.size.x * (previous.x - min_update) / span, plot.end.y - plot.size.y * previous.y / chart_max)
				var corner := Vector2(end.x, start.y)
				draw_dashed_line(start, corner, BEST, 2.0, 5.0, true, true)
				draw_line(corner, end, BEST, 2.0, true)
		var line := PackedVector2Array()
		for point in points:
			line.append(Vector2(plot.position.x + plot.size.x * (point.x - min_update) / span, plot.end.y - plot.size.y * point.y / chart_max))
		if line.size() > 1:
			draw_polyline(line, ACCENT, 2.0, true)
		draw_circle(line[-1], 3.0, ACCENT, true, -1, true)
		draw_string(font, Vector2(plot.end.x - 64, size.y - 6), str(int(points[-1].x)), HORIZONTAL_ALIGNMENT_RIGHT, 64, 11, MUTED)
	draw_string(font, Vector2(plot.position.x, size.y - 6), axis_label, HORIZONTAL_ALIGNMENT_LEFT, -1, 11, MUTED)

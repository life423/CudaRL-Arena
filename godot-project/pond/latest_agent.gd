extends "res://pond/deep_pond.gd"

const LATEST_POLICY := "res://../docs/experiments/swimmer/gentle-currents/bank-recovery/selected-policy.bin"
var last_session_time := -1.0

func _start_lab() -> bool:
	paused = true
	var backend := "auto"
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--pond-backend="):
			backend = argument.get_slice("=", 1)
	return lab.watch_progress_controller(LATEST_POLICY, backend)

func _ready() -> void:
	super._ready()
	get_window().title = "CudaRL · Latest trained swimmer · live"
	(sidebar_scroll.get_child(0).get_child(0) as Label).text = "Live trained swimmer"
	var self_link := find_child("WatchLatestAgent", true, false)
	if self_link:
		self_link.hide()
	_poll()

func _poll() -> void:
	super._poll()
	if not lab or not lab.get_error().is_empty() or state.is_empty():
		title_status.text = "Live agent could not start"
		return
	var untrained: bool = stats.get("showcase_untrained", false)
	title_status.text = "Live · random initial weights" if untrained else "Live · latest saved PPO agent"
	training_heading.text = "Live inference"
	training_stats.text = "AI and physics · %s\nSelector update %d\nSwimming retained from %d\nCalm water · three foods" % [stats.compute_backend, 0 if untrained else stats.selector_update, 0 if untrained else stats.policy_update]
	save_status.text = "Random initial networks · no training updates." if untrained else "Bank-trained candidate · weights fixed.\nPhysical simulation runs live.\nOriginal policy and saves preserved."
	var elapsed: float = state.elapsed
	if elapsed < last_session_time:
		curve.points.clear()
	curve.best_points.clear()
	curve.best_score = -1
	curve.show_best_score = false
	curve.plot_title = "Live collection rate"
	curve.status_caption = "Random weights · no learning updates" if untrained else "Saved PPO weights · no learning updates"
	curve.plot_caption = "This session · cumulative food / minute"
	curve.axis_label = "Elapsed seconds"
	curve.score_label = "Session"
	curve.empty_message = "Watching the live swimmer…"
	curve.tooltip_text = "Measured from food actually collected in this live session. This is not a training curve or a fixed validation score."
	curve.current_score = state.eaten * 60.0 / elapsed if elapsed >= 1 else -1
	if elapsed >= 1 and (curve.points.is_empty() or floori(elapsed / 5) > floori(curve.points[-1].x / 5)):
		curve.points.append(Vector2(elapsed, curve.current_score))
		if curve.points.size() > 360:
			curve.points.remove_at(0)
	last_session_time = elapsed
	curve.queue_redraw()

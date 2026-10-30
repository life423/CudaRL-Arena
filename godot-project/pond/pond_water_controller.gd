extends RefCounted

# UI/current composition shared by live viewing and interactive learning.
# Owns no session, checkpoint, physics integrator or training loop.
signal changed
const SPEEDS := [0.5, 0.75]
var lab: RefCounted
var toggle: CheckButton
var strength: OptionButton
var status: Label
var flow: Node3D

func configure(session: RefCounted, controls: Dictionary) -> void:
	lab = session
	toggle = controls.switch
	strength = controls.strength
	status = controls.status
	if lab == null:
		toggle.disabled = true
		strength.disabled = true

func set_enabled(enabled: bool) -> void:
	if lab == null or not lab.set_steady_current(SPEEDS[strength.selected] if enabled else 0.0, true):
		toggle.set_pressed_no_signal(not enabled)
		status.text = "Current could not change"
		status.tooltip_text = lab.get_error() if lab else "The pond session is unavailable."
		return
	changed.emit()

func select_strength(index: int, stats: Dictionary) -> void:
	if lab == null or not toggle.button_pressed:
		return
	if not lab.set_steady_current(SPEEDS[index], true):
		strength.select(1 if stats.get("water_current_cm_s", 0.0) > 0.625 else 0)
		status.text = "Current could not change"
		status.tooltip_text = lab.get_error()
		return
	changed.emit()

func refresh(stats: Dictionary) -> void:
	var flowing: bool = stats.get("water_current_cm_s", 0.0) > 0
	var drifting: bool = stats.get("food_drifting", false)
	if flow:
		flow.set_current(stats.get("water_current", Vector3.ZERO))
	toggle.set_pressed_no_signal(flowing)
	strength.disabled = not flowing
	if flowing:
		strength.select(1 if stats.get("water_current_cm_s", 0.0) > 0.625 else 0)
	var speed := str(snappedf(stats.get("water_current_cm_s", 0.0), 0.01))
	status.text = (speed + " cm/s · food drifts →" if drifting else speed + " cm/s · steady flow →") if flowing else "Calm water"
	status.tooltip_text = ("Suspended food follows the horizontal current, with clearance at the bank." if drifting else "Stationary-food reference mode.") if flowing else "No imposed current; food remains at its present position."

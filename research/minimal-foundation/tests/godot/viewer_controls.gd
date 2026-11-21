extends SceneTree
# Zero-step check for the explicit absence of a verified PPO checkpoint.
var view: Node
func _initialize() -> void:
    view = load("res://main.tscn").instantiate()
    view.process_mode = Node.PROCESS_MODE_DISABLED
    root.add_child.call_deferred(view)
    _verify.call_deferred()
func _verify() -> void:
    var ok := view.swimmer != null and not bool(view.state.get("loaded",true))
    ok = ok and view.start_button.disabled and view.pause_button.disabled and view.reset_button.disabled
    ok = ok and not view.speed_control.editable and view.continuous_control.disabled
    ok = ok and int(view.state.get("substeps",-1)) == 0 and view.total_substeps == 0
    ok = ok and view.status.text.begins_with("No verified PPO checkpoint installed")
    view._start()
    ok = ok and int(view.swimmer.call("get_state").substeps) == 0
    print(JSON.stringify({"missing_PPO_asset_check_passed":ok,"physical_substeps":0}))
    quit(0 if ok else 3)

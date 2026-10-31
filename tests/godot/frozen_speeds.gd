extends SceneTree
# Existing native frozen cases through the actual viewer/UI binding. One run:
# 1520 controls / 15200 substeps: recorded cases plus one post-brake upshift.
# 20s wall cap, no training or retries.
var view: Node
var expected := {}
var controls := 0
var substeps := 0
var started := 0
var cases := 0
var switches := 0
var capture := ""

func _initialize() -> void:
    for argument in OS.get_cmdline_user_args():
        if argument.begins_with("--verification-image="): capture = argument.trim_prefix("--verification-image=")
    view = load("res://main.tscn").instantiate()
    view.process_mode = Node.PROCESS_MODE_DISABLED
    root.add_child.call_deferred(view)
    _verify.call_deferred()

func _fail(reason: String) -> bool:
    if view.swimmer != null: view.swimmer.call("pause")
    print(JSON.stringify({"error":reason,"controls":controls,"physical_substeps":substeps}))
    quit(3)
    return false

func _initialize_case(pose: int, target: float) -> void:
    view._pause()
    view.swimmer.call("set_speed",target/1000)
    view.swimmer.call("reset",pose)
    view._refresh()
    view._start()

func _switch(target: float) -> bool:
    var before: Dictionary = view.swimmer.call("get_state")
    view.speed_control.value = target
    var after: Dictionary = view.swimmer.call("get_state")
    for key in ["head_x","head_y","head_z","qx","qy","qz","qw","j0","j1","j2","j3","phase","amplitude","frequency","amplitude_rate","frequency_rate","controls","substeps","episode_time_s","time_s","paused"]:
        if before[key] != after[key]: return _fail("Speed request changed physical/running state: "+key)
    if bool(after.paused) or absf(float(after.target_speed_m_s)*1000-target)>0.001: return _fail("Mid-swim request failed")
    switches += 1
    return true

func _segment(kind: String,pose: int,target: float,count: int) -> bool:
    var maximum_lateral := 0.0
    var hold_start := 0.0
    var hold_error := 0.0
    for tick in range(count):
        if controls>=1520 or substeps+10>15200 or Time.get_ticks_msec()-started>20000: return _fail("Bridge verification cap")
        var before: Dictionary = view.swimmer.call("get_state")
        controls += 1
        var advanced: bool = view.swimmer.call("step")
        var state: Dictionary = view.swimmer.call("get_state")
        substeps += int(state.substeps)-int(before.substeps)
        if not advanced or bool(state.paused) or not str(state.error).is_empty(): return _fail("Bridge physical/controller stop")
        maximum_lateral = maxf(maximum_lateral,float(state.lateral_m))
        if tick==79: hold_start = float(state.forward_m)
        if tick>=80: hold_error += float(state.speed_error_m_s)*float(state.speed_error_m_s)
    var state: Dictionary = view.swimmer.call("get_state")
    var key := "%s/%d/%d" % [kind,pose,int(target)]
    var work_per_m := float(state.segment_work_J)/maxf(float(state.forward_m),0.000000001)
    if not bool(state.passed): return _fail("Bridge task gate failed "+key)
    if kind != "UI_flow_upshift":
        if not expected.has(key): return _fail("Missing native reference "+key)
        var reference: Array = expected[key]
        if absf(float(state.steady_speed_m_s)*1000-float(reference[5]))>0.001 or absf(float(state.speed_rmse_m_s)*1000-float(reference[6]))>0.001 or absf(work_per_m-float(reference[7]))>0.00000001 or absf(maximum_lateral*1000-float(reference[8]))>0.001:
            return _fail("Bridge differs from frozen native case "+key)
    if count>80:
        var held_speed := (float(state.forward_m)-hold_start)/2*1000
        if held_speed<0.9*target or held_speed>1.1*target or sqrt(hold_error/40)*1000>0.3125*target: return _fail("Independent settling check failed")
    cases += 1
    view._refresh(state)
    return true

func _verify() -> void:
    started = Time.get_ticks_msec()
    if view.swimmer == null or not bool(view.state.get("loaded",false)) or int(view.state.action_mapping)!=2:
        _fail("Accepted signed actor did not load")
        return
    var file := FileAccess.open("res://../assets/signed-speed-v1/frozen-evaluation.csv",FileAccess.READ)
    if file==null:
        _fail("Missing accepted native recording")
        return
    file.get_csv_line()
    while not file.eof_reached():
        var row := file.get_csv_line()
        if row.size()==1 and row[0].is_empty(): continue
        if row.size()!=9:
            _fail("Malformed native recording")
            return
        expected["%s/%d/%d" % [row[0],int(row[1]),int(float(row[2]))]] = Array(row)
    for pose in [0,3]:
        for target in [4.0,5.0,6.0,7.0,8.0]:
            _initialize_case(pose,target)
            if not _segment("constant",pose,target,80): return
    for pose in [1,2]:
        _initialize_case(pose,4)
        if not _segment("heldout_warmup",pose,4,80): return
        for target in [8.0,4.0]:
            if not _switch(target) or not _segment("heldout_transition",pose,target,120): return
    # Same Start remains active: explicitly exercise 8->4->8 without Start/reset.
    if not _switch(8) or not _segment("UI_flow_upshift",2,8,80): return
    view._pause()
    var before: Dictionary = view.swimmer.call("get_state")
    view.speed_control.value = 4
    if not bool(view.state.paused) or view.state.controls!=before.controls:
        _fail("Paused request did not stay paused")
        return
    view._refresh()
    print(JSON.stringify({"bridge_cases_passed":cases,"running_switches_without_Start":switches,"controls":controls,"physical_substeps":substeps,"native_reference_parity":true,"training":false,"wall_seconds":float(Time.get_ticks_msec()-started)/1000}))
    if not capture.is_empty():
        await RenderingServer.frame_post_draw
        if get_root().get_texture().get_image().save_png(capture)!=OK:
            _fail("Verification screenshot failed")
            return
    quit()

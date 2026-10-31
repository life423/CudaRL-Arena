extends SceneTree
# One opt-in pose2 sequence: 400 controls /4000 substeps maximum. No learning.
var view: Node
var output := ""
var log: FileAccess
var started := 0
var controls := 0
var substeps := 0
var segments: Array = []

func _initialize() -> void:
    for argument in OS.get_cmdline_user_args():
        if argument.begins_with("--trace-output="): output=argument.trim_prefix("--trace-output=")
    view=load("res://main.tscn").instantiate()
    view.process_mode=Node.PROCESS_MODE_DISABLED
    root.add_child.call_deferred(view)
    _run.call_deferred()

func _stop(reason: String) -> void:
    view.swimmer.call("pause")
    if log!=null: log.flush()
    print(JSON.stringify({"fault":reason,"controls":controls,"physical_substeps":substeps,"segments":segments}))
    quit(3)

func _run() -> void:
    started=Time.get_ticks_msec()
    log=FileAccess.open(output,FileAccess.WRITE)
    if log==null or view.swimmer==null or not bool(view.state.get("loaded",false)):
        _stop("Recording/native load failed")
        return
    view.swimmer.call("set_speed",0.004)
    view.swimmer.call("reset",2)
    view._refresh()
    view._start()
    for stage in range(4):
        var target: float=[4.0,8.0,4.0,8.0][stage]
        var count: int=[80,120,120,80][stage]
        var before: Dictionary=view.swimmer.call("get_state")
        if stage>0: view.speed_control.value=target
        var requested: Dictionary=view.swimmer.call("get_state")
        log.store_line(JSON.stringify({"event":"request","stage":stage,"target_mm_s":target,"before":before,"after":requested}))
        for tick in range(count):
            if controls>=400 or substeps+10>4000 or Time.get_ticks_msec()-started>5000:
                _stop("Frozen reproduction cap")
                return
            var prior: Dictionary=view.swimmer.call("get_state")
            controls+=1
            var advanced: bool=view.swimmer.call("step")
            var state: Dictionary=view.swimmer.call("get_state")
            substeps+=int(state.substeps)-int(prior.substeps)
            log.store_line(JSON.stringify({"event":"step","stage":stage,"tick":tick+1,"state":state}))
            if not advanced or bool(state.paused) or not str(state.error).is_empty():
                _stop("Physical/controller stop")
                return
        var final: Dictionary=view.swimmer.call("get_state")
        var report: Dictionary={"stage":stage,"target_mm_s":target,"controls":count,"loops_before":before.loops,"loops_after":final.loops,"score":final.score,"passed":final.passed,"episode_time_s":final.episode_time_s,"view_time_s":final.time_s}
        segments.append(report)
        log.store_line(JSON.stringify({"event":"segment_end","data":report}))
        log.flush()
        view._refresh(final)
    view._pause()
    print(JSON.stringify({"reproduction_complete":true,"controls":controls,"physical_substeps":substeps,"training":false,"wall_seconds":float(Time.get_ticks_msec()-started)/1000,"segments":segments}))
    quit()

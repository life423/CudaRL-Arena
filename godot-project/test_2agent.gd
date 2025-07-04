# test_2agent.gd
extends SceneTree

func _init():
    var arena = ArenaGDExtension.new()
    arena.reset_environment()
    # human=0, ai=1
    var results = arena.step_environment([0, 1])
    print("HEADLESS TEST OUTPUT → ", results)
    quit()

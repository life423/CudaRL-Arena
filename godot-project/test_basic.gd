extends SceneTree

func _init():
    print("=== CudaRL Arena Integration Test ===")
    
    # Test the ArenaGDExtension
    var arena = ArenaGDExtension.new()
    print(arena.hello_cuda())
    arena.run_training(10)
    
    print("✅ Integration test complete!")
    quit()
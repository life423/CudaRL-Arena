extends Control

func _ready():
    print("=== CudaRL Arena Smoke Test ===")
    
    # Wait for extensions to fully load
    await get_tree().process_frame
    
    # Test the ArenaGDExtension
    var arena = ArenaGDExtension.new()
    print(arena.hello_cuda())
    arena.run_training(10)
    
    print("✅ CudaRL Plugin loaded and ready!")
    get_tree().quit()

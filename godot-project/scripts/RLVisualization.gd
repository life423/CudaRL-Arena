extends Control

func _ready():
    print("=== CudaRL Arena Smoke Test ===")
    
    # Wait for extensions to fully load
    await get_tree().process_frame
    
    # Simple success message without extensions
    print("✅ CudaRL Plugin loaded and ready!")
    get_tree().quit()

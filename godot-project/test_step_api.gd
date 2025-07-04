extends SceneTree

func _initialize():
	test_step_api()
	quit()

func test_step_api():
	print("=== Testing Step & Reset API ===")
	
	var arena = ArenaGDExtension.new()
	print("🚀 CUDA Arena Extension created")
	
	# Test reset
	arena.reset_environment()
	print("✅ Environment reset")
	
	# Test step
	var result = arena.step_environment(1)
	print("Step result: ", result)
	print("  State: ", result["state"])
	print("  Reward: ", result["reward"])
	print("  Done: ", result["done"])
	
	# Test multiple steps
	for i in range(3):
		result = arena.step_environment(i % 4)
		print("Step ", i, " with action ", i % 4, ": reward=", result["reward"])
	
	print("✅ Step API test complete!")
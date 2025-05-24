from python.cudarl import Environment

# Test if bindings work
try:
    env = Environment(width=5, height=5)
    obs = env.reset()
    print("Bindings working! Observation shape:", obs.shape)
except Exception as e:
    print(f"Binding error: {e}")
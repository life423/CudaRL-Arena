class Environment:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        print(f"Mock environment created with size {width}x{height}")
    
    def reset(self):
        print("Mock environment reset")
        return [0.0] * (self.width * self.height)
    
    def step(self, action):
        print(f"Mock environment step with action {action}")
        return [0.0] * (self.width * self.height), -0.01, False, ''
    
    def get_agent_position(self):
        return (0, 0)
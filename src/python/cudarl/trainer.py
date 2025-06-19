"""
Training orchestration for CudaRL-Arena.
"""
class Trainer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def train(self, episodes: int = 100):
        for ep in range(episodes):
            obs = self.env.reset()
            done = False
            while not done:
                action = self.agent.act(obs)
                obs, reward, done, info = self.env.step(action)

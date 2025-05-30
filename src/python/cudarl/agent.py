"""
Agent implementations for CudaRL-Arena.
"""
class Agent:
    def __init__(self):
        pass

    def act(self, obs):
        raise NotImplementedError

class QTableAgent(Agent):
    def __init__(self, env):
        super().__init__()
        self.env = env
        # initialize Q-table here

    def act(self, obs):
        # stub: pick random action
        return self.env.action_space - 1

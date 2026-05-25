from robolab.eval import InferenceClient

class RobolabEnvManager:
    def __init__(self, envs, client: InferenceClient):
        self.envs = envs
        self.client = client

    def reset(self):
        return self.envs.reset()

    def step(self, actions):
        return self.envs.step(actions)

    def close(self):
        return self.envs.close()
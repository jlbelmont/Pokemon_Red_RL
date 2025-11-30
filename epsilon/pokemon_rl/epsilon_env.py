import gymnasium as gym

class EpsilonEnv(gym.Env):
    def __init__(self, pyboy_env, reward_modules):
        super().__init__()
        self.env = pyboy_env
        self.reward_modules = reward_modules
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for r in self.reward_modules:
            r.reset()
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward_components = []
        reward_total = 0.0
        for module in self.reward_modules:
            value = module.compute(obs, info)
            reward_components.append((module.__class__.__name__, float(value)))
            reward_total += value

        info = dict(info)
        info["reward_components"] = reward_components
        info["reward_sum"] = float(reward_total)
        return obs, reward_total, terminated, truncated, info

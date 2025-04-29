import gymnasium
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType


class CustomWrapper(BaseWrapper):
    """
    Wrapper to use to add state pre-processing (feature engineering).
    """

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        flat_obs = obs.flatten()
        return flat_obs


class CustomPredictFunction:
    """
    Function to use to load the trained model and predict the action.
    """

    def __init__(self, env: gymnasium.Env):
        self.env = env

    def __call__(self, observation, agent, *args, **kwargs):
        #print(observation[1])
        #print(agent)
        #if observation[4] != 1:
        if observation[1] > 0.078125:
            if observation[3] != 1.0:
                return 2
            else:
                return 1
        else:
            return 4
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gdax_client import GdaxClient 

import numpy as np

class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.client = GdaxClient()
        self.action_state = spaces.Box(low = -1.0, high = 1.0, shape = (1,))
        # TODO decide observation dimensions, do on ETH-EUR historical data
        self.observation_space = None 
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _step(self, action):
        pass
    
    def _reset(self):
        pass
    
    def _render(self, mode='human', close=False):
        pass

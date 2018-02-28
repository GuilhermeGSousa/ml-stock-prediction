import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gdax_client import GdaxClient 


class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.client = GdaxClient()
        self.action_state = spaces.Box(-1,1)
        # TODO decide observation dimensions, do on ETH-EUR historical data
        
    def _configure(self, display=None):
        pass
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _step(self, action):
        pass
    
    def _reset(self):
        pass
    
    def _render(self, mode='human', close=False):
        pass
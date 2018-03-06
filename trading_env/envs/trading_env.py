import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gdax_client import GdaxClient 

import numpy as np

import datetime

class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.client = GdaxClient()
        # TODO Make this configurable
        self.episode_steps = 10 #TODO Set this up
        self.portfolio_value = 0
        self.fiat = 0
        self.crypto = 1
        self.window_size = 100
        self.n_features = 2
        
        self.action_state = spaces.Box(low = -1.0, 
                                       high = 1.0, 
                                       shape = (1,), 
                                       dtype = np.float32)
        self.observation_space = spaces.Box(low = 0.0, 
                                            high = np.finfo(np.float32).max, 
                                            shape = (self.window_size, self.n_features,), 
                                            dtype = np.float32)
        self.last_date = datetime.datetime(1971, 1, 1, 0, 0)
        self.portfolio_value = self._get_portfolio_value()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        start = datetime.datetime.utcnow() - datetime.timedelta(minutes=15) * self.window_size
        end = datetime.datetime.utcnow()
        df = self.client.get_historical_data(start,end)
        s = np.transpose(np.array([df["close"],df["volume"]]))
        current_value = self._get_portfolio_value()
        r = (current_value - self.portfolio_value)/self.portfolio_value
        self.portfolio_value = current_value
        return s, r
    
    def reset(self):
        pass
    
    def render(self, mode='human', close=False):
        pass
    
    def _get_portfolio_value(self):
        current_price = self.client.get_market_price()
        value = self.fiat + self.crypto * current_price
        return value

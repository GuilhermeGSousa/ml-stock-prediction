import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gdax_client import GdaxClient 

import numpy as np
import pandas as pd
from random import randint

from os import getcwd
from os.path import join, realpath, dirname, isfile

import datetime

class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.client = GdaxClient()
        self.portfolio_value = 0
        self.fiat = 0
        self.crypto = 1
        self.window_size = 100
        self.n_features = 2
        self.granularity = 900
        # self.episode_steps = 3600/granularity * 24 * 30 #One month episodes
        
        self.action_space = spaces.Box(low = -1.0, 
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
    
    
class TestTradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.client = GdaxClient()

        self.portfolio_value = 0
        self.start_fiat = 0
        self.start_crypto = 1
        self.fiat = self.start_fiat
        self.crypto = self.start_crypto
        self.window_size = 100
        self.n_features = 2
        self.granularity = 900
        self.episode_steps = 3600/self.granularity * 24 * 30 #One month episodes
        
        self.pair = 'ETH-USD'
        self.begin = datetime.datetime(2017, 1, 1, 0, 0) # (YYYY/MM/DD/h/min)
        self.end = datetime.datetime(2018, 3, 1, 0, 0)
        
        self.action_space = spaces.Box(low = -1.0, 
                                       high = 1.0, 
                                       shape = (1,), 
                                       dtype = np.float32)
        self.observation_space = spaces.Box(low = 0.0, 
                                            high = np.finfo(np.float32).max, 
                                            shape = (self.window_size, self.n_features,), 
                                            dtype = np.float32)
        self.start_index = None
        self.steps = 0
        self._set_data()
        self._set_start_index()
        self.portfolio_value = self._get_portfolio_value()
            
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        done = False
        self._set_portfolio(action)
        #One timestep goes by...
        self.steps += 1
        s = self.historical_data.loc[self.start_index + self.steps - self.window_size + 1 : self.start_index + self.steps,
                                     ["close","volume"]].values
        current_value = self._get_portfolio_value()
        
        r = (current_value - self.portfolio_value)/self.portfolio_value
        self.portfolio_value = current_value
        
        if (self.steps >= self.episode_steps):
            done = True

        return s, r, done, {}
    
    def reset(self, date=None):
        """
        If date is set to None a random period of one month is chosen for the episode,
        otherwise the episode starts at the given date
        Returns first state of simulation
        """
        if date is None:
            self._set_start_index()
        else:
            tmp_index = next((i for i, x in self.historical_data.iterrows() if not x["time"] < date), None)
            if (tmp_index is not None and 
                (tmp_index + self.episode_steps < self.historical_data.shape[0]) and 
                (tmp_index >=self.window_size)):
                    self.start_index = tmp_index
            else:
                raise ValueError('Incorrect date entered.')
                
        self.fiat = self.start_fiat
        self.crypto = self.start_crypto
        self.steps = 0
        self.portfolio_value = self._get_portfolio_value()
        s = self.historical_data.loc[self.start_index + self.steps - self.window_size + 1 : self.start_index + self.steps,
                                     ["close","volume"]].values
        return s
        
    def render(self, mode='human', close=False):
        pass
    
    def _set_start_index(self):
        self.start_index = randint(self.window_size, len(self.historical_data["time"])-self.episode_steps-1)
        
    def _set_portfolio(self, action):
        a = action[0]
        current_price = self.historical_data["close"][self.start_index + self.steps]
        prev_fiat = self.fiat
        prev_crypto = self.crypto
        if(a > 0):
            self.crypto = a * 1/current_price * prev_fiat + prev_crypto
            self.fiat = (1 - a) * prev_fiat
        else:
            a=-a
            self.fiat = a * current_price * prev_crypto + prev_fiat
            self.crypto = (1 - a) * prev_crypto
        
    def _get_portfolio_value(self):
        current_price = self.historical_data["close"][self.start_index + self.steps]
        value = self.fiat + self.crypto * current_price
        return value
    
    def _set_data(self):
        current_dir = realpath(__file__)
        main_dir = dirname(dirname(dirname(current_dir)))
        data_file = join(main_dir, "data", "train.pkl")
        if not isfile(data_file):
            print("Downloading historical data")
            self.historical_data = self.client.get_historical_data(self.begin, 
                                                               self.end, 
                                                               granularity = self.granularity, 
                                                               pair = self.pair)
            self.historical_data.to_pickle(data_file)
        else:
            print("Loading historical data file")
            self.historical_data = pd.read_pickle(data_file)

# ml-stock-prediction

This repository aims to use Reinforcement Learning to backtest and live trade cryptocurrencies with custom OpenAI Gym environments.

Prerequisites
=============

To get this project up and running on your machine, and mainly to be able to use the trading environments, a few dependencies must first be installed. Firstly to install if not already installed NumPy, Pandas, Jupyter and matplotlib by doing:

```
pip install numpy matplotlib ipython jupyter pandas
```
OpenAI Gym must also be installed, which can be done with:

```
pip install gym
```
Please refer to the OpenAI Gym [repo](https://github.com/openai/gym#installation) for more information.

To get historical data, place orders or making other requests from GDAX, the [unoficial GDAX python client](https://github.com/danpaquin/gdax-python) by Daniel Paquin is used, which can be installed once again using `pip`:

```
pip install gdax
```
OpenAI Environments
===================

### The testing environment

Similarly to other OpenAI environments, this one can be instanced by

```python
import gym
import trading_env

env_trading = gym.make('test_trading-v0')
```
It uses GDAX historical data from January 2017 to March 2018, and a start date can be specified when reseting the environment. A random start date is otherwise chosen. 

```python
env_trading.reset(date=datetime.datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0))
```

One episode represents one month starting at date with timesteps of 15 minutes. Every step returns the last 200 closing prices and volume up until the current step, relative to (divided by) the current closing price and volume. The actions range from -1 to 1 and represent the percentage of the portfolio's crypto sold or fiat used to buy, respectively.

| States          | Actions       |  Rewards  |
| :-------------: |:-------------:| :--------:|
| Box(100, 2)     | Box(1)        |(-inf, inf)|

The reward is computed as the relative difference of portfolio value since the last timestep:

```
reward = (current portfolio value - previous portfolio value)/previous portfolio value
```

### The Live Environment

This on is still under development

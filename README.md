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

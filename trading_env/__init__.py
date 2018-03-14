from gym.envs.registration import register

register(
    id='trading-v0',
    entry_point='trading_env.envs:TradingEnv',
)
register(
    id='test_trading-v0',
    entry_point='trading_env.envs:TestTradingEnv',
)
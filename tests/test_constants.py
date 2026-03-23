import core.constants as constants

def test_constants_values():
    assert constants.OBSERVATION_CLIP_RANGE == 10.0
    assert constants.TRADING_DAYS_PER_YEAR == 252
    assert constants.HOURS_PER_TRADING_DAY == 6.5
    assert constants.STEPS_PER_TRADING_WEEK_DEFAULT == 78
    assert constants.DSR_VARIANCE_FLOOR == 1e-4
    assert constants.PORTFOLIO_HISTORY_WINDOW == 252
    assert constants.RETURNS_HISTORY_WINDOW == 100
    assert constants.MIN_POSITION_THRESHOLD == 1e-6
    assert constants.EQUITY_EPSILON == 1e-8
    assert constants.BANKRUPTCY_THRESHOLD == 0.5
    assert constants.MAX_REWARD_CLIP == 10.0
    assert constants.DEFAULT_VOL_CEILING == 0.05

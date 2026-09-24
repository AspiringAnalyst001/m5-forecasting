from m5.models.baselines import Forecaster, SeasonalWindowAverage, get_baselines


def get_models() -> dict[str, Forecaster]:
    from m5.models.lgbm import LightGBMForecaster
    from m5.models.rules import ClosureRule

    models = dict(get_baselines())
    for m in (
        LightGBMForecaster(name="lgbm_v1"),
        LightGBMForecaster(name="lgbm_fast", train_days=365, num_rounds=200),
        LightGBMForecaster(name="lgbm_fast_s1", train_days=365, num_rounds=200, params={"seed": 1}),
        LightGBMForecaster(name="lgbm_fast_s2", train_days=365, num_rounds=200, params={"seed": 2}),
        ClosureRule(SeasonalWindowAverage(7, 8)),
        ClosureRule(LightGBMForecaster(name="lgbm_fast", train_days=365, num_rounds=200)),
    ):
        models[m.name] = m
    return models
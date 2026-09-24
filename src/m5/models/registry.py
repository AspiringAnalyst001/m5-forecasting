from m5.models.baselines import Forecaster, get_baselines


def get_models() -> dict[str, Forecaster]:
    from m5.models.lgbm import LightGBMForecaster

    models = dict(get_baselines())
    for m in (
        LightGBMForecaster(name="lgbm_v1"),
        LightGBMForecaster(name="lgbm_fast", train_days=365, num_rounds=200),
    ):
        models[m.name] = m
    return models
"""Minimal MLflow PyFunc wrapper for the per-store LightGBM forecaster.

MLflow's model registry (3.x) requires models to be logged through a
flavor-specific log_model call to create a "Logged Model" entity — plain
mlflow.log_artifacts() stores files but can't be registered. This wrapper
exists only to satisfy that requirement for versioning/aliasing (e.g.
"champion"). It does NOT implement real inference: this forecaster's feature
engineering is stateful and array-based, a poor fit for pyfunc's simple
interface. The FastAPI service (Phase 7) loads the saved boosters directly
via m5.models.lgbm.LightGBMForecaster instead of calling this wrapper's predict().
"""
import json
from pathlib import Path

import lightgbm as lgb
import mlflow.pyfunc


class M5ForecasterWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_dir = Path(context.artifacts["model_dir"])
        with open(model_dir / "config.json") as f:
            self.config = json.load(f)
        self.boosters = {
            txt_file.stem: lgb.Booster(model_file=str(txt_file))
            for txt_file in model_dir.glob("*.txt")
        }

    def predict(self, context, model_input):
        raise NotImplementedError(
            "This registry entry is for versioning/aliasing only. "
            "Use m5.models.lgbm.LightGBMForecaster with the saved boosters "
            "directly for real inference (see the FastAPI service)."
        )
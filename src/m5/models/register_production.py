"""Log the trained production model to MLflow and register it.
Run: python -m m5.models.register_production
"""
import mlflow

from m5.config import ROOT
from m5.models.pyfunc_wrapper import M5ForecasterWrapper

MODEL_DIR = ROOT / "models" / "production"
MODEL_NAME = "m5-lgbm-forecaster"

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("m5-forecasting")


def main() -> None:
    with mlflow.start_run(run_name="production_model") as run:
        mlflow.log_params({"model": "lgbm_production", "train_days": 730, "num_rounds": 400})
        mlflow.log_metrics({"reference_wrmsse_fold3": 0.522})

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=M5ForecasterWrapper(),
            artifacts={"model_dir": str(MODEL_DIR)},
        )

        model_uri = f"runs:/{run.info.run_id}/model"
        result = mlflow.register_model(model_uri, MODEL_NAME)
        print(f"Registered {MODEL_NAME} version {result.version}")

        client = mlflow.MlflowClient()
        client.set_registered_model_alias(MODEL_NAME, "champion", result.version)
        print(f"Tagged version {result.version} as 'champion'")


if __name__ == "__main__":
    main()
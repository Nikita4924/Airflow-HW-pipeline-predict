import glob
import json
import os
from datetime import datetime

import dill
import pandas as pd

path = os.environ.get("PROJECT_PATH", ".")


def get_model_path() -> str:
    model_files = glob.glob(f"{path}/data/models/*.pkl")
    if not model_files:
        raise FileNotFoundError(f"No model files found in {path}/data/models")

    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model


def load_model():
    model_path = get_model_path()
    with open(model_path, "rb") as file:
        model = dill.load(file)
    return model


def load_test_data() -> pd.DataFrame:
    test_files = glob.glob(f"{path}/data/test/*.json")
    if not test_files:
        raise FileNotFoundError(f"No test files found in {path}/data/test")

    rows = []
    for file_name in test_files:
        with open(file_name, "r", encoding="utf-8") as file:
            data = json.load(file)
            rows.append(data)

    df = pd.DataFrame(rows)
    return df


def save_predictions(df: pd.DataFrame) -> str:
    predictions_dir = f"{path}/data/predictions"
    os.makedirs(predictions_dir, exist_ok=True)

    output_path = f"{predictions_dir}/preds_{datetime.now().strftime('%Y%m%d%H%M')}.csv"
    df.to_csv(output_path, index=False)
    return output_path


def predict() -> None:
    model = load_model()
    test_df = load_test_data()

    predictions = model.predict(test_df)

    result = pd.DataFrame(
        {
            "id": test_df["id"],
            "prediction": predictions,
        }
    )

    output_path = save_predictions(result)
    print(result.head())
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    predict()
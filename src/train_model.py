"""Train a house-price regression model and create a Kaggle submission."""

from pathlib import Path


# Create the expected project folders when this script is run from a fresh checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
SOURCE_DIR = PROJECT_ROOT / "src"
DATASET_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = ["GrLivArea", "BedroomAbvGr", "TotalBathrooms"]
SOURCE_FEATURE_COLUMNS = ["GrLivArea", "BedroomAbvGr", "FullBath", "HalfBath"]


def make_features(data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    missing_columns = sorted(set(SOURCE_FEATURE_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required feature columns: "
            f"{', '.join(missing_columns)}"
        )

    features = data[SOURCE_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(
        {
            "GrLivArea": features["GrLivArea"],
            "BedroomAbvGr": features["BedroomAbvGr"],
            "TotalBathrooms": features["FullBath"] + 0.5 * features["HalfBath"],
        },
        index=data.index,
    )[FEATURE_COLUMNS]


def impute_features(
    training_features: pd.DataFrame, other_features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    medians = training_features.median()
    unavailable_medians = medians[medians.isna()].index.tolist()
    if unavailable_medians:
        raise ValueError(
            "Cannot compute training medians for features with no valid values: "
            f"{', '.join(unavailable_medians)}"
        )

    return (
        training_features.fillna(medians),
        other_features.fillna(medians),
    )


def main() -> None:
    train_path = DATASET_DIR / "train.csv"
    test_path = DATASET_DIR / "test.csv"
    if not train_path.is_file() or not test_path.is_file():
        missing = [str(path) for path in (train_path, test_path) if not path.is_file()]
        raise FileNotFoundError(
            "Required dataset file(s) not found: "
            f"{', '.join(missing)}. Place train.csv and test.csv in {DATASET_DIR}."
        )

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    if "SalePrice" not in train_data.columns:
        raise ValueError("train.csv is missing the required target column: SalePrice")
    if "Id" not in test_data.columns:
        raise ValueError("test.csv is missing the required output column: Id")

    target = pd.to_numeric(train_data["SalePrice"], errors="coerce")
    if target.isna().any():
        raise ValueError("train.csv contains missing or non-numeric SalePrice values.")

    training_features = make_features(train_data, "train.csv")
    test_features = make_features(test_data, "test.csv")
    training_features, test_features = impute_features(training_features, test_features)

    x_train, x_validation, y_train, y_validation = train_test_split(
        training_features,
        target,
        test_size=0.2,
        random_state=42,
    )
    model = LinearRegression()
    model.fit(x_train, y_train)

    validation_predictions = model.predict(x_validation)
    rmse = np.sqrt(mean_squared_error(y_validation, validation_predictions))
    r2 = r2_score(y_validation, validation_predictions)
    print(f"Validation RMSE: {rmse:.2f}")
    print(f"Validation R2: {r2:.4f}")

    model.fit(training_features, target)
    test_predictions = model.predict(test_features)
    submission = pd.DataFrame(
        {
            "Id": test_data["Id"],
            "SalePrice": test_predictions,
        }
    )

    output_path = DATASET_DIR / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()

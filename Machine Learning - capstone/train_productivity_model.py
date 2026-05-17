"""
Train a farmer-scale productivity model.

This model predicts yield in metric tons per hectare from planning-time fields,
then the API scales that yield by the farmer's desired square meters. It avoids
using Area harvested as a user input because that is only known after harvest.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "fulldataset.csv"
MODEL_DIR = BASE_DIR / "model_artifacts"

CATEGORICAL_FEATURES = ["MUNICIPALITY", "FARM TYPE", "MONTH", "CROP"]
NUMERICAL_FEATURES = ["YEAR"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET_COLUMN = "Productivity(mt/ha)"


def load_clean_dataset():
    df = pd.read_csv(DATASET_PATH)

    for column in [
        "Production(mt)",
        "Area planted(ha)",
        "Area harvested(ha)",
        "Productivity(mt/ha)",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].astype(str).str.upper().str.strip()

    before = len(df)
    df = df.dropna(
        subset=[
            "Production(mt)",
            "Area planted(ha)",
            "Area harvested(ha)",
            "Productivity(mt/ha)",
            "YEAR",
        ]
    )
    df = df[
        (df["Production(mt)"] > 0)
        & (df["Area planted(ha)"] > 0)
        & (df["Area harvested(ha)"] > 0)
        & (df["Productivity(mt/ha)"] > 0)
    ].copy()
    df = df.drop_duplicates()

    placeholder_mask = (
        df["Area planted(ha)"].eq(5)
        & df["Area harvested(ha)"].eq(5)
        & df["Production(mt)"].eq(55)
        & df["Productivity(mt/ha)"].eq(11)
    )
    placeholders_removed = int(placeholder_mask.sum())
    df = df[~placeholder_mask].copy()

    clipped_groups = []
    outliers_removed = 0
    for _, crop_df in df.groupby("CROP", sort=False):
        lower = crop_df[TARGET_COLUMN].quantile(0.01)
        upper = crop_df[TARGET_COLUMN].quantile(0.99)
        kept = crop_df[
            (crop_df[TARGET_COLUMN] >= lower)
            & (crop_df[TARGET_COLUMN] <= upper)
        ]
        outliers_removed += len(crop_df) - len(kept)
        clipped_groups.append(kept)

    df = pd.concat(clipped_groups, ignore_index=True)

    cleaning_summary = {
        "raw_rows": before,
        "clean_rows": len(df),
        "exact_placeholder_rows_removed": placeholders_removed,
        "crop_quantile_outlier_rows_removed": int(outliers_removed),
        "notes": [
            "Removed non-numeric or non-positive production/area/productivity rows.",
            "Removed repeated 5 ha / 5 ha / 55 mt / 11 mt-ha pattern.",
            "Removed crop-level productivity values outside the 1st-99th percentile range.",
        ],
    }
    return df, cleaning_summary


def build_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(name, pipeline, x_train, x_test, y_train, y_test):
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    return {
        "name": name,
        "pipeline": pipeline,
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
    }


def main():
    print("=" * 80)
    print("TRAINING FARMER-SCALE PRODUCTIVITY MODEL")
    print("=" * 80)

    df, cleaning_summary = load_clean_dataset()
    print(f"Rows after cleaning: {len(df):,}")

    x = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    candidates = [
        (
            "Random Forest Productivity",
            build_pipeline(
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=30,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                )
            ),
        ),
        (
            "Extra Trees Productivity",
            build_pipeline(
                ExtraTreesRegressor(
                    n_estimators=300,
                    max_depth=30,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                )
            ),
        ),
    ]

    results = [
        evaluate_model(name, pipeline, x_train, x_test, y_train, y_test)
        for name, pipeline in candidates
    ]
    best = max(results, key=lambda item: item["r2"])

    train_years = df[df["YEAR"] <= 2022]
    test_years = df[df["YEAR"] >= 2023]
    time_holdout = None
    if len(test_years) > 0:
        time_pipeline = build_pipeline(
            ExtraTreesRegressor(
                n_estimators=300,
                max_depth=30,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )
        )
        time_pipeline.fit(train_years[FEATURE_COLUMNS], train_years[TARGET_COLUMN])
        time_predictions = time_pipeline.predict(test_years[FEATURE_COLUMNS])
        time_holdout = {
            "train_years": "2015-2022",
            "test_years": "2023-2024",
            "r2": float(r2_score(test_years[TARGET_COLUMN], time_predictions)),
            "mae": float(mean_absolute_error(test_years[TARGET_COLUMN], time_predictions)),
            "rmse": float(np.sqrt(mean_squared_error(test_years[TARGET_COLUMN], time_predictions))),
            "test_rows": int(len(test_years)),
        }

    print("\nCandidate results:")
    for result in results:
        print(
            f"  {result['name']}: "
            f"R2={result['r2']:.4f}, "
            f"MAE={result['mae']:.2f}, "
            f"RMSE={result['rmse']:.2f}"
        )

    if time_holdout:
        print(
            "\nTime holdout: "
            f"R2={time_holdout['r2']:.4f}, "
            f"MAE={time_holdout['mae']:.2f}, "
            f"RMSE={time_holdout['rmse']:.2f}"
        )

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "productivity_model.pkl.xz"
    joblib.dump(best["pipeline"], model_path, compress=("xz", 3))

    metadata = {
        "model_type": best["name"],
        "training_date": datetime.now().isoformat(timespec="seconds"),
        "target_variable": TARGET_COLUMN,
        "prediction_strategy": "Predict productivity_mt_per_ha, then scale by Area_sqm / 10000.",
        "input_features": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "n_samples_train": int(len(x_train)),
        "n_samples_test": int(len(x_test)),
        "test_r2_score": best["r2"],
        "test_mae": best["mae"],
        "test_rmse": best["rmse"],
        "time_holdout": time_holdout,
        "candidate_results": [
            {
                "name": result["name"],
                "test_r2_score": result["r2"],
                "test_mae": result["mae"],
                "test_rmse": result["rmse"],
            }
            for result in results
        ],
        "cleaning_summary": cleaning_summary,
        "note": "Uses planning-time fields only. Area harvested is not requested from farmers.",
    }

    with open(MODEL_DIR / "productivity_model_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)

    categorical_values = {
        column: sorted(df[column].dropna().unique().tolist())
        for column in CATEGORICAL_FEATURES
    }
    with open(MODEL_DIR / "productivity_categorical_values.json", "w", encoding="utf-8") as handle:
        json.dump(categorical_values, handle, indent=4)

    print(f"\nSaved model: {model_path}")
    print("Saved metadata and categorical values.")
    print("=" * 80)


if __name__ == "__main__":
    main()

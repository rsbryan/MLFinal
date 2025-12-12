from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from catboost import CatBoostRegressor  # type: ignore

    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


@dataclass(frozen=True)
class Metrics:
    r2: float
    mae: float
    rmse: float


def _evaluate(y_true, y_pred) -> Metrics:
    return Metrics(
        r2=float(r2_score(y_true, y_pred)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(root_mean_squared_error(y_true, y_pred)),
    )

def _add_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col not in df.columns:
        return df

    dt = pd.to_datetime(df[date_col], errors="coerce")
    df = df.copy()
    df[f"{date_col}_year"] = dt.dt.year
    df[f"{date_col}_month"] = dt.dt.month
    df[f"{date_col}_day"] = dt.dt.day
    df = df.drop(columns=[date_col])
    return df


def main() -> int:
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=".*encountered in matmul.*",
    )

    parser = argparse.ArgumentParser(
        description="Train house price models (Linear/Ridge, Random Forest, CatBoost)."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/usa_housing.csv"),
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="price",
        help="Name of the target column.",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default="date",
        help="Optional date column to expand into year/month/day features.",
    )
    parser.add_argument(
        "--include-categorical",
        action="store_true",
        help="If set, include non-numeric columns via one-hot encoding (can be high-dimensional).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ridge",
        choices=["ridge", "linear", "rf", "catboost"],
        help="Model to train: ridge, linear, rf (RandomForest), catboost.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Regularization strength for Ridge (only used when --model ridge).",
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=300,
        help="Number of trees for Random Forest (only used when --model rf).",
    )
    parser.add_argument(
        "--drop-cols",
        type=str,
        default="street",
        help="Comma-separated columns to always drop (e.g. 'street,country').",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for the test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to save the trained model pipeline (defaults to artifacts/<model>.joblib).",
    )
    args = parser.parse_args()

    if args.out is None:
        args.out = Path(f"artifacts/{args.model}.joblib")

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise SystemExit(
            f"Target column '{args.target}' not found. Available: {df.columns.tolist()}"
        )

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    drop_cols = [c for c in drop_cols if c in df.columns and c != args.target]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    y = df[args.target]
    X = df.drop(columns=[args.target])
    X = _add_date_features(X, args.date_col)

    # Basic preprocessing -> numeric impute and categorical one hot encoding
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols] if args.include_categorical else []

    # Tree models in sklearn generally require dense matrices.
    onehot_dense = args.model in ("rf", "catboost")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=not onehot_dense,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    if args.model == "ridge":
        regressor = Ridge(alpha=args.ridge_alpha, random_state=args.random_state)
    elif args.model == "linear":
        regressor = LinearRegression()
    elif args.model == "rf":
        regressor = RandomForestRegressor(
            n_estimators=args.rf_n_estimators,
            random_state=args.random_state,
            n_jobs=-1,
        )
    else:
        if not HAS_CATBOOST:
            raise SystemExit(
                "CatBoost is not installed. Install it with: pip install catboost\n"
                "Then rerun with: --model catboost"
            )
        regressor = CatBoostRegressor(
            verbose=False,
            random_seed=args.random_state,
        )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", regressor),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = _evaluate(y_test, preds)

    print("=== Model Results ===")
    if args.model == "ridge":
        model_str = f"ridge (alpha={args.ridge_alpha})"
    elif args.model == "rf":
        model_str = f"rf (n_estimators={args.rf_n_estimators})"
    else:
        model_str = args.model
    print(f"model: {model_str}")
    print(f"rows: {len(df)} | features: {X.shape[1]} | target: {args.target}")
    print(f"split: train={len(X_train)} test={len(X_test)}")
    print(f"R^2 : {metrics.r2:.4f}")
    print(f"MAE : {metrics.mae:,.2f}")
    print(f"RMSE: {metrics.rmse:,.2f}")
    try:
        sample_x = X_test.iloc[[0]]
        sample_pred = float(model.predict(sample_x)[0])
        sample_actual = float(y_test.iloc[0])
        print(f"sample prediction: predicted={sample_pred:,.0f} actual={sample_actual:,.0f}")
    except Exception:
        pass

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(f"saved model -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



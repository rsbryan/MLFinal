from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
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
    from catboost import CatBoostRegressor, Pool  

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
        default=Path("data/realtor-data.zip.csv"),
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
        default="prev_sold_date",
        help="Optional date column to expand into year/month/day features.",
    )
    parser.add_argument(
        "--include-categorical",
        action="store_true",
        help="If set, include non-numeric columns via one-hot encoding (can be high-dimensional).",
    )
    parser.add_argument(
        "--log-target",
        action="store_true",
        help="If set, train on log1p(target) and invert predictions for reporting.",
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
        default=100,
        help="Number of trees for Random Forest (only used when --model rf).",
    )
    parser.add_argument(
        "--cb-iterations",
        type=int,
        default=2000,
        help="CatBoost iterations (only used when --model catboost).",
    )
    parser.add_argument(
        "--cb-depth",
        type=int,
        default=8,
        help="CatBoost tree depth (only used when --model catboost).",
    )
    parser.add_argument(
        "--cb-learning-rate",
        type=float,
        default=0.1,
        help="CatBoost learning rate (only used when --model catboost).",
    )
    parser.add_argument(
        "--cb-early-stopping",
        type=int,
        default=100,
        help="CatBoost early stopping rounds (only used when --model catboost).",
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
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of rows to load (useful for very large CSVs).",
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

    # For huge datasets, it's easy to run out of RAM. need to be careful with the dataset
    max_rows = args.max_rows
    try:
        if max_rows is None and args.data.exists() and args.data.stat().st_size > 100 * 1024 * 1024:
            max_rows = 200_000
            print(f"note: large dataset detected; loading first {max_rows:,} rows (override with --max-rows).")
    except Exception:
        pass

    df = pd.read_csv(args.data, nrows=max_rows)
    if args.target not in df.columns:
        raise SystemExit(
            f"Target column '{args.target}' not found. Available: {df.columns.tolist()}"
        )

    # Ensure target is numeric and drop missing targets 
    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    drop_cols = [c for c in drop_cols if c in df.columns and c != args.target]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    y = df[args.target]
    keep = y.notna()
    df = df.loc[keep].copy()
    y = df[args.target]
    X = df.drop(columns=[args.target])
    X = _add_date_features(X, args.date_col)

    # Basic preprocessing -> numeric impute and categorical one hot encoding
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols] if args.include_categorical else []

    # --- Model-specific training paths ---
    # CatBoost: use native categorical handling which is better than one hot for this dataset
    if args.model == "catboost":
        if not HAS_CATBOOST:
            raise SystemExit(
                "CatBoost is not installed. Install it with: pip install catboost\n"
                "Then rerun with: --model catboost"
            )

        # Build a clean numeric + categorical frame for CatBoost
        # - numeric: coerce to numbers, fill missing with median
        # - categorical: strings, fill missing with sentinel value
        X_num = X[numeric_cols].copy()
        for c in numeric_cols:
            X_num[c] = pd.to_numeric(X_num[c], errors="coerce")
        X_num = X_num.fillna(X_num.median(numeric_only=True))

        X_cat = X[categorical_cols].copy()
        for c in categorical_cols:
            X_cat[c] = X_cat[c].astype(str).fillna("missing")

        X_cb = pd.concat([X_num, X_cat], axis=1)
        # CatBoost will expect categorical feature indices
        cat_features = list(range(len(numeric_cols), len(numeric_cols) + len(categorical_cols)))

        # Hold out split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_cb,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        # Optionally train on log1p to reduce the impact of extreme outliers
        y_train_used = y_train.to_numpy()
        y_test_used = y_test.to_numpy()
        if args.log_target:
            y_train_used = np.log1p(y_train_used)
            y_test_used = np.log1p(y_test_used)

        # Use CatBoost Pool so it knows which columns are categorical
        train_pool = Pool(X_train, y_train_used, cat_features=cat_features)
        test_pool = Pool(X_test, y_test_used, cat_features=cat_features)

        # CatBoost training with early stopping.
        # allow_writing_files=False prevents creating catboost_info/ logs
        regressor = CatBoostRegressor(
            iterations=args.cb_iterations,
            depth=args.cb_depth,
            learning_rate=args.cb_learning_rate,
            loss_function="RMSE",
            eval_metric="R2",
            random_seed=args.random_state,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,  # prevents catboost_info/
        )

        regressor.fit(
            train_pool,
            eval_set=test_pool,
            use_best_model=True,
            early_stopping_rounds=args.cb_early_stopping,
        )

        # Predict on holdout set and invert log transform if used.
        preds = regressor.predict(X_test)
        if args.log_target:
            preds = np.expm1(preds)
            y_for_metrics = y_test.to_numpy()
        else:
            y_for_metrics = y_test.to_numpy()

        # Print quick evaluation metrics for the holdout split.
        metrics = _evaluate(y_for_metrics, preds)

        print("=== Model Results ===")
        print("model: catboost (native categorical)")
        print(f"rows: {len(df)} | features: {X_cb.shape[1]} | target: {args.target}")
        print(f"split: train={len(X_train)} test={len(X_test)}")
        print(f"R^2 : {metrics.r2:.4f}")
        print(f"MAE : {metrics.mae:,.2f}")
        print(f"RMSE: {metrics.rmse:,.2f}")
        try:
            print(
                f"sample prediction: predicted={float(preds[0]):,.0f} actual={float(y_for_metrics[0]):,.0f}"
            )
        except Exception:
            pass

        # Save model + minimal metadata needed to reproduce preprocessing at inference time
        args.out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": regressor,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "cat_features": cat_features,
                "log_target": bool(args.log_target),
            },
            args.out,
        )
        print(f"saved model -> {args.out}")
        return 0

    # --- Sklearn models (ridge/linear/rf) ---
    # RandomForestRegressor expects dense matrices, so force dense one-hot when RF is selected
    onehot_dense = args.model == "rf"

    # Preprocessing for sklearn models:
    # - numeric: median impute + standardize
    # - categorical (optional): most-frequent impute + one-hot encode
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

    # Choose the sklearn regressor
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
        raise SystemExit("Internal error: catboost should be handled in the native catboost branch.")

    # End-to-end sklearn pipeline
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", regressor),
        ]
    )

    # Train/test split, then fit + evaluate on the holdout test set
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = _evaluate(y_test, preds)

    # Print evaluation and persist the trained pipeline for later use
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



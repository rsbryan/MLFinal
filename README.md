## House Price Prediction (Linear Regression MVP)

Simple starter project: train a linear regression model to predict `price` from the provided housing features.

### Project layout
- **`data/usa_housing.csv`**: dataset (copied into the repo)
- **`train_linear_regression.py`**: trains + evaluates + saves model
- **`artifacts/linear_regression.joblib`**: saved model pipeline (created after training)

### Setup

```bash
cd into /MLFinal"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train the MVP model

```bash
./.venv/bin/python train_linear_regression.py --data data/usa_housing.csv
```

### Output
- Prints **R²**, **MAE**, and **RMSE** on a holdout test split
- Saves the trained pipeline to **`artifacts/linear_regression.joblib`**

### Notes (what the MVP uses)
- By default, the MVP uses **numeric features** plus optional **date-derived features** (`date_year`, `date_month`, `date_day`).
- The default model is **Ridge regression** (still a linear regression model) for better numerical stability.
- If you want plain least-squares Linear Regression, run:

```bash
./.venv/bin/python train_linear_regression.py --data data/usa_housing.csv --model linear
```
- To include text/categorical columns via one-hot encoding (can be very high-dimensional), run:

```bash
./.venv/bin/python train_linear_regression.py --data data/usa_housing.csv --include-categorical
```



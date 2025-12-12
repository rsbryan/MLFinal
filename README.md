### House Price Prediction (Linear Regression)

Train a simple linear regression model to predict `price` from housing features.

### Setup

```bash
cd into /MLFinal"
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt 
```

### Run (simple commands)

- **1) Linear Regression** (plain least squares):

```bash
./.venv/bin/python run.py linear
```

- **2) Linear + categorical features** (usually better; expected R² ~0.50–0.56):

```bash
./.venv/bin/python run.py linear_cat
```

- **3) Random Forest + categorical features** (expected R² often ~0.50–0.75):

```bash
./.venv/bin/python run.py rf
```

- **4) CatBoost + categorical features** (install required; expected R² often ~0.55–0.85):

```bash
./.venv/bin/python run.py catboost
```

All runs save a `.joblib` model to `artifacts/` by default.



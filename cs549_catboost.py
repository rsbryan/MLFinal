import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from catboost import CatBoostRegressor, Pool

import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("realtor-data.zip.csv")
df.head()

# feature selection and cleaning
df_model = df[["price", "state", "city", "zip_code", "bed", "bath", "house_size"]].copy()

df_model = df_model.dropna(subset=["price", "state", "city", "zip_code", "bed", "bath", "house_size"])
df_model["zip_code"] = df_model["zip_code"].astype(str)
df_model["state"] = df_model["state"].astype(str)
df_model["city"] = df_model["city"].astype(str)
df_model = df_model.copy()
print(df_model.columns)

# sets the number of examples to 25,000 so the program is more manageable
max_per_state = 25000

df_model = (
    df_model
    .groupby("state", group_keys=False)
    .apply(lambda g: g.sample(min(len(g), max_per_state), random_state=42))
    .reset_index(drop=True)
)

print(df_model.columns)

df_model = df_model.reset_index(drop=True) 

print(df_model.columns)

# logarithmic value of the price
X = df_model.drop("price", axis=1)
y = np.log1p(df_model["price"])

from sklearn.model_selection import train_test_split

#limits total examples to be training to 1,100,000 entries
N = 1_100_000
if len(X) > N:
    X_sample = X.sample(n=N, random_state=42)
    y_sample = y.loc[X_sample.index]
else:
    X_sample = X
    y_sample = y

X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42
)

# tell CatBoost which columns are categorical
# CatBoost model that is trained based on the given data
cat_features = ["state", "city", "zip_code"]

train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

model = CatBoostRegressor(
    depth=8,
    learning_rate=0.12,
    n_estimators=500,
    loss_function="MAE",
    random_seed=42,
    verbose=100
)

model.fit(train_pool, eval_set=test_pool)

# evalutes the model
y_pred_log = model.predict(test_pool)

# convert predictions and true values back to dollars
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

print("MAE:", mean_absolute_error(y_true, y_pred))
print("R2 score:", r2_score(y_true, y_pred))

from catboost import Pool

# helper function that creates an interactive space for the user
# to input housing features to obtain an estimated price on the house
def ask_and_predict():
    print("Enter house information to get an estimated price:\n")

    state = input("State: ")
    city = input("City: ")
    zip_code = input("Zip code: ")
    bed = float(input("Number of bedrooms: "))
    bath = float(input("Number of bathrooms: "))
    house_size = float(input("House size (sqft): "))

    sample = pd.DataFrame({
        "state": [state],
        "city": [city],
        "zip_code": [str(zip_code)],
        "bed": [bed],
        "bath": [bath],
        "house_size": [house_size],
        # add extra features here if df_model has them,
        # e.g. "acre_lot", "status", etc.
    })

    sample_pool = Pool(sample, cat_features=cat_features)

    # model output is log(price + 1)
    predicted_log = model.predict(sample_pool)[0]

    # convert back to dollars
    predicted_price = np.expm1(predicted_log)

    print("\nEstimated price: ${:,.0f}".format(predicted_price))
    return predicted_price

# call to the function
ask_and_predict()
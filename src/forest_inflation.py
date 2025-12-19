import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split


def optimize_correlation(df, target_column="price"):
    """Находит линейную комбинацию признаков, которая максимально коррелирует с заданным целевым признаком."""
    features = df.drop(columns=target_column).columns
    X = df[features].values
    P = df[target_column].values
    n = len(features)

    cov_matrix = np.cov(X, rowvar=False)
    cov_XP = np.cov(X, P, rowvar=False)[:n, n:].flatten()

    def target_function(a_rest):
        weights = np.insert(a_rest, 0, 1.0)
        L = weights @ cov_XP
        S = weights @ cov_matrix @ weights
        return -np.abs(L) / np.sqrt(S)  # Минимизируем -|L|/sqrt(S)

    initial_guess = np.zeros(n - 1)
    result = minimize(target_function, initial_guess)
    weights = np.insert(result.x, 0, 1.0)
    linear_combination = X @ weights
    correlation = np.corrcoef(linear_combination, P)[0, 1]
    return correlation, weights
inflation_dct = {
    (2018, 1): 1,
    (2018, 2): 1.0031,
    (2018, 3): 1.00520651,
    (2018, 4): 1.008121608879,
    (2018, 5): 1.01195247099274,
    (2018, 6): 1.0157978903825124,
    (2018, 7): 1.0207753000453867,
    (2018, 8): 1.0235313933555092,
    (2018, 9): 1.0236337464948448,
    (2018, 10): 1.0252715604892366,
    (2018, 11): 1.028860010950949,
    (2018, 12): 1.0340043110057036,
    (2019, 1): 1.0426899472181514,
    (2019, 2): 1.0532211156850548,
    (2019, 3): 1.057855288594069,
    (2019, 4): 1.0612404255175703,
    (2019, 5): 1.0643180227515712,
    (2019, 6): 1.0679367040289265,
    (2019, 7): 1.068363878710538,
    (2019, 8): 1.0705006064679592,
    (2019, 9): 1.0679314050124362,
    (2019, 10): 1.0662227147644163,
    (2019, 11): 1.06760880429361,
    (2019, 12): 1.070598108945632,
    (2020, 1): 1.0744522621378363,
    (2020, 2): 1.0787500711863875,
    (2020, 3): 1.0823099464213026,
    (2020, 4): 1.0882626511266198,
    (2020, 5): 1.0972952311309707,
    (2020, 6): 1.1002579282550242,
    (2020, 7): 1.1026784956971853,
    (2020, 8): 1.1065378704321256,
    (2020, 9): 1.1060952552839527,
    (2020, 10): 1.105320988605254,
    (2020, 11): 1.1100738688562566,
    (2020, 12): 1.117955393325136,
    (2021, 1): 1.1272344230897346,
    (2021, 2): 1.1347868937244359,
    (2021, 3): 1.1436382314954865,
    (2021, 4): 1.1511862438233567,
    (2021, 5): 1.157863124037532,
    (2021, 6): 1.1664313111554099,
    (2021, 7): 1.1744796872023822,
    (2021, 8): 1.1781205742327097,
    (2021, 9): 1.1801233792089052,
    (2021, 10): 1.1872041194841587,
    (2021, 11): 1.200382085210433,
    (2021, 12): 1.2119057532284532,
    (2022, 1): 1.2348433804049265,
    (2022, 2): 1.248433804049265,
    (2022, 3): 1.343433804049265,
    (2022, 4): 1.3644,
    (2022, 5): 1.366,
    (2022, 6): 1.3612,
    (2022, 7): 1.3559,
    (2022, 8): 1.3488,
    (2022, 9): 1.3495,
    (2022, 10): 1.3519,
    (2022, 11): 1.3569,
    (2022, 12): 1.3675,
    (2023, 1): 1.379,
    (2023, 2): 1.3853,
    (2023, 3): 1.3904,
    (2023, 4): 1.3957,
    (2023, 5): 1.4,
    (2023, 6): 1.4052,
    (2023, 7): 1.4141,
    (2023, 8): 1.4181,
    (2023, 9): 1.4304,
    (2023, 10): 1.4423,
    (2023, 11): 1.4583,
    (2023, 12): 1.4689,
    (2024, 1): 1.4815,
    (2024, 2): 1.4916,
    (2024, 3): 1.4974,
    (2024, 4): 1.5049,
    (2024, 5): 1.516,
    (2024, 6): 1.5257,
    (2024, 7): 1.5431,
    (2024, 8): 1.5462,
    (2024, 9): 1.5536,
    (2024, 10): 1.5653,
    (2024, 11): 1.5877,
    (2024, 12): 1.6087,
}

df = pd.read_csv("all_v2.csv")

df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

df = df.drop(["date", "region", "time", "level"], axis=1)
df = df[
    (df["price"].quantile(0.1) <= df["price"])
    & (df["price"] <= df["price"].quantile(0.9))
]
df = df[
    (df["area"].quantile(0.05) <= df["area"])
    & (df["area"] <= df["area"].quantile(0.95))
]
df = df[
    (df["kitchen_area"].quantile(0.05) <= df["kitchen_area"])
    & (df["kitchen_area"] <= df["kitchen_area"].quantile(0.95))
]
df = df[df["kitchen_area"] < df["area"]]

df["price"] = df["price"] / df.apply(
    lambda row: inflation_dct.get((row["year"], row["month"]), 1), axis=1
)
print("Датасет обработан")

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

target_train = df_train["price"]
features_train = df_train.drop(["price"], axis=1)

target_test = df_test["price"]
features_test = df_test.drop(["price"], axis=1)


X_train = df_train.drop(
    [
        "price",
    ],
    axis=1,
)
y_train = target_train

X_test = df_test.drop(
    [
        "price",
    ],
    axis=1,
)
y_test = target_test

n_estimators = 10
model = RandomForestRegressor(
    n_estimators=n_estimators,
    random_state=42,
    n_jobs=14,
)
model.fit(X_train, y_train)
print(X_train.columns)
print("Научил лес")
predictions = model.predict(X_test)
print("Предсказание с учётом инфляции:")
rmse = root_mean_squared_error(target_test, predictions)
mae = mean_absolute_error(target_test, predictions)
r2 = r2_score(target_test, predictions)
rmse_relative = rmse / target_test.mean()
mae_relative = mae / target_test.mean()
mape = np.mean(np.abs((predictions - target_test) / target_test))
smape = np.mean(np.abs(predictions - target_test)) / np.mean(predictions + target_test)
results = {
    "rmse": rmse,
    "rmse_relative": rmse_relative,
    "mae": mae,
    "mae_relative": mae_relative,
    "mape": mape,
    "smape": smape,
    "r2": r2,
}

print(n_estimators,"деревьев,",)
for el in results:
    print(el, ':', results[el])
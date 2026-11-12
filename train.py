"""Train and save two basic coffee rating models."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


DATA_URL = (
    "https://raw.githubusercontent.com/leontoddjohnson/"
    "datasets/refs/heads/main/data/coffee_analysis.csv"
)
MODEL_DIR = Path(__file__).resolve().parent
ROAST_CATEGORIES = {
    "Light": 0,
    "Medium-Light": 1,
    "Medium": 2,
    "Medium-Dark": 3,
    "Dark": 4,
}


def roast_category(value):
    """Encode a roast, using NaN for missing or unknown roast values."""
    return ROAST_CATEGORIES.get(value, np.nan)


def train_models():
    """Download coffee data, evaluate the models, and save fitted models."""
    data = pd.read_csv(DATA_URL)
    for column in ["100g_USD", "rating"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["100g_USD", "rating"]).copy()
    data["roast_cat"] = data["roast"].map(roast_category)

    training, testing = train_test_split(
        data, test_size=0.2, random_state=42
    )
    models = [
        ("model_1.pickle", LinearRegression(), ["100g_USD"]),
        (
            "model_2.pickle",
            DecisionTreeRegressor(random_state=42),
            ["100g_USD", "roast_cat"],
        ),
    ]

    for filename, model, features in models:
        model.fit(training[features].to_numpy(), training["rating"])
        predictions = model.predict(testing[features].to_numpy())
        error = mean_absolute_error(testing["rating"], predictions)
        print(f"{filename}: test MAE = {error:.3f} rating points")

        # Refit on all usable rows before saving the final model.
        model.fit(data[features].to_numpy(), data["rating"])
        with (MODEL_DIR / filename).open("wb") as file:
            pickle.dump(model, file)
        print(f"Saved {filename}")


if __name__ == "__main__":
    train_models()

    # Check predictions from the saved models on new input.
    with (MODEL_DIR / "model_1.pickle").open("rb") as file:
        linear_model = pickle.load(file)
    with (MODEL_DIR / "model_2.pickle").open("rb") as file:
        tree_model = pickle.load(file)

    print("Price-only predictions:", linear_model.predict([[10.0], [15.0]]))
    new_coffees = [[10.0, 1], [15.0, 3], [8.5, np.nan]]
    print("Price and roast predictions:", tree_model.predict(new_coffees))
    unusual_roasts = [[10.0, -99], [15.0, 2938.24]]
    print("Unusual roast codes:", tree_model.predict(unusual_roasts))

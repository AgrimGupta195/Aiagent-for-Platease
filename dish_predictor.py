import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# ---------------------- Config: Ingredients per Dish ----------------------
ingredients_per_dish = {
    'Vegetable Taco': {'Tortilla': 1, 'Paneer': 0.1},
    'Veg Wrap': {'Wrap': 1, 'Sauce': 0.05},
    'Cheesy Blueberry Cake': {'Blueberry': 0.1, 'Cream': 0.05},
    'Strawberry Mojito': {'Strawberry': 0.1, 'Mint': 0.02},
    'Mango Smoothie': {'Mango': 0.2, 'Milk': 0.25},
    'Chinese Noodles': {'Noodles': 1, 'Oil': 0.1, 'Veggies': 0.2},
    'Triple Chocolate Brownie': {'Chocolate': 0.3, 'Flour': 0.2},
    'Hazelnut Frappe': {'Hazelnut': 0.1, 'Milk': 0.2},
    'Americano': {'Coffee': 0.05, 'Water': 0.2},
}

# ---------------------- Load & Pivot Data ----------------------
def load_and_pivot(filepath):
    df = pd.read_csv(filepath, parse_dates=['date'])
    print("Loaded rows:", len(df))
    df_wide = df.pivot_table(index='date', columns='dish', values='quantity').fillna(0)
    return df, df_wide

# ---------------------- Train Linear Regression Models Per Dish ----------------------
def train_models(df_wide):
    models = {}
    for dish in df_wide.columns:
        X = np.arange(len(df_wide)).reshape(-1, 1)
        y = df_wide[dish].values
        model = LinearRegression()
        model.fit(X, y)
        models[dish] = model
    return models

# ---------------------- Predict Future Demand with Non-Negative Constraint ----------------------
def predict_future(models, df_wide, future_days=30):
    future_dates = [df_wide.index[-1] + timedelta(days=i) for i in range(1, future_days + 1)]
    predictions = {}
    for dish, model in models.items():
        X_future = np.arange(len(df_wide), len(df_wide) + future_days).reshape(-1, 1)
        preds = model.predict(X_future).round().astype(int)
        preds = np.clip(preds, 0, None)  # No negative predictions
        predictions[dish] = preds
    df_pred = pd.DataFrame(predictions, index=future_dates)
    return df_pred

# ---------------------- Estimate Ingredient Usage ----------------------
def estimate_ingredient_usage(df_pred):
    ingredient_totals = {}
    for dish in df_pred.columns:
        total_qty = df_pred[dish].sum()
        for ing, amount_per_dish in ingredients_per_dish.get(dish, {}).items():
            ingredient_totals.setdefault(ing, 0)
            ingredient_totals[ing] += total_qty * amount_per_dish
    return ingredient_totals

# ---------------------- Generate Rush Charts ----------------------
def generate_rush_charts(df):
    df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce').dt.hour
    df['weekday'] = df['date'].dt.day_name()

    hourly_trend = df.groupby('hour').size().reindex(range(24), fill_value=0)
    weekly_trend = df.groupby('weekday').size().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], fill_value=0
    )

    plt.figure(figsize=(10, 4))
    hourly_trend.plot(kind='line', title='Rush by Hour', marker='o')
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Orders")
    plt.grid(True)
    plt.xticks(range(0, 24))
    plt.show()

    plt.figure(figsize=(8, 4))
    weekly_trend.plot(kind='bar', title='Rush by Weekday', color='orange')
    plt.xlabel("Weekday")
    plt.ylabel("Number of Orders")
    plt.grid(axis='y')
    plt.show()

# ---------------------- Plot Demand Trend for a Dish ----------------------
def plot_trend(df_wide, dish):
    plt.figure(figsize=(10, 4))
    plt.plot(df_wide.index, df_wide[dish], label='Historical')
    plt.title(f"Demand Trend for {dish}")
    plt.xlabel("Date")
    plt.ylabel("Quantity Ordered")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------- Main Function ----------------------
def main():
    filepath = 'phase 1/orders.csv'  # Path to your dataset

    # ✅ Load and pivot data
    df, df_wide = load_and_pivot(filepath)
    print(f"Loaded {len(df)} rows after cleaning.")

    # ✅ Train ML models
    models = train_models(df_wide)

    # ✅ Predict demand
    df_pred = predict_future(models, df_wide, future_days=30)
    df_pred.to_csv("future_demand_predictions.csv")
    print("Future demand predictions saved to 'future_demand_predictions.csv'.")

    # ✅ Estimate ingredient usage
    ingredient_usage = estimate_ingredient_usage(df_pred)
    print("\nEstimated Ingredient Usage for Next 30 Days:")
    for ing, qty in ingredient_usage.items():
        print(f"{ing}: {qty:.2f} units")

    # ✅ Generate rush analysis charts
    print("\nGenerating rush hour and weekday charts...")
    generate_rush_charts(df)

    # ✅ Plot historical trend
    plot_trend(df_wide, 'Vegetable Taco')

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# HOUSE PRICE PREDICTION - MODEL TRAINING
# ============================================================

print("=" * 60)
print("HOUSE PRICE PREDICTION - MODEL TRAINING")
print("=" * 60)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs("models", exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading Dataset...")

df = pd.read_csv("data/housing.csv")

print("\nOriginal Dataset Shape:")
print(df.shape)


# ============================================================
# DATA CLEANING
# ============================================================

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with missing values
df = df.dropna()

print("\nDataset Shape After Cleaning:")
print(df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Create Rooms per Bedroom feature
df["Rooms_per_Bedroom"] = (
    df["Avg. Area Number of Rooms"] /
    df["Avg. Area Number of Bedrooms"]
)

# Create Population per Room feature
df["Population_per_Room"] = (
    df["Area Population"] /
    df["Avg. Area Number of Rooms"]
)

# Replace infinite values
df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# Remove invalid rows
df = df.dropna()

print("\nNew Engineered Features:")
print("- Rooms_per_Bedroom")
print("- Population_per_Room")


# ============================================================
# SELECT FEATURES AND TARGET
# ============================================================

features = [
    "Avg. Area Income",
    "Avg. Area House Age",
    "Avg. Area Number of Rooms",
    "Avg. Area Number of Bedrooms",
    "Area Population",
    "Rooms_per_Bedroom",
    "Population_per_Room"
]

target = "Price"

X = df[features]
y = df[target]


print("\nSelected Features:")

for feature in features:
    print("-", feature)

print("\nTarget:")
print("-", target)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ============================================================
# MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, model):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\n" + name)
    print("-" * 40)

    print(f"MAE  : ${mae:,.2f}")
    print(f"RMSE : ${rmse:,.2f}")
    print(f"R²   : {r2:.4f}")

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("LINEAR REGRESSION")
print("=" * 60)

linear_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

linear_model.fit(
    X_train,
    y_train
)

linear_result = evaluate_model(
    "Linear Regression",
    linear_model
)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

rf_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    (
        "model",
        RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        )
    )
])

rf_model.fit(
    X_train,
    y_train
)

rf_result = evaluate_model(
    "Random Forest",
    rf_model
)


# ============================================================
# GRADIENT BOOSTING
# ============================================================

print("\n" + "=" * 60)
print("GRADIENT BOOSTING")
print("=" * 60)

gb_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    (
        "model",
        GradientBoostingRegressor(
            random_state=42
        )
    )
])

gb_model.fit(
    X_train,
    y_train
)

gb_result = evaluate_model(
    "Gradient Boosting",
    gb_model
)


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING - RANDOM FOREST")
print("=" * 60)

rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    (
        "model",
        RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        )
    )
])


# Parameters to test

param_grid = {
    "model__n_estimators": [
        100,
        200
    ],

    "model__max_depth": [
        None,
        10,
        20
    ],

    "model__min_samples_split": [
        2,
        5
    ],

    "model__min_samples_leaf": [
        1,
        2
    ]
}


print("\nGridSearchCV is running...")
print("Please wait...")


grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(
    X_train,
    y_train
)


# ============================================================
# BEST TUNED MODEL
# ============================================================

best_model = grid_search.best_estimator_


print("\n" + "=" * 60)
print("BEST HYPERPARAMETERS")
print("=" * 60)

print(
    grid_search.best_params_
)


print("\nBest Cross-Validation R²:")
print(
    f"{grid_search.best_score_:.4f}"
)


# ============================================================
# EVALUATE TUNED MODEL
# ============================================================

print("\n" + "=" * 60)
print("TUNED RANDOM FOREST EVALUATION")
print("=" * 60)

tuned_result = evaluate_model(
    "Tuned Random Forest",
    best_model
)


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results = pd.DataFrame([
    linear_result,
    rf_result,
    gb_result,
    tuned_result
])

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# FIND BEST MODEL
# ============================================================

best_result = results.loc[
    results["R2"].idxmax()
]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    f"Model: {best_result['Model']}"
)

print(
    f"MAE  : ${best_result['MAE']:,.2f}"
)

print(
    f"RMSE : ${best_result['RMSE']:,.2f}"
)

print(
    f"R²   : {best_result['R2']:.4f}"
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)
# ============================================================
# SAVE BEST PERFORMING MODEL
# ============================================================

if best_result["Model"] == "Linear Regression":
    final_model = linear_model

elif best_result["Model"] == "Random Forest":
    final_model = rf_model

elif best_result["Model"] == "Gradient Boosting":
    final_model = gb_model

else:
    final_model = best_model


joblib.dump(
    final_model,
    "models/house_price_model.pkl"
)

print("\nModel saved successfully!")
print("Final Model:", best_result["Model"])
print("Location: models/house_price_model.pkl")

print("\nModel saved successfully!")

print(
    "Location: models/house_price_model.pkl"
)


# ============================================================
# SAVE MODEL RESULTS
# ============================================================

results.to_csv(
    "models/model_results.csv",
    index=False
)

print(
    "Results saved successfully!"
)

print(
    "Location: models/model_results.csv"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nCompleted Steps:")
print("✓ Data Cleaning")
print("✓ Feature Engineering")
print("✓ Train/Test Split")
print("✓ Linear Regression")
print("✓ Random Forest")
print("✓ Gradient Boosting")
print("✓ Hyperparameter Tuning")
print("✓ MAE Evaluation")
print("✓ RMSE Evaluation")
print("✓ R² Evaluation")
print("✓ Best Model Selection")
print("✓ Model Saved")
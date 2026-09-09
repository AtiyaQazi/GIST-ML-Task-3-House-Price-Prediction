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

# HOUSE PRICE PREDICTION - MODEL TRAINING
print("=" * 60)
print("HOUSE PRICE PREDICTION - MODEL TRAINING")
print("=" * 60)

# Create model directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# LOAD DATASET
df = pd.read_csv("data/housing.csv")

print("\nOriginal Dataset Shape:")
print(df.shape)

# DATA CLEANING

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with missing values
df = df.dropna()

# Remove infinite values
df = df.replace([np.inf, -np.inf], np.nan)

# Remove rows containing invalid values
df = df.dropna()
print("\nDataset Shape After Cleaning:")
print(df.shape)
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# FEATURE ENGINEERING

# Create Rooms per Bedroom
df["Rooms_per_Bedroom"] = (
    df["Avg. Area Number of Rooms"] /
    df["Avg. Area Number of Bedrooms"]
)
# Create Population per Room
df["Population_per_Room"] = (
    df["Area Population"] /
    df["Avg. Area Number of Rooms"]
)
# Remove infinite values created during feature engineering
df = df.replace([np.inf, -np.inf], np.nan)

# Remove invalid rows
df = df.dropna()

# SELECT FEATURES AND TARGET
features = [
    "Avg. Area Income",
    "Avg. Area House Age",
    "Avg. Area Number of Rooms",
    "Avg. Area Number of Bedrooms",
    "Area Population",
    "Rooms_per_Bedroom",
    "Population_per_Room"
]
X = df[features]
y = df["Price"]
print("\nSelected Features:")
for feature in features:
    print("-", feature)
print("\nTarget:")
print("Price")
print("\nEngineered Features:")
print("- Rooms_per_Bedroom")
print("- Population_per_Room")

# TRAIN / TEST SPLIT
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

# MODEL EVALUATION FUNCTION
def evaluate_model(name, model, X_test, y_test):
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

# LINEAR REGRESSION
linear_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
linear_model.fit(
    X_train,
    y_train
)

# RANDOM FOREST
rf_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    ))
])
rf_model.fit(
    X_train,
    y_train
)

# GRADIENT BOOSTING
gb_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", GradientBoostingRegressor(
        random_state=42
    ))
])
gb_model.fit(
    X_train,
    y_train
)

# INITIAL MODEL EVALUATION
print("\n" + "=" * 60)
print("INITIAL MODEL EVALUATION")
print("=" * 60)
results = []
results.append(
    evaluate_model(
        "Linear Regression",
        linear_model,
        X_test,
        y_test
    )
)
results.append(
    evaluate_model(
        "Random Forest",
        rf_model,
        X_test,
        y_test
    )
)
results.append(
    evaluate_model(
        "Gradient Boosting",
        gb_model,
        X_test,
        y_test
    )
)

# HYPERPARAMETER TUNING
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING - RANDOM FOREST")
print("=" * 60)
rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    ))
])
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}
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

# BEST TUNED MODEL
best_model = grid_search.best_estimator_
print("\nBest Hyperparameters:")
print(grid_search.best_params__)
print("\nBest Cross-Validation R²:")
print(f"{grid_search.best_score_:.4f}")

# TUNED MODEL EVALUATION
print("\n" + "=" * 60)
print("TUNED MODEL EVALUATION")
print("=" * 60)
tuned_result = evaluate_model(
    "Tuned Random Forest",
    best_model,
    X_test,
    y_test
)
results.append(tuned_result)

# MODEL COMPARISON
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
results_df = pd.DataFrame(results)
print(
    results_df.to_string(index=False)
)

# FIND BEST MODEL
best_result = results_df.loc[
    results_df["R2"].idxmax()
]
print("\nBest Model:")
print(best_result["Model"])

# SAVE BEST MODEL
joblib.dump(
    best_model,
    "models/house_price_model.pkl"
)

# SAVE MODEL RESULTS
results_df.to_csv(
    "models/model_results.csv",
    index=False
)
print("\n" + "=" * 60)
print("MODEL SAVING")
print("=" * 60)
print("\nBest model saved successfully!")
print("Model: Tuned Random Forest")
print("Location: models/house_price_model.pkl")
print("\nModel evaluation results saved successfully!")
print("Location: models/model_results.csv")

# FINAL RESULTS
print("\n" + "=" * 60)
print("FINAL MODEL RESULTS")
print("=" * 60)
print("\nMAE:")
print(f"${tuned_result['MAE']:,.2f}")
print("\nRMSE:")
print(f"${tuned_result['RMSE']:,.2f}")
print("\nR² Score:")
print(f"{tuned_result['R2']:.4f}")
print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
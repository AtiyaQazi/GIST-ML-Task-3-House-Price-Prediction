
# GIST-ML-Task-3-House-Price-Prediction

An end-to-end Machine Learning project that predicts house prices based on property and area-related features.

The project covers the complete machine learning workflow:

**Data Cleaning → Exploratory Data Analysis → Feature Engineering → Model Training → Hyperparameter Tuning → Model Evaluation → Prediction Interface**

---

## Project Overview

The House Price Prediction System is a machine learning application developed using Python and Scikit-learn.

The system uses housing data to train multiple regression models and selects the best-performing model based on evaluation metrics.

A Streamlit web interface allows users to enter house information and receive an estimated house price.

---

## Objectives

The main objectives of this project are:

- Clean and prepare the housing dataset
- Perform exploratory data analysis
- Create meaningful engineered features
- Train multiple machine learning regression models
- Perform hyperparameter tuning
- Evaluate models using MAE, RMSE, and R²
- Select the best-performing model
- Save the trained model
- Build an interactive prediction interface using Streamlit

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

---

## Project Structure

```text
Task-3-House-Price-Prediction/
│
├── data/
│   └── housing.csv
│
├── models/
│   ├── house_price_model.pkl
│   └── model_results.csv
│
├── screenshots/
│
├── age_vs_price.png
├── income_vs_price.png
├── price_distribution.png
├── rooms_vs_price.png
├── correlation_heatmap.png
│
├── app.py
├── data_analysis.py
├── model_training.py
├── train_model.py
│
├── requirements.txt
├── .gitignore
└── README.md
````

---

## Dataset

The project uses a housing dataset containing 5,000 records and the following original columns:

| Feature                      | Description                       |
| ---------------------------- | --------------------------------- |
| Avg. Area Income             | Average income of the area        |
| Avg. Area House Age          | Average age of houses in the area |
| Avg. Area Number of Rooms    | Average number of rooms           |
| Avg. Area Number of Bedrooms | Average number of bedrooms        |
| Area Population              | Population of the area            |
| Price                        | House price and target variable   |
| Address                      | Property address                  |

The `Address` column is not used as a machine learning feature because it is a text-based location field.

---

## Data Cleaning

The dataset was checked for:

* Missing values
* Duplicate records
* Invalid data
* Unnecessary columns

The dataset contained no missing values.

After cleaning, the dataset contained:

```text
5,000 records
```

---

## Exploratory Data Analysis

Exploratory data analysis was performed to understand the dataset and relationships between the features and house prices.

The following visualizations were generated:

* Price Distribution
* Average Income vs Price
* House Age vs Price
* Number of Rooms vs Price
* Feature Correlation Heatmap

The visualizations are generated using `data_analysis.py`.

---

## Feature Engineering

Two additional features were created to improve the model input.

### Rooms per Bedroom

```text
Rooms_per_Bedroom = Rooms / Bedrooms
```

This feature represents the relationship between the total number of rooms and bedrooms.

### Population per Room

```text
Population_per_Room = Population / Rooms
```

This feature represents the population relative to the number of rooms in the area.

### Final Features

The final model uses seven features:

```text
1. Avg. Area Income
2. Avg. Area House Age
3. Avg. Area Number of Rooms
4. Avg. Area Number of Bedrooms
5. Area Population
6. Rooms_per_Bedroom
7. Population_per_Room
```

---

## Train/Test Split

The dataset was divided into training and testing sets using an 80/20 split.

```text
Training Data: 4,000 records
Testing Data:  1,000 records
```

The split uses `random_state=42` to ensure reproducibility.

---

## Machine Learning Models

Three regression algorithms were trained and evaluated.

### Linear Regression

Linear Regression achieved the best overall performance.

```text
MAE  : $80,914.27
RMSE : $100,478.05
R²   : 0.9179
```

### Random Forest Regressor

The initial Random Forest model achieved:

```text
MAE  : $94,958.49
RMSE : $120,759.75
R²   : 0.8815
```

### Gradient Boosting Regressor

Gradient Boosting achieved:

```text
MAE  : $87,315.28
RMSE : $109,227.24
R²   : 0.9030
```

---

## Hyperparameter Tuning

Hyperparameter tuning was performed on the Random Forest model using `GridSearchCV`.

The following parameters were evaluated:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
```

The tuning process evaluated:

```text
24 parameter combinations
3-fold cross-validation
72 total fits
```

### Best Random Forest Parameters

```text
max_depth = 20
min_samples_leaf = 1
min_samples_split = 2
n_estimators = 200
```

### Best Cross-Validation R²

```text
0.8774
```

After tuning, the Random Forest model achieved:

```text
MAE  : $94,530.16
RMSE : $120,391.33
R²   : 0.8822
```

---

## Model Comparison

| Model               |        MAE |        RMSE |     R² |
| ------------------- | ---------: | ----------: | -----: |
| Linear Regression   | $80,914.27 | $100,478.05 | 0.9179 |
| Gradient Boosting   | $87,315.28 | $109,227.24 | 0.9030 |
| Tuned Random Forest | $94,530.16 | $120,391.33 | 0.8822 |
| Random Forest       | $94,958.49 | $120,759.75 | 0.8815 |

---

## Final Model

Based on the test-set evaluation, Linear Regression achieved the highest R² score and the lowest MAE and RMSE among the evaluated models.

Therefore, Linear Regression was selected as the final prediction model.

### Final Performance

```text
Model : Linear Regression

MAE  : $80,914.27
RMSE : $100,478.05
R²   : 0.9179
```

The final trained model is saved at:

```text
models/house_price_model.pkl
```

The saved model is a Scikit-learn pipeline containing:

```text
SimpleImputer
      |
StandardScaler
      |
LinearRegression
```

---

## Streamlit Application

The project includes an interactive web interface built with Streamlit.

Users can enter:

* Average Area Income
* Average Area House Age
* Average Area Number of Rooms
* Average Area Number of Bedrooms
* Area Population

The application automatically calculates the engineered features and uses the trained model to generate an estimated house price.

---

## Running the Project

### Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd Task-3-House-Price-Prediction
```

### Create a Virtual Environment

For Windows:

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python model_training.py
```

This process will:

* Load the dataset
* Clean the data
* Perform feature engineering
* Train multiple models
* Perform hyperparameter tuning
* Evaluate the models
* Select the best model
* Save the final model

### Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in the browser.

---

## Important Files

### `data_analysis.py`

Performs exploratory data analysis and generates the project visualizations.

### `model_training.py`

Handles:

* Data cleaning
* Feature engineering
* Model training
* Hyperparameter tuning
* Model evaluation
* Best model selection
* Model saving

### `train_model.py`

Provides the model training workflow.

### `app.py`

Runs the Streamlit prediction interface.

### `models/house_price_model.pkl`

Contains the final trained Linear Regression pipeline.

### `models/model_results.csv`

Contains the model comparison results.

---

## Future Improvements

Possible improvements for future versions include:

* Add more advanced regression models
* Perform additional feature selection
* Apply cross-validation to all models
* Add prediction confidence ranges
* Improve the Streamlit user interface
* Add interactive EDA charts
* Include additional housing features
* Deploy the application online

---

## Author

Attia Qamar-un-nisa

House Price Prediction System

Built using Python, Scikit-learn, and Streamlit.



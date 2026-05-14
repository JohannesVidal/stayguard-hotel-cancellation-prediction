# StayGuard: Predicting Hotel Booking Cancellations with Machine Learning

## Project Overview

StayGuard is a machine learning project focused on predicting hotel booking cancellations using reservation-level data.

Hotel cancellations create uncertainty for hotels by affecting revenue forecasting, occupancy planning, staffing, and pricing decisions. The goal of this project is to build a supervised classification model that can identify bookings with a high risk of cancellation before the guest arrival date.

## Business Objective

The main business question is:

**Can we predict whether a hotel booking will be cancelled using information available at the time of reservation?**

A reliable cancellation prediction model could help hotels:

- Identify high-risk bookings in advance.
- Improve revenue and occupancy planning.
- Support smarter overbooking strategies.
- Adjust guest communication based on cancellation risk.
- Reduce operational uncertainty.

## Dataset

The dataset used in this project is the **Hotel Reservations Classification Dataset** from Kaggle by Ahsan81.

Each row represents a hotel booking and includes information such as:

- Number of adults and children.
- Weekend and weekday nights.
- Meal plan.
- Required car parking space.
- Reserved room type.
- Lead time.
- Arrival date information.
- Market segment.
- Repeated guest status.
- Previous cancellations and previous non-cancelled bookings.
- Average room price.
- Number of special requests.
- Booking status.

The target variable is:

`booking_status`

It contains two classes:

- `Canceled`
- `Not_Canceled`

## Machine Learning Problem

This is a **supervised binary classification problem**.

The target variable was encoded as:

- `0`: Not_Canceled
- `1`: Canceled

The objective is to predict whether a booking belongs to the `Canceled` class.

## Repository Structure

    stayguard-hotel-cancellation-prediction/
    │
    ├── data/
    │   ├── raw/
    │   └── processed/
    │
    ├── models/
    │   └── random_forest_final_model.pkl
    │
    ├── notebooks/
    │   ├── 01_eda_and_cleaning.ipynb
    │   └── 02_modeling.ipynb
    │
    ├── reports/
    │   └── figures/
    │   └── presentation/
    │       └── StayGuard Hotel Cancellation Prediction.pdf        
    │       └── StayGuard Hotel Cancellation Prediction.ppt
    │
    ├── src/
    │   ├── __init__.py
    │   ├── data_cleaning.py
    │   ├── feature_engineering.py
    │   └── modeling.py
    │
    ├── README.md
    ├── requirements.txt
    └── .gitignore

## Presentation

The final project presentation is available in:

`reports/presentation/StayGuard_Hotel_Cancellation_Prediction.pdf`
`reports/presentation/StayGuard_Hotel_Cancellation_Prediction.ppt`

## Source Code

The `src/` folder contains reusable Python functions used to keep the project more organized and modular.

- `data_cleaning.py`: functions for column standardization, dataset overview, target summary, sanity checks, and saving cleaned data.
- `feature_engineering.py`: functions for target encoding, feature-target splitting, feature type identification, and EDA grouping variables.
- `modeling.py`: functions for model evaluation, classification reports, confusion matrices, and model comparison.

Although the main workflow is shown in the notebooks, these scripts support cleaner and more reusable project structure.

## Workflow

### 1. Data Loading and Initial Inspection

The dataset was loaded and inspected for:

- Shape and data types.
- Missing values.
- Duplicated rows and duplicated booking IDs.
- Target distribution.
- Categorical feature consistency.
- Potentially unusual numerical values.
- Both bagging-based and boosting-based ensemble approaches were tested through Random Forest and Gradient Boosting.

The dataset contained:

- 36,275 rows.
- 19 columns.
- No missing values.
- No duplicated booking IDs.

### 2. Data Cleaning

The dataset was already in good condition. The main cleaning steps were:

- Standardizing column names.
- Inspecting rare or unusual values.
- Reviewing zero-price bookings.
- Reviewing bookings with zero adults.
- Saving a cleaned version of the dataset.

No rows were removed during the initial cleaning stage because the unusual values represented a very small percentage of the dataset and were considered plausible business cases.

### 3. Exploratory Data Analysis

The EDA focused on understanding cancellation patterns across key variables.

Main findings:

- Bookings with longer lead times showed much higher cancellation rates.
- Bookings with more special requests were less likely to be cancelled.
- Online bookings had the highest cancellation rate among market segments.
- Average room price contained useful predictive information, although the relationship was not fully linear.
- Meal plan and room type showed some variation in cancellation rates.

### 4. Preprocessing

The preprocessing pipeline included:

- `StandardScaler` for numerical features.
- `OneHotEncoder` for categorical features.
- `ColumnTransformer` to combine transformations.
- `Pipeline` to avoid data leakage.

The `booking_id` column was removed because it is only an identifier.

### 5. Model Training and Evaluation

Several classification models were trained and compared:

- K-Nearest Neighbors baseline.
- Tuned K-Nearest Neighbors.
- Logistic Regression baseline.
- Decision Tree baseline.
- Tuned Decision Tree.
- Random Forest ensemble.
- Gradient Boosting ensemble.

The dataset was split into training and testing sets using stratification to preserve the original target distribution.

Because the target variable was moderately imbalanced, model selection focused mainly on **F1-score for the `Canceled` class**, instead of accuracy alone.

## Model Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest Ensemble | 0.9068 | 0.8882 | 0.8187 | 0.8520 |
| Decision Tree Tuned | 0.8841 | 0.8328 | 0.8086 | 0.8205 |
| KNN Tuned | 0.8816 | 0.8401 | 0.7888 | 0.8136 |
| Decision Tree Baseline | 0.8744 | 0.8109 | 0.8044 | 0.8076 |
| KNN Baseline | 0.8567 | 0.7996 | 0.7505 | 0.7743 |
| Gradient Boosting Ensemble | 0.8584 | 0.8296 | 0.7148 | 0.7679 |
| Logistic Regression Baseline | 0.8143 | 0.7537 | 0.6437 | 0.6943 |

## Best Model

The best-performing model was:

**Random Forest Ensemble**

Final test performance:

- Accuracy: **90.68%**
- Precision: **88.82%**
- Recall: **81.87%**
- F1-score: **85.20%**

The Random Forest ensemble achieved the best balance between detecting actual cancellations and avoiding excessive false cancellation predictions.

## Feature Importance

The most important features for the Random Forest model were:

1. `lead_time`
2. `avg_price_per_room`
3. `no_of_special_requests`
4. `arrival_date`
5. `arrival_month`
6. `no_of_week_nights`
7. `no_of_weekend_nights`
8. `market_segment_type_Online`

The strongest predictor was `lead_time`, confirming the EDA finding that bookings made further in advance are much more likely to be cancelled.

## Key Insights

- Lead time is the strongest driver of cancellation risk.
- Guests with more special requests are less likely to cancel.
- Online bookings show higher cancellation rates than corporate bookings.
- Random Forest outperformed both simpler models and the Gradient Boosting ensemble by better balancing precision and recall for cancellation detection.
- Accuracy alone was not enough to evaluate the models because the target variable was moderately imbalanced.

## Business Impact

This model could help hotels identify high-risk bookings before arrival.

Potential applications include:

- Flagging reservations with high cancellation risk.
- Improving overbooking decisions.
- Supporting revenue management.
- Prioritizing communication with guests more likely to cancel.
- Reducing uncertainty in occupancy planning.

## Future Improvements

Future versions of this project could include:

- Threshold optimization to adjust the balance between precision and recall.
- Additional advanced ensemble models such as XGBoost, LightGBM, or CatBoost.
- External data such as local events, holidays, weather, or seasonality.
- Model explainability using SHAP values.
- Validation on more recent hotel booking data.
- Deployment as a simple prediction app or dashboard.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## How to Run the Project

1. Clone the repository.

        git clone https://github.com/JohannesVidal/stayguard-hotel-cancellation-prediction.git

2. Navigate into the project folder.

        cd stayguard-hotel-cancellation-prediction

3. Install the required dependencies.

        pip install -r requirements.txt

4. Add the dataset file to:

        data/raw/Hotel Reservations.csv

5. Run the notebooks in order:

        notebooks/01_eda_and_cleaning.ipynb
        notebooks/02_modeling.ipynb

## Author

Johannes Vidal
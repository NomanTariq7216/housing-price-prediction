import os
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

MODEL_FILE = 'model.pkl'
PIPELINE_FILE = 'pipeline.pkl'

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', cat_pipeline, cat_attribs)
    ])  
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Lets train the model
    
    # Load the dataset
    housing_data = pd.read_csv('housing.csv')

    # Stratified Shuffle Split
    housing_data["income_cat"] = pd.cut(housing_data["median_income"],
                                        bins=[0.0,1.5,3.0,4.5,6.0, np.inf],
                                        labels=[1,2,3,4,5])
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing_data, housing_data["income_cat"]):
        housing_data.loc[test_index].drop(['income_cat', 'median_house_value'], axis=1).to_csv("input.csv", index=False)
        housing_data = housing_data.loc[train_index].drop('income_cat', axis=1)
        

    # Separate features and target variable
    housing_labels = housing_data['median_house_value'].copy()
    housing_features = housing_data.drop('median_house_value', axis=1)

    # separate numerical and categorical columns
    num_cols = housing_features.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = housing_features.select_dtypes(include=['object']).columns

    # Build the pipeline
    pipeline = build_pipeline(num_cols, cat_cols)

    #fit and transform the training data
    housing_prepared = pipeline.fit_transform(housing_features)

    # Train the model
    model = RandomForestRegressor(random_state=42)

    rmse_scores = -cross_val_score(
    model,
    housing_prepared,
    housing_labels,
    scoring='neg_root_mean_squared_error',
    cv=10
)

    print("Cross-validation RMSE scores:", rmse_scores)
    print("Average RMSE:", rmse_scores.mean())

    model.fit(housing_prepared, housing_labels)

    # Save the model and pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model and pipeline trained and saved successfully.")

else:
    # Lets do inference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['median_house_value'] = predictions

    # calculate the RMSE for the predictions
    actual_values = pd.read_csv('input(Copy)_Actual_test_set.csv')['median_house_value'].values
    rmse = root_mean_squared_error(actual_values, predictions)
    print("RMSE for the predictions:", rmse)

    # calculate MAE for the predictions
    mae = mean_absolute_error(actual_values, predictions)
    print("MAE for the predictions:", mae)

    # calculate model accuracy using R^2 score 
    r2_score = model.score(transformed_input, actual_values)
    print("R² Score:", r2_score)

    # calculate error percentage
    error_percentage = (rmse / np.mean(actual_values)) * 100
    print("Error percentage:", error_percentage)

    input_data.to_csv("output.csv", index=False)
    print("Inference completed and output saved to output.csv.")

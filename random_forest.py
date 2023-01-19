import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


# Load the best_params dictionary from the previous script
best_params = joblib.load("best_params.joblib")

# Load the data
df = pd.read_csv('input_data.csv')
df['target'] = df['target'].str.extract('(\d+)').astype(int)

#initialize variables
scores = {}
predictions_df = pd.DataFrame(columns=['target', 'predictions'])
seed = 42

#iterate over targets
for target in best_params.keys():
    #split data by target
    data = df.query(f'target=={target}')
    X = data.drop(["activity",'molecule'], axis=1)
    y = data["activity"]

    # Initialize the model with the best hyperparameters for the current target
    model = RandomForestRegressor(**best_params[target], random_state=seed)

    # build the model
    # fit the model
    model.fit(X, y)
    # Make predictions on the molecule1_descriptors
    molecule1_descriptors = pd.read_csv("molecule1_descriptors.txt", sep="\t", header = 0)
    y_pred_molecule1 = model.predict(molecule1_descriptors)
    # Add the target column and the predictions to the predictions DataFrame
    predictions_df = predictions_df.append({'target': target, 'predictions': y_pred_molecule1}, ignore_index=True)


# Write the DataFrame to a single CSV file
predictions_df.to_csv('predictions.csv', index=False)

import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import subprocess

# Load the best_params dictionary from the previous script
best_params = joblib.load("best_params.joblib")

# Load the data
df = pd.read_csv('input_data.csv')
df['target'] = df['target'].str.extract('(\d+)').astype(int)

#load the molecules
subprocess.run(['python', 'cats.py', 'mols.csv', 'output.txt'])

#initialize variables
scores = {}
predictions_df = pd.DataFrame(columns=['target', 'predictions', 'variance'])
seed = 42

#iterate over targets
for target in best_params.keys():
    #split data by target
    data = df.query(f'target=={target}')
    X = data.drop(["activity",'molecule'], axis=1)
    y = data["activity"]

    # Initialize the model with the best hyperparameters for the current target
    model = RandomForestRegressor(**best_params[target], random_state=seed)

    # fit the model
    model.fit(X, y)
    y_pred = model.predict(X)
    estimators = model.estimators_
    variance_list = [explained_variance_score(y_pred, est.predict(X)) for est in estimators]



    # Make predictions on the molecule1_descriptors
    molecule_descriptors = pd.read_csv("output.txt", sep="\t", header = 0)
    y_pred_molecule = model.predict(molecule_descriptors)

    min_activity = data['activity'].min()
    max_activity = data['activity'].max()

    # Add the target column and the predictions to the predictions DataFrame
    predictions_df = predictions_df.append({'target': target, 'predictions': y_pred_molecule, 'variance': variance_list, 'min_activity': min_activity, 'max_activity': max_activity}, ignore_index=True)



# Write the DataFrame to a single CSV file
predictions_df.to_csv('predictions.csv', index=False)

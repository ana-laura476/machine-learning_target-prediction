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
molecule_dfs = {}
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

    variances = []
    for i, e in enumerate(model.estimators_):
        y_pred_tree = e.predict(X)
        variance = np.var(y_pred_tree)
        variances.append(variance)
    mean_variance = np.mean(variances)
    min_activity = data['activity'].min()
    max_activity = data['activity'].max()

    #Make predictions iterating over the output.txt file
    with open('output.txt') as f:
        f.readline()
        for i, line in enumerate(f):
            molecule_descriptors = [float(x) for x in line.split()]
            molecule_descriptors = np.array(molecule_descriptors).reshape(1, -1)
            y_pred_molecule = model.predict(molecule_descriptors)
            molecule = f'molecule{i}'
            if molecule not in molecule_dfs:
                molecule_dfs[molecule] = pd.DataFrame(columns=['target', 'predictions', 'variance', 'min_activity', 'max_activity'])
            molecule_dfs[molecule] = molecule_dfs[molecule].append({'target': target, 'predictions': y_pred_molecule, 'variance': np.mean(variances), 'min_activity': min_activity, 'max_activity': max_activity}, ignore_index=True)

    for molecule in molecule_dfs:
        molecule_dfs[molecule].to_csv(f'{molecule}_predictions.csv', index=False)

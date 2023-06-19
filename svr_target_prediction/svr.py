import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import KFold
import subprocess
import scipy
import scipy.stats as stats
from sklearn.metrics import pairwise_distances


# Load the best_params dictionary from the previous script
best_params = joblib.load("best_params.joblib")

# Load the data
df = pd.read_csv('input_data.csv')
df['target'] = df['target'].str.extract('(\d+)').astype(int)

#load the molecules
subprocess.run(['python', 'cats.py', 'mols.csv', 'mols2.csv', 'output.txt', 'output2.txt'])

#initialize variables
scores = {}
molecule_dfs = {}
seed = 42

#iterate over targets
for target in best_params.keys():
    #split data by target
    data = df.query(f'target=={target}')
    X = data.drop(["activity",'target', 'molecule'], axis=1)
    print(X.shape)
    X=X.values
    y = data["activity"]

    # Initialize the model with the best hyperparameters for the current target
    model = SVR(**best_params[target])

    # fit the model
    model.fit(X, y)

    #average distance to the support vector
    min_activity = data['activity'].min()
    max_activity = data['activity'].max()


    #Predict activity for dark chemical matter (dcm)

    y_pred_dcm = pd.DataFrame()
    with open('output2.txt') as f:
        for i, line in enumerate(f):
            dcm_descriptors = [float(x) for x in line.split()]
            dcm_descriptors = np.array(dcm_descriptors).reshape(1, -1)
            y_pred_dcm = y_pred_dcm.append(pd.DataFrame(model.predict(dcm_descriptors)))
        mean_dcm = float(y_pred_dcm.mean())
        print(mean_dcm)
        std_dcm = float(y_pred_dcm.std())
        print(std_dcm)

    #Make predictions iterating over the output.txt file
    with open('output.txt') as f:
        for i, line in enumerate(f):
            molecule_descriptors = [float(x) for x in line.split()]
            molecule_descriptors = np.array(molecule_descriptors).reshape(1, -1)
            y_pred_molecule = float(model.predict(molecule_descriptors))
            z_score = ((y_pred_molecule - mean_dcm)/std_dcm)
            print(z_score)
            p_value = scipy.stats.norm.sf(z_score)
            print(p_value)
            molecule = f'molecule{i}'
            if molecule not in molecule_dfs:
               molecule_dfs[molecule] = pd.DataFrame(columns=['target', 'predictions', 'dcm mean prediction', 'dcm std prediction', 'z-score', 'p-value', 'min_activity', 'max_activity'])
            molecule_dfs[molecule] = molecule_dfs[molecule].append({'target': target, 'predictions': y_pred_molecule, 'dcm mean prediction': mean_dcm, 'dcm std prediction': std_dcm, 'z-score': z_score, 'p-value': p_value, 'min_activity': min_activity, 'max_activity': max_activity}, ignore_index=True)

    for molecule in molecule_dfs:
        molecule_dfs[molecule].to_csv(f'{molecule}_predictions.csv', index=False)

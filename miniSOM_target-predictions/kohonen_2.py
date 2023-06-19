import sys
from minisom import MiniSom
import pandas as pd
import numpy as np
import os 
import csv
import numpy as np
import joblib
import subprocess
import scipy
import math
from scipy.spatial import distance
from scipy.spatial.distance import cdist
import scipy.stats



def fit_som(X, data, max_iter):
    l = (int((math.sqrt(len(X)/100))))
    features = data.shape[1]
    som = MiniSom(l, l, features, sigma=5, learning_rate=0.5, random_seed = seed)  
    print(f"Grid is {l} by {l} with {features} features ")
    som.train(data, max_iter, verbose=True)
    print("train is over")
    return som

seed=42
#load the molecules and input data
subprocess.run(['python', 'cats.py', 'mols.csv', 'mols2.csv', 'output.txt', 'output2.txt'])
#load input data
df = pd.read_csv('input_data.csv', delimiter=",")
print(df)
molecule=df['molecule']
epochs = len(molecule)
target=df['target']
X = df.drop(['target', 'molecule', 'activity'], axis=1).values
new_molecules = np.loadtxt('output.txt')

som = fit_som(molecule, X, epochs)
train_coords = np.array([som.winner(x) for x in X])
newmolecule_coords = np.array([som.winner(x) for x in new_molecules])
train_coords = np.squeeze(train_coords)
newmolecule_coords = np.squeeze(newmolecule_coords)


print(train_coords.shape)
print(train_coords[:5])
print(newmolecule_coords.shape)
print(newmolecule_coords[:5])

train_coords_str = [f"{x},{y}" for x, y in train_coords]
train_coords_str = np.array(train_coords_str).reshape(-1, 1)
train_coords_str = train_coords_str.reshape(-1,)

newmolecule_coords_str = [f"{x},{y}" for x, y in newmolecule_coords]
newmolecule_coords_str = np.array(newmolecule_coords_str).reshape(-1, 1)
newmolecule_coords_str = newmolecule_coords_str.reshape(-1,)

train_neuron_map = pd.DataFrame({'identity': molecule, 'neuron': train_coords_str, 'target': target})
newmolecule_neuron_map = pd.DataFrame({'neuron': newmolecule_coords_str})

X=pd.DataFrame(X)
new_molecules=pd.DataFrame(new_molecules)
train_neuron_map.reset_index(drop=True, inplace=True)
train_neuron_map = pd.concat([train_neuron_map, X], axis=1)
train_neuron_map.to_csv("neuron_map_2.csv", index=False)

#load the new molecules to predict
newmolecule_neuron_map.reset_index(drop=True, inplace=True)
newmolecule_neuron_map = pd.concat([newmolecule_neuron_map, new_molecules], axis=1)
newmolecule_neuron_map.to_csv("newmolecule_neuron_map_2.csv", index=False)

molecules_dfs = {}
#load the new molecules to predict


for neuron in newmolecule_neuron_map['neuron'].unique():
    print(f'the neuron is: {neuron}')
    new_molecules_neuron_rows = newmolecule_neuron_map[newmolecule_neuron_map['neuron'] == neuron]
    # for each neuron, trainign molecules there:
    neuron_rows = train_neuron_map[train_neuron_map['neuron'] == neuron]
    new_molecules_dfs = {}
    
    new_molecules_dfs = pd.DataFrame(columns=['code', 'neuron', 'molecule_within_neuron', 'target', 'distance_to_neuron', 'mean_distance_diff_targets', 'std_distance_diff_targets', 'z-score', 'p-value'])
    for target in neuron_rows['target'].unique():
        neuron_rows_same_label = neuron_rows[neuron_rows['target'] == target]
        neuron_rows_diff_label = neuron_rows[neuron_rows['target'] != target]
        # for each cluster, descriptors of training molecules there:
        neuron_data = neuron_rows_same_label.drop(['identity', 'neuron', 'target'], axis=1).values
        neuron_data_diff_label = neuron_rows_diff_label.drop(['identity', 'neuron', 'target'], axis=1).values

        # for each cluster, descriptors of new molecules there:
        new_moleucule_neuron_data = new_molecules_neuron_rows.drop(['neuron'], axis=1).values
        distances_to_neuron = cdist(neuron_data, new_moleucule_neuron_data, metric='euclidean')
        distances_to_neuron_background = cdist(neuron_data, neuron_data_diff_label, metric='euclidean')

        # for each cluster, distance of training data and new molecules:
        print(distances_to_neuron)
        print(distances_to_neuron.shape)
        # Retrieve the training molecules inside the cluster
        molecule_train = neuron_rows_same_label['identity']
        molecule_train.reset_index(drop=True, inplace=True)
        print(molecule_train)
        target_train = neuron_rows_same_label['target']
        target_train.reset_index(drop=True, inplace=True)
        print(target_train)

        mean_distances_to_neuron_background = distances_to_neuron_background.mean(axis=1)
        std_distances_to_neuron_background = distances_to_neuron_background.std(axis=1)

        # Save the new molecule dataframes to CSV files outside the inner loop

        for i in range(len(distances_to_neuron)):
            #iterate over each line i, over each training element
            print(f'i is: {i}')
            mol_train = molecule_train[i]
            targ_train = target_train[i]
            print(mol_train)
            print(targ_train)
            mean_distance_background = float(mean_distances_to_neuron_background[i])
            std_distance_background = float(std_distances_to_neuron_background[i])
            print(mean_distance_background)
            # Loop over each new molecule entry in the row i, one new molecule-one training molecule, access information with [i][j]
            for j in range(len(distances_to_neuron[i])):
                print(f'j is: {j}')
                code = f'{j}_{neuron}'
                # Compute the z-score for the j-th distance between row i and the validation entry j
                distance = float(distances_to_neuron[i][j])
                print(distance)
                if std_distance_background != 0:
                    z_score = -1*((distance - mean_distance_background) / std_distance_background)     
                    p_value = scipy.stats.norm.sf(z_score)
                    
                else:
                    z_score = 'homogenous neuron'  
                    p_value = 'homogenous neuron'  
                print(z_score)
                print(p_value)
                # Add the results to the prediction list
                new_molecules_dfs = new_molecules_dfs.append({'code': code, 'neuron': neuron, 'molecule_within_neuron': mol_train, 'target': targ_train, 'distance_to_neuron': distance, 'mean_distance_diff_targets': mean_distance_background, 'std_distance_diff_targets': std_distance_background,'z-score': z_score, 'p-value': p_value}, ignore_index=True)

        for code in new_molecules_dfs['code'].unique():
            predictions = new_molecules_dfs[new_molecules_dfs['code'] == code]
            # Save predictions dataframe to csv
            predictions.to_csv(f'moleclue_{code}_neuron_2.csv', index=False)
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
    som = MiniSom(l, l, features, sigma=5, learning_rate=0.5, random_seed=seed)  
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
dcm = np.loadtxt('output2.txt')
new_molecules = np.loadtxt('output.txt')

som = fit_som(molecule, X, epochs)
train_coords = np.array([som.winner(x) for x in X])
newmolecule_coords = np.array([som.winner(x) for x in new_molecules])
dcm_coords = np.array([som.winner(x) for x in dcm])
train_coords = np.squeeze(train_coords)
newmolecule_coords = np.squeeze(newmolecule_coords)
dcm_coords = np.squeeze(dcm_coords)


print(train_coords.shape)
print(train_coords[:5])
print(newmolecule_coords.shape)
print(newmolecule_coords[:5])
print(dcm_coords.shape)
print(dcm_coords[:5])

train_coords_str = [f"{x},{y}" for x, y in train_coords]
train_coords_str = np.array(train_coords_str).reshape(-1, 1)
train_coords_str = train_coords_str.reshape(-1,)

newmolecule_coords_str = [f"{x},{y}" for x, y in newmolecule_coords]
newmolecule_coords_str = np.array(newmolecule_coords_str).reshape(-1, 1)
newmolecule_coords_str = newmolecule_coords_str.reshape(-1,)

dcm_coords_str = [f"{x},{y}" for x, y in dcm_coords]
dcm_coords_str = np.array(dcm_coords_str).reshape(-1, 1)
dcm_coords_str = dcm_coords_str.reshape(-1,)

train_neuron_map = pd.DataFrame({'identity': molecule, 'neuron': train_coords_str, 'target': target})
newmolecule_neuron_map = pd.DataFrame({'neuron': newmolecule_coords_str})
dcm_neuron_map = pd.DataFrame({'neuron': dcm_coords_str})

X=pd.DataFrame(X)
new_molecules=pd.DataFrame(new_molecules)
dcm=pd.DataFrame(dcm)
train_neuron_map.reset_index(drop=True, inplace=True)
train_neuron_map = pd.concat([train_neuron_map, X], axis=1)
train_neuron_map.to_csv("neuron_map.csv", index=False)

#load the new molecules to predict
newmolecule_neuron_map.reset_index(drop=True, inplace=True)
newmolecule_neuron_map = pd.concat([newmolecule_neuron_map, new_molecules], axis=1)
newmolecule_neuron_map.to_csv("newmolecule_neuron_map.csv", index=False)

dcm_neuron_map.reset_index(drop=True, inplace=True)
dcm_neuron_map = pd.concat([dcm_neuron_map, dcm], axis=1)
dcm_neuron_map.to_csv("dcm_neuron_map_newpredictions.csv", index=False)

molecules_dfs = {}
#load the new molecules to predict


for neuron in newmolecule_neuron_map['neuron'].unique():
    print(f'the neuron is: {neuron}')
    new_molecules_neuron_rows = newmolecule_neuron_map[newmolecule_neuron_map['neuron'] == neuron]
    # for each neuron, trainign molecules there:
    neuron_rows = train_neuron_map[train_neuron_map['neuron'] == neuron]
    # for each cluster, dcm molecules there:
    dcm_neuron_rows = dcm_neuron_map[dcm_neuron_map['neuron'] == neuron]
    n_dcm_neuron = len(dcm_neuron_rows)

    # for each cluster, descriptors of training molecules there:
    neuron_data = neuron_rows.drop(['identity', 'neuron', 'target'], axis=1).values
    # for each cluster, descriptors of new molecules there:
    new_moleucule_neuron_data = new_molecules_neuron_rows.drop(['neuron'], axis=1).values
    # for each cluster, descriptors of dcm there:
    dcm_neuron_data = dcm_neuron_rows.drop(['neuron'], axis=1).values
    # for each cluster, distance of training data and ALL dcm:
    distances_to_neuron_dcm = cdist(neuron_data, dcm, metric='euclidean')
    print(distances_to_neuron_dcm)
    print(distances_to_neuron_dcm.shape)
    # for each cluster, distance of training data and dcm also in same cluster:
    distances_to_same_neuron_dcm = cdist(neuron_data, dcm_neuron_data, metric='euclidean')
    print(distances_to_same_neuron_dcm)
    print(distances_to_same_neuron_dcm.shape)
    # for each cluster, distance of training data and new molecules:
    distances_to_neuron = cdist(neuron_data, new_moleucule_neuron_data, metric='euclidean')
    print(distances_to_neuron)
    print(distances_to_neuron.shape)
    # Retrieve the training molecules inside the cluster
    molecule_train = neuron_rows['identity']
    molecule_train.reset_index(drop=True, inplace=True)
    print(molecule_train)
    target_train = neuron_rows['target']
    target_train.reset_index(drop=True, inplace=True)
    print(target_train)
    # mean and std of the distances of dcm (same cluster and all) for each molecule of the training data
    mean_distance_dcm_same_neuron = distances_to_same_neuron_dcm.mean(axis=1)
    std_distance_dcm_same_neuron = distances_to_same_neuron_dcm.std(axis=1)

    mean_distance_dcm_neuron = distances_to_neuron_dcm.mean(axis=1)
    std_distance_dcm_neuron = distances_to_neuron_dcm.std(axis=1)

    print(mean_distance_dcm_same_neuron)
    print(mean_distance_dcm_same_neuron.shape)

    print(mean_distance_dcm_neuron)
    print(mean_distance_dcm_neuron.shape)
    molecules_dfs = {}
    # Save the new molecule dataframes to CSV files outside the inner loop
    new_molecules_dfs = {}
    new_molecules_dfs = pd.DataFrame(columns=['code', 'neuron', 'n_dcm_neuron', 'molecule_within_neuron', 'target', 'distance_to_neuron', 'dcm_mean_distance', 'dcm_std_distance', 'dcm_same_neuron_mean_distance', 'dcm_same_neuron_std_distance', 'z-score', 'p-value', 'z-score_neuron', 'p-value_neuron'])

    for i in range(len(distances_to_neuron)):
        #iterate over each line i, over each training element
        print(f'i is: {i}')
        # get the mean and std values of dcm distances for each target, both same cluster dcm and all dcm
        mol_train = molecule_train[i]
        targ_train = target_train[i]
        print(mol_train)
        print(targ_train)
        mean_distance_same_neuron = float(mean_distance_dcm_same_neuron[i])
        std_distance_same_neuron = float(std_distance_dcm_same_neuron[i])
        print(mean_distance_same_neuron)
        mean_distance_all_dcm = float(mean_distance_dcm_neuron[i])
        std_distance_all_dcm = float(std_distance_dcm_neuron[i])
        print(mean_distance_all_dcm)
        # Loop over each new molecule entry in the row i, one new molecule-one training molecule, access information with [i][j]
        for j in range(len(distances_to_neuron[i])):
            print(f'j is: {j}')
            code = f'{j}_{neuron}'
            # Compute the z-score for the j-th distance between row i and the validation entry j
            distance = float(distances_to_neuron[i][j])
            print(distance)
            if std_distance_same_neuron != 0:
                z_score_same_neuron = -1*((distance - mean_distance_same_neuron) / std_distance_same_neuron)     
                p_value_same_neuron = scipy.stats.norm.sf(z_score_same_neuron)
                
            else:
                z_score_same_neuron = 'no dcm molecule in same cluster'  
                p_value_same_neuron = 'no dcm molecule in same cluster'  
            z_score_all_dcm = -1*((distance - mean_distance_all_dcm) / std_distance_all_dcm)
            p_value_all_dcm = scipy.stats.norm.sf(z_score_all_dcm)
            print(p_value_same_neuron)
            print(p_value_all_dcm)
            # Add the results to the prediction list
            new_molecules_dfs = new_molecules_dfs.append({'code': code, 'neuron': neuron, 'n_dcm_neuron': n_dcm_neuron, 'molecule_within_neuron': mol_train, 'target': targ_train, 'distance_to_neuron': distance, 'dcm_mean_distance': mean_distance_all_dcm, 'dcm_std_distance': std_distance_all_dcm, 'dcm_same_neuron_mean_distance': mean_distance_same_neuron, 'dcm_same_neuron_std_distance': std_distance_same_neuron, 'z-score': z_score_all_dcm, 'p-value': p_value_all_dcm, 'z-score_neuron': z_score_same_neuron, 'p-value_neuron': p_value_same_neuron}, ignore_index=True)

    for code in new_molecules_dfs['code'].unique():
        predictions = new_molecules_dfs[new_molecules_dfs['code'] == code]
        # Save predictions dataframe to csv
        predictions.to_csv(f'moleclue_{code}_neuron.csv', index=False)

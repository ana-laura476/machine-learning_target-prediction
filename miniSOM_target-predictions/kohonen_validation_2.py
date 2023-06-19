from minisom import MiniSom
import pandas as pd
import numpy as np
import os 
import csv
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import subprocess
import scipy
import math
from scipy.spatial import distance
from multiprocessing import Process, Queue, cpu_count
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

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
train_df, valid_df = train_test_split(df, test_size=0.1, random_state=seed)
# Extract features and labels from the training set
molecule_train = train_df['molecule']
target_train = train_df['target']
X_train = np.array(train_df.drop(['target', 'molecule', 'activity'], axis=1))
X_train = X_train.copy(order='C')

# Extract features and labels from the validation set
molecule_valid = valid_df['molecule']
target_valid = valid_df['target']
X_valid = np.array(valid_df.drop(['target', 'molecule', 'activity'], axis=1))
X_valid = X_valid.copy(order='C')
som = fit_som(molecule, X_train, epochs)  


train_coords = np.array([som.winner(x) for x in X_train])
valid_coords = np.array([som.winner(x) for x in X_valid])
train_coords = np.squeeze(train_coords)
valid_coords = np.squeeze(valid_coords)


print(train_coords.shape)
print(train_coords[:5])
print(valid_coords.shape)
print(valid_coords[:5])

train_coords_str = [f"{x},{y}" for x, y in train_coords]
train_coords_str = np.array(train_coords_str).reshape(-1, 1)
train_coords_str = train_coords_str.reshape(-1,)

valid_coords_str = [f"{x},{y}" for x, y in valid_coords]
valid_coords_str = np.array(valid_coords_str).reshape(-1, 1)
valid_coords_str = valid_coords_str.reshape(-1,)

train_neuron_map = pd.DataFrame({'identity': molecule_train, 'neuron': train_coords_str, 'target': target_train})
valid_neuron_map = pd.DataFrame({'identity': molecule_valid, 'neuron': valid_coords_str, 'target': target_valid})

X_train=pd.DataFrame(X_train)
X_valid=pd.DataFrame(X_valid)
train_neuron_map.reset_index(drop=True, inplace=True)
train_neuron_map = pd.concat([train_neuron_map, X_train], axis=1)
# Save neuron map to file
train_neuron_map.to_csv("train_neuron_map_2.csv", index=False)
#load the new molecules to predict
valid_neuron_map.reset_index(drop=True, inplace=True)
valid_neuron_map = pd.concat([valid_neuron_map, X_valid], axis=1)
valid_neuron_map.to_csv("valid_neuron_map_2.csv", index=False)
molecules_dfs = {}
#load the new molecules to predict
predictions_df = pd.DataFrame(columns=['identity', 'label', 'significant_targets', 'non_significant_targets', 'no_p-value','true_positive', 'false_negative'])

predictions_df = pd.DataFrame(columns=['identity', 'label', 'significant_targets', 'non_significant_targets', 'no_p-value', 'true_positive', 'false_negative'])

for neuron in valid_neuron_map['neuron'].unique():
    # for each neuron, training molecules there:
    print(f'the neuron is: {neuron}')
    neuron_rows = train_neuron_map[train_neuron_map['neuron'] == neuron]  #train molecules there
    molecules_neuron = valid_neuron_map[valid_neuron_map['neuron'] == neuron]   #validation molecules there
    # for each cluster, descriptors of validation molecules there:
    molecules_dfs = {}
    molecules_dfs = pd.DataFrame(columns=['identity', 'target', 'molecule_within_neuron', 'target_train', 'p-value'])

    for target in neuron_rows['target'].unique():
        print(f'the target is: {target}')
        neuron_rows_same_label = neuron_rows[neuron_rows['target'] == target]
        neuron_rows_diff_label = neuron_rows[neuron_rows['target'] != target]
        print(neuron_rows)
        # for each cluster, descriptors of training molecules there:
        neuron_data = neuron_rows_same_label.drop(['identity', 'neuron', 'target'], axis=1).values
        neuron_data_diff_label = neuron_rows_diff_label.drop(['identity', 'neuron', 'target'], axis=1).values
        # for each cluster and target, distance of training data and validation data:
        mols_descriptors = molecules_neuron.drop(['identity', 'neuron', 'target'], axis=1)
        distances_to_neuron = cdist(neuron_data, mols_descriptors, metric='euclidean')
        # for each cluster, distance of training data and dcm also in same cluster:
        # for each cluster and target, distance of training data with the label and the training data without the label:
        distances_to_neuron_background = cdist(neuron_data, neuron_data_diff_label, metric='euclidean')
        # create a new DataFrame for each row in cluster_map_valid
        molecule_val = molecules_neuron['identity']
        molecule_val.reset_index(drop=True, inplace=True)
        target_label = molecules_neuron['target']
        target_label.reset_index(drop=True, inplace=True)
        # Retrieve the training molecules inside the cluster
        molecule_train = neuron_rows_same_label['identity']
        molecule_train.reset_index(drop=True, inplace=True)
        target_train = neuron_rows_same_label['target']
        target_train.reset_index(drop=True, inplace=True)


        # mean and std of the distances of dcm (same cluster and all) for each molecule of the training data
        mean_distances_to_neuron_background = distances_to_neuron_background.mean(axis=1)
        std_distances_to_neuron_background = distances_to_neuron_background.std(axis=1)

        print(mean_distances_to_neuron_background)
        print(mean_distances_to_neuron_background.shape)

        for i in range(len(distances_to_neuron)):
            #iterate over each line i, over each training element
            print(f'i is: {i}')
            # get the mean and std values of dcm distances for each target, both same cluster dcm and all dcm
            mol_train = molecule_train[i]
            targ_train = target_train[i]
            print(mol_train)
            print(targ_train)
            mean_distance_background = float(mean_distances_to_neuron_background[i])
            std_distance_background = float(std_distances_to_neuron_background[i])
            print(mean_distance_background)
            # Loop over each validation entry in the row i, one validation-one training molecule, access information with [i][j]
            for j in range(len(distances_to_neuron[i])):
                print(f'j is: {j}')
                # Compute the z-score for the j-th distance between row i and the validation entry j
                iden_valid = molecule_val[j]
                targ_valid = target_label[j]
                print(iden_valid)
                print(targ_valid)
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
                molecules_dfs = molecules_dfs.append({'identity': iden_valid, 'target': targ_valid, 'molecule_within_neuron': mol_train, 'target_train': targ_train, 'p-value': p_value}, ignore_index=True)

    # Initialize variables
    print(molecules_dfs)
    print(molecules_dfs.shape)

    # Loop over results and dataframes
    for identity in molecules_dfs['identity'].unique():
        counts = {'true_positive': 0, 'false_negative': 0, 'significant_targets': [], 'non_significant_targets': [], 'no_p-value': []}
        val = molecules_dfs[molecules_dfs['identity'] == identity]
        label = val['target'].iloc[0]
        print(val)
        print(val.shape)
        print(label)
        numeric_rows = val[pd.to_numeric(val['p-value'], errors='coerce').notna()]
        non_numeric_rows = val[pd.to_numeric(val['p-value'], errors='coerce').isna()]

        significant_rows = numeric_rows.loc[numeric_rows['p-value'] <= 0.05]
        non_significant_rows = numeric_rows.loc[numeric_rows['p-value'] > 0.05]
        counts['significant_targets'].append(list(significant_rows['target_train']))
        counts['non_significant_targets'].append(list(non_significant_rows['target_train']))
        counts['no_p-value'].append(list(non_numeric_rows['target_train']))  #targets in homogenous clusters

        accurate_rows = numeric_rows.loc[(numeric_rows['p-value'] <= 0.05) & (numeric_rows['target_train'] == targ_valid)]
        accurate_rows_ns = numeric_rows.loc[(numeric_rows['p-value'] > 0.05) & (numeric_rows['target_train'] == targ_valid)]
        counts['true_positive'] += len(accurate_rows)
        counts['false_negative'] += len(accurate_rows_ns)
        # Create predictions row
        true_positive = 'yes' if counts['true_positive'] > 0 else 'no'
        false_negative = 'yes' if counts['false_negative'] > 0 else 'no'
        predictions_row = pd.DataFrame([[identity, label, counts['significant_targets'], counts['non_significant_targets'], counts['no_p-value'], true_positive, false_negative]],
                               columns=['identity', 'label', 'significant_targets', 'non_significant_targets','no_p-value', 'true_positive', 'false_negative'])
        #Add row to predictions dataframe
        predictions_df = pd.concat([predictions_df, predictions_row], ignore_index=True)


# Save predictions dataframe to csv
predictions_df.to_csv("predictions_2.csv", index=False)


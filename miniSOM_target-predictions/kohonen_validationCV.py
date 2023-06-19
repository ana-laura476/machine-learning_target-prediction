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
from sklearn.model_selection import KFold


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
kf = KFold(n_splits=10, random_state=seed, shuffle=True)
# Change the starting fold index to 4
starting_fold = 6
for fold, (train_idx, valid_idx) in enumerate(kf.split(df)):    # Extract features and labels from the training set
    # Skip the folds that you don't want to redo
    if fold > starting_fold:
        print(f'Fold {fold + 1}')
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]
        molecule_train = train_df['molecule']
        target_train = train_df['target']
        X_train = np.array(train_df.drop(['target', 'molecule', 'activity'], axis=1))
        X_train = X_train.copy(order='C')

        # Extract features and labels from the validation set
        molecule_valid = valid_df['molecule']
        target_valid = valid_df['target']
        X_valid = np.array(valid_df.drop(['target', 'molecule', 'activity'], axis=1))
        X_valid = X_valid.copy(order='C')
        # Find the cluster assignments for dcm molecules
        dcm = np.loadtxt('output2.txt')
        som = fit_som(molecule, X_train, epochs)  


        train_coords = np.array([som.winner(x) for x in X_train])
        valid_coords = np.array([som.winner(x) for x in X_valid])
        dcm_coords = np.array([som.winner(x) for x in dcm])
        train_coords = np.squeeze(train_coords)
        valid_coords = np.squeeze(valid_coords)
        dcm_coords = np.squeeze(dcm_coords)


        print(train_coords.shape)
        print(train_coords[:5])
        print(valid_coords.shape)
        print(valid_coords[:5])
        print(dcm_coords.shape)
        print(dcm_coords[:5])

        train_coords_str = [f"{x},{y}" for x, y in train_coords]
        train_coords_str = np.array(train_coords_str).reshape(-1, 1)
        train_coords_str = train_coords_str.reshape(-1,)

        valid_coords_str = [f"{x},{y}" for x, y in valid_coords]
        valid_coords_str = np.array(valid_coords_str).reshape(-1, 1)
        valid_coords_str = valid_coords_str.reshape(-1,)

        dcm_coords_str = [f"{x},{y}" for x, y in dcm_coords]
        dcm_coords_str = np.array(dcm_coords_str).reshape(-1, 1)
        dcm_coords_str = dcm_coords_str.reshape(-1,)

        train_neuron_map = pd.DataFrame({'identity': molecule_train, 'neuron': train_coords_str, 'target': target_train})
        valid_neuron_map = pd.DataFrame({'identity': molecule_valid, 'neuron': valid_coords_str, 'target': target_valid})
        dcm_neuron_map = pd.DataFrame({'neuron': dcm_coords_str})

        X_train=pd.DataFrame(X_train)
        X_valid=pd.DataFrame(X_valid)
        dcm=pd.DataFrame(dcm)
        train_neuron_map.reset_index(drop=True, inplace=True)
        train_neuron_map = pd.concat([train_neuron_map, X_train], axis=1)
        # Save neuron map to file
        train_neuron_map.to_csv(f"train_neuron_map_fold{fold}.csv", index=False)
        #load the new molecules to predict
        valid_neuron_map.reset_index(drop=True, inplace=True)
        valid_neuron_map = pd.concat([valid_neuron_map, X_valid], axis=1)
        valid_neuron_map.to_csv(f"valid_neuron_map_fold{fold}.csv", index=False)

        dcm_neuron_map.reset_index(drop=True, inplace=True)
        dcm_neuron_map = pd.concat([dcm_neuron_map, dcm], axis=1)
        dcm_neuron_map.to_csv(f"dcm_neuron_map_fold{fold}.csv", index=False)
        molecules_dfs = {}
        #load the new molecules to predict

        predictions_df = pd.DataFrame(columns=['identity', 'label', 'significant_targets', 'significant_targets_neuron', 'non_significant_targets', 'non_significant_targets_neuron', 'true_positive', 'true_positive_neuron', 'false_negative', 'false_negative_neuron'])

        for neuron in valid_neuron_map['neuron'].unique():
            # for each neuron, training molecules there:
            print(f'the neuron is: {neuron}')
            neuron_rows = train_neuron_map[train_neuron_map['neuron'] == neuron]
            print(neuron_rows)
            # for each cluster, dcm molecules there:
            dcm_neuron_rows = dcm_neuron_map[dcm_neuron_map['neuron'] == neuron]
            # for each cluster, descriptors of training molecules there:
            neuron_data = neuron_rows.drop(['identity', 'neuron', 'target'], axis=1).values
            # for each cluster, descriptors of dcm there:
            dcm_neuron_data = dcm_neuron_rows.drop(['neuron'], axis=1).values
            # for each cluster, distance of training data and ALL dcm:
            distances_to_neuron_dcm = cdist(neuron_data, dcm, metric='euclidean')
            # for each cluster, distance of training data and dcm also in same cluster:
            distances_to_same_neuron_dcm = cdist(neuron_data, dcm_neuron_data, metric='euclidean')
            # for each cluster, validation molecules there:
            molecules_neuron = valid_neuron_map[valid_neuron_map['neuron'] == neuron]
            # for each cluster, descriptors of validation molecules there:
            mols_descriptors = molecules_neuron.drop(['identity', 'neuron', 'target'], axis=1)
            # for each cluster, distance of training data and validation data:
            distances_to_neuron = cdist(neuron_data, mols_descriptors, metric='euclidean')


            # create a new DataFrame for each row in cluster_map_valid
            molecule_val = molecules_neuron['identity']
            molecule_val.reset_index(drop=True, inplace=True)
            target_label = molecules_neuron['target']
            target_label.reset_index(drop=True, inplace=True)
            # Retrieve the training molecules inside the cluster
            molecule_train = neuron_rows['identity']
            molecule_train.reset_index(drop=True, inplace=True)
            target_train = neuron_rows['target']
            target_train.reset_index(drop=True, inplace=True)


            # mean and std of the distances of dcm (same cluster and all) for each molecule of the training data
            mean_distance_dcm_same_neuron = distances_to_same_neuron_dcm.mean(axis=1)
            std_distance_dcm_same_neuron = distances_to_same_neuron_dcm.std(axis=1)

            mean_distance_dcm_neuron = distances_to_neuron_dcm.mean(axis=1)
            std_distance_dcm_neuron = distances_to_neuron_dcm.std(axis=1)

            molecules_dfs = {}
            molecules_dfs = pd.DataFrame(columns=['identity', 'target', 'molecule_within_neuron', 'target_train', 'p-value', 'p-value_neuron'])

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
                mean_distance_all_dcm = float(mean_distance_dcm_neuron[i])
                std_distance_all_dcm = float(std_distance_dcm_neuron[i])
                print(mean_distance_all_dcm)
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
                    molecules_dfs = molecules_dfs.append({'identity': iden_valid, 'target': targ_valid, 'molecule_within_neuron': mol_train, 'target_train': targ_train, 'p-value': p_value_all_dcm, 'p-value_neuron': p_value_same_neuron}, ignore_index=True)


            # Initialize variables
            print(molecules_dfs)
            print(molecules_dfs.shape)

            # Loop over results and dataframes
            for identity in molecules_dfs['identity'].unique():
                counts = {'true_positive': 0, 'true_positive_neuron': 0, 'false_negative': 0, 'false_negative_neuron':0, 'significant_targets': [], 'significant_targets_neuron': [], 'non_significant_targets': [], 'non_significant_targets_neuron': []}
                val = molecules_dfs[molecules_dfs['identity'] == identity]
                label = val['target'].iloc[0]
                print(val)
                print(val.shape)
                print(label)
                significant_rows = val.loc[val['p-value'] <= 0.05]
                non_significant_rows = val.loc[val['p-value'] > 0.05]
                print(significant_rows)
                counts['significant_targets'].append(list(significant_rows['target_train']))
                counts['non_significant_targets'].append(list(non_significant_rows['target_train']))
                # Filter out non-numeric rows
                numeric_rows = val[pd.to_numeric(val['p-value_neuron'], errors='coerce').notna()]
                significant_neuron_rows = numeric_rows.loc[numeric_rows['p-value_neuron'] <= 0.05]
                non_significant_neuron_rows = numeric_rows.loc[numeric_rows['p-value_neuron'] > 0.05]
                counts['significant_targets_neuron'].append(list(significant_neuron_rows['target_train']))
                counts['non_significant_targets_neuron'].append(list(non_significant_neuron_rows['target_train']))
                accurate_rows = val.loc[(val['p-value'] <= 0.05) & (val['target_train'] == targ_valid)]
                accurate_rows_ns = val.loc[(val['p-value'] > 0.05) & (val['target_train'] == targ_valid)]
                counts['true_positive'] += len(accurate_rows)
                counts['false_negative'] += len(accurate_rows_ns)
                accurate_neuron_rows = numeric_rows.loc[(numeric_rows['p-value_neuron'] <= 0.05) & (numeric_rows['target_train'] == targ_valid)]
                accurate_neuron_rows_ns = numeric_rows.loc[(numeric_rows['p-value_neuron'] > 0.05) & (numeric_rows['target_train'] == targ_valid)]
                counts['true_positive_neuron'] += len(accurate_neuron_rows)
                counts['false_negative_neuron'] += len(accurate_neuron_rows_ns)
                # Create predictions row
                true_positive = 'yes' if counts['true_positive'] > 0 else 'no'
                false_negative = 'yes' if counts['false_negative'] > 0 else 'no'

                true_positive_neuron = 'yes' if counts['true_positive_neuron'] > 0 else 'no'
                false_negative_neuron= 'yes' if counts['false_negative_neuron'] > 0 else 'no'
                predictions_row = pd.DataFrame([[identity, label, counts['significant_targets'], counts['significant_targets_neuron'], counts['non_significant_targets'], counts['non_significant_targets_neuron'],  true_positive, true_positive_neuron, false_negative, false_negative_neuron]],
                                       columns=['identity', 'label', 'significant_targets', 'significant_targets_neuron', 'non_significant_targets', 'non_significant_targets_neuron', 'true_positive', 'true_positive_neuron', 'false_negative', 'false_negative_neuron'])
                #Add row to predictions dataframe
                predictions_df = pd.concat([predictions_df, predictions_row], ignore_index=True)


        # Save predictions dataframe to csv
        predictions_df.to_csv(f"predictions_fold{fold}.csv", index=False)


import pandas as pd
import numpy as np
import csv
from joblib import dump
from sklearn.cluster import BisectingKMeans
import subprocess
import scipy
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed
from sklearn.model_selection import KFold


n_jobs = -1
seed=42

#load input data
df = pd.read_csv('input_data.csv', delimiter=",")
print(df)
molecule = df['molecule']

# Split data into training and validation sets
#initialize KMeans model
n_clusters=int(len(molecule)/100)
kmeans = BisectingKMeans(n_clusters=n_clusters, max_iter=300, n_init=10, random_state=seed)
kf = KFold(n_splits=10, random_state=seed, shuffle=True)
for fold, (train_idx, valid_idx) in enumerate(kf.split(df)):
    print(f'Fold {fold + 1}')
    train_df = df.iloc[train_idx]
    valid_df = df.iloc[valid_idx]
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

    print(f'n_cluster: {n_clusters}')
    kmeans.fit(X_train)
    combined_data = np.vstack((X_train, X_valid))
    print(combined_data)
    combined_cluster_assign = kmeans.predict(combined_data)
    # Split the cluster assignments back into X_train_cluster_assign and dcm_cluster_assign
    cluster_assign_train = combined_cluster_assign[:len(X_train)]
    cluster_assign_valid = combined_cluster_assign[len(X_train):]




    # Find the cluster assignments for the training set
    cluster_assign_train = cluster_assign_train.reshape(-1)

    # Find the cluster assignments for the training set
    cluster_assign_valid = cluster_assign_valid.reshape(-1)


    # Create cluster map dataframe for the training set
    cluster_map_train = pd.DataFrame({'identity': molecule_train, 'cluster': cluster_assign_train, 'target': target_train})
    cluster_map_train.reset_index(drop=True, inplace=True)
    X_train = pd.DataFrame(X_train)
    X_train.reset_index(drop=True, inplace=True)
    cluster_map_train = pd.concat([cluster_map_train, X_train], axis=1)
    # Save cluster map to file
    cluster_map_train.to_csv(f"cluster_map_train_2_fold{fold}.csv", index=False)


    # Create cluster map dataframe for the validation set
    cluster_map_valid = pd.DataFrame({'identity': molecule_valid, 'cluster': cluster_assign_valid, 'target': target_valid})
    cluster_map_valid.reset_index(drop=True, inplace=True)
    X_valid = pd.DataFrame(X_valid)
    X_valid.reset_index(drop=True, inplace=True)
    cluster_map_valid = pd.concat([cluster_map_valid, X_valid], axis=1)
    cluster_map_valid.to_csv(f"cluster_map_valid_2_fold{fold}.csv", index=False)


    predictions_df = pd.DataFrame(columns=['identity', 'label', 'significant_targets', 'non_significant_targets', 'no_p-value','true_positive', 'false_negative'])

    for cluster in cluster_map_valid['cluster'].unique():
        print(f'the cluster is: {cluster}')
        # for each cluster, descriptors of training molecules there:
        cluster_rows = cluster_map_train[cluster_map_train['cluster'] == cluster]
        # for each cluster, validation molecules there:
        molecules_cluster = cluster_map_valid[cluster_map_valid['cluster'] == cluster]
        molecules_dfs = {}
        molecules_dfs = pd.DataFrame(columns=['identity', 'target', 'molecule_within_cluster', 'target_train', 'p-value'])

        for target in cluster_rows['target'].unique():
            # for each cluster, training molecules there:
            print(f'the target is: {target}')
            cluster_rows_same_label = cluster_rows[cluster_rows['target'] == target]
            cluster_rows_diff_label = cluster_rows[cluster_rows['target'] != target]
            cluster_data = cluster_rows_same_label.drop(['identity', 'cluster', 'target'], axis=1).values
            cluster_data_diff_label = cluster_rows_diff_label.drop(['identity', 'cluster', 'target'], axis=1).values

            # for each cluster, descriptors of validation molecules there:
            mols_descriptors = molecules_cluster.drop(['identity', 'cluster', 'target'], axis=1)
            # for each cluster and target, distance of training data and validation data:
            distances_to_cluster = cdist(cluster_data, mols_descriptors, metric='euclidean')
            print(distances_to_cluster)
            print(distances_to_cluster.shape)
            # for each cluster and target, distance of training data with the label and the training data without the label:
            distances_to_cluster_background = cdist(cluster_data, cluster_data_diff_label, metric='euclidean')
            print(distances_to_cluster)
            print(distances_to_cluster.shape)

            # create a new DataFrame for each row in cluster_map_valid
            molecule_val = molecules_cluster['identity']
            molecule_val.reset_index(drop=True, inplace=True)
            print(molecule_val)
            target_label = molecules_cluster['target']
            target_label.reset_index(drop=True, inplace=True)
            print(target_label) 
            # Retrieve the training molecules inside the cluster
            molecule_train = cluster_rows_same_label['identity']
            molecule_train.reset_index(drop=True, inplace=True)
            print(molecule_train)
            target_train = cluster_rows_same_label['target']
            target_train.reset_index(drop=True, inplace=True)
            print(target_train)


            # mean and std of the distances of dcm (same cluster and all) for each molecule of the training data
            mean_distances_to_cluster_background = distances_to_cluster_background.mean(axis=1)
            std_distances_to_cluster_background = distances_to_cluster_background.std(axis=1)

            print(mean_distances_to_cluster_background)
            print(mean_distances_to_cluster_background.shape)

            for i in range(len(distances_to_cluster)):
                #iterate over each line i, over each training element
                print(f'i is: {i}')
                # get the mean and std values of dcm distances for each target, both same cluster dcm and all dcm
                mol_train = molecule_train[i]
                targ_train = target_train[i]
                print(mol_train)
                print(targ_train)
                mean_distance_background = float(mean_distances_to_cluster_background[i])
                std_distance_background = float(std_distances_to_cluster_background[i])
                print(mean_distance_background)
                # Loop over each validation entry in the row i, one validation-one training molecule, access information with [i][j]
                for j in range(len(distances_to_cluster[i])):
                    print(f'j is: {j}')
                    # Compute the z-score for the j-th distance between row i and the validation entry j
                    iden_valid = molecule_val[j]
                    targ_valid = target_label[j]
                    print(iden_valid)
                    print(targ_valid)
                    distance = float(distances_to_cluster[i][j])
                    print(distance)
                    if std_distance_background != 0:
                        z_score = -1*((distance - mean_distance_background) / std_distance_background)     
                        p_value = scipy.stats.norm.sf(z_score)
                    
                    else:
                        z_score = 'homogenous cluster'  
                        p_value = 'homogenous cluster'  
                        
                    
                    print(z_score)
                    print(p_value)
                    # Add the results to the prediction list
                    molecules_dfs = molecules_dfs.append({'identity': iden_valid, 'target': targ_valid, 'molecule_within_cluster': mol_train, 'target_train': targ_train, 'p-value': p_value}, ignore_index=True)


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
            # Filter out non-numeric rows
            numeric_rows = val[pd.to_numeric(val['p-value'], errors='coerce').notna()]
            non_numeric_rows = val[pd.to_numeric(val['p-value'], errors='coerce').isna()]
            significant_rows = numeric_rows.loc[(numeric_rows['p-value'] <= 0.05)] 
            non_significant_rows = numeric_rows.loc[(numeric_rows['p-value'] > 0.05)] 
            print(significant_rows)
            counts['significant_targets'].append(list(significant_rows['target_train']))   #targets with p-value less than 0.05
            counts['non_significant_targets'].append(list(non_significant_rows['target_train']))   #targets with p-value over 0.05
            counts['no_p-value'].append(list(non_numeric_rows['target_train']))  #targets in homogenous clusters
            # Filter out non-numeric rows 
            accurate_rows = numeric_rows.loc[(numeric_rows['p-value'] <= 0.05) & (numeric_rows['target_train'] == targ_valid)]
            accurate_rows_ns = numeric_rows.loc[(numeric_rows['p-value'] > 0.05) & (numeric_rows['target_train'] == targ_valid)]
            counts['true_positive'] += len(accurate_rows)
            counts['false_negative'] += len(accurate_rows_ns)
            # Create predictions row
            true_positive = 'yes' if counts['true_positive'] > 0 else 'no'
            false_negative = 'yes' if counts['false_negative'] > 0 else 'no'
            predictions_row = pd.DataFrame([[identity, label, counts['significant_targets'], counts['non_significant_targets'], counts['no_p-value'], true_positive, false_negative]],
                               columns=['identity', 'label', 'significant_targets', 'non_significant_targets', 'no_p-value', 'true_positive', 'false_negative'])
            #Add row to predictions dataframe
            predictions_df = pd.concat([predictions_df, predictions_row], ignore_index=True)


    # Save predictions dataframe to csv
    predictions_df.to_csv(f"predictions_2_fold{fold}.csv", index=False)
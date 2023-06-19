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



subprocess.run(['python', 'cats.py', 'mols.csv', 'mols2.csv', 'output.txt', 'output2.txt'])
predictions_df = pd.DataFrame(columns=['identity', 'label', 'significant_targets', 'significant_targets_cluster', 'non_significant_targets', 'non_significant_targets_cluster', 'true_positive', 'true_positive_cluster', 'false_negative', 'false_negative_cluster'])

if __name__ == '__main__':

    seed=42

    #load input data
    df = pd.read_csv('input_data.csv', delimiter=",")
    print(df)
    molecule = df['molecule']

    # Split data into training and validation sets
    train_df, valid_df = train_test_split(df, test_size=0.1, random_state=seed)
    print(train_df)
    print(valid_df)

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

    #initialize KMeans model
    n_clusters=int(len(molecule)/100)
    print(f'n_cluster: {n_clusters}')
    kmeans = BisectingKMeans(n_clusters=n_clusters, max_iter=300, n_init=10, random_state=seed)
    print(kmeans)
    kmeans.fit(X_train)

    dcm = np.loadtxt('output2.txt')
    print(dcm)
    combined_data = np.vstack((X_train, dcm))
    print(combined_data)
    combined_cluster_assign = kmeans.predict(combined_data)
    print(combined_cluster_assign)
    # Split the cluster assignments back into X_train_cluster_assign and dcm_cluster_assign
    cluster_assign_train = combined_cluster_assign[:len(X_train)]
    dcm_cluster_assign = combined_cluster_assign[len(X_train):]



    #dcm_cluster_assign = kmeans.predict(dcm)
    print('stop2')
    print(dcm_cluster_assign)
    dcm_cluster_assign=dcm_cluster_assign.reshape(-1)
    dcm_cluster_map = pd.DataFrame({'cluster': dcm_cluster_assign})
    dcm = pd.DataFrame(dcm)
    dcm_cluster_map = pd.concat([dcm_cluster_map, dcm], axis=1)
    # Save cluster map to file
    dcm_cluster_map.to_csv("dcm_cluster_map.csv", index=False)

    # Find the cluster assignments for the training set
    #cluster_assign_train = kmeans.predict(X_train)
    #print(cluster_assign_train)

    cluster_assign_train = cluster_assign_train.reshape(-1)
    print(cluster_assign_train)

    # Create cluster map dataframe for the training set
    cluster_map_train = pd.DataFrame({'identity': molecule_train, 'cluster': cluster_assign_train, 'target': target_train})
    cluster_map_train.reset_index(drop=True, inplace=True)
    X_train = pd.DataFrame(X_train)
    X_train.reset_index(drop=True, inplace=True)
    cluster_map_train = pd.concat([cluster_map_train, X_train], axis=1)
    # Save cluster map to file
    cluster_map_train.to_csv("cluster_map_train.csv", index=False)


    # Find the cluster assignments for the training set
    cluster_assign_valid = kmeans.predict(X_valid)
    print(cluster_assign_valid)
    cluster_assign_valid = cluster_assign_valid.reshape(-1)
    print(cluster_assign_valid)



    # Create cluster map dataframe for the validation set
    cluster_map_valid = pd.DataFrame({'identity': molecule_valid, 'cluster': cluster_assign_valid, 'target': target_valid})
    cluster_map_valid.reset_index(drop=True, inplace=True)
    X_valid = pd.DataFrame(X_valid)
    X_valid.reset_index(drop=True, inplace=True)
    cluster_map_valid = pd.concat([cluster_map_valid, X_valid], axis=1)
    cluster_map_valid.to_csv("cluster_map_valid.csv", index=False)

    # Find the cluster assignments for dcm molecules


    for cluster in cluster_map_valid['cluster'].unique():
        # for each cluster, training molecules there:
        print(f'the cluster is: {cluster}')
        cluster_rows = cluster_map_train[cluster_map_train['cluster'] == cluster]
        print(cluster_rows)
        # for each cluster, dcm molecules there:
        dcm_cluster_rows = dcm_cluster_map[dcm_cluster_map['cluster'] == cluster]
        # for each cluster, descriptors of training molecules there:
        cluster_data = cluster_rows.drop(['identity', 'cluster', 'target'], axis=1).values
        print(cluster_data)
        # for each cluster, descriptors of dcm there:
        dcm_cluster_data = dcm_cluster_rows.drop(['cluster'], axis=1).values
        # for each cluster, distance of training data and ALL dcm:
        print('here')
        print(cluster_data.shape)
        print(dcm.shape)
        distances_to_cluster_dcm = cdist(cluster_data, dcm, metric='euclidean')
        print(distances_to_cluster_dcm)
        print(distances_to_cluster_dcm.shape)
        # for each cluster, distance of training data and dcm also in same cluster:
        distances_to_same_cluster_dcm = cdist(cluster_data, dcm_cluster_data, metric='euclidean')
        print(distances_to_same_cluster_dcm)
        print(distances_to_same_cluster_dcm.shape)
        # for each cluster, validation molecules there:
        molecules_cluster = cluster_map_valid[cluster_map_valid['cluster'] == cluster]
        # for each cluster, descriptors of validation molecules there:
        mols_descriptors = molecules_cluster.drop(['identity', 'cluster', 'target'], axis=1)
        # for each cluster, distance of training data and validation data:
        distances_to_cluster = cdist(cluster_data, mols_descriptors, metric='euclidean')
        print(distances_to_cluster)
        print(distances_to_cluster.shape)
        # dictionary of distances betwen training data and validation data with cluster information:


        # create a new DataFrame for each row in cluster_map_valid
        molecule_val = molecules_cluster['identity']
        molecule_val.reset_index(drop=True, inplace=True)
        print(molecule_val)
        target_label = molecules_cluster['target']
        target_label.reset_index(drop=True, inplace=True)
        print(target_label) 
        # Retrieve the training molecules inside the cluster
        molecule_train = cluster_rows['identity']
        molecule_train.reset_index(drop=True, inplace=True)
        print(molecule_train)
        target_train = cluster_rows['target']
        target_train.reset_index(drop=True, inplace=True)
        print(target_train)


        # mean and std of the distances of dcm (same cluster and all) for each molecule of the training data
        mean_distance_dcm_same_cluster = distances_to_same_cluster_dcm.mean(axis=1)
        std_distance_dcm_same_cluster = distances_to_same_cluster_dcm.std(axis=1)

        mean_distance_dcm_cluster = distances_to_cluster_dcm.mean(axis=1)
        std_distance_dcm_cluster = distances_to_cluster_dcm.std(axis=1)

        print(mean_distance_dcm_same_cluster)
        print(mean_distance_dcm_same_cluster.shape)

        print(mean_distance_dcm_cluster)
        print(mean_distance_dcm_cluster.shape)
        molecules_dfs = {}
        molecules_dfs = pd.DataFrame(columns=['identity', 'target', 'molecule_within_cluster', 'target_train', 'p-value', 'p-value_cluster'])

        for i in range(len(distances_to_cluster)):
            #iterate over each line i, over each training element
            print(f'i is: {i}')
            # get the mean and std values of dcm distances for each target, both same cluster dcm and all dcm
            mol_train = molecule_train[i]
            targ_train = target_train[i]
            print(mol_train)
            print(targ_train)
            mean_distance_same_cluster = float(mean_distance_dcm_same_cluster[i])
            std_distance_same_cluster = float(std_distance_dcm_same_cluster[i])
            print(mean_distance_same_cluster)
            mean_distance_all_dcm = float(mean_distance_dcm_cluster[i])
            std_distance_all_dcm = float(std_distance_dcm_cluster[i])
            print(mean_distance_all_dcm)
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
                if std_distance_same_cluster != 0:
                    z_score_same_cluster = -1*((distance - mean_distance_same_cluster) / std_distance_same_cluster)     
                    p_value_same_cluster = scipy.stats.norm.sf(z_score_same_cluster)
                    
                else:
                    z_score_same_cluster = 'no dcm molecule in same cluster'  
                    p_value_same_cluster = 'no dcm molecule in same cluster'  
                z_score_all_dcm = -1*((distance - mean_distance_all_dcm) / std_distance_all_dcm)
                p_value_all_dcm = scipy.stats.norm.sf(z_score_all_dcm)
                print(p_value_same_cluster)
                print(p_value_all_dcm)
                # Add the results to the prediction list
                molecules_dfs = molecules_dfs.append({'identity': iden_valid, 'target': targ_valid, 'molecule_within_cluster': mol_train, 'target_train': targ_train, 'p-value': p_value_all_dcm, 'p-value_cluster': p_value_same_cluster}, ignore_index=True)


        # Initialize variables
        print(molecules_dfs)
        print(molecules_dfs.shape)

        # Loop over results and dataframes
        for identity in molecules_dfs['identity'].unique():
            counts = {'true_positive': 0, 'true_positive_cluster': 0, 'false_negative': 0, 'false_negative_cluster':0, 'significant_targets': [], 'significant_targets_cluster': [], 'non_significant_targets': [], 'non_significant_targets_cluster': []}
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
            numeric_rows = val[pd.to_numeric(val['p-value_cluster'], errors='coerce').notna()]
            significant_cluster_rows = numeric_rows.loc[numeric_rows['p-value_cluster'] <= 0.05]
            non_significant_cluster_rows = numeric_rows.loc[numeric_rows['p-value_cluster'] > 0.05]
            print(significant_cluster_rows)
            counts['significant_targets_cluster'].append(list(significant_cluster_rows['target_train']))
            counts['non_significant_targets_cluster'].append(list(non_significant_cluster_rows['target_train']))
            accurate_rows = val.loc[(val['p-value'] <= 0.05) & (val['target_train'] == targ_valid)]
            accurate_rows_ns = val.loc[(val['p-value'] > 0.05) & (val['target_train'] == targ_valid)]
            counts['true_positive'] += len(accurate_rows)
            counts['false_negative'] += len(accurate_rows_ns)
            accurate_cluster_rows = numeric_rows.loc[(numeric_rows['p-value_cluster'] <= 0.05) & (numeric_rows['target_train'] == targ_valid)]
            accurate_cluster_rows_ns = numeric_rows.loc[(numeric_rows['p-value_cluster'] > 0.05) & (numeric_rows['target_train'] == targ_valid)]
            counts['true_positive_cluster'] += len(accurate_cluster_rows)
            counts['false_negative_cluster'] += len(accurate_cluster_rows_ns)
            # Create predictions row
            true_positive = 'yes' if counts['true_positive'] > 0 else 'no'
            false_negative = 'yes' if counts['false_negative'] > 0 else 'no'

            true_positive_cluster = 'yes' if counts['true_positive_cluster'] > 0 else 'no'
            false_negative_cluster = 'yes' if counts['false_negative_cluster'] > 0 else 'no'
            predictions_row = pd.DataFrame([[identity, label, counts['significant_targets'], counts['significant_targets_cluster'], counts['non_significant_targets'], counts['non_significant_targets_cluster'],  true_positive, true_positive_cluster, false_negative, false_negative_cluster]],
                                   columns=['identity', 'label', 'significant_targets', 'significant_targets_cluster', 'non_significant_targets', 'non_significant_targets_cluster', 'true_positive', 'true_positive_cluster', 'false_negative', 'false_negative_cluster'])
            #Add row to predictions dataframe
            predictions_df = pd.concat([predictions_df, predictions_row], ignore_index=True)


    # Save predictions dataframe to csv
    predictions_df.to_csv("predictions.csv", index=False)
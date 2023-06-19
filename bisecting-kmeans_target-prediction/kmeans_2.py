import pandas as pd
import numpy as np
import csv
from joblib import dump
from sklearn.cluster import BisectingKMeans
import subprocess
import scipy
from scipy.spatial import distance
from scipy.spatial.distance import cdist



subprocess.run(['python', 'cats.py', 'mols.csv', 'mols2.csv', 'output.txt', 'output2.txt'])

seed=42
molecules_dfs = {}

#load input data
df = pd.read_csv('input_data.csv', delimiter=",")
molecule = df['molecule']
target = df['target']
X = np.array(df.drop(['target', 'molecule', 'activity'], axis=1))
X = X.copy(order='C')
#initialize KMeans model
n_clusters=int(len(molecule)/100)
print(f'n_cluster: {n_clusters}')
kmeans = BisectingKMeans(n_clusters=n_clusters, max_iter=300, n_init=10, random_state=42)
kmeans.fit(X)

new_molecules = np.loadtxt('output.txt')
print(new_molecules)
combined_data = np.vstack((X, new_molecules))
print(combined_data)
combined_cluster_assign = kmeans.predict(combined_data)
print(combined_cluster_assign)
# Split the cluster assignments back into X_train_cluster_assign and dcm_cluster_assign
cluster_assignments = combined_cluster_assign[:len(X)]
new_molecules_cluster_assignments = combined_cluster_assign[len(X):]

# find the cluster assignments for the data points
cluster_assignments = cluster_assignments.reshape(-1)
print(cluster_assignments)
print(cluster_assignments.shape)
new_molecules_cluster_assignments=new_molecules_cluster_assignments.reshape(-1)
print(new_molecules_cluster_assignments)
print(new_molecules_cluster_assignments.shape)

# Create cluster map dataframe
cluster_map = pd.DataFrame({'identity': molecule, 'cluster': cluster_assignments, 'target': target})
X = pd.DataFrame(X)
cluster_map = pd.concat([cluster_map, X], axis=1)
# Save cluster map to file
cluster_map.to_csv("cluster_map_2.csv", index=False)

new_molecules_cluster_map = pd.DataFrame({'cluster': new_molecules_cluster_assignments})
new_molecules = pd.DataFrame(new_molecules)
new_molecules_cluster_map = pd.concat([new_molecules_cluster_map, new_molecules], axis=1)
# Save cluster map to file
new_molecules_cluster_map.to_csv("new_molecules_cluster_map_2.csv", index=False)

for cluster in new_molecules_cluster_map['cluster'].unique():
    # for each molecule cluster, the new molecules there
    print(f'the cluster is: {cluster}')
    new_molecules_cluster_rows = new_molecules_cluster_map[new_molecules_cluster_map['cluster'] == cluster]
    # for each cluster, trainign molecules there:
    cluster_rows = cluster_map[cluster_map['cluster'] == cluster]
    new_molecules_dfs = {}
    new_molecules_dfs = pd.DataFrame(columns=['code', 'cluster', 'molecule_within_cluster', 'target', 'distance_to_cluster', 'mean_distance_diff_targets', 'std_distance_diff_targets','z-score', 'p-value'])
    for target in cluster_rows['target'].unique():
        # for each cluster, training molecules there:
        print(f'the target is: {target}')
        cluster_rows_same_label = cluster_rows[cluster_rows['target'] == target]
        cluster_rows_diff_label = cluster_rows[cluster_rows['target'] != target]
        cluster_data = cluster_rows_same_label.drop(['identity', 'cluster', 'target'], axis=1).values
        cluster_data_diff_label = cluster_rows_diff_label.drop(['identity', 'cluster', 'target'], axis=1).values

        # for each cluster, descriptors of validation molecules there:
        new_moleucule_cluster_data = new_molecules_cluster_rows.drop(['cluster'], axis=1)
        # for each cluster and target, distance of training data and validation data:
        distances_to_cluster = cdist(cluster_data, new_moleucule_cluster_data, metric='euclidean')
        print(distances_to_cluster)
        print(distances_to_cluster.shape)
        # for each cluster and target, distance of training data with the label and the training data without the label:
        distances_to_cluster_background = cdist(cluster_data, cluster_data_diff_label, metric='euclidean')
        print(distances_to_cluster)
        print(distances_to_cluster.shape)

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
                code = f'{j}_{cluster}'
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
                new_molecules_dfs = new_molecules_dfs.append({'code': code, 'cluster': cluster, 'molecule_within_cluster': mol_train, 'target': targ_train, 'distance_to_cluster': distance, 'mean_distance_diff_targets': mean_distance_background, 'std_distance_diff_targets': std_distance_background, 'z-score': z_score, 'p-value': p_value}, ignore_index=True)

    for code in new_molecules_dfs['code'].unique():
        predictions = new_molecules_dfs[new_molecules_dfs['code'] == code]
        # Save predictions dataframe to csv
        predictions.to_csv(f'moleclue_{code}_clusters_2.csv', index=False)

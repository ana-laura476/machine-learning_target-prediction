import pandas as pd
import numpy as np
import csv
from joblib import dump
from sklearn.cluster import KMeans
import faiss
import subprocess
import scipy
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform
subprocess.run(['python', 'cats.py', 'mols.csv', 'mols2.csv', 'output.txt', 'output2.txt'])
# initialize the Faiss index
class FaissKMeans:
    def __init__(self, n_clusters=4439, n_init=10, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]


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
kmeans = FaissKMeans(n_clusters=n_clusters, max_iter=300, n_init=10)
kmeans.fit(X_train)


# Find the cluster assignments for the training set
cluster_assign_train = kmeans.predict(X_train)
cluster_assign_train = cluster_assign_train.reshape(-1)

# Find the cluster assignments for the training set
cluster_assign_valid = kmeans.predict(X_valid)
cluster_assign_valid = cluster_assign_valid.reshape(-1)


# Create cluster map dataframe for the training set
cluster_map_train = pd.DataFrame({'identity': molecule_train, 'cluster': cluster_assign_train, 'target': target_train})
cluster_map_train.reset_index(drop=True, inplace=True)
X_train = pd.DataFrame(X_train)
X_train.reset_index(drop=True, inplace=True)
cluster_map_train = pd.concat([cluster_map_train, X_train], axis=1)
# Save cluster map to file
cluster_map_train.to_csv("cluster_map_train.csv", index=False)


# Create cluster map dataframe for the validation set
cluster_map_valid = pd.DataFrame({'identity': molecule_valid, 'cluster': cluster_assign_valid, 'target': target_valid})
cluster_map_valid.reset_index(drop=True, inplace=True)
X_valid = pd.DataFrame(X_valid)
X_valid.reset_index(drop=True, inplace=True)
cluster_map_valid = pd.concat([cluster_map_valid, X_valid], axis=1)
cluster_map_valid.to_csv("cluster_map_valid.csv", index=False)

# Find the cluster assignments for dcm molecules
dcm = np.loadtxt('output2.txt')
dcm_cluster_assign = kmeans.predict(dcm)
dcm_cluster_assign=dcm_cluster_assign.reshape(-1)
dcm_cluster_map = pd.DataFrame({'cluster': dcm_cluster_assign})
dcm = pd.DataFrame(dcm)
dcm_cluster_map = pd.concat([dcm_cluster_map, dcm], axis=1)
# Save cluster map to file
dcm_cluster_map.to_csv("dcm_cluster_map.csv", index=False)




# precompute the distances between all pairs of dcm and training molecules in each cluster and dcm molecules inside the same cluster
# precompute the distances between the validation data and the training of the same cluster
dcm_dist_cluster = {}
dcm_dist_same_cluster = {}
dists_cluster = {}
for cluster in cluster_map_train['cluster'].unique():
    # for each cluster, training molecules there:
    cluster_rows = cluster_map_train[cluster_map_train['cluster'] == cluster]
    print(cluster_rows)
    # for each cluster, dcm molecules there:
    dcm_cluster_rows = dcm_cluster_map[dcm_cluster_map['cluster'] == cluster]
    # for each cluster, descriptors of training molecules there:
    cluster_data = cluster_rows.drop(['identity', 'cluster', 'target'], axis=1).values
    # for each cluster, descriptors of dcm there:
    dcm_cluster_data = dcm_cluster_rows.drop(['cluster'], axis=1).values
    # for each cluster, distance of training data and ALL dcm:
    dist_cluster_dcm = cdist(cluster_data, dcm, metric='euclidean')
    print(dist_cluster_dcm)
    print(dist_cluster_dcm.shape)
    # for each cluster, distance of training data and dcm also in same cluster:
    dist_same_cluster_dcm = cdist(cluster_data, dcm_cluster_data, metric='euclidean')
    print(dist_same_cluster_dcm)
    print(dist_same_cluster_dcm.shape)
    # dictionary of distances betwen training data and all dcm with cluster information:
    dcm_dist_cluster[cluster] = dist_cluster_dcm
    # dictionary of distances betwen training data and dcm also in the same cluster with cluster information:
    dcm_dist_same_cluster[cluster] = dist_same_cluster_dcm
    # for each cluster, validation molecules there:
    molecules_cluster = cluster_map_valid[cluster_map_valid['cluster'] == cluster]
    # for each cluster, descriptors of validation molecules there:
    mols_descriptors = molecules_cluster.drop(['identity', 'cluster', 'target'], axis=1)
    # for each cluster, distance of training data and validation data:
    dist_cluster = cdist(cluster_data, mols_descriptors, metric='euclidean')
    print(dist_cluster)
    print(dist_cluster.shape)
    # dictionary of distances betwen training data and validation data with cluster information:
    dists_cluster[cluster] = dist_cluster

# create a new DataFrame for each row in cluster_map_valid
predictions_df = pd.DataFrame(columns=['identity', 'label', 'significant_targets', 'significant_targets_cluster', 'true_label_included', 'true_label_included_cluster'])
for clus in cluster_map_valid['cluster'].unique():
    print(f'the cluster is: {clus}')
    val_rows = cluster_map_valid[cluster_map_valid['cluster'] == clus]
    print(val_rows)
    print(val_rows.shape)
    molecule_val = val_rows['identity']
    molecule_val.reset_index(drop=True, inplace=True)
    print(molecule_val)
    target_label = val_rows['target']
    target_label.reset_index(drop=True, inplace=True)
    print(target_label) 
    # Retrieve the training molecules inside the cluster
    train_rows = cluster_map_train[cluster_map_train['cluster'] == clus]
    print(train_rows)
    print(train_rows.shape)
    molecule_train = train_rows['identity']
    molecule_train.reset_index(drop=True, inplace=True)
    print(molecule_train)
    target_train = train_rows['target']
    target_train.reset_index(drop=True, inplace=True)
    print(target_train)
    # Retrieve the dcm inside the cluster
    dcm_rows = dcm_cluster_map[dcm_cluster_map['cluster'] == clus]
    n_dcm_cluster = len(dcm_rows)

    # Get the distances between the validation molecules and training molecules in the same cluster
    distances_to_cluster = dists_cluster[clus]
    # Get the distances between the dcm molecule and  training molecules in the same cluster
    distances_to_same_cluster_dcm = dcm_dist_same_cluster[clus]
    # Get the distances between ALL dcm molecules and training molecules in the same cluster
    distances_to_cluster_dcm = dcm_dist_cluster[clus]
    print(distances_to_cluster)
    print(distances_to_cluster.shape)

    print(distances_to_same_cluster_dcm)
    print(distances_to_same_cluster_dcm.shape)

    print(distances_to_cluster_dcm)
    print(distances_to_cluster_dcm.shape)

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
            z_score_same_cluster = -1*((distance - mean_distance_same_cluster) / std_distance_same_cluster)
            z_score_all_dcm = -1*((distance - mean_distance_all_dcm) / std_distance_all_dcm)
            p_value_same_cluster = scipy.stats.norm.sf(z_score_same_cluster)
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
        counts = {'accurate_count': 0, 'accurate_count_cluster': 0, 'significant_targets': [], 'significant_targets_cluster': []}
        val = molecules_dfs[molecules_dfs['identity'] == identity]
        label = val['target'].iloc[0]
        print(val)
        print(val.shape)
        print(label)
        significant_rows = val.loc[val['p-value'] <= 0.05]
        print(significant_rows)
        counts['significant_targets'].append(list(significant_rows['target_train']))
        significant_cluster_rows = val.loc[val['p-value_cluster'] <= 0.05]
        print(significant_cluster_rows)
        counts['significant_targets_cluster'].append(list(significant_cluster_rows['target_train']))
        accurate_rows = val.loc[(val['p-value'] <= 0.05) & (val['target_train'] == targ_valid)]
        counts['accurate_count'] += len(accurate_rows)
        accurate_cluster_rows = val.loc[(val['p-value_cluster'] <= 0.05) & (val['target_train'] == targ_valid)]
        counts['accurate_count_cluster'] += len(accurate_cluster_rows)
        # Create predictions row
        include = 'yes' if counts['accurate_count'] > 0 else 'no'
        include_cluster = 'yes' if counts['accurate_count_cluster'] > 0 else 'no'
        predictions_row = pd.DataFrame([[identity, label, counts['significant_targets'], counts['significant_targets_cluster'], include, include_cluster]],
                               columns=['identity', 'label', 'significant_targets', 'significant_targets_cluster', 'true_label_included', 'true_label_included_cluster'])
        #Add row to predictions dataframe
        predictions_df = pd.concat([predictions_df, predictions_row], ignore_index=True)


# Save predictions dataframe to csv
predictions_df.to_csv("predictions.csv", index=False)
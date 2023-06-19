import pandas as pd
import numpy as np
import csv
from joblib import dump
from sklearn.cluster import KMeans
import faiss
import subprocess
import scipy
from scipy.spatial import distance
from scipy.spatial.distance import cdist



subprocess.run(['python', 'cats.py', 'mols.csv', 'mols2.csv', 'output.txt', 'output2.txt'])

# initialize the Faiss index
class FaissKMeans:
    def __init__(self, n_clusters=4439, n_init=10, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = seed
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
molecule = df['molecule']
target = df['target']
X = np.array(df.drop(['target', 'molecule', 'activity'], axis=1))
X = X.copy(order='C')
#initialize KMeans model
n_clusters=int(len(molecule)/100)
print(f'n_cluster: {n_clusters}')
kmeans = FaissKMeans(n_clusters=n_clusters, max_iter=300, n_init=10, random_state=seed)
kmeans.fit(X)

# find the cluster assignments for the data points
cluster_assignments = kmeans.predict(X)
cluster_assignments = cluster_assignments.reshape(-1)

print(cluster_assignments)
print(cluster_assignments.shape)

# Create cluster map dataframe
cluster_map = pd.DataFrame({'identity': molecule, 'cluster': cluster_assignments, 'target': target})
X = pd.DataFrame(X)

cluster_map = pd.concat([cluster_map, X], axis=1)

# Save cluster map to file
cluster_map.to_csv("cluster_map.csv", index=False)
molecules_dfs = {}

dcm = np.loadtxt('output2.txt')
dcm_cluster_assignments = kmeans.predict(dcm)
dcm_cluster_assignments=dcm_cluster_assignments.reshape(-1)
print(dcm_cluster_assignments)
print(dcm_cluster_assignments.shape)
dcm_cluster_map = pd.DataFrame({'cluster': dcm_cluster_assignments})
dcm = pd.DataFrame(dcm)
dcm_cluster_map = pd.concat([dcm_cluster_map, dcm], axis=1)
dcm_cluster_map.to_csv("dcm_cluster_map.csv", index=False)

new_molecules = np.loadtxt('output.txt')
new_molecules_cluster_assignments = kmeans.predict(new_molecules)
new_molecules_cluster_assignments=new_molecules_cluster_assignments.reshape(-1)
new_molecules_cluster_map = pd.DataFrame({'cluster': new_molecules_cluster_assignments})
new_molecules = pd.DataFrame(new_molecules)
new_molecules_cluster_map = pd.concat([new_molecules_cluster_map, new_molecules], axis=1)
# Save cluster map to file
new_molecules_cluster_map.to_csv("new_molecules_cluster_map.csv", index=False)

for cluster in new_molecules_cluster_map['cluster'].unique():
    # for each molecule cluster, the new molecules there
    print(f'the cluster is: {cluster}')
    new_molecules_cluster_rows = new_molecules_cluster_map[new_molecules_cluster_map['cluster'] == cluster]
    # for each cluster, trainign molecules there:
    cluster_rows = cluster_map[cluster_map['cluster'] == cluster]
    # for each cluster, dcm molecules there:
    dcm_cluster_rows = dcm_cluster_map[dcm_cluster_map['cluster'] == cluster]
    n_dcm_cluster = len(dcm_cluster_rows)

    # for each cluster, descriptors of training molecules there:
    cluster_data = cluster_rows.drop(['identity', 'cluster', 'target'], axis=1).values
    # for each cluster, descriptors of new molecules there:
    new_moleucule_cluster_data = new_molecules_cluster_rows.drop(['cluster'], axis=1).values
    # for each cluster, descriptors of dcm there:
    dcm_cluster_data = dcm_cluster_rows.drop(['cluster'], axis=1).values

    # for each cluster, distance of training data and ALL dcm:
    distances_to_cluster_dcm = cdist(cluster_data, dcm, metric='euclidean')
    print(distances_to_cluster_dcm)
    print(distances_to_cluster_dcm.shape)
    # for each cluster, distance of training data and dcm also in same cluster:
    distances_to_same_cluster_dcm = cdist(cluster_data, dcm_cluster_data, metric='euclidean')
    print(distances_to_same_cluster_dcm)
    print(distances_to_same_cluster_dcm.shape)
    # for each cluster, distance of training data and new molecules:
    distances_to_cluster = cdist(cluster_data, new_moleucule_cluster_data, metric='euclidean')
    print(distances_to_cluster)
    print(distances_to_cluster.shape)
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
    # Save the new molecule dataframes to CSV files outside the inner loop
    new_molecules_dfs = {}
    new_molecules_dfs = pd.DataFrame(columns=['code', 'cluster', 'n_dcm_cluster', 'molecule_within_cluster', 'target', 'distance_to_cluster', 'dcm_mean_distance', 'dcm_std_distance', 'dcm_same_cluster_mean_distance', 'dcm_same_cluster_std_distance', 'z-score', 'p-value', 'z-score_cluster', 'p-value_cluster'])

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
        # Loop over each new molecule entry in the row i, one new molecule-one training molecule, access information with [i][j]
        for j in range(len(distances_to_cluster[i])):
            print(f'j is: {j}')
            code = f'{j}_{cluster}'
            # Compute the z-score for the j-th distance between row i and the validation entry j
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
            new_molecules_dfs = new_molecules_dfs.append({'code': code, 'cluster': cluster, 'n_dcm_cluster': n_dcm_cluster, 'molecule_within_cluster': mol_train, 'target': targ_train, 'distance_to_cluster': distance, 'dcm_mean_distance': mean_distance_all_dcm, 'dcm_std_distance': std_distance_all_dcm, 'dcm_same_cluster_mean_distance': mean_distance_same_cluster, 'dcm_same_cluster_std_distance': std_distance_same_cluster, 'z-score': z_score_all_dcm, 'p-value': p_value_all_dcm, 'z-score_cluster': z_score_same_cluster, 'p-value_cluster': p_value_same_cluster}, ignore_index=True)

    for code in new_molecules_dfs['code'].unique():
        predictions = new_molecules_dfs[new_molecules_dfs['code'] == code]
        # Save predictions dataframe to csv
        predictions.to_csv(f'moleclue_{code}_clusters.csv', index=False)

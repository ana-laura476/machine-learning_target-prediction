import pandas as pd
import numpy as np
import csv
from joblib import dump
from sklearn.cluster import KMeans
import faiss
import subprocess
import scipy
from scipy.spatial import distance



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
molecule = df['molecule']

print(molecule)
print(molecule.shape)
target = df['target']

print(target)
print(target.shape)
X = np.array(df.drop(['target', 'molecule', 'activity'], axis=1))
X = X.copy(order='C')
print(X)
print(X.shape)
#initialize KMeans model 
print('kmeans ok1')
n_clusters=int(len(molecule)/100)
print(f'n_cluster: {n_clusters}')
kmeans = FaissKMeans(n_clusters=n_clusters, max_iter=300, n_init=10)
print('kmeans ok2')
kmeans.fit(X)

print('kmeans ok3')
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

#load the new molecules to predict
with open('output.txt') as f:
    for i, line in enumerate(f):
        molecule_descriptor = [float(x) for x in line.split()]
        molecule_descriptor = np.array(molecule_descriptor).reshape(1, -1)
        print(molecule_descriptor[0])
        print(molecule_descriptor[0].shape)
        molecule_cluster = kmeans.predict(molecule_descriptor)
        molecule_cluster = int(molecule_cluster[0])
        print(f"molecule_cluster is {molecule_cluster}")
        #get distance to new molecule and backgroud noise with dark chemical matter (dcm)

        cluster_rows = cluster_map[cluster_map['cluster'] == molecule_cluster]
        print(cluster_rows)
        print(cluster_rows.shape)
        for index, row in cluster_rows.iterrows():
            print(row)
            iden=row['identity']
            clus=row['cluster']
            tar=row['target']
            cluster_descriptor = np.array(row.drop(['identity', 'cluster', 'target']))
            print(cluster_descriptor)
            print(cluster_descriptor.shape)
            distances = distance.euclidean(molecule_descriptor[0], cluster_descriptor)
            #get backgroud noise with dark chemical matter (dcm)
            distance_to_cluster_dcm = []
            with open('output2.txt') as g:
                for k, line in enumerate(g):
                    dcm_descriptor = [float(x) for x in line.split()]
                    dcm_descriptor = np.array(dcm_descriptor).reshape(1, -1)
                    distance_dcm = distance.euclidean(dcm_descriptor[0], cluster_descriptor)
                    distance_to_cluster_dcm.append(distance_dcm)
                    
            mean_distance_dcm=np.mean(distance_to_cluster_dcm)          
            std_distance_dcm=np.std(distance_to_cluster_dcm)
            #get p-vale for each molecule of the cluster
            z_score = abs((float(distances) - float(mean_distance_dcm))/float(std_distance_dcm))
            print(z_score)
            p_value = scipy.stats.norm.sf(z_score)
            print(p_value)
            molecules = f'molecules{i}'
            if molecules not in molecules_dfs:
                molecules_dfs[molecules] = pd.DataFrame(columns=['cluster', 'molecule_within_cluster', 'target', 'distance to cluster', 'dcm mean distance', 'dcm std distance', 'z-score', 'p-value'])
            molecules_dfs[molecules] = molecules_dfs[molecules].append({'cluster': clus,'molecule_within_cluster': iden, 'target': tar, 'distance to cluster': distances, 'dcm mean distance': mean_distance_dcm, 'dcm std distance': std_distance_dcm, 'z-score': z_score, 'p-value': p_value}, ignore_index=True)
         
          
        for molecules in molecules_dfs:
            molecules_dfs[molecules].to_csv(f'{molecules}_clusters.csv', index=False)


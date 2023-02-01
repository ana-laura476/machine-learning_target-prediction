import pandas as pd
import numpy as np
import os 
import csv
from joblib import dump
from sklearn.cluster import KMeans
import subprocess
import scipy
import scipy.stats as stats
import scipy.spatial.distance as dist


seed=42

#load input data
df = pd.read_csv('input_data.csv', delimiter=",")
target = df['target']
molecule = df['molecule']
activity = df['activity']
X = df.drop(['target', 'molecule', 'activity'], axis=1)
print(X)
print(X.shape)
num_clusters=4439

#load the molecules and input data
subprocess.run(['python', 'cats.py', 'mols.csv', 'mols2.csv', 'output.txt', 'output2.txt'])

#initialize KMeans model 
kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
kmeans.fit(X)
cluster_assignments = kmeans.labels_
cluster_map = pd.DataFrame(columns=['identity', 'cluster', 'target'])
for i, row in enumerate(X.values):
    cluster_map = cluster_map.append({'identity': molecule[i], 'cluster': cluster_assignments[i], 'target': target[i]}, ignore_index=True)
cluster_map = pd.concat([cluster_map, X], axis=1)
cluster_map.to_csv("cluster_map.csv", index=False)



#load the new molecules to predict
with open('output.txt') as f:
    for i, line in enumerate(f):
        molecule_descriptor = [float(x) for x in line.split()]
        molecule_descriptor = np.array(molecule_descriptor).reshape(1, -1)
        molecule_cluster = kmeans.predict(molecule_descriptor)
        
        for index, row in cluster_map[cluster_map.cluster == molecule_cluster].iterrows():
            cluster_descriptor = row.drop(['identity', 'cluster', 'target'], axis=1)
            distance = dist.euclidean(molecule_descriptor, cluster_descriptor)
            distance_to_cluster = np.mean(distance)
        
        #get backgroud noise with dark chemical matter (dcm)
        
        with open('output2.txt') as f:
            for i, line in enumerate(f):
                dcm_descriptor = [float(x) for x in line.split()]
                dcm_descriptor = np.array(dcm_descriptor).reshape(1, -1)
                distance_to_cluster_dcm = []
                
                for index, row in cluster_map[cluster_map.cluster == molecule_cluster].iterrows():
                    cluster_descriptor = row.drop(['identity', 'cluster', 'target'], axis=1)
                    distance_dcm = dist.euclidean(dcm_descriptor, cluster_descriptor)
                    distance_to_cluster_dcm.append(np.mean(distance_dcm))
                mean_distance_dcm = np.mean(distance_to_cluster_dcm)
                std_distance_dcm = np.std(mean_distance_dcm)

                #get p-vale for each molecule of the cluster
                z_score = (float(distance_to_cluster) - float(mean_distance_dcm)/float(std_distance_dcm))
                print(z_score)
                p_value = scipy.stats.norm.sf(z_score)
                print(p_value)
 
                molecules = f'molecules{i}'
                if molecules not in molecules_dfs:
                    molecules_df = pd.DataFrame(columns=['cluster', 'molecule_within_cluster', 'target', 'distance', 'dcm_mean_distance', 'dcm_std_distance', 'z-score', 'p-value'])
                molecules_df = molecule_df.append({'cluster': molecule_cluster,'molecule_within_cluster': molecule, 'target': target, 'distance to cluster': distance_to_cluster, 'dcm mean distance': mean_distance_dcm, 'dcm std distance': std_distance_dcm, 'z-score': z_score, 'p-value': p_value}, ignore_index=True)
 
          
molecules_df.to_csv("molecule_clusters.csv", index=False)

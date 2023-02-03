from ugtm import eGTM
import pandas as pd
import numpy as np
import os 
import csv
import numpy as np
import subprocess
import pickle
from scipy.spatial import distance


seed=42
subprocess.run(['python', 'cats.py', 'mols.csv', 'mols2.csv', 'output.txt', 'output2.txt'])

#load input data
df = pd.read_csv('input_data.csv', delimiter=",")
X = df.drop(['target', 'molecule', 'activity'], axis=1)
print(X)
print(X.shape)

k=70
gtm_model = eGTM(k=k, m=0, niter=300, random_state=seed, model='means') 
gtm_model.fit(X)
transform = gtm_model.transform(X)


prototypes = gtm_model.prototypes_


#load the new molecules to predict

with open('output.txt') as f:
    for i, line in enumerate(f):
        molecule_descriptor = [float(x) for x in line.split()]
        molecule_descriptor = np.array(molecule_descriptor).reshape(1, -1)
        molecule_transform = gtm_model.transform(molecule_descriptor)
        distances = distance.cdist(prototypes, molecule_transform, 'euclidean')
        indices = np.argsort(distances)[:100]
        close_molecules = df.iloc[indices, :]
        closest_molecules = close_molecules.drop(['target', 'molecule', 'activity'])

        for index, row in closest_molecules.iterrows():
            e_distance = dist.euclidean(molecule_descriptor[0], row.values)
            #get backgroud noise with dark chemical matter (dcm)

            with open('output2.txt') as f:
                for i, line in enumerate(f):
                    dcm_descriptor = [float(x) for x in line.split()]
                    dcm_descriptor = np.array(dcm_descriptor).reshape(1, -1)
                    distance_to_mol_dcm = []
                    distance_dcm = dist.euclidean(dcm_descriptor[0], row.values)
                    distance_to_mol_dcm.append(distance_dcm)
                    mean_distance_dcm=np.mean(distance_to_mol_dcm)          
                    std_distance_dcm=np.std(distance_to_mol_dcm)
                    #get p-vale for each molecule of the cluster
                    z_score = (float(distance) - float(mean_distance_dcm))/float(std_distance_dcm)
                    print(z_score)
                    p_value = scipy.stats.norm.sf(z_score)
                    print(p_value)
                    molecules = f'molecules{i}'
                    if molecules not in molecules_dfs:
                        molecules_dfs[molecules] = pd.DataFrame(columns=['closest molecule', 'target', 'distance', 'dcm_mean_distance', 'dcm_std_distance', 'z-score', 'p-value'])
                    molecules_dfs[molecules] = molecules_dfs.append({'closest molecule': close_molecules['molecule'], 'target': close_molecule['target'], 'distance': e_distance, 'dcm mean distance': mean_distance_dcm, 'dcm std distance': std_distance_dcm, 'z-score': z_score, 'p-value': p_value}, ignore_index=True)
     
          
for molecules in molecule_dfs:
    molecules_dfs[molecules].to_csv(f'{molecules}_clusters.csv', index=False)
